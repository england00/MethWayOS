import datetime
import os
import sys
import time
import wandb
import yaml
import numpy as np
import torch.cuda
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
import warnings
from logs.methods.log_storer import DualOutput
from sksurv.metrics import concordance_index_censored
from src.mcat.modules.loss import CrossEntropySurvivalLoss, SurvivalClassificationTobitLoss
from src.mcat.modules.utils import l1_reg
from src.mcat.modules.mcat import MultimodalCoAttentionTransformer
from src.mcat.modules.dataset import MultimodalDataset
from torch.utils.data import DataLoader


## CONFIGURATION
LOG_PATH = f'../../logs/files/{os.path.basename(__file__)}.txt'
CLASSES_NUMBER = 20
BATCH_SIZE = 1


## FUNCTIONS
def title(text):
    print('\n' + f' {text} '.center(80, '#'))


''' TRAINING DEFINITION '''
def train(epoch, config, device, train_loader, model, loss_function, optimizer, scheduler, reg_function):
    model.train()
    grad_acc_step = config['training']['grad_acc_step']
    use_scheduler = config['training']['scheduler']
    lambda_reg = config['training']['lambda']
    checkpoint_epoch = config['model']['checkpoint_epoch']
    train_loss = 0.0
    risk_scores = torch.zeros(len(train_loader), device=device)
    censorships = torch.zeros(len(train_loader), device=device)
    event_times = torch.zeros(len(train_loader), device=device)
    start_batch_time = time.time()
    for batch_index, (survival_months, survival_class, censorship, omics_data, patches_embeddings) in enumerate(
            train_loader):

        # print(survival_class)
        survival_months = survival_months.to(device, non_blocking=True)
        survival_class = survival_class.to(device, non_blocking=True)
        survival_class = survival_class.unsqueeze(0).to(torch.int64)
        censorship = censorship.type(torch.FloatTensor).to(device, non_blocking=True)
        patches_embeddings = patches_embeddings.to(device)
        omics_data = [omic_data.to(device) for omic_data in omics_data]
        hazards, survs, Y, attention_scores = model(wsi=patches_embeddings, omics=omics_data)

        # Choosing Loss Function
        if config['training']['loss'] == 'ce':
            loss = loss_function(Y, survival_class.long())
        elif config['training']['loss'] == 'ces':
            loss = loss_function(hazards, survs, survival_class, c=censorship)
        elif config['training']['loss'] == 'sct':
            loss = loss_function(Y, survival_class, c=censorship)
        else:
            raise RuntimeError(f'Loss "{config["training"]["loss"]}" not implemented')
        loss_value = loss.item()

        if reg_function is None:
            loss_reg = 0
        else:
            loss_reg = reg_function(model) * lambda_reg

        risk = -torch.sum(survs, dim=1)
        risk_scores[batch_index] = risk
        censorships[batch_index] = censorship
        event_times[batch_index] = survival_months

        train_loss += loss_value + loss_reg

        if (batch_index + 1) % 50 == 0:

            print(f'--> Batch: {batch_index}, Loss: {loss_value + loss_reg:.4f}, Label: {survival_class.item()}, Survival Months: {survival_months.item():.2f}, Risk: {float(risk.item()):.4f}')
            end_batch_time = time.time()
            print(f'\t--> Average Speed: {((end_batch_time - start_batch_time) / 32):.2f}s per batch')
            print(f'\t--> Time from Last Check: {(end_batch_time - start_batch_time):.2f}s')
            start_batch_time = time.time()
        loss = loss / grad_acc_step + loss_reg
        loss.backward()

        if (batch_index + 1) % grad_acc_step == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Calculate loss and error for epoch
    train_loss /= len(train_loader)
    risk_scores = risk_scores.detach().cpu().numpy()
    censorships = censorships.detach().cpu().numpy()
    event_times = event_times.detach().cpu().numpy()
    c_index = concordance_index_censored((1 - censorships).astype(bool), event_times, risk_scores)[0]
    if use_scheduler:
        lr = optimizer.param_groups[0]["lr"]
        if scheduler == 'exp':
            scheduler.step()
        if scheduler == 'rop':
            scheduler.step(train_loss)
        print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Epoch: {epoch + 1}, lr: {lr:.8f}, train_loss: {train_loss:.4f}, train_c_index: {c_index:.4f}')
    else:
        print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Epoch: {epoch + 1}, train_loss: {train_loss:.4f}, train_c_index: {c_index:.4f}')
    if checkpoint_epoch > 0:
        if (epoch + 1) % checkpoint_epoch == 0 and epoch != 0:
            now = datetime.datetime.now().strftime('%Y%m%d%H%M')
            filename = f'{config["model"]["name"]}_{config["dataset"]["name"]}_E{epoch + 1}_{now}.pt'
            checkpoint_dir = config['model']['checkpoint_dir']
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            print(f'--> Saving model into {checkpoint_path}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
    wandb_enabled = config['wandb']['enabled']
    if wandb_enabled:
        wandb.log({"train_loss": train_loss, "train_c_index": c_index})


''' VALIDATION DEFINITION '''
def validate(epoch, config, device, val_loader, model, loss_function, reg_function):
    model.eval()
    val_loss = 0.0
    lambda_reg = config['training']['lambda']
    risk_scores = np.zeros((len(val_loader)))
    censorships = np.zeros((len(val_loader)))
    event_times = np.zeros((len(val_loader)))
    for batch_index, (survival_months, survival_class, censorship, omics_data, patches_embeddings) in enumerate(
            val_loader):
        survival_months = survival_months.to(device)
        survival_class = survival_class.to(device)
        survival_class = survival_class.unsqueeze(0).to(torch.int64)
        censorship = censorship.type(torch.FloatTensor).to(device)
        patches_embeddings = patches_embeddings.to(device)
        omics_data = [omic_data.to(device) for omic_data in omics_data]
        with torch.no_grad():
            hazards, survs, Y, attention_scores = model(wsi=patches_embeddings, omics=omics_data)

        if config['training']['loss'] == 'ce':
            loss = loss_function(Y, survival_class.long())
        elif config['training']['loss'] == 'ces':
            loss = loss_function(hazards, survs, survival_class, c=censorship)
        elif config['training']['loss'] == 'sct':
            loss = loss_function(Y, survival_class, c=censorship)
        else:
            raise RuntimeError(f'--> Loss "{config["training"]["loss"]}" not implemented')
        loss_value = loss.item()

        if reg_function is None:
            loss_reg = 0
        else:
            loss_reg = reg_function(model) * lambda_reg

        risk = -torch.sum(survs, dim=1).cpu().numpy()
        risk_scores[batch_index] = risk.item()
        censorships[batch_index] = censorship.item()
        event_times[batch_index] = survival_months.item()

        val_loss += loss_value + loss_reg

    # Calculating Loss and Error
    val_loss /= len(val_loader)
    c_index = concordance_index_censored((1 - censorships).astype(bool), event_times, risk_scores)[0]
    if epoch == 'final validation':
        print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Epoch: {epoch}, val_loss: {val_loss:.4f}, val_c_index: {c_index:.4f}')
    else:
        print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Epoch: {epoch}, val_loss: {val_loss:.4f}, val_c_index: {c_index:.4f}')
    wandb_enabled = config['wandb']['enabled']
    if wandb_enabled:
        wandb.log({"val_loss": val_loss, "val_c_index": c_index})


''' TESTING DEFINITION '''
def test(config, device, epoch, val_loader, model, patient, save=False):
    model.eval()
    output_dir = config['training']['test_output_dir']
    model_name = config['model']['name']
    now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    for batch_index, (survival_months, survival_class, censorship, omics_data, patches_embeddings) in enumerate(
            val_loader):
        survival_months = survival_months.to(device)
        survival_class = survival_class.to(device)
        survival_class = survival_class.unsqueeze(0).to(torch.int64)
        censorship = censorship.type(torch.FloatTensor).to(device)
        patches_embeddings = patches_embeddings.to(device)
        omics_data = [omic_data.to(device) for omic_data in omics_data]
        print(f'[{batch_index}] Survival months: {survival_months.item()}, Survival class: {survival_class.item()}, '
              f'Censorship: {censorship.item()}')
        with torch.no_grad():
            hazards, survs, Y, attention_scores = model(wsi=patches_embeddings, omics=omics_data, inference=True)
            risk = -torch.sum(survs, dim=1).cpu().numpy()
            print(f'Hazards: {hazards}, Survs: {survs}, Risk: {risk}, Y: {Y}')
            print(f'Attn min: {attention_scores["coattn"].min()}, Attn max: {attention_scores["coattn"].max()}, Attn '
                  f'mean: {attention_scores["coattn"].mean()}')

            if save:
                output_file = os.path.join(output_dir, f'ATTN_{model_name}_{patient}_{now}_E{epoch}_{batch_index}.pt')
                print(f'Saving attention in {output_file}')
                torch.save(attention_scores['coattn'], output_file)


''' W&B CONFIGURATION '''
def wandb_init(config):
    slurm_job_name = os.getenv('SLURM_JOB_NAME', 'default_job_name')  # Slurm Job Name
    slurm_job_id = os.getenv('SLURM_JOB_ID', 'default_job_id')  # Slurm Job Id
    wandb.init(
        project=config['wandb']['project'],
        name=f"{slurm_job_name}_{slurm_job_id}",
        settings=wandb.Settings(init_timeout=300),
        config={
            'model': config['model']['name'],
            'dataset': config['dataset']['name'],
            'normalization': config['dataset']['normalize'],
            'standardization': config['dataset']['standardize'],
            'decider_only': config['dataset']['decider_only'],
            'tcga_only': config['dataset']['tcga_only'],
            'diagnostic_only': config['dataset']['diagnostic_only'],
            'optimizer': config['training']['optimizer'],
            'learning_rate': config['training']['lr'],
            'weight_decay': config['training']['weight_decay'],
            'gradient_acceleration_step': config['training']['grad_acc_step'],
            'epochs': config['training']['epochs'],
            'architecture': config['model']['name'],
            'fusion': config['model']['fusion'],
            'loss': config['training']['loss'],
            'scheduler': config['training']['scheduler'],
            'alpha': config['training']['alpha'],
            'lambda': config['training']['lambda'],
            'gamma': config['training']['gamma'],
            'model_size': config['model']['model_size'],
            'leave_one_out': config['training']['leave_one_out']
        }
    )


''' MAIN DEFINITION '''
def main(config_path: str):
    # Loading Configuration file
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Starting W&B
    wandb_enabled = config['wandb']['enabled']
    if wandb_enabled:
        print('WANDB: setting up for report')
        os.environ['WANDB__SERVICE_WAIT'] = '300'
        wandb_init(config)
    else:
        print('WANDB: disabled')

    # CUDA
    print('')
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA: not available')
        device = 'cpu'
    elif device == 'cuda' and torch.cuda.is_available():
        print('CUDA: available')
        print(f'--> Device count: {torch.cuda.device_count()}')
        for device_index in range(torch.cuda.device_count()):
            print(f'--> Using device: {torch.cuda.get_device_name(device_index)}')
    print(f'--> Running on {device.upper()}')

    # Loading Dataset
    print('')
    file_csv = config['dataset']['file']
    dataset = MultimodalDataset(file_csv,
                                config,
                                classes_number=CLASSES_NUMBER,
                                use_signatures=True,
                                remove_incomplete_samples=True)  # Dataset object
    train_size = config['training']['train_size']
    print(f'--> Using {int(train_size * 100)}% train, {100 - int(train_size * 100)}% validation')
    leave_one_out = config['training']['leave_one_out'] is not None  # Managing Leave One Out
    test_patient = config['training']['leave_one_out']
    train_dataset, val_dataset, test_dataset = dataset.split(train_size, test=leave_one_out, patient=test_patient)
    print(f'--> Training Set: [{len(train_dataset)}], Validation Set: [{len(val_dataset)}]')
    if test_dataset is not None:
        print(f'--> Leave One Out available: testing patient {test_patient}')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)
    output_attn_epoch = config['training']['output_attn_epoch']

    # Model
    print('')
    model_name = config['model']['name']
    print(f'MODEL: {model_name}')
    model_size = config['model']['model_size']
    omics_sizes = dataset.signature_sizes
    fusion = config['model']['fusion']
    model = MultimodalCoAttentionTransformer(model_size=model_size,
                                             n_classes=CLASSES_NUMBER,
                                             omic_sizes=omics_sizes,
                                             fusion=fusion,
                                             device=device)
    print(f'--> Trainable parameters of {model_name}: {model.get_trainable_parameters()}')
    checkpoint_path = config['model']['load_from_checkpoint']  # Loading previous checkpoint if specified
    checkpoint = None
    if checkpoint_path is not None:
        print(f'--> Loading model checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device=device)
    alpha = config['training']['alpha']

    # Loss function
    if config['training']['loss'] == 'ce':
        print('--> Loss function: CrossEntropyLoss')
        loss_function = nn.CrossEntropyLoss()
    elif config['training']['loss'] == 'ces':
        print('--> Loss function: CrossEntropySurvivalLoss')
        loss_function = CrossEntropySurvivalLoss(alpha=alpha)
    elif config['training']['loss'] == 'sct':
        print('--> Loss function: SurvivalClassificationTobitLoss')
        loss_function = SurvivalClassificationTobitLoss()
    else:
        raise RuntimeError(f'--> Loss function: "{config["training"]["loss"]}" not implemented')

    # Optimizer
    learning_rate = config['training']['lr']
    weight_decay = config['training']['weight_decay']
    optimizer_name = config['training']['optimizer']
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    elif optimizer_name == 'adadelta':
        optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adamax':
        optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer_name = 'adam'
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
    print(f'--> Optimizer: {optimizer_name}')

    # Scheduler for variable learning rate
    scheduler = config['training']['scheduler']
    if scheduler == 'exp':
        gamma = config['training']['gamma']
        scheduler = lrs.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler == 'rop':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)
    else:
        scheduler = None

    # Optimizer in a given checkpoint
    starting_epoch = 0
    if checkpoint_path is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']

    # Lambda parameter
    lambda_param = config['training']['lambda']
    if lambda_param:
        reg_function = l1_reg
    else:
        reg_function = None

    # Training
    title(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Training started')
    model.train()
    epochs = config['training']['epochs']
    for epoch in range(starting_epoch, epochs):
        print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Epoch: {epoch + 1}')
        start_time = time.time()
        train(epoch, config, device, train_loader, model, loss_function, optimizer, scheduler, reg_function)
        validate(epoch, config, device, val_loader, model, loss_function, reg_function)
        if leave_one_out:
            save = False
            if (epoch + 1) % output_attn_epoch == 0:
                save = True
            test_patient = config['training']['leave_one_out']
            test(config, device, epoch + 1, test_loader, model, test_patient, save=save)
        end_time = time.time()
        print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Time elapsed for epoch {epoch + 1}: {end_time - start_time:.0f}s')
    title(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Training started')

    # Validation
    validate('final validation', config, device, val_loader, model, loss_function, reg_function)

    # Ending W&B
    if wandb_enabled:
        wandb.finish()


## MAIN
if __name__ == '__main__':
    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Execution
    title(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - MCAT started')
    main('../../config/files/mcat.yaml')
    title(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - MCAT finished')

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
