import datetime
import os
import time
import wandb
import yaml
import numpy as np
import socket
import sys
import torch.cuda
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
import warnings
sys.path.append('/homes/linghilterra/AIforBioinformatics')
from logs.methods.log_storer import DualOutput
from src.mcat.gene_expression_and_methylation_functions.training_gene_expression_and_methylation import training
from src.mcat.gene_expression_and_methylation_functions.validation_gene_expression_and_methylation import validation
from src.mcat.original_modules.loss import CrossEntropySurvivalLoss, SurvivalClassificationTobitLoss
from src.mcat.original_modules.utils import l1_reg
from src.mcat.gene_expression_and_methylation_modules.mcat_gene_expression_and_methylation import MultimodalCoAttentionTransformer
from src.mcat.gene_expression_and_methylation_modules.dataset_gene_expression_and_methylation import MultimodalDataset
from torch.utils.data import DataLoader


## CONFIGURATION
''' General '''
C_INDEX_THRESHOLD = 0.65
LOG_PATH = f'../../logs/files/{os.path.basename(__file__)}.txt'
MCAT_MULTIMODAL_YAML = '../../config/files/mcat_gene_expression_and_methylation.yaml'
PID = None


## FUNCTIONS
def title(text):
    print('\n' + f' {text} '.center(80, '#'))


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
            'gene_expression': config['dataset']['gene_expression'],
            'gene_expression_signatures': config['dataset']['gene_expression_signatures'],
            'methylation': config['dataset']['methylation'],
            'methylation_signatures': config['dataset']['methylation_signatures'],
            'normalization': config['dataset']['normalize'],
            'standardization': config['dataset']['standardize'],
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
            'early_stopping_patience': config['training']['early_stopping_patience'],
            'batch_size': config['training']['batch_size'],
            'classes_number': config['training']['classes_number'],
        }
    )


''' MAIN DEFINITION '''
def main(config_path: str):
    # Loading Configuration file
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Managing Runs on Local Machine
    print(f"HOSTNAME: {socket.gethostname()}")
    if socket.gethostname() == 'DELL-XPS-15':
        config['wandb']['enabled'] = False
        process_id = f'{socket.gethostname()}_{os.getpid()}'
    else:
        process_id = f'{os.getenv("SLURM_JOB_NAME", "default_job_name")}_{os.getenv("SLURM_JOB_ID", "default_job_id")}'

    # Starting W&B
    print('')
    if config['wandb']['enabled']:
        print('WANDB: setting up for report')
        os.environ['WANDB__SERVICE_WAIT'] = '300'
        wandb_init(config)
    else:
        print('WANDB: disabled')

    # CUDA
    print('')
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        print('CUDA: not available')
        config['device'] = 'cpu'
    elif config['device'] == 'cuda' and torch.cuda.is_available():
        print('CUDA: available')
        print(f'--> Device count: {torch.cuda.device_count()}')
        for device_index in range(torch.cuda.device_count()):
            print(f'--> Using device: {torch.cuda.get_device_name(device_index)}')
    print(f'--> Running on {config["device"].upper()}')

    # Loading Dataset
    print('')
    dataset = MultimodalDataset(config,
                                classes_number=config['training']['classes_number'],
                                use_signatures=True,
                                remove_incomplete_samples=True)  # Dataset object
    print(f'--> Using {int(config["training"]["train_size"] * 100)}% training, {100 - int(config["training"]["train_size"] * 100)}% validation')

    # Splitting Dataset
    training_dataset, testing_dataset = dataset.split(config['training']['train_size'])
    print(f'--> Training Set: [{len(training_dataset)}], Testing Set: [{len(testing_dataset)}]')
    training_loader = DataLoader(training_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=6, pin_memory=True)
    validation_loader = DataLoader(testing_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=6, pin_memory=True)

    # Model
    print('')
    model_name = config['model']['name']
    print(f'MODEL: {model_name}')
    model = MultimodalCoAttentionTransformer(model_size=config['model']['model_size'],
                                             n_classes=config['training']['classes_number'],
                                             rnaseq_sizes=dataset.gene_expression_signature_sizes,
                                             meth_sizes=dataset.methylation_signature_sizes,
                                             dropout=config['training']['dropout'],
                                             fusion=config['model']['fusion'],
                                             device=config['device'])
    print(f'--> Trainable parameters of {model_name}: {model.get_trainable_parameters()}')

    # Starting from Checkpoint
    checkpoint = None
    if config['model']['load_from_checkpoint'] is not None:
        print(f'--> Loading model checkpoint from {config["model"]["load_from_checkpoint"]}')
        checkpoint = torch.load(config['model']['load_from_checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])

    # Moving Model on GPU
    if config['device'] == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device=config['device'])

    # Loss function
    if config['training']['loss'] == 'ce':
        print('--> Loss function: CrossEntropyLoss')
        loss_function = nn.CrossEntropyLoss()
    elif config['training']['loss'] == 'ces':
        print('--> Loss function: CrossEntropySurvivalLoss')
        loss_function = CrossEntropySurvivalLoss(alpha=config['training']['alpha'])
    elif config['training']['loss'] == 'sct':
        print('--> Loss function: SurvivalClassificationTobitLoss')
        loss_function = SurvivalClassificationTobitLoss()
    else:
        raise RuntimeError(f'--> Loss function: "{config["training"]["loss"]}" not implemented')

    # Optimizer
    if config['training']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['lr'])
    elif config['training']['optimizer'] == 'adadelta':
        optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    elif config['training']['optimizer'] == 'adamax':
        optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    else:
        config['training']['optimizer'] = 'adam'
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    print(f'--> Optimizer: {config["training"]["optimizer"]}')

    # Scheduler for variable learning rate
    if config['training']['scheduler'] == 'exp':
        gamma = config['training']['gamma']
        scheduler = lrs.ExponentialLR(optimizer, gamma=gamma)
    else:
        scheduler = None

    # Optimizer in a given checkpoint
    starting_epoch = 0
    if config['model']['load_from_checkpoint'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']

    # Lambda parameter
    if config['training']['lambda']:
        reg_function = l1_reg
    else:
        reg_function = None

    # Training
    title(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Training started')
    model.train()
    best_c_index = 0.0
    best_loss = np.inf
    best_score = -np.inf
    epochs_without_improvement = 0
    patience = config['training']['early_stopping_patience']
    for epoch in range(starting_epoch, config['training']['epochs']):
        print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Epoch: {epoch + 1}')
        start_time = time.time()

        # Training and validation in this epoch
        training(epoch, config, training_loader, model, loss_function, optimizer, scheduler, reg_function)
        validation_loss, validation_c_index = validation(epoch, config, validation_loader, model, loss_function, reg_function)

        # Saving best model dependently on Validation Loss or Validation C-Index
        validation_score = config['training']['weight_c_index'] * validation_c_index - config['training']['weight_loss'] * validation_loss
        if validation_score > best_score:
            best_score = validation_score
            best_loss = validation_loss
            best_c_index = validation_c_index
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'validation_loss': validation_loss,
                'validation_c_index': validation_c_index,
            }, f'{config["model"]["checkpoint_best_model"]}_{process_id}.pt')
            print(f'--> New BEST Validation Score: {validation_score:.4f}, validation_loss: {validation_loss:.4f}, validation_c_index: {validation_c_index:.4f}')
        else:
            epochs_without_improvement += 1
            print(f'--> No improvement in validation_loss or validation_c_index for {epochs_without_improvement} epoch(s)')
        print(f'--> Current BEST Score: {best_score:.4f}\n\t--> validation_loss: {best_loss:.4f}\n\t--> best validation_c_index: {best_c_index:.4f}')

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f'--> Early stopping triggered after {patience} epochs without improvement')
            break

        end_time = time.time()
        print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Time elapsed for epoch {epoch + 1}: {end_time - start_time:.0f}s')

    # Restoring Best Model
    best_model_state = torch.load(f'{config["model"]["checkpoint_best_model"]}_{process_id}.pt')
    print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Restoring best model from epoch {best_model_state["epoch"] + 1}')
    model.load_state_dict(best_model_state['model_state_dict'])
    if best_model_state["validation_c_index"] >= C_INDEX_THRESHOLD:
        new_filename = f'{config["model"]["checkpoint_best_model"]}_{process_id}_{best_model_state["validation_c_index"]:4f}.pt'
        os.rename(f'{config["model"]["checkpoint_best_model"]}_{process_id}.pt', f'{config["model"]["checkpoint_best_model"]}_{process_id}_{best_model_state["validation_c_index"]:4f}.pt')
        print(f'--> Model saved as {new_filename}')
    else:
        os.remove(f'{config["model"]["checkpoint_best_model"]}_{process_id}.pt')
        print('--> Model discarded due to low validation C-Index.')        
    title(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Training completed')

    # Final Validation
    title(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Validation started')
    start_time = time.time()
    validation('testing', config, validation_loader, model, loss_function, reg_function)
    end_time = time.time()
    print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Time elapsed for Validation: {end_time - start_time:.0f}s')
    title(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Validation completed')

    # Ending W&B
    if config['wandb']['enabled']:
        wandb.finish()


## MAIN
if __name__ == '__main__':
    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Execution
    title(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - MCAT started')
    main(MCAT_MULTIMODAL_YAML)
    title(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - MCAT terminated')

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
