import datetime
import os
import time
import wandb
import yaml
import numpy as np
import random
import scipy.stats as stats
import socket
import sys
import torch.cuda
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
import warnings
sys.path.append('/homes/linghilterra/AIforBioinformatics')
from logs.methods.log_storer import DualOutput
from src.surv_path.gene_expression_and_methylation_modules.training import training
from src.surv_path.gene_expression_and_methylation_modules.validation import validation
from src.surv_path.original_modules.loss import CrossEntropySurvivalLoss, SurvivalClassificationTobitLoss
from src.surv_path.original_modules.utils import l2_reg
from src.surv_path.gene_expression_and_methylation_modules.surv_path import SurvPath
from src.surv_path.gene_expression_and_methylation_modules.dataset import MultimodalDataset
from torch.utils.data import DataLoader


## CONFIGURATION
''' General '''
C_INDEX_THRESHOLD = 1.0
LOG_PATH = f'../../logs/files/{os.path.basename(__file__)}.txt'
MCAT_MULTIMODAL_YAML = '../../config/files/surv_path_gene_expression_and_methylation.yaml'
PID = None


## FUNCTIONS
def title(text):
    print('\n' + f' {text} '.center(80, '#'))


''' RANDOM CONFIGURATION for deterministic results '''
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


''' STATISTICS '''
def calculate_statistics(data):
    mean = np.mean(data)
    variance = np.var(data, ddof=1)
    cardinality = len(data)
    std_error = np.std(data, ddof=1) / np.sqrt(cardinality)
    ci_lower, ci_upper = stats.norm.interval(0.95, loc=mean, scale=std_error)
    return mean, variance, ci_lower, ci_upper


''' W&B CONFIGURATION '''
def wandb_init(config):
    os.makedirs("./wandb", exist_ok=True)
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
            'dropout': config['training']['dropout'],
            'classes_number': config['training']['classes_number'],
            'folds': config['training']['folds'],
        }
    )


''' MAIN DEFINITION '''
def main(config_path: str):
    ## Configuration
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        set_seed(config['dataset']['random_seed'])

    ## Local Machine Runs
    print(f"HOSTNAME: {socket.gethostname()}")
    if socket.gethostname() == 'DELL-XPS-15':
        config['wandb']['enabled'] = False
        process_id = f'{socket.gethostname()}_{os.getpid()}'
    else:
        process_id = f'{os.getenv("SLURM_JOB_NAME", "default_job_name")}_{os.getenv("SLURM_JOB_ID", "default_job_id")}'

    ## Starting W&B
    print('')
    if config['wandb']['enabled']:
        print('WANDB: setting up for report')
        os.environ['WANDB__SERVICE_WAIT'] = '300'
        wandb_init(config)
    else:
        print('WANDB: disabled')

    ## CUDA
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

    ## DATASET
    print('')
    dataset = MultimodalDataset(config,
                                classes_number=config['training']['classes_number'],
                                use_signatures=True,
                                random_seed=config['dataset']['random_seed'],
                                remove_incomplete_samples=True)  # Dataset object
    print(f'--> Using {int(config["training"]["train_size"] * 100)}% training, {100 - int(config["training"]["train_size"] * 100)}% validation')

    ## GRID SEARCH
    title(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Grid Search')
    grid_search = {
        'alpha': config['training']['alpha'],
        'dropout': config['training']['dropout'],
        'gamma': config['training']['gamma'],
        'lambda': config['training']['lambda'],
        'learning_rate': config['training']['lr'],
        'weight_decay': config['training']['weight_decay'],
    }
    for alpha in grid_search['alpha']:
        for dropout in grid_search['dropout']:
            for gamma in grid_search['gamma']:
                for lambda_ in grid_search['lambda']:
                    for lr in grid_search['learning_rate']:
                        for weight_decay in grid_search['weight_decay']:
                            config['training']['alpha'] = float(alpha)
                            config['training']['dropout'] = float(dropout)
                            config['training']['gamma'] = float(gamma)
                            config['training']['lambda'] = float(lambda_)
                            config['training']['lr'] = float(lr)
                            config['training']['weight_decay'] = float(weight_decay)

                            title(f'Cross Validation with Current Configuration started')
                            print(f"--> alpha: {alpha}, dropout: {dropout}, gamma: {gamma}, lambda: {lambda_}, lr: {lr}, weight_decay: {weight_decay}")

                            ## CROSS VALIDATION
                            training_c_index_list = []
                            training_loss_list = []
                            validation_c_index_list = []
                            validation_loss_list = []
                            folds = dataset.k_fold_split(dataset, k=config['training']['folds'])
                            for fold_index, (training_fold, validation_fold) in enumerate(folds):
                                title(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Fold n°{fold_index + 1}')

                                ## MODEL
                                print('')
                                model_name = config['model']['name']
                                print(f'MODEL: {model_name}')
                                model = SurvPath(n_classes=config['training']['classes_number'],
                                                 rnaseq_sizes=dataset.gene_expression_signature_sizes,
                                                 meth_sizes=dataset.methylation_signature_sizes,
                                                 dropout=config['training']['dropout'],
                                                 fusion=config['model']['fusion'],
                                                 device=config['device'])
                                print(f'--> Trainable parameters of {model_name}: {model.get_trainable_parameters()}')
                                checkpoint = None
                                if config['model']['load_from_checkpoint'] is not None:  # Starting Model from Checkpoint
                                    print(f'--> Loading model checkpoint from {config["model"]["load_from_checkpoint"]}')
                                    checkpoint = torch.load(config['model']['load_from_checkpoint'])
                                    model.load_state_dict(checkpoint['model_state_dict'])
                                if config['device'] == 'cuda' and torch.cuda.device_count() > 1:  # Moving Model on GPU
                                    model = nn.DataParallel(model)
                                model.to(device=config['device'])

                                ## Loss Function
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

                                ## Optimizer
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
                                starting_epoch = 0
                                if config['model']['load_from_checkpoint'] is not None:  # Starting Optimizer from Checkpoint
                                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                                    starting_epoch = checkpoint['epoch']

                                ## Scheduler
                                if config['training']['scheduler'] == 'exp':
                                    gamma = config['training']['gamma']
                                    scheduler = lrs.ExponentialLR(optimizer, gamma=gamma)
                                else:
                                    scheduler = None

                                ## Regularization
                                if config['training']['lambda']:
                                    reg_function = l2_reg
                                else:
                                    reg_function = None

                                ## Fold's TRAINING
                                title(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Training started (Fold n°{fold_index + 1})')
                                training_loader = DataLoader(training_fold, batch_size=config['training']['batch_size'], shuffle=True, num_workers=6, pin_memory=True)
                                validation_loader = DataLoader(validation_fold, batch_size=config['training']['batch_size'], shuffle=False, num_workers=6, pin_memory=True)
                                model.train()
                                best_training_loss = np.inf
                                best_training_c_index = 0.0
                                best_validation_loss = np.inf
                                best_validation_c_index = 0.0
                                best_score = -np.inf
                                epochs_without_improvement = 0
                                patience = config['training']['early_stopping_patience']
                                for epoch in range(starting_epoch, config['training']['epochs']):
                                    print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Epoch: {epoch + 1}')
                                    start_time = time.time()

                                    # TRAINING & VALIDATION in this epoch
                                    training_loss, training_c_index =  training(epoch, config, training_loader, model, loss_function, optimizer, scheduler, reg_function, grid_search=True)
                                    validation_loss, validation_c_index = validation(epoch, config, validation_loader, model, loss_function, reg_function)

                                    # BEST MODEL chosen on VALIDATION Loss or C-Index
                                    os.makedirs(config["model"]["checkpoint_dir"], exist_ok=True)
                                    validation_score = config['training']['weight_c_index'] * validation_c_index - config['training']['weight_loss'] * validation_loss - config['training']['weight_difference'] * (abs(validation_c_index - training_c_index))
                                    if validation_score > best_score:
                                        best_score = validation_score
                                        best_training_loss = training_loss
                                        best_training_c_index = training_c_index
                                        best_validation_loss = validation_loss
                                        best_validation_c_index = validation_c_index
                                        epochs_without_improvement = 0
                                        torch.save({
                                            'epoch': epoch,
                                            'fold_number': fold_index + 1,
                                            'model_state_dict': model.state_dict(),
                                            'optimizer_state_dict': optimizer.state_dict(),
                                            'scheduler_state_dict': scheduler.state_dict(),
                                            'training_loss': training_loss,
                                            'training_c_index': training_c_index,
                                            'validation_loss': validation_loss,
                                            'validation_c_index': validation_c_index,
                                        }, f'{config["model"]["checkpoint_best_model"]}_{process_id}_{fold_index}.pt')
                                        print(f'--> New BEST Validation Score: {validation_score:.4f}, validation_loss: {validation_loss:.4f}, validation_c_index: {validation_c_index:.4f}, training_loss: {training_loss:.4f}, training_c_index: {training_c_index:.4f}')
                                    else:
                                        epochs_without_improvement += 1
                                        print(f'--> Score: {validation_score:.4f}. No improvement in validation_loss or validation_c_index for {epochs_without_improvement} epoch(s)')
                                    print(f'--> Current BEST Score: {best_score:.4f}\n\t--> validation_loss: {best_validation_loss:.4f}\n\t--> validation_c_index: {best_validation_c_index:.4f}\n\t--> training_loss: {best_training_loss:.4f}\n\t--> training_c_index: {best_training_c_index:.4f}')

                                    # Early stopping
                                    if epochs_without_improvement >= patience:
                                        print(f'--> Early stopping triggered after {patience} epochs without improvement')
                                        break

                                    end_time = time.time()
                                    print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Time elapsed for epoch {epoch + 1}: {end_time - start_time:.0f}s')

                                ## Fold's BEST MODEL
                                best_model_state = torch.load(f'{config["model"]["checkpoint_best_model"]}_{process_id}_{fold_index}.pt')
                                print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Restoring best model from epoch {best_model_state["epoch"] + 1}')
                                model.load_state_dict(best_model_state['model_state_dict'])
                                if best_model_state["validation_c_index"] >= C_INDEX_THRESHOLD:
                                    new_filename = f'{config["model"]["checkpoint_best_model"]}_{process_id}_{best_model_state["validation_c_index"]:4f}.pt'
                                    os.rename(f'{config["model"]["checkpoint_best_model"]}_{process_id}_{fold_index}.pt',
                                              f'{config["model"]["checkpoint_best_model"]}_{process_id}_{fold_index}_{best_model_state["validation_c_index"]:4f}.pt')
                                    print(f'--> Model saved as {new_filename}')
                                else:
                                    os.remove(f'{config["model"]["checkpoint_best_model"]}_{process_id}_{fold_index}.pt')
                                    print('--> Model discarded due to low validation C-Index.')
                                title(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Training completed (Fold n°{fold_index + 1})')

                                # Final Validation
                                title(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Validation started (Fold n°{fold_index + 1})')
                                start_time = time.time()
                                training_c_index_list.append(best_model_state["training_c_index"])
                                training_loss_list.append(best_model_state["training_loss"])
                                validation_loss, validation_c_index = validation('testing', config, validation_loader, model, loss_function, reg_function)
                                validation_c_index_list.append(validation_c_index)
                                validation_loss_list.append(validation_loss)
                                end_time = time.time()
                                print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Time elapsed for Validation: {end_time - start_time:.0f}s')
                                title(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Validation completed (Fold n°{fold_index + 1})')

                            ## Final Metrics
                            title(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Final Metrics')
                            ''' TRAINING '''
                            training_loss_list = [tensor.detach().cpu().numpy() for tensor in training_loss_list if isinstance(tensor, torch.Tensor)]
                            training_c_index_mean, training_c_index_variance, training_c_index_ci_lower, training_c_index_ci_upper = calculate_statistics(training_c_index_list)
                            training_loss_mean, training_loss_variance, training_loss_ci_lower, training_loss_ci_upper = calculate_statistics(training_loss_list)
                            print("TRAINING:")
                            print(f"--> C-Index: [", ", ".join(f"{value:.4f}" for value in training_c_index_list), "]")
                            print(f"    Mean: {training_c_index_mean:.4f}, Variance: {training_c_index_variance:.4f}, 95% Confidence Interval: [{training_c_index_ci_lower:.4f}, {training_c_index_ci_upper:.4f}]")
                            print(f"\n--> Loss: [", ", ".join(f"{value:.4f}" for value in training_loss_list), "]")
                            print(f"    Mean: {training_loss_mean:.4f}, Variance: {training_loss_variance:.4f}, 95% Confidence Interval: [{training_loss_ci_lower:.4f}, {training_loss_ci_upper:.4f}]")
                            ''' VALIDATION '''
                            validation_loss_list = [tensor.detach().cpu().numpy() for tensor in validation_loss_list if isinstance(tensor, torch.Tensor)]
                            validation_c_index_mean, validation_c_index_variance, validation_c_index_ci_lower, validation_c_index_ci_upper = calculate_statistics(validation_c_index_list)
                            validation_loss_mean, validation_loss_variance, validation_loss_ci_lower, validation_loss_ci_upper = calculate_statistics(validation_loss_list)
                            print("\nVALIDATION:")
                            print(f"--> C-Index: [", ", ".join(f"{value:.4f}" for value in validation_c_index_list), "]")
                            print(f"    Mean: {validation_c_index_mean:.4f}, Variance: {validation_c_index_variance:.4f}, 95% Confidence Interval: [{validation_c_index_ci_lower:.4f}, {validation_c_index_ci_upper:.4f}]")
                            print(f"\n--> Loss: [", ", ".join(f"{value:.4f}" for value in validation_loss_list), "]")
                            print(f"    Mean: {validation_loss_mean:.4f}, Variance: {validation_loss_variance:.4f}, 95% Confidence Interval: [{validation_loss_ci_lower:.4f}, {validation_loss_ci_upper:.4f}]")

                            title(f'Cross Validation with Current Configuration ended')
                            print(f"--> alpha: {alpha}, dropout: {dropout}, gamma: {gamma}, lambda: {lambda_}, lr: {lr}, weight_decay: {weight_decay}")

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
