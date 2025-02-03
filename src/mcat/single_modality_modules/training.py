import datetime
import math
import time
import wandb
import torch.cuda
from sksurv.metrics import concordance_index_censored


''' TRAINING DEFINITION '''
def training(epoch, config, training_loader, model, loss_function, optimizer, scheduler, reg_function, grid_search: bool = False):
    # Initialization
    model.train()
    training_loss = 0.0
    risk_scores = torch.zeros(len(training_loader), device=config['device'])
    censorships = torch.zeros(len(training_loader), device=config['device'])
    event_times = torch.zeros(len(training_loader), device=config['device'])

    # Starting
    start_batch_time = time.time()
    for batch_index, (survival_months, survival_class, censorship, single_modality_data) in enumerate(training_loader):

        ''' ######################################### FORWARD PASS ################################################# '''
        survival_months = survival_months.to(config['device'], non_blocking=True)
        survival_class = survival_class.to(config['device'], non_blocking=True)
        survival_class = survival_class.unsqueeze(0).to(torch.int64)
        censorship = censorship.type(torch.FloatTensor).to(config['device'], non_blocking=True)
        single_modality_data = [gene.to(config['device']) for gene in single_modality_data]
        hazards, surv, Y, attention_scores = model(data=single_modality_data)
        surv = torch.nan_to_num(surv, nan=0.0)

        # Choosing Loss Function
        try:
            if config['training']['loss'] == 'ce':
                loss = loss_function(Y, survival_class.long())
            elif config['training']['loss'] == 'ces':
                loss = loss_function(hazards, surv, survival_class, c=censorship)
            elif config['training']['loss'] == 'sct':
                loss = loss_function(Y, survival_class, c=censorship)
            else:
                raise RuntimeError(f'Loss "{config["training"]["loss"]}" not implemented')
        except RuntimeError as e:
            print(f"Error computing loss, skipping: {e}")
            continue
        loss_value = loss.item()

        # Regularization function
        if reg_function is None:
            loss_reg = 0
        else:
            loss_reg = reg_function(model) * config['training']['lambda']

        # Scores
        risk = -torch.sum(surv, dim=1)
        risk_scores[batch_index] = risk
        censorships[batch_index] = censorship
        event_times[batch_index] = survival_months
        training_loss += loss_value + loss_reg

        # Logger
        if (batch_index + 1) % 50 == 0:
            print(f'--> Batch: {batch_index}, Loss: {loss_value + loss_reg:.4f}, Label: {survival_class.item()}, Survival Months: {survival_months.item():.2f}, Risk: {float(risk.item()):.4f}')
            end_batch_time = time.time()
            print(f'\t--> Average Speed: {((end_batch_time - start_batch_time) / 32):.2f}s per batch')
            print(f'\t--> Time from Last Check: {(end_batch_time - start_batch_time):.2f}s')
            start_batch_time = time.time()

        ''' ######################################## BACKWARD PASS ################################################# '''
        loss = loss / config['training']['grad_acc_step'] + loss_reg
        loss.backward()

        # Scheduler
        if (batch_index + 1) % config['training']['grad_acc_step'] == 0:
            optimizer.step()
            optimizer.zero_grad()

    ''' ############################################## METRICS ##################################################### '''
    # Calculate loss and error for epoch
    if isinstance(training_loss, float):
        if math.isnan(training_loss):
            print("NaN detected in training_loss (float), skipping")
            training_loss = 0.0
    elif torch.is_tensor(training_loss):
        if torch.isnan(training_loss).any():
            print("NaN detected in training_loss (tensor), skipping")
            training_loss = torch.tensor(0.0, device=training_loss.device)
    training_loss /= len(training_loader)
    risk_scores = risk_scores.detach().cpu().numpy()
    censorships = censorships.detach().cpu().numpy()
    event_times = event_times.detach().cpu().numpy()
    training_c_index = concordance_index_censored((1 - censorships).astype(bool), event_times, risk_scores)[0]
    if config['training']['scheduler']:
        lr = optimizer.param_groups[0]["lr"]
        if config['training']['scheduler'] == 'exp':
            scheduler.step()
        print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Epoch: {epoch + 1}, lr: {lr:.8f}, training_loss: {training_loss:.4f}, training_c_index: {training_c_index:.4f}')
    else:
        print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Epoch: {epoch + 1}, training_loss: {training_loss:.4f}, training_c_index: {training_c_index:.4f}')

    # W&B
    wandb_enabled = config['wandb']['enabled']
    if wandb_enabled:
        wandb.log({"training_loss": training_loss, "training_c_index": training_c_index})

    if grid_search:
        return training_loss, training_c_index
