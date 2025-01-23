import datetime
import wandb
import numpy as np
import torch.cuda
from sksurv.metrics import concordance_index_censored


''' VALIDATION DEFINITION '''
def validation(epoch, config, validation_loader, model, loss_function, reg_function):
    # Initialization
    model.eval()
    validation_loss = 0.0
    risk_scores = np.zeros((len(validation_loader)))
    censorships = np.zeros((len(validation_loader)))
    event_times = np.zeros((len(validation_loader)))

    # Starting
    for batch_index, (survival_months, survival_class, censorship, methylation_data) in enumerate(validation_loader):
        survival_months = survival_months.to(config['device'])
        survival_class = survival_class.to(config['device'])
        survival_class = survival_class.unsqueeze(0).to(torch.int64)
        censorship = censorship.type(torch.FloatTensor).to(config['device'])
        methylation_data = [island.to(config['device']) for island in methylation_data]
        with torch.no_grad():
            hazards, surv, Y, attention_scores = model(islands=methylation_data)

        # Choosing Loss Function
        if config['training']['loss'] == 'ce':
            loss = loss_function(Y, survival_class.long())
        elif config['training']['loss'] == 'ces':
            loss = loss_function(hazards, surv, survival_class, c=censorship)
        elif config['training']['loss'] == 'sct':
            loss = loss_function(Y, survival_class, c=censorship)
        else:
            raise RuntimeError(f'--> Loss "{config["training"]["loss"]}" not implemented')
        loss_value = loss.item()

        # Regularization function
        if reg_function is None:
            loss_reg = 0
        else:
            loss_reg = reg_function(model) * config['training']['lambda']

        # Scores
        risk = -torch.sum(surv, dim=1).cpu().numpy()
        if np.isnan(risk).any():
            print(f"Warning: NaN detected in risk at batch {batch_index}. Risk: {risk}")
            risk = np.nan_to_num(risk, nan=0.0)
        risk_scores[batch_index] = risk.item()
        censorships[batch_index] = censorship.item()
        event_times[batch_index] = survival_months.item()
        validation_loss += loss_value + loss_reg

    # Check for NaN in risk_scores before concordance index
    if np.isnan(risk_scores).any():
        print("Warning: NaN detected in risk_scores. Replacing with zeros.")
        risk_scores = np.nan_to_num(risk_scores, nan=0.0)

    # Calculating Loss and Error
    validation_loss /= len(validation_loader)
    validation_c_index = concordance_index_censored((1 - censorships).astype(bool), event_times, risk_scores)[0]
    if epoch == 'testing':
        print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Testing, testing_loss: {validation_loss:.4f}, testing_c_index: {validation_c_index:.4f}')
    else:
        print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Epoch: {epoch + 1}, validation_loss: {validation_loss:.4f}, validation_c_index: {validation_c_index:.4f}')

    # W&B
    wandb_enabled = config['wandb']['enabled']
    if wandb_enabled:
        wandb.log({"validation_loss": validation_loss, "validation_c_index": validation_c_index})

    return validation_loss, validation_c_index
