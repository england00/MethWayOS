import datetime
import os
import torch.cuda
import matplotlib.pyplot as plt
import seaborn as sns


''' TESTING DEFINITION '''
def testing(epoch, config, testing_loader, model, patient, process_id, fold_index, rnaseq_signatures, cpg_signatures, save=False):
    # Initialization
    model.eval()
    now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    # Starting
    for batch_index, (survival_months, survival_class, censorship, gene_expression_data, methylation_data) in enumerate(testing_loader):

        ''' ######################################### FORWARD PASS ################################################# '''
        survival_months = survival_months.to(config['device'])
        survival_class = survival_class.to(config['device'])
        survival_class = survival_class.unsqueeze(0).to(torch.int64)
        censorship = censorship.type(torch.FloatTensor).to(config['device'])
        gene_expression_data = [gene.to(config['device']) for gene in gene_expression_data]
        methylation_data = [island.to(config['device']) for island in methylation_data]
        print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Testing Batch Index: {batch_index}, Survival months: {survival_months.item()}, Survival class: {survival_class.item()}, Censorship: {censorship.item()}')
        with torch.no_grad():
            hazards, surv, Y, attention_scores = model(islands=methylation_data, genes=gene_expression_data, inference=True)
            surv = torch.nan_to_num(surv, nan=0.0)
            risk = -torch.sum(surv, dim=1).cpu().numpy()
            print(f'--> Hazards: {hazards}')
            print(f'--> Surveys: {surv}')
            print(f'--> Risk: {risk}')
            print(f'--> Y: {Y}')
            print(f'--> Co-Attention Min: {attention_scores["coattn"].min()}, Co-Attention Max: {attention_scores["coattn"].max()}, Co-Attention Mean: {attention_scores["coattn"].mean()}')

            # Saving Model
            if save:
                tensor_np = attention_scores['coattn'].cpu().numpy()
                plt.figure(figsize=(10, 4))
                sns.heatmap(tensor_np, cmap="coolwarm", annot=True, fmt=".4f", linewidths=0.5, xticklabels=cpg_signatures, yticklabels=rnaseq_signatures)
                plt.title("Co-Attention Heatmap")
                os.makedirs(config['training']['test_output_dir'], exist_ok=True)
                plt.savefig(os.path.join(config['training']['test_output_dir'], f'{patient}_{process_id}_{fold_index}.png'), dpi=300, bbox_inches="tight")
                torch.save(attention_scores['coattn'], os.path.join(config['training']['test_output_dir'], f'{patient}_{process_id}_{fold_index}.pt'))
                print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Saving Co-Attention matrix in {os.path.join(config["training"]["test_output_dir"], f"{patient}_{process_id}_{fold_index}.pt")}')
