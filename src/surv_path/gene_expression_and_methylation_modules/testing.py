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
            print(f'--> RNA-Seq Self-Attention Min: {attention_scores["self_attention_rnaseq"].min()}, '
                  f'RNA-Seq Self-Attention Max: {attention_scores["self_attention_rnaseq"].max()}, '
                  f'RNA-Seq Self-Attention Mean: {attention_scores["self_attention_rnaseq"].mean()}')
            print(f'--> RNA-Seq Cross-Attention Min: {attention_scores["cross_attention_rnaseq"].min()}, '
                  f'RNA-Seq Cross-Attention Max: {attention_scores["cross_attention_rnaseq"].max()}, '
                  f'RNA-Seq Cross-Attention Mean: {attention_scores["cross_attention_rnaseq"].mean()}')
            print(f'--> CpG Sites Cross-Attention Min: {attention_scores["cross_attention_meth"].min()}, '
                  f'CpG Sites Cross-Attention Max: {attention_scores["cross_attention_meth"].max()}, '
                  f'CpG Sites Cross-Attention Mean: {attention_scores["cross_attention_meth"].mean()}')

            # Saving Model
            if save:
                '''RNA-Seq Self-Attention'''
                tensor_np = attention_scores['self_attention_rnaseq'].cpu().numpy()
                plt.figure(figsize=(10, 4))
                sns.heatmap(tensor_np, cmap="coolwarm", annot=True, fmt=".4f", linewidths=0.5)
                plt.title("RNA-Seq Self-Attention Heatmap")
                os.makedirs(config['training']['test_output_dir'], exist_ok=True)
                plt.savefig(os.path.join(config['training']['test_output_dir'], f'RNA_Seq_Self_Attention_{patient}_{process_id}_{fold_index}.png'), dpi=300, bbox_inches="tight")
                torch.save(attention_scores['self_attention_rnaseq'], os.path.join(config['training']['test_output_dir'], f'RNA_Seq_Self_Attention_{patient}_{process_id}_{fold_index}.pt'))
                print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Saving RNA-Seq Self-Attention matrix in {os.path.join(config["training"]["test_output_dir"], f"RNA_Seq_Self_Attention_{patient}_{process_id}_{fold_index}.pt")}')

                '''RNA-Seq Cross-Attention'''
                tensor_np = attention_scores['cross_attention_rnaseq'].cpu().numpy()
                plt.figure(figsize=(10, 4))
                sns.heatmap(tensor_np, cmap="coolwarm", annot=True, fmt=".4f", linewidths=0.5)
                plt.title("RNA-Seq Cross-Attention Heatmap")
                os.makedirs(config['training']['test_output_dir'], exist_ok=True)
                plt.savefig(os.path.join(config['training']['test_output_dir'], f'RNA_Seq_Cross_Attention_{patient}_{process_id}_{fold_index}.png'), dpi=300, bbox_inches="tight")
                torch.save(attention_scores['cross_attention_rnaseq'], os.path.join(config['training']['test_output_dir'], f'RNA_Seq_Cross_Attention_{patient}_{process_id}_{fold_index}.pt'))
                print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Saving RNA-Seq Cross-Attention matrix in {os.path.join(config["training"]["test_output_dir"], f"RNA_Seq_Cross_Attention_{patient}_{process_id}_{fold_index}.pt")}')

                '''CpG Sites Cross-Attention'''
                tensor_np = attention_scores['cross_attention_meth'].cpu().numpy()
                plt.figure(figsize=(10, 4))
                sns.heatmap(tensor_np, cmap="coolwarm", annot=True, fmt=".4f", linewidths=0.5)
                plt.title("CpG Sites Cross-Attention Heatmap")
                os.makedirs(config['training']['test_output_dir'], exist_ok=True)
                plt.savefig(os.path.join(config['training']['test_output_dir'], f'CpG_Sites_Cross_Attention_{patient}_{process_id}_{fold_index}.png'), dpi=300, bbox_inches="tight")
                torch.save(attention_scores['cross_attention_meth'], os.path.join(config['training']['test_output_dir'], f'CpG_Sites_Cross_Attention_{patient}_{process_id}_{fold_index}.pt'))
                print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Saving CpG Sites Cross-Attention matrix in {os.path.join(config["training"]["test_output_dir"], f"CpG_Sites_Cross_Attention_{patient}_{process_id}_{fold_index}.pt")}')
