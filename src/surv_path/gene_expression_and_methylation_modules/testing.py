import datetime
import os
import torch.cuda

'''
 if leave_one_out:
     save = False
     if (epoch + 1) % config['training']['output_attn_epoch'] == 0:
         save = True
     testing_patient = config['training']['leave_one_out']
     test(epoch + 1, config, testing_loader, model, testing_patient, save=save)
 '''

''' TESTING DEFINITION '''
def test(epoch, config, testing_loader, model, patient, save=False):
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

        # Testing
        with torch.no_grad():
            hazards, surveys, Y, attention_scores = model(islands=methylation_data, genes=gene_expression_data, inference=True)
            risk = -torch.sum(surveys, dim=1).cpu().numpy()
            print(f'--> Hazards: {hazards}')
            print(f'--> Surveys: {surveys}')
            print(f'--> Risk: {risk}')
            print(f'--> Y: {Y}')
            print(f'--> Attention Min: {attention_scores["coattn"].min()}, Attention Max: {attention_scores["coattn"].max()}, Attention Mean: {attention_scores["coattn"].mean()}')

            # Saving Model
            if save:
                output_file = os.path.join(config['training']['test_output_dir'], f'ATTN_{config["model"]["name"]}_{patient}_{now}_E{epoch}_{batch_index}.pt')
                print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Saving attention in {output_file}')
                torch.save(attention_scores['coattn'], output_file)
