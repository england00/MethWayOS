import datetime
import os
import torch.cuda


''' TESTING DEFINITION '''
def test(epoch, config, testing_loader, model, patient, save=False):
    # Initialization
    model.eval()
    now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    # Starting
    for batch_index, (survival_months, survival_class, censorship, omics_data, patches_embeddings) in enumerate(testing_loader):
        survival_months = survival_months.to(config['device'])
        survival_class = survival_class.to(config['device'])
        survival_class = survival_class.unsqueeze(0).to(torch.int64)
        censorship = censorship.type(torch.FloatTensor).to(config['device'])
        patches_embeddings = patches_embeddings.to(config['device'])
        omics_data = [omic_data.to(config['device']) for omic_data in omics_data]
        print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Testing Batch Index: {batch_index}, Survival months: {survival_months.item()}, Survival class: {survival_class.item()}, Censorship: {censorship.item()}')

        # Testing
        with torch.no_grad():
            hazards, surveys, Y, attention_scores = model(wsi=patches_embeddings, omics=omics_data, inference=True)
            risk = -torch.sum(surveys, dim=1).cpu().numpy()
            print(f'--> Hazards: {hazards}')
            print(f'--> Surveys: {surveys}')
            print(f'--> Risk: {risk}')
            print(f'--> Y: {Y}')
            print(f'--> Attention Min: {attention_scores["coattn"].min()}, Attention Max: {attention_scores["coattn"].max()}, Attention Mean: {attention_scores["coattn"].mean()}')
            print(f'--> Full Attention: {attention_scores["coattn"]}')

            # Saving Model
            if save:
                output_file = os.path.join(config['training']['test_output_dir'], f'ATTN_{config["model"]["name"]}_{patient}_{now}_E{epoch}_{batch_index}.pt')
                print(f'[{datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}] - Saving attention in {output_file}')
                torch.save(attention_scores['coattn'], output_file)
