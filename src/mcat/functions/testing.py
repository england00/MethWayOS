import datetime
import os
import torch.cuda


''' TESTING DEFINITION '''
def test(epoch, config, validation_loader, model, patient, save=False):
    # Initialization
    model.eval()
    output_dir = config['training']['test_output_dir']
    model_name = config['model']['name']
    now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    # Starting
    for batch_index, (survival_months, survival_class, censorship, omics_data, patches_embeddings) in enumerate(
            validation_loader):
        survival_months = survival_months.to(config['device'])
        survival_class = survival_class.to(config['device'])
        survival_class = survival_class.unsqueeze(0).to(torch.int64)
        censorship = censorship.type(torch.FloatTensor).to(config['device'])
        patches_embeddings = patches_embeddings.to(config['device'])
        omics_data = [omic_data.to(config['device']) for omic_data in omics_data]
        print(f'[{batch_index}] Survival months: {survival_months.item()}, Survival class: {survival_class.item()}, '
              f'Censorship: {censorship.item()}')

        # Testing
        with torch.no_grad():
            hazards, survs, Y, attention_scores = model(wsi=patches_embeddings, omics=omics_data, inference=True)
            risk = -torch.sum(survs, dim=1).cpu().numpy()
            print(f'Hazards: {hazards}, Survs: {survs}, Risk: {risk}, Y: {Y}')
            print(f'Attn min: {attention_scores["coattn"].min()}, Attn max: {attention_scores["coattn"].max()}, Attn '
                  f'mean: {attention_scores["coattn"].mean()}')

            # Saving Model
            if save:
                output_file = os.path.join(output_dir, f'ATTN_{model_name}_{patient}_{now}_E{epoch}_{batch_index}.pt')
                print(f'Saving attention in {output_file}')
                torch.save(attention_scores['coattn'], output_file)
