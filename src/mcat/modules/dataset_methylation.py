import numpy as np
import pandas as pd
import time
import torch
import yaml
from torch.utils.data import Dataset, DataLoader


## CLASSES
class MultimodalDataset(Dataset):
    def __init__(self, config, classes_number: int = 4, use_signatures: bool = False, remove_incomplete_samples: bool = True):
        # GENERAL: Managing CSV Dataset
        self.gene_expression = pd.read_csv(config['dataset']['gene_expression'])
        self.methylation = pd.read_csv(config['dataset']['methylation'])
        print('DATASET: Loading data')
        self.gene_expression.reset_index(drop=True, inplace=True)
        self.methylation.reset_index(drop=True, inplace=True)

        # GENERAL: Managing incomplete examples (only 'case_id' within Gene Expression and Methylation datasets)
        if remove_incomplete_samples:
            common_ids = set(self.gene_expression['case_id']) & set(self.methylation['case_id'])
            self.gene_expression = self.gene_expression[self.gene_expression['case_id'].isin(common_ids)].reset_index(drop=True)
            self.methylation = self.methylation[self.methylation['case_id'].isin(common_ids)].reset_index(drop=True)

            # METHYLATION: Aligning indices of the two DataFrames based on Gene Expression order
            self.methylation = self.methylation.set_index('case_id').reindex(self.gene_expression['case_id']).reset_index()
            print(f'--> Remaining samples after removing incomplete: {len(self.gene_expression)}')

        # GENERAL: Managing Classes creation (based with Gene Expression)
        self.classes_number = classes_number
        survival_class, class_intervals = pd.qcut(self.gene_expression['survival_months'], q=self.classes_number, retbins=True, labels=False)
        self.gene_expression['survival_class'] = survival_class
        print('--> Class intervals: [')
        for i in range(0, classes_number):
            print('\t{}: [{:.2f} - {:.2f}]'.format(i, class_intervals[i], class_intervals[i + 1]))
        print('    ]')
        self.survival_months = self.gene_expression['survival_months'].values
        self.survival_class = self.gene_expression['survival_class'].values
        self.censorship = self.gene_expression['censorship'].values

        # GENE EXPRESSION: Managing columns (Standardization or Normalization)
        rnaseq_columns = [col for col in self.gene_expression.columns if col.endswith('_rnaseq')]
        if config['dataset']['standardize']:
            print('--> Standardizing RNA-seq data')
            for col in rnaseq_columns:
                mean = self.gene_expression[col].mean()
                std = self.gene_expression[col].std()
                if std == 0:
                    self.gene_expression[col] = 0
                else:
                    self.gene_expression[col] = (self.gene_expression[col] - mean) / std
        elif config['dataset']['normalize']:
            print('--> Normalizing RNA-seq data')
            for col in rnaseq_columns:
                std = self.gene_expression[col].std()
                if std == 0:
                    self.gene_expression[col] = 0.5
                else:
                    self.gene_expression[col] = 2 * (self.gene_expression[col] - self.gene_expression[col].min()) / (self.gene_expression[col].max() - self.gene_expression[col].min()) - 1

        # METHYLATION: Managing columns (Standardization or Normalization)
        meth_columns = [col for col in self.methylation.columns if col.endswith('_meth')]
        if config['dataset']['standardize']:
            print('--> Standardizing Methylation Islands beta values')
            for col in meth_columns:
                mean = self.methylation[col].mean()
                std = self.methylation[col].std()
                if std == 0:
                    self.methylation[col] = 0
                else:
                    self.methylation[col] = (self.methylation[col] - mean) / std
        elif config['dataset']['normalize']:
            print('--> Normalizing Methylation Islands beta values')
            for col in meth_columns:
                std = self.methylation[col].std()
                if std == 0:
                    self.methylation[col] = 0.5
                else:
                    self.methylation[col] = 2 * (self.methylation[col] - self.methylation[col].min()) / (self.methylation[col].max() - self.methylation[col].min()) - 1

        # GENE EXPRESSION: RNA-Seq
        self.rnaseq = self.gene_expression.iloc[:, self.gene_expression.columns.str.endswith('_rnaseq')].astype(float)
        self.rnaseq_size = len(self.rnaseq.columns)
        self.rnaseq = torch.tensor(self.rnaseq.values, dtype=torch.float32)

        # METHYLATION: Meth-Islands
        self.meth = self.methylation.iloc[:, self.methylation.columns.str.endswith('_meth')].astype(float)
        self.meth_size = len(self.meth.columns)
        self.meth = torch.tensor(self.meth.values, dtype=torch.float32)

        # GENE EXPRESSION: Managing Signatures
        self.use_signatures = use_signatures
        self.gene_expression_signature_sizes = []
        self.gene_expression_signature_data = {}
        signatures_df = pd.read_csv(config['dataset']['gene_expression_signatures'])
        self.gene_expression_signatures = signatures_df.columns
        for signature_name in self.gene_expression_signatures:
            columns = {}
            for gene in signatures_df[signature_name].dropna():
                gene += '_rnaseq'
                if gene in self.gene_expression.columns:
                    columns[gene] = self.gene_expression[gene]
            self.gene_expression_signature_data[signature_name] = torch.tensor(pd.DataFrame(columns).values, dtype=torch.float32)
            self.gene_expression_signature_sizes.append(self.gene_expression_signature_data[signature_name].shape[1])
        print(f'--> Gene Expression Signatures size: {self.gene_expression_signature_sizes}')

        # METHYLATION: Managing Signatures
        self.methylation_signature_sizes = []
        self.methylation_signature_data = {}
        signatures_df = pd.read_csv(config['dataset']['methylation_signatures'])
        self.methylation_signatures = signatures_df.columns
        for signature_name in self.methylation_signatures:
            columns = {}
            for island in signatures_df[signature_name].dropna():
                island += '_meth'
                if island in self.methylation.columns:
                    columns[island] = self.methylation[island]
            self.methylation_signature_data[signature_name] = torch.tensor(pd.DataFrame(columns).values, dtype=torch.float32)
            self.methylation_signature_sizes.append(self.methylation_signature_data[signature_name].shape[1])
        print(f'--> Methylation Signatures size: {self.methylation_signature_sizes}')

    def __getitem__(self, index):
        survival_months = self.survival_months[index]
        survival_class = self.survival_class[index]
        censorship = self.censorship[index]

        # GENE EXPRESSION: Managing data
        gene_expression_data = []
        for signature in self.gene_expression_signatures:
            signature_data = self.gene_expression_signature_data[signature][index]
            gene_expression_data.append(signature_data)

        # METHYLATION: Managing data
        methylation_data = []
        for signature in self.methylation_signatures:
            signature_data = self.methylation_signature_data[signature][index]
            methylation_data.append(signature_data)

        return survival_months, survival_class, censorship, gene_expression_data, methylation_data

    def __len__(self):
        return len(self.gene_expression)

    def split(self, training_size, testing: bool = False, patient: str = ''):
        # GENERAL: Ensure training_size is a valid ratio
        if not 0 < training_size < 1:
            raise ValueError("training_size should be a float between 0 and 1.")

        # GENERAL: Get unique patients and Shuffle randomly
        unique_patients = self.gene_expression['case_id'].unique()
        np.random.shuffle(unique_patients)
        training_patient_count = int(len(unique_patients) * training_size)

        # GENERAL: Split patients into train and validation sets
        training_patients = unique_patients[:training_patient_count]
        validation_patients = unique_patients[training_patient_count:]

        # GENERAL: Filter the data into train, validation, and optionally test sets for both datasets
        testing_dataset = None
        if testing:
            # GENE EXPRESSION: Filtering
            training_data_ge = self.gene_expression[self.gene_expression['case_id'].isin(training_patients)].copy()
            training_data_ge = training_data_ge[training_data_ge['case_id'] != patient]
            validation_data_ge = self.gene_expression[self.gene_expression['case_id'].isin(validation_patients)].copy()
            validation_data_ge = validation_data_ge[validation_data_ge['case_id'] != patient]
            testing_data_ge = self.gene_expression[self.gene_expression['case_id'] == patient].copy()
            testing_data_ge.reset_index(drop=True, inplace=True)

            # METHYLATION: Filtering
            training_data_me = self.methylation[self.methylation['case_id'].isin(training_patients)].copy()
            training_data_me = training_data_me[training_data_me['case_id'] != patient]
            validation_data_me = self.methylation[self.methylation['case_id'].isin(validation_patients)].copy()
            validation_data_me = validation_data_me[validation_data_me['case_id'] != patient]
            testing_data_me = self.methylation[self.methylation['case_id'] == patient].copy()
            testing_data_me.reset_index(drop=True, inplace=True)

            # GENERAL: Create testing dataset
            testing_dataset = MultimodalDataset.from_dataframes(testing_data_ge, testing_data_me, self)
        else:
            # GENE EXPRESSION: Filtering
            training_data_ge = self.gene_expression[self.gene_expression['case_id'].isin(training_patients)].copy()
            validation_data_ge = self.gene_expression[self.gene_expression['case_id'].isin(validation_patients)].copy()

            # METHYLATION: Filtering
            training_data_me = self.methylation[self.methylation['case_id'].isin(training_patients)].copy()
            validation_data_me = self.methylation[self.methylation['case_id'].isin(validation_patients)].copy()

        # GENE EXPRESSION: Reset indices for training datasets
        training_data_ge.reset_index(drop=True, inplace=True)
        training_data_me.reset_index(drop=True, inplace=True)
        training_data_me = training_data_me.set_index('case_id').reindex(training_data_ge['case_id']).reset_index()

        # METHYLATION: Reset indices for validation datasets
        validation_data_ge.reset_index(drop=True, inplace=True)
        validation_data_me.reset_index(drop=True, inplace=True)
        validation_data_me = validation_data_me.set_index('case_id').reindex(validation_data_ge['case_id']).reset_index()

        # GENERAL: Create new instances of MultimodalDataset with the train and validation data
        training_dataset = MultimodalDataset.from_dataframes(training_data_ge, training_data_me, self)
        validation_dataset = MultimodalDataset.from_dataframes(validation_data_ge, validation_data_me, self)

        return training_dataset, validation_dataset, testing_dataset

    @classmethod
    def from_dataframes(cls, ge_dataframe, me_dataframe, original_instance):
        # Create a new MultimodalDataset instance from an existing DataFrame
        # while preserving the original configuration and parameters (without calling __init__)
        instance = cls.__new__(cls)
        instance.use_signatures = original_instance.use_signatures

        # GENE EXPRESSION: Reset indices in the DataFrame
        ge_dataframe = ge_dataframe.reset_index(drop=True)
        instance.gene_expression = ge_dataframe
        if original_instance.use_signatures:
            instance.gene_expression_signatures = original_instance.gene_expression_signatures
            instance.gene_expression_signature_sizes = original_instance.gene_expression_signature_sizes

        # METHYLATION: Reset indices in the DataFrame
        me_dataframe = me_dataframe.reset_index(drop=True)
        instance.methylation = me_dataframe
        if original_instance.use_signatures:
            instance.methylation_signatures = original_instance.methylation_signatures
            instance.methylation_signature_sizes = original_instance.methylation_signature_sizes

        # GENERAL: Copy RNA within the new subset of data
        instance.survival_months = ge_dataframe['survival_months'].values
        instance.survival_class = ge_dataframe['survival_class'].values
        instance.censorship = ge_dataframe['censorship'].values

        # GENE EXPRESSION: RNA-Seq
        rnaseq = ge_dataframe.iloc[:, ge_dataframe.columns.str.endswith('_rnaseq')].astype(float)
        if original_instance.rnaseq_size == len(rnaseq.columns):
            instance.rnaseq = torch.tensor(rnaseq.values, dtype=torch.float32)
        else:
            instance.rnaseq = torch.tensor(np.zeros((len(ge_dataframe), original_instance.rnaseq_size)), dtype=torch.float32)

        # METHYLATION: Meth-Islands
        meth = me_dataframe.iloc[:, me_dataframe.columns.str.endswith('_meth')].astype(float)
        if original_instance.meth_size == len(meth.columns):
            instance.meth = torch.tensor(meth.values, dtype=torch.float32)
        else:
            instance.meth = torch.tensor(np.zeros((len(me_dataframe), original_instance.meth_size)), dtype=torch.float32)

        # GENE EXPRESSION: Copy signature data if use_signatures is True
        if original_instance.use_signatures:
            instance.gene_expression_signature_sizes = original_instance.gene_expression_signature_sizes
            instance.gene_expression_signature_data = {}
            for signature_name in original_instance.gene_expression_signatures:
                signature_data = original_instance.gene_expression_signature_data[signature_name]
                indices = ge_dataframe.index
                instance.gene_expression_signature_data[signature_name] = signature_data[indices]

        # METHYLATION: Copy signature data if use_signatures is True
        if original_instance.use_signatures:
            instance.methylation_signature_sizes = original_instance.methylation_signature_sizes
            instance.methylation_signature_data = {}
            for signature_name in original_instance.methylation_signatures:
                signature_data = original_instance.methylation_signature_data[signature_name]
                indices = me_dataframe.index
                instance.methylation_signature_data[signature_name] = signature_data[indices]

        return instance


## TEST FUNCTIONS
def test_multimodal_dataset():
    print('Testing MultimodalDataset...')

    # Loading Configuration file
    with open('../../../config/files/mcat_methylation.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config['dataset']['file'] = '../../../data/datasets/MCAT_gene_expression_and_methylation_and_overall_survival_dataset.csv'
    config['dataset']['signatures'] = '../../../data/tables/gene_expression_keys_methylation.csv'
    config['dataset']['methylation'] = '../../../data/datasets/MCAT_methylation450_and_overall_survival_dataset.csv'
    config['dataset']['signatures_methylation'] = '../../../data/tables/methylation450_keys.csv'

    # Testing
    dataset = MultimodalDataset(config['dataset']['file'], config, use_signatures=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    start_dataload_time = time.time()
    for batch_index, (survival_months, survival_class, censorship, gene_expression_data, methylation_data) in enumerate(loader):
        print('GE: ', gene_expression_data, ', lenght: ', gene_expression_data[0].numel(), ', index: ', batch_index)
        print('ME: ', methylation_data, ', lenght: ', methylation_data[0].numel(), ', index: ', batch_index)
    end_dataload_time = time.time()
    print('Average dataload time: {:.2f}'.format((end_dataload_time - start_dataload_time) / len(dataset)))

    print('Test successful')


def test_multimodal_dataset_split():
    print('Testing MultimodalDataset...')

    # Loading Configuration file
    with open('../../../config/files/mcat_methylation.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config['dataset']['file'] = '../../../data/datasets/MCAT_gene_expression_and_methylation_and_overall_survival_dataset.csv'
    config['dataset']['signatures'] = '../../../data/tables/gene_expression_keys_methylation.csv'
    config['dataset']['methylation'] = '../../../data/datasets/MCAT_methylation450_and_overall_survival_dataset.csv'
    config['dataset']['signatures_methylation'] = '../../../data/tables/methylation450_keys.csv'

    # Testing
    dataset = MultimodalDataset(config['dataset']['file'], config, use_signatures=True)
    training_dataset, validation_dataset, testing_dataset = dataset.split(config['training']['train_size'])
    assert len(training_dataset) > len(validation_dataset)

    # Training set
    print("\nTRAINING SET:")
    training_loader = DataLoader(training_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=6, pin_memory=True)
    start_dataload_time = time.time()
    for batch_index, (survival_months, survival_class, censorship, gene_expression_data, methylation_data) in enumerate(training_loader):
        print('GE: ', gene_expression_data, ', lenght: ', gene_expression_data[0].numel(), ', index: ', batch_index)
        print('ME: ', methylation_data, ', lenght: ', methylation_data[0].numel(), ', index: ', batch_index)
    end_dataload_time = time.time()
    print('Average dataload time: {:.2f}'.format((end_dataload_time - start_dataload_time) / len(dataset)))

    # Validation set
    print("\nVALIDATION SET:")
    validation_loader = DataLoader(validation_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=6, pin_memory=True)
    start_dataload_time = time.time()
    for batch_index, (survival_months, survival_class, censorship, gene_expression_data, methylation_data) in enumerate(validation_loader):
        print('GE: ', gene_expression_data, ', lenght: ', gene_expression_data[0].numel(), ', index: ', batch_index)
        print('ME: ', methylation_data, ', lenght: ', methylation_data[0].numel(), ', index: ', batch_index)
    end_dataload_time = time.time()
    print('Average dataload time: {:.2f}'.format((end_dataload_time - start_dataload_time) / len(dataset)))

    print('Test successful')


## MAIN
if __name__ == '__main__':
    # test_multimodal_dataset()
    test_multimodal_dataset_split()
