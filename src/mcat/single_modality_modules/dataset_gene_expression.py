import numpy as np
import pandas as pd
import time
import torch
import yaml
from data.methods.csv_dataset_loader import csv_loader
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader


## CLASSES
class RnaSeqDataset(Dataset):
    def __init__(self, config, classes_number: int = 4, use_signatures: bool = False, random_seed: int = None):
        # GENERAL
        self.random_seed = random_seed

        # GENE EXPRESSION: Loading CSV Dataset
        print('DATASET: Loading data')
        self.gene_expression, _ = csv_loader(config['dataset']['gene_expression'], verbose=False)
        self.gene_expression.reset_index(drop=True, inplace=True)
        if 'Unnamed: 0' in self.gene_expression.columns:
            self.gene_expression = self.gene_expression.drop(columns=['Unnamed: 0'])
        print('--> Gene Expression loaded')
        print(f'--> Current samples number: {len(self.gene_expression)}')

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

        # GENE EXPRESSION: RNA-Seq
        self.rnaseq = self.gene_expression.iloc[:, self.gene_expression.columns.str.endswith('_rnaseq')].astype(float)
        self.rnaseq_size = len(self.rnaseq.columns)
        self.rnaseq = torch.tensor(self.rnaseq.values, dtype=torch.float32)
        print('--> RNA-seq loaded')

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

    def __getitem__(self, index):
        survival_months = self.survival_months[index]
        survival_class = self.survival_class[index]
        censorship = self.censorship[index]

        # GENE EXPRESSION: Managing data
        gene_expression_data = []
        for signature in self.gene_expression_signatures:
            signature_data = self.gene_expression_signature_data[signature][index]
            gene_expression_data.append(signature_data)

        return survival_months, survival_class, censorship, gene_expression_data

    def __len__(self):
        return len(self.gene_expression)

    def split(self, training_size):
        # GENERAL: Ensure training_size is a valid ratio
        if not 0 < training_size < 1:
            raise ValueError("training_size should be a float between 0 and 1.")

        # GENERAL: Get unique patients and Shuffle randomly
        unique_patients = self.gene_expression['case_id'].unique()
        np.random.seed(self.random_seed)
        np.random.shuffle(unique_patients)
        training_patient_count = int(len(unique_patients) * training_size)

        # GENERAL: Split patients into train and validation sets
        training_patients = unique_patients[:training_patient_count]
        testing_patients = unique_patients[training_patient_count:]

        # GENERAL: Filter and Reset indices for training datasets
        training_data_ge = self.gene_expression[self.gene_expression['case_id'].isin(training_patients)].reset_index(drop=True)

        # GENERAL: Filter and Reset indices for testing datasets
        testing_data_ge = self.gene_expression[self.gene_expression['case_id'].isin(testing_patients)].reset_index(drop=True)

        # GENERAL: Create new instances of MultimodalDataset with the train and validation data
        training_dataset = RnaSeqDataset.from_dataframes(training_data_ge, self)
        testing_dataset = RnaSeqDataset.from_dataframes(testing_data_ge, self)

        return training_dataset, testing_dataset

    def k_fold_split(self, training_dataset, k):
        # GENERAL: Ensure k is a valid ratio
        if k < 2:
            raise ValueError("Number of folds k must be at least 2")

        # GENERAL: Extract dataframes from training dataset
        ge_dataframe = training_dataset.gene_expression

        # GENERAL: Extract unique patient IDs
        unique_patients = ge_dataframe['case_id'].unique()
        k_fold = KFold(n_splits=k, shuffle=True, random_state=self.random_seed)

        # GENERAL: Extract folds
        folds = []
        for training_indices, validation_indices in k_fold.split(unique_patients):
            # GENERAL: Splitting patients into train and validation sets
            training_patients = unique_patients[training_indices]
            validation_patients = unique_patients[validation_indices]

            # GENERAL: Filter and Reset indices for training datasets
            training_ge = ge_dataframe[ge_dataframe['case_id'].isin(training_patients)].reset_index(drop=True)

            # GENERAL: Filter and Reset indices for validation datasets
            validation_ge = ge_dataframe[ge_dataframe['case_id'].isin(validation_patients)].reset_index(drop=True)

            # GENERAL: Create MultimodalDataset instances
            training_fold = RnaSeqDataset.from_dataframes(training_ge, self)
            validation_fold = RnaSeqDataset.from_dataframes(validation_ge, self)

            # GENERAL: Storing current fold
            folds.append((training_fold, validation_fold))

        return folds

    @classmethod
    def from_dataframes(cls, ge_dataframe, original_instance):
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

        # GENE EXPRESSION: Copy signature data if use_signatures is True
        if original_instance.use_signatures:
            instance.gene_expression_signature_sizes = original_instance.gene_expression_signature_sizes
            instance.gene_expression_signature_data = {}
            for signature_name in original_instance.gene_expression_signatures:
                signature_data = original_instance.gene_expression_signature_data[signature_name]
                indices = ge_dataframe.index
                instance.gene_expression_signature_data[signature_name] = signature_data[indices]

        return instance


## TEST FUNCTIONS
def test_rnaseq_dataset():
    print('Testing MultimodalDataset...')

    # Loading Configuration file
    with open('../../../config/files/mcat_gene_expression.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config['dataset']['gene_expression'] = '../../../data/datasets/MCAT_gene_expression_and_overall_survival_dataset.csv'
    config['dataset']['gene_expression_signatures'] = '../../../data/tables/gene_expression_keys_methylation.csv'

    # Testing
    dataset = RnaSeqDataset(config,
                            classes_number=config['training']['classes_number'],
                            use_signatures=True,
                            random_seed=config['dataset']['random_seed'])  # Dataset object
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    start_dataload_time = time.time()
    for batch_index, (survival_months, survival_class, censorship, gene_expression_data) in enumerate(loader):
        print('GE: ', gene_expression_data, ', lenght: ', gene_expression_data[0].numel(), ', index: ', batch_index)
    end_dataload_time = time.time()
    print('Average dataload time: {:.2f}'.format((end_dataload_time - start_dataload_time) / len(dataset)))

    print('Test successful')


def test_rnaseq_dataset_split():
    print('Testing MultimodalDataset...')

    # Loading Configuration file
    with open('../../../config/files/mcat_gene_expression.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config['dataset']['gene_expression'] = '../../../data/datasets/MCAT_gene_expression_and_overall_survival_dataset.csv'
    config['dataset']['gene_expression_signatures'] = '../../../data/tables/gene_expression_keys_methylation.csv'

    # Testing
    dataset = RnaSeqDataset(config,
                            classes_number=config['training']['classes_number'],
                            use_signatures=True,
                            random_seed=config['dataset']['random_seed'])  # Dataset object
    training_dataset, testing_dataset = dataset.split(config['training']['train_size'])
    assert len(training_dataset) > len(testing_dataset)

    # Training set
    print("\nTRAINING SET:")
    training_loader = DataLoader(training_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=6, pin_memory=True)
    start_dataload_time = time.time()
    for batch_index, (survival_months, survival_class, censorship, gene_expression_data) in enumerate(training_loader):
        print('GE: ', gene_expression_data, ', lenght: ', gene_expression_data[0].numel(), ', index: ', batch_index)
    end_dataload_time = time.time()
    print('Average dataload time: {:.2f}'.format((end_dataload_time - start_dataload_time) / len(dataset)))

    # Validation set
    print("\nVALIDATION SET:")
    validation_loader = DataLoader(testing_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=6, pin_memory=True)
    start_dataload_time = time.time()
    for batch_index, (survival_months, survival_class, censorship, gene_expression_data) in enumerate(validation_loader):
        print('GE: ', gene_expression_data, ', lenght: ', gene_expression_data[0].numel(), ', index: ', batch_index)
    end_dataload_time = time.time()
    print('Average dataload time: {:.2f}'.format((end_dataload_time - start_dataload_time) / len(dataset)))

    print('Test successful')


## MAIN
if __name__ == '__main__':
    # test_rnaseq_dataset()
    test_rnaseq_dataset_split()
