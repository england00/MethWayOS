import numpy as np
import cupy as cp
import dask.dataframe as dd
import pandas as pd
import time
import torch
import yaml
from data.methods.csv_dataset_loader import csv_loader
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader


## CLASSES
class MethylationDataset(Dataset):
    def __init__(self, config, classes_number: int = 4, use_signatures: bool = False, random_seed: int = None, block_size: int = 10000):
        # GENERAL
        self.random_seed = random_seed

        # METHYLATION: Loading CSV Dataset
        self.methylation, _ = csv_loader(config['dataset']['methylation'], verbose=False)
        self.methylation.reset_index(drop=True, inplace=True)
        if 'Unnamed: 0' in self.methylation.columns:
            self.methylation = self.methylation.drop(columns=['Unnamed: 0'])
        print('--> Methylation loaded')

        # GENERAL: Managing Classes creation (based with Gene Expression)
        self.classes_number = classes_number
        survival_class, class_intervals = pd.qcut(self.methylation['survival_months'], q=self.classes_number, retbins=True, labels=False)
        self.methylation['survival_class'] = survival_class
        print('--> Class intervals: [')
        for i in range(0, classes_number):
            print('\t{}: [{:.2f} - {:.2f}]'.format(i, class_intervals[i], class_intervals[i + 1]))
        print('    ]')
        self.survival_months = self.methylation['survival_months'].values
        self.survival_class = self.methylation['survival_class'].values
        self.censorship = self.methylation['censorship'].values

        # METHYLATION: Managing columns (Standardization or Normalization)
        meth_columns = [col for col in self.methylation.columns if col.endswith('_meth')]
        methylation_gpu = cp.array(self.methylation[meth_columns].values)
        self.meth_size = methylation_gpu.shape[1]
        for start in range(0, self.meth_size, block_size):
            end = min(start + block_size, self.meth_size)
            block = methylation_gpu[:, start:end]
            if config['dataset']['standardize']:
                print(f'--> Standardizing Methylation Islands columns from {start} to {end}')
                means = cp.mean(block, axis=0)
                stds = cp.std(block, axis=0)
                standardized = (block - means) / cp.where(stds == 0, cp.nan, stds)
                standardized = cp.nan_to_num(standardized)  # Change NaN with 0
                methylation_gpu[:, start:end] = standardized
            elif config['dataset']['normalize']:
                print(f'--> Normalizing Methylation Islands columns from {start} to {end}')
                minimum = cp.min(block, axis=0)
                maximus = cp.max(block, axis=0)
                ranges = cp.where(maximus - minimum == 0, cp.nan, maximus - minimum)
                normalized = 2 * (block - minimum) / ranges - 1
                normalized = cp.nan_to_num(normalized, nan=0.5)  # Change NaN with 0.5
                methylation_gpu[:, start:end] = normalized
        methylation_gpu = methylation_gpu.astype(cp.float32)
        chunks = []
        for start in range(0, methylation_gpu.shape[1], block_size):
            end = min(start + block_size, methylation_gpu.shape[1])
            print(f"--> Processing Methylation Islands columns from {start} to {end}")
            chunk = methylation_gpu[:, start:end]
            dask_chunk = dd.from_pandas(pd.DataFrame(cp.asnumpy(chunk), columns=meth_columns[start:end]), npartitions=8)
            chunks.append(dask_chunk)
        dask_df = dd.concat(chunks, axis=1)
        self.methylation[meth_columns] = dask_df.compute()

        # METHYLATION: Meth-Islands
        meth_columns_mask = self.methylation.columns.str.endswith('_meth')  # Boolean mask
        meth_data = self.methylation.loc[:, meth_columns_mask].to_numpy(dtype='float32')
        self.meth = torch.from_numpy(meth_data)
        self.meth_size = self.meth.shape[1]
        print('--> Methylation loaded')

        # METHYLATION: Managing Signatures
        self.use_signatures = use_signatures
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

        # METHYLATION: Managing data
        methylation_data = []
        for signature in self.methylation_signatures:
            signature_data = self.methylation_signature_data[signature][index]
            methylation_data.append(signature_data)

        return survival_months, survival_class, censorship, methylation_data

    def __len__(self):
        return len(self.methylation)

    def split(self, training_size):
        # GENERAL: Ensure training_size is a valid ratio
        if not 0 < training_size < 1:
            raise ValueError("training_size should be a float between 0 and 1.")

        # GENERAL: Get unique patients and Shuffle randomly
        unique_patients = self.methylation['case_id'].unique()
        np.random.seed(self.random_seed)
        np.random.shuffle(unique_patients)
        training_patient_count = int(len(unique_patients) * training_size)

        # GENERAL: Split patients into train and validation sets
        training_patients = unique_patients[:training_patient_count]
        testing_patients = unique_patients[training_patient_count:]

        # GENERAL: Filter and Reset indices for training datasets
        training_data_meth = self.methylation[self.methylation['case_id'].isin(training_patients)].reset_index(drop=True)

        # GENERAL: Filter and Reset indices for testing datasets
        testing_data_meth = self.methylation[self.methylation['case_id'].isin(testing_patients)].reset_index(drop=True)

        # GENERAL: Create new instances of MultimodalDataset with the train and validation data
        training_dataset = MethylationDataset.from_dataframes(training_data_meth, self)
        testing_dataset = MethylationDataset.from_dataframes(testing_data_meth, self)

        return training_dataset, testing_dataset

    def k_fold_split(self, training_dataset, k):
        # GENERAL: Ensure k is a valid ratio
        if k < 2:
            raise ValueError("Number of folds k must be at least 2")

        # GENERAL: Extract dataframes from training dataset
        meth_dataframe = training_dataset.methylation

        # GENERAL: Extract unique patient IDs
        unique_patients = meth_dataframe['case_id'].unique()
        k_fold = KFold(n_splits=k, shuffle=True, random_state=self.random_seed)

        # GENERAL: Extract folds
        folds = []
        for training_indices, validation_indices in k_fold.split(unique_patients):
            # GENERAL: Splitting patients into train and validation sets
            training_patients = unique_patients[training_indices]
            validation_patients = unique_patients[validation_indices]

            # GENERAL: Filter and Reset indices for training datasets
            training_meth = meth_dataframe[meth_dataframe['case_id'].isin(training_patients)].reset_index(drop=True)

            # GENERAL: Filter and Reset indices for validation datasets
            validation_meth = meth_dataframe[meth_dataframe['case_id'].isin(validation_patients)].reset_index(drop=True)

            # GENERAL: Create MultimodalDataset instances
            training_fold = MethylationDataset.from_dataframes(training_meth, self)
            validation_fold = MethylationDataset.from_dataframes(validation_meth, self)

            # GENERAL: Storing current fold
            folds.append((training_fold, validation_fold))

        return folds

    @classmethod
    def from_dataframes(cls, meth_dataframe, original_instance):
        # Create a new MultimodalDataset instance from an existing DataFrame
        # while preserving the original configuration and parameters (without calling __init__)
        instance = cls.__new__(cls)
        instance.use_signatures = original_instance.use_signatures

        # METHYLATION: Reset indices in the DataFrame
        meth_dataframe = meth_dataframe.reset_index(drop=True)
        instance.methylation = meth_dataframe
        if original_instance.use_signatures:
            instance.methylation_signatures = original_instance.methylation_signatures
            instance.methylation_signature_sizes = original_instance.methylation_signature_sizes

        # GENERAL: Copy RNA within the new subset of data
        instance.survival_months = meth_dataframe['survival_months'].values
        instance.survival_class = meth_dataframe['survival_class'].values
        instance.censorship = meth_dataframe['censorship'].values

        # METHYLATION: Meth-Islands
        meth = meth_dataframe.iloc[:, meth_dataframe.columns.str.endswith('_meth')].astype(float)
        if original_instance.meth_size == len(meth.columns):
            instance.meth = torch.tensor(meth.values, dtype=torch.float32)
        else:
            instance.meth = torch.tensor(np.zeros((len(meth_dataframe), original_instance.meth_size)), dtype=torch.float32)

        # METHYLATION: Copy signature data if use_signatures is True
        if original_instance.use_signatures:
            instance.methylation_signature_sizes = original_instance.methylation_signature_sizes
            instance.methylation_signature_data = {}
            for signature_name in original_instance.methylation_signatures:
                signature_data = original_instance.methylation_signature_data[signature_name]
                indices = meth_dataframe.index
                instance.methylation_signature_data[signature_name] = signature_data[indices]

        return instance


## TEST FUNCTIONS
def test_methylation_dataset():
    print('Testing MultimodalDataset...')

    # Loading Configuration file
    with open('../../../config/files/smt_methylation.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config['dataset']['methylation'] = '../../../data/datasets/MCAT_methylation27_full_and_overall_survival_dataset.csv'
    config['dataset']['methylation_signatures'] = '../../../data/tables/methylation27_full_keys.csv'

    # Testing
    dataset = MethylationDataset(config,
                                 classes_number=config['training']['classes_number'],
                                 use_signatures=True,
                                 random_seed=config['dataset']['random_seed'])  # Dataset object
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    start_dataload_time = time.time()
    for batch_index, (survival_months, survival_class, censorship, methylation_data) in enumerate(loader):
        print('ME: ', methylation_data, ', lenght: ', methylation_data[0].numel(), ', index: ', batch_index)
    end_dataload_time = time.time()
    print('Average dataload time: {:.2f}'.format((end_dataload_time - start_dataload_time) / len(dataset)))

    print('Test successful')


def test_methylation_dataset_split():
    print('Testing MultimodalDataset...')

    # Loading Configuration file
    with open('../../../config/files/smt_methylation.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config['dataset']['methylation'] = '../../../data/datasets/MCAT_methylation27_full_and_overall_survival_dataset.csv'
    config['dataset']['methylation_signatures'] = '../../../data/tables/methylation27_full_keys.csv'

    # Testing
    dataset = MethylationDataset(config,
                                 classes_number=config['training']['classes_number'],
                                 use_signatures=True,
                                 random_seed=config['dataset']['random_seed'])  # Dataset object
    training_dataset, testing_dataset = dataset.split(config['training']['train_size'])
    assert len(training_dataset) > len(testing_dataset)

    # Training set
    print("\nTRAINING SET:")
    training_loader = DataLoader(training_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=6, pin_memory=True)
    start_dataload_time = time.time()
    for batch_index, (survival_months, survival_class, censorship, methylation_data) in enumerate(training_loader):
        print('ME: ', methylation_data, ', lenght: ', methylation_data[0].numel(), ', index: ', batch_index)
    end_dataload_time = time.time()
    print('Average dataload time: {:.2f}'.format((end_dataload_time - start_dataload_time) / len(dataset)))

    # Validation set
    print("\nVALIDATION SET:")
    validation_loader = DataLoader(testing_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=6, pin_memory=True)
    start_dataload_time = time.time()
    for batch_index, (survival_months, survival_class, censorship, methylation_data) in enumerate(validation_loader):
        print('ME: ', methylation_data, ', lenght: ', methylation_data[0].numel(), ', index: ', batch_index)
    end_dataload_time = time.time()
    print('Average dataload time: {:.2f}'.format((end_dataload_time - start_dataload_time) / len(dataset)))

    print('Test successful')


## MAIN
if __name__ == '__main__':
    # test_methylation_dataset()
    test_methylation_dataset_split()
