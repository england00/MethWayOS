import numpy as np
import cupy as cp
import dask.dataframe as dd
import pandas as pd
import time
import torch
import yaml
from data.methods.csv_dataset_loader import csv_loader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader


## CLASSES
class MultimodalDataset(Dataset):
    def __init__(self, config, classes_number: int = 4, use_signatures: bool = False, random_seed: int = None, remove_incomplete_samples: bool = True, block_size: int = 10000):
        # GENERAL
        self.random_seed = random_seed

        # GENE EXPRESSION: Loading CSV Dataset
        print('DATASET: Loading data')
        self.gene_expression, _ = csv_loader(config['dataset']['gene_expression'], verbose=False)
        self.gene_expression.reset_index(drop=True, inplace=True)
        if 'Unnamed: 0' in self.gene_expression.columns:
            self.gene_expression = self.gene_expression.drop(columns=['Unnamed: 0'])
        print('--> Gene Expression loaded')

        # METHYLATION: Loading CSV Dataset
        self.methylation, _ = csv_loader(config['dataset']['methylation'], verbose=False)
        self.methylation.reset_index(drop=True, inplace=True)
        if 'Unnamed: 0' in self.methylation.columns:
            self.methylation = self.methylation.drop(columns=['Unnamed: 0'])
        print('--> Methylation loaded')

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
        if self.classes_number != 4 and self.classes_number >= 2:
            self.gene_expression['survival_class'], class_intervals = pd.qcut(self.gene_expression['survival_months'], q=self.classes_number, retbins=True, labels=False)
            print('--> Class intervals: [')
            for i in range(0, classes_number):
                print('\t{}: [{:.2f} - {:.2f}]'.format(i, class_intervals[i], class_intervals[i + 1]))
            print('    ]')
        else:
            intervals = [0, 12, 36, 60, np.inf]
            labels = ['[0, 12) --> High Risk',
                      '[12, 36) --> Medium Risk',
                      '[36, 60) --> Low Risk',
                      '[60, inf) --> No Risk']
            survival_class = pd.cut(self.gene_expression['survival_months'], bins=intervals, labels=labels, right=False)
            class_frequencies = survival_class.value_counts()
            print('--> Class intervals: ', class_frequencies)
            self.gene_expression['survival_class'] = survival_class.cat.codes.astype('int64')
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

        # GENE EXPRESSION: RNA-Seq
        self.rnaseq = self.gene_expression.iloc[:, self.gene_expression.columns.str.endswith('_rnaseq')].astype(float)
        self.rnaseq_size = len(self.rnaseq.columns)
        self.rnaseq = torch.tensor(self.rnaseq.values, dtype=torch.float32)
        print('--> RNA-seq loaded')

        # METHYLATION: Meth-Islands
        meth_columns_mask = self.methylation.columns.str.endswith('_meth')  # Boolean mask
        meth_data = self.methylation.loc[:, meth_columns_mask].to_numpy(dtype='float32')
        self.meth = torch.from_numpy(meth_data)
        self.meth_size = self.meth.shape[1]
        print('--> Methylation loaded')

        # GENE EXPRESSION: Managing Signatures
        self.use_signatures = use_signatures
        self.gene_expression_signature_sizes = []
        self.gene_expression_signature_data = {}
        signatures_df = pd.read_csv(config['dataset']['gene_expression_signatures'])
        self.gene_expression_signatures = list(signatures_df.columns)
        for signature in self.gene_expression_signatures:
            columns = {}
            for gene in signatures_df[signature].dropna():
                gene += '_rnaseq'
                if gene in self.gene_expression.columns:
                    if self.gene_expression[gene].std() > 0 and pd.Series(self.survival_months, index=self.gene_expression.index).std() > 0:
                        correlation = self.gene_expression[gene].corr(pd.Series(self.survival_months, index=self.gene_expression.index))
                    else:
                        correlation = 0
                    if abs(correlation) >= config['dataset']['gene_expression_correlation_threshold']:
                        columns[gene] = self.gene_expression[gene]
            dataframe = torch.tensor(pd.DataFrame(columns).values, dtype=torch.float32)
            if dataframe.shape[0] != 0:
                self.gene_expression_signature_data[signature] = dataframe
                self.gene_expression_signature_sizes.append(self.gene_expression_signature_data[signature].shape[1])
        self.gene_expression_signatures = list(self.gene_expression_signature_data.keys())
        print(f'--> {len(self.gene_expression_signature_sizes)} Gene Expression Signatures with sizes: {self.gene_expression_signature_sizes}')

        # METHYLATION: Managing Signatures
        self.methylation_signature_sizes = []
        self.methylation_signature_data = {}
        signatures_df = pd.read_csv(config['dataset']['methylation_signatures'], dtype=str)
        self.methylation_signatures = list(signatures_df.columns)
        for signature in self.methylation_signatures:
            if str(config['dataset']['gene_expression_signatures']) == '../../data/signatures/mcat_signatures_transformed.csv':
                if signature in self.gene_expression_signatures :
                    columns = {}
                    for island in signatures_df[signature].dropna():
                        island += '_meth'
                        if island in self.methylation.columns:
                            if self.methylation[island].std() > 0 and pd.Series(self.survival_months, index=self.methylation.index).std() > 0:
                                correlation = self.methylation[island].corr(pd.Series(self.survival_months, index=self.methylation.index))
                            else:
                                correlation = 0
                            if abs(correlation) >= config['dataset']['methylation_islands_correlation_threshold']:
                                columns[island] = self.methylation[island]
                    if config['dataset']['methylation_islands_statistics']:
                        dataframe = pd.DataFrame(columns)
                    else:
                        dataframe = torch.tensor(pd.DataFrame(columns).values, dtype=torch.float32)
                    if dataframe.shape[0] != 0 and dataframe.shape[1] >= int(config['dataset']['methylation_island_number_per_column']):
                        if config['dataset']['methylation_islands_statistics']:
                            tensor_data = torch.tensor(dataframe.values, dtype=torch.float32)
                            mean_values = torch.mean(tensor_data, dim=1, keepdim=True)
                            std_values = torch.std(tensor_data, dim=1, unbiased=False, keepdim=True)
                            max_values = torch.max(tensor_data, dim=1, keepdim=True)[0]
                            min_values = torch.min(tensor_data, dim=1, keepdim=True)[0]
                            summary_tensor = torch.cat([mean_values, std_values, max_values, min_values], dim=1)
                            self.methylation_signature_data[signature] = summary_tensor
                            self.methylation_signature_sizes.append(summary_tensor.shape[1])
                        else:
                            self.methylation_signature_data[signature] = dataframe
                            self.methylation_signature_sizes.append(self.methylation_signature_data[signature].shape[1])
            else:
                columns = {}
                for island in signatures_df[signature].dropna():
                    island += '_meth'
                    if island in self.methylation.columns:
                        if self.methylation[island].std() > 0 and pd.Series(self.survival_months,
                                                                            index=self.methylation.index).std() > 0:
                            correlation = self.methylation[island].corr(
                                pd.Series(self.survival_months, index=self.methylation.index))
                        else:
                            correlation = 0
                        if abs(correlation) >= config['dataset']['methylation_islands_correlation_threshold']:
                            columns[island] = self.methylation[island]
                if config['dataset']['methylation_islands_statistics']:
                    dataframe = pd.DataFrame(columns)
                else:
                    dataframe = torch.tensor(pd.DataFrame(columns).values, dtype=torch.float32)
                if dataframe.shape[0] != 0 and dataframe.shape[1] >= int(
                        config['dataset']['methylation_island_number_per_column']):
                    if config['dataset']['methylation_islands_statistics']:
                        tensor_data = torch.tensor(dataframe.values, dtype=torch.float32)
                        mean_values = torch.mean(tensor_data, dim=1, keepdim=True)
                        std_values = torch.std(tensor_data, dim=1, unbiased=False, keepdim=True)
                        max_values = torch.max(tensor_data, dim=1, keepdim=True)[0]
                        min_values = torch.min(tensor_data, dim=1, keepdim=True)[0]
                        summary_tensor = torch.cat([mean_values, std_values, max_values, min_values], dim=1)
                        self.methylation_signature_data[signature] = summary_tensor
                        self.methylation_signature_sizes.append(summary_tensor.shape[1])
                    else:
                        self.methylation_signature_data[signature] = dataframe
                        self.methylation_signature_sizes.append(self.methylation_signature_data[signature].shape[1])
        self.methylation_signatures = list(self.methylation_signature_data.keys())
        print(f'--> {len(self.methylation_signature_sizes)} Methylation Signatures with sizes: {self.methylation_signature_sizes}')

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

    def split(self, training_size):
        # GENERAL: Ensure training_size is a valid ratio
        if not 0 < training_size < 1:
            raise ValueError("training_size should be a float between 0 and 1.")

        # GENERAL: Get unique patients and their corresponding survival class labels
        unique_patients = self.gene_expression['case_id'].unique()
        patient_survival_class = self.gene_expression.groupby('case_id')['survival_class'].first().loc[unique_patients]

        # GENERAL: Perform stratified split
        training_patients, testing_patients = train_test_split(
            unique_patients,
            train_size=training_size,
            stratify=patient_survival_class,
            random_state=self.random_seed
        )

        # GENERAL: Filter and Reset indices for training datasets
        training_data_ge = self.gene_expression[self.gene_expression['case_id'].isin(training_patients)].reset_index(drop=True)
        training_data_meth = self.methylation[self.methylation['case_id'].isin(training_patients)].reset_index(drop=True)
        training_data_meth = training_data_meth.set_index('case_id').reindex(training_data_ge['case_id']).reset_index()

        # GENERAL: Filter and Reset indices for testing datasets
        testing_data_ge = self.gene_expression[self.gene_expression['case_id'].isin(testing_patients)].reset_index(drop=True)
        testing_data_meth = self.methylation[self.methylation['case_id'].isin(testing_patients)].reset_index(drop=True)
        testing_data_meth = testing_data_meth.set_index('case_id').reindex(testing_data_ge['case_id']).reset_index()

        # GENERAL: Create new instances of MultimodalDataset with the train and validation data
        training_dataset = MultimodalDataset.from_dataframes(training_data_ge, training_data_meth, self)
        testing_dataset = MultimodalDataset.from_dataframes(testing_data_ge, testing_data_meth, self)

        return training_dataset, testing_dataset

    def k_fold_split(self, training_dataset, k):
        # GENERAL: Ensure k is a valid ratio
        if k < 2:
            raise ValueError("Number of folds k must be at least 2")

        # GENERAL: Extract dataframes from training dataset
        ge_dataframe = training_dataset.gene_expression
        meth_dataframe = training_dataset.methylation

        # GENERAL: Extract unique patient IDs and their corresponding survival class labels
        unique_patients = ge_dataframe['case_id'].unique()
        patient_survival_class = ge_dataframe.groupby('case_id')['survival_class'].first().loc[unique_patients]

        # GENERAL: Define stratified k-fold split
        stratified_k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_seed)

        # GENERAL: Extract folds
        folds = []
        for training_indices, validation_indices in stratified_k_fold.split(unique_patients, patient_survival_class):

            # GENERAL: Splitting patients into train and validation sets
            training_patients = unique_patients[training_indices]
            validation_patients = unique_patients[validation_indices]

            # GENERAL: Filter and Reset indices for training datasets
            training_ge = ge_dataframe[ge_dataframe['case_id'].isin(training_patients)].reset_index(drop=True)
            training_meth = meth_dataframe[meth_dataframe['case_id'].isin(training_patients)].reset_index(drop=True)
            training_meth = training_meth.set_index('case_id').reindex(training_ge['case_id']).reset_index()

            # GENERAL: Filter and Reset indices for validation datasets
            validation_ge = ge_dataframe[ge_dataframe['case_id'].isin(validation_patients)].reset_index(drop=True)
            validation_meth = meth_dataframe[meth_dataframe['case_id'].isin(validation_patients)].reset_index(drop=True)
            validation_meth = validation_meth.set_index('case_id').reindex(validation_ge['case_id']).reset_index()

            # GENERAL: Create MultimodalDataset instances
            training_fold = MultimodalDataset.from_dataframes(training_ge, training_meth, self)
            validation_fold = MultimodalDataset.from_dataframes(validation_ge, validation_meth, self)

            # GENERAL: Storing current fold
            folds.append((training_fold, validation_fold))

        return folds

    @classmethod
    def from_dataframes(cls, ge_dataframe, meth_dataframe, original_instance):
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
        meth_dataframe = meth_dataframe.reset_index(drop=True)
        instance.methylation = meth_dataframe
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
        meth = meth_dataframe.iloc[:, meth_dataframe.columns.str.endswith('_meth')].astype(float)
        if original_instance.meth_size == len(meth.columns):
            instance.meth = torch.tensor(meth.values, dtype=torch.float32)
        else:
            instance.meth = torch.tensor(np.zeros((len(meth_dataframe), original_instance.meth_size)), dtype=torch.float32)

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
                indices = meth_dataframe.index
                instance.methylation_signature_data[signature_name] = signature_data[indices]

        return instance


## TEST FUNCTIONS
def test_multimodal_dataset():
    print('Testing MultimodalDataset...')

    # Loading Configuration file
    with open('../../../config/files/mcat_gene_expression_and_methylation.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config['dataset']['gene_expression'] = '../../../data/datasets/MCAT_gene_expression_and_overall_survival_dataset.csv'
    config['dataset']['gene_expression_signatures'] = '../../../data/tables/gene_expression_keys_methylation.csv'
    config['dataset']['methylation'] = '../../../data/datasets/MCAT_methylation450_and_overall_survival_dataset.csv'
    config['dataset']['methylation_signatures'] = '../../../data/tables/methylation450_keys.csv'

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
    with open('../../../config/files/mcat_gene_expression_and_methylation.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config['dataset']['gene_expression'] = '../../../data/datasets/MCAT_gene_expression_and_overall_survival_dataset.csv'
    config['dataset']['gene_expression_signatures'] = '../../../data/tables/gene_expression_keys_methylation.csv'
    config['dataset']['methylation'] = '../../../data/datasets/MCAT_methylation450_and_overall_survival_dataset.csv'
    config['dataset']['methylation_signatures'] = '../../../data/tables/methylation450_keys.csv'

    # Testing
    dataset = MultimodalDataset(config['dataset']['file'], config, use_signatures=True)
    training_dataset, testing_dataset = dataset.split(config['training']['train_size'])
    assert len(training_dataset) > len(testing_dataset)

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
    validation_loader = DataLoader(testing_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=6, pin_memory=True)
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
