<div id="top">

<!-- HEADER STYLE: COMPACT -->
<img src="docs/logo.jpg" width="30%" align="left" style="margin-right: 15px">

# METHWAYOS
<em></em>

<!-- BADGES -->
<!-- local repository, no metadata badges. -->

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/Flask-000000.svg?style=flat-square&logo=Flask&logoColor=white" alt="Flask">
<img src="https://img.shields.io/badge/JSON-000000.svg?style=flat-square&logo=JSON&logoColor=white" alt="JSON">
<img src="https://img.shields.io/badge/Dask-FC6E6B.svg?style=flat-square&logo=Dask&logoColor=white" alt="Dask">
<img src="https://img.shields.io/badge/TOML-9C4121.svg?style=flat-square&logo=TOML&logoColor=white" alt="TOML">
<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikitlearn">
<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=flat-square&logo=tqdm&logoColor=black" alt="tqdm">
<img src="https://img.shields.io/badge/SymPy-3B5526.svg?style=flat-square&logo=SymPy&logoColor=white" alt="SymPy">
<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=flat-square&logo=GNU-Bash&logoColor=white" alt="GNU%20Bash">
<br>
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat-square&logo=NumPy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Numba-00A3E0.svg?style=flat-square&logo=Numba&logoColor=white" alt="Numba">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/AIOHTTP-2C5BB4.svg?style=flat-square&logo=AIOHTTP&logoColor=white" alt="AIOHTTP">
<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat-square&logo=SciPy&logoColor=white" alt="SciPy">
<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat-square&logo=pandas&logoColor=white" alt="pandas">
<img src="https://img.shields.io/badge/YAML-CB171E.svg?style=flat-square&logo=YAML&logoColor=white" alt="YAML">

<br clear="left"/>

## ☀️ Table of Contents

- [☀ ️ Table of Contents](#-table-of-contents)
- [🌞 Overview](#-overview)
- [🔥 Features](#-features)
- [🌅 Project Structure](#-project-structure)
    - [🌄 Project Index](#-project-index)
- [✨ Credits](#-credits)

---

## 🌞 Overview

Predicting overall survival in medical research has been a significant challenge, often tackled by leveraging multi-modal 
data with advanced Machine Learning and Deep Learning techniques. Despite notable progress, persistent difficulties in
model accuracy and interpretability remain, prompting the search for alternative and more effective approaches.

In response to this, I developed MethWayOS, a novel model that integrates an attention mechanism within cross-modal 
analysis. This model focuses on the relationship between two critical omics datasets: RNA gene expression and DNA
methylation. By incorporating an attention-driven framework, MethWayOS is capable of dynamically weighing the
contributions of each modality, enabling the model to better capture complex biological interactions and improve both 
its performance and interpretability.

MethWayOS has been validated using case study data from The Cancer Genome Atlas, achieving promising results in
overall survival prediction. The model not only targets state-of-the-art performance but also seeks to enhance the 
interpretability of the relationship between gene expression and DNA methylation, two key omics data in cancer research. 
Additionally, the learned attention maps provide insights into the biological processes driving the survival predictions, 
which could aid in clinical decision-making.

---

## 🔥 Features

| Feature                       | Description                                                                                 |
|-------------------------------|---------------------------------------------------------------------------------------------|
| **Primary Language**          | Python                                                                                      |
| **Execution Environment**     | SLURM HPC                            |
| **Machine Learning**          | Scikit-Learn, deep learning workflows  |
| **Omics Data Types**          | DNA Methylation, Gene Expression                                                            |
| **Survival Analysis**         | Yes                                  |                                             |
| **Batch/Script System**       | Shell scripts, SLURM batch scripts for reproducible compute workflows            |
| **Environment Management**    | Conda                                                    |


---

## 🌅 Project Structure

```sh
└── MethWayOS/
    ├── bash.sh
    ├── config
    │   ├── files
    │   │   ├── mcat.yaml
    │   │   ├── mcat_gene_expression_and_methylation.yaml
    │   │   ├── mcat_gene_expression_and_methylation_single_execution.yaml
    │   │   ├── smt_gene_expression.yaml
    │   │   ├── smt_methylation.yaml
    │   │   └── methway_os_gene_expression_and_methylation.yaml
    │   ├── methods
    │   │   └── configuration_loader.py
    │   └── paths
    │       ├── dataset_paths.yaml
    │       ├── datastore_paths.yaml
    │       ├── directories_paths.yaml
    │       ├── json_paths.yaml
    │       └── table_paths.yaml
    ├── data
    │   ├── eda
    │   │   ├── attention4_map.png
    │   │   ├── attention_map.png
    │   │   ├── attention_maps.py
    │   │   ├── os_distribution.png
    │   │   ├── os_distribution.py
    │   │   ├── os_distribution_censored.png
    │   │   ├── os_distribution_censored_average.png
    │   │   ├── os_distribution_filtered.py
    │   │   ├── os_distribution_months.png
    │   │   ├── venn_diagram.png
    │   │   └── venn_diagram.py
    │   ├── methods
    │   │   ├── csv_dataset_loader.py
    │   │   ├── csv_dataset_storer.py
    │   │   ├── directory_loader.py
    │   │   ├── tsv_loader.py
    │   │   └── xml_loader.py
    │   └── signatures
    │       ├── gene_expression_keys_methylation.csv
    │       ├── gene_expression_keys_methylation_transformed.csv
    │       ├── gene_expression_keys_wsi.csv
    │       ├── mcat_signatures.csv
    │       ├── mcat_signatures_transformed.csv
    │       ├── methylation27_full_keys.csv
    │       ├── methylation27_keys.csv
    │       ├── methylation450_full_keys.csv
    │       ├── methylation450_keys.csv
    │       ├── methylation_signatures_complete.csv
    │       ├── methylation_signatures_full_27.csv
    │       ├── methylation_signatures_full_27_full.csv
    │       ├── methylation_signatures_full_450.csv
    │       ├── methylation_signatures_full_450_full.csv
    │       ├── methylation_signatures_promoters_27.csv
    │       ├── methylation_signatures_promoters_27_full.csv
    │       ├── methylation_signatures_promoters_450.csv
    │       ├── methylation_signatures_promoters_450_full.csv
    │       ├── methylation_signatures_with_functional_group.csv
    │       ├── methylation_signatures_with_pathway_combine.csv
    │       ├── methylation_signatures_with_pathway_hallmarks.csv
    │       ├── methylation_signatures_with_pathway_xena.csv
    │       ├── methway_os_pathway_combine_signatures.csv
    │       ├── methway_os_pathway_hallmarks_signatures.csv
    │       ├── methway_os_pathway_xena_signatures.csv
    │       └── methway_os_signatures.csv
    ├── docs
    │   ├── Grid Search.xlsx
    │   ├── Guidelines to Connect AImage Lab Server Space
    │   │   ├── Guidelines_to_Connect_AImage_Lab_Server_Space.md
    │   │   ├── img.png
    │   │   ├── img_1.png
    │   │   ├── img_2.png
    │   │   ├── img_3.png
    │   │   └── img_4.png
    │   └── logo.jpg
    ├── environment_deepprog.yml
    ├── error
    │   ├── configuration_file_error.py
    │   └── general_error.py
    ├── json_dir
    │   ├── indexes
    │   │   └── gene_expression_tss_names.json
    │   ├── methods
    │   │   ├── json_loader.py
    │   │   └── json_storer.py
    │   └── paths
    │       ├── gene_expression.json
    │       ├── methylation.json
    │       └── overall_survival.json
    ├── LICENSE
    ├── logs
    │   ├── methods
    │   │   └── log_storer.py
    │   ├── slurm
    │   │   ├── slurm_err
    │   │   │   ├── *
    │   │   └── slurm_out
    │   │       ├── *
    │   └── tests
    │       ├── GENE EXPRESSION & METHYLATION & OS - Binary Classification
    │       │   └── Torch Trials
    │       │       ├── MLP (GPU) - 70,00% - 300 features.txt
    │       │       ├── MLP (GPU) - 87,50% - 39 features & 81,25% - 18,29 features.txt
    │       │       ├── MLP (GPU) - BEST RESULTS (200 features) - 87,50% .txt
    │       │       └── MLP (GPU) - BEST RESULTS (300 features) - 70,00%.txt
    │       ├── GENE EXPRESSION & METHYLATION STATISTICS & OS - Binary Classification
    │       │   └── Torch Trials
    │       │       ├── MLP (GPU) - 93,75 - 48 features.txt
    │       │       └── MLP (GPU) - BEST RESULTS (200 features) - 93,75%.txt
    │       ├── GENE EXPRESSION & OS - Binary Classification
    │       │   ├── Sklearn Trials
    │       │   │   ├── Decision Tree - 83% - 21 features.txt
    │       │   │   ├── MLP - 75% - 20 features.txt
    │       │   │   ├── MLP - 75% - 26 features.txt
    │       │   │   ├── Random Forest - 79% - 24 features.txt
    │       │   │   ├── Random Forest - 83% - 12 features.txt
    │       │   │   ├── SVC - 75% - 15 features.txt
    │       │   │   ├── SVC - 79% - 14 features.txt
    │       │   │   └── SVC - 79% - 30 features.txt
    │       │   └── Torch Trials
    │       │       ├── MLP (GPU) - 70,83% - 20 features.txt
    │       │       ├── MLP (GPU) - 70,83% - 26 features.txt
    │       │       ├── MLP (GPU) - 71,43% - 200 features.txt
    │       │       ├── MLP (GPU) - 74,07% - 200 features.txt.txt
    │       │       ├── MLP (GPU) - 75% - 20 features.txt
    │       │       ├── MLP (GPU) - 75% - 8,9,17 features.txt
    │       │       ├── MLP (GPU) - 77,78% - 200 features.txt
    │       │       ├── MLP (GPU) - 82,61% - 200 features - 1.txt
    │       │       ├── MLP (GPU) - 82,61% - 200 features.txt
    │       │       └── MLP (GPU) - BEST RESULTS (200 features) - 91,67% .txt
    │       ├── METHYLATION & OS - Binary Classification
    │       │   └── Torch Trials
    │       │       ├── MLP (GPU) - 68,75% - 10 features.txt
    │       │       ├── MLP (GPU) - 68,75% - 11 features.txt
    │       │       ├── MLP (GPU) - 68,75% - 6 features.txt
    │       │       ├── MLP (GPU) - 68,75% - 8 features.txt
    │       │       ├── MLP (GPU) - 72,00% - 50 features.txt
    │       │       ├── MLP (GPU) - 75,00% - 27 features & 68,75% - 37 features.txt
    │       │       └── MLP (GPU) - BEST RESULTS (200 features) - 87,50% .txt
    │       └── src
    │           ├── cuda_test.py
    │           └── logs_manager.py
    ├── README.md
    ├── requirements.txt
    └── src
        ├── binary_classification
        │   ├── 2.1_ge_&_os_sklearn.py
        │   ├── 2.1_ge_&_os_sklearn.sbatch
        │   ├── 2.2_methylation_&_os_sklearn.py
        │   ├── 2.2_methylation_&_os_sklearn.sbatch
        │   ├── 3.1_ge_&_os_gpu.py
        │   ├── 3.1_ge_&_os_gpu.sbatch
        │   ├── 3.2_methylation_&_os_gpu.py
        │   ├── 3.2_methylation_&_os_gpu.sbatch
        │   ├── 3.3_ge_&_methylation_&_os_gpu.py
        │   ├── 3.3_ge_&_methylation_&_os_gpu.sbatch
        │   ├── 4.1_ge_&_os_gpu_v1.py
        │   ├── 4.1_ge_&_os_gpu_v1.sbatch
        │   ├── 4.2_methylation_&_os_gpu_v1.py
        │   ├── 4.2_methylation_&_os_gpu_v1.sbatch
        │   ├── 4.3_ge_&_methylation_&_os_gpu_v1.py
        │   ├── 4.3_ge_&_methylation_&_os_gpu_v1.sbatch
        │   ├── 5_ge_&_methylation_statistics_&_os_gpu.py
        │   ├── 5_ge_&_methylation_statistics_&_os_gpu.sbatch
        │   ├── functions_sklearn
        │   │   ├── f10_testing.py
        │   │   ├── f1_dataset_acquisition.py
        │   │   ├── f2_exploratory_data_analysis.py
        │   │   ├── f3_features_preprocessing.py
        │   │   ├── f4_features_selection.py
        │   │   ├── f5_dataset_splitting.py
        │   │   ├── f6_models.py
        │   │   ├── f7_grid_search.py
        │   │   ├── f8_cross_validation_model_assessment.py
        │   │   └── f9_training.py
        │   ├── functions_torch
        │   │   ├── f10_training_bagging_ensemble.py
        │   │   ├── f10_training_kfold_voting.py
        │   │   ├── f10_training_single_model.py
        │   │   ├── f11_testing_kfold_voting.py
        │   │   ├── f11_testing_single_model.py
        │   │   ├── f1_dataset_acquisition_and_splitting.py
        │   │   ├── f4_features_selection_training_set.py
        │   │   ├── f5_features_selection_testing_set.py
        │   │   ├── f6_sklearn_to_torch.py
        │   │   ├── f7_hyperparameters.py
        │   │   ├── f8_grid_search.py
        │   │   └── f9_mlp_models.py
        │   └── model_weights
        │       └── Methylation & OS Binary Classification.pth
        ├── data_extractors
        │   ├── 0.0_case_id_extractor.py
        │   ├── 0.1_gene_expression_extractor.py
        │   ├── 0.2.1_methylation_extractor.py
        │   ├── 0.2.2_cgp950_methylation_islands_counter_per_gene.py
        │   ├── 0.2.3.1_methylation_vectors_for_each_gene.py
        │   ├── 0.2.3.2_methylation_vectors_for_each_gene.py
        │   ├── 0.2.4.1_gene_associated_methylation_extractor.py
        │   ├── 0.2.4.2_gene_associated_methylation_statistics_extractor.py
        │   ├── 0.3_overall_survival_extractor.py
        │   ├── 0.4_whole_slide_image_extractor.py
        │   ├── 0.5.1_mcat_genes_signatures.py
        │   └── 0.5.2_mcat_methylation_signatures.py
        ├── dataset_creators
        │   ├── 1.1_gene_expression_&_overall_survival.py
        │   ├── 1.2_methylation_&_overall_survival.py
        │   ├── 1.3.1_gene_expression_&_methylation_&_overall_survival.py
        │   ├── 1.3.2_gene_expression_&_methylation_&_overall_survival.py
        │   ├── 1.4.1_mcat_gene_expression_&_overall_survival_for_wsi.py
        │   ├── 1.4.2_mcat_gene_expression_&_overall_survival_for_methylation.py
        │   ├── 1.5_mcat_methylation_&_overall_survival.py
        │   ├── 1.6_deep_prog_gene_expression_&_overall_survival.py
        │   └── 1.7_deep_prog_methylation.py
        ├── deep_prog
        │   ├── 10.1_simple_deepprog_model.py
        │   ├── 10.2_ensemble_deepprog_model.py
        │   ├── 10.3_hcc_ensemble_deepprog_model.py
        │   ├── examples
        │   │   └── data
        │   │       ├── metadata_dummy.tsv
        │   │       ├── meth_dummy.tsv
        │   │       ├── mir_dummy.tsv
        │   │       ├── mir_test_dummy.tsv
        │   │       ├── rna_dummy.tsv
        │   │       ├── rna_test_dummy.tsv
        │   │       ├── rna_test_dummy2.tsv
        │   │       ├── survival_dummy.tsv
        │   │       ├── survival_test_dummy.tsv
        │   │       └── survival_test_dummy2.tsv
        │   ├── results
        │   │   ├── DONE_20_test_KIRCKIRP_stacked
        │   │   │   ├── test_KIRCKIRP_stacked_features_anticorrelated_scores_per_clusters.tsv
        │   │   │   ├── test_KIRCKIRP_stacked_features_scores_per_clusters.tsv
        │   │   │   ├── test_KIRCKIRP_stacked_full_labels.tsv
        │   │   │   ├── test_KIRCKIRP_stacked_KM_plot_boosting_full.pdf
        │   │   │   ├── test_KIRCKIRP_stacked_proba_KM_plot_boosting_full.pdf
        │   │   │   ├── test_KIRCKIRP_stacked_test_RNA_only_KM_plot_boosting_test.pdf
        │   │   │   ├── test_KIRCKIRP_stacked_test_RNA_only_proba_KM_plot_boosting_test.pdf
        │   │   │   ├── test_KIRCKIRP_stacked_test_RNA_only_supervised_test_kdeplot.html
        │   │   │   ├── test_KIRCKIRP_stacked_test_RNA_only_test_labels.tsv
        │   │   │   └── test_KIRCKIRP_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html
        │   │   ├── DONE_25_9_test_KIRCKIRP_stacked
        │   │   │   ├── test_KIRCKIRP_stacked_features_anticorrelated_scores_per_clusters.tsv
        │   │   │   ├── test_KIRCKIRP_stacked_features_scores_per_clusters.tsv
        │   │   │   ├── test_KIRCKIRP_stacked_full_labels.tsv
        │   │   │   ├── test_KIRCKIRP_stacked_KM_plot_boosting_full.pdf
        │   │   │   ├── test_KIRCKIRP_stacked_proba_KM_plot_boosting_full.pdf
        │   │   │   ├── test_KIRCKIRP_stacked_test_RNA_only_KM_plot_boosting_test.pdf
        │   │   │   ├── test_KIRCKIRP_stacked_test_RNA_only_proba_KM_plot_boosting_test.pdf
        │   │   │   ├── test_KIRCKIRP_stacked_test_RNA_only_supervised_test_kdeplot.html
        │   │   │   ├── test_KIRCKIRP_stacked_test_RNA_only_test_labels.tsv
        │   │   │   └── test_KIRCKIRP_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html
        │   │   ├── DONE_GBMLGG_stacked
        │   │   │   ├── test_GBMLGG_stacked_features_anticorrelated_scores_per_clusters.tsv
        │   │   │   ├── test_GBMLGG_stacked_features_scores_per_clusters.tsv
        │   │   │   ├── test_GBMLGG_stacked_full_labels.tsv
        │   │   │   ├── test_GBMLGG_stacked_KM_plot_boosting_full.pdf
        │   │   │   ├── test_GBMLGG_stacked_proba_KM_plot_boosting_full.pdf
        │   │   │   ├── test_GBMLGG_stacked_test_RNA_only_KM_plot_boosting_test.pdf
        │   │   │   ├── test_GBMLGG_stacked_test_RNA_only_proba_KM_plot_boosting_test.pdf
        │   │   │   ├── test_GBMLGG_stacked_test_RNA_only_supervised_test_kdeplot.html
        │   │   │   ├── test_GBMLGG_stacked_test_RNA_only_test_labels.tsv
        │   │   │   └── test_GBMLGG_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html
        │   │   ├── DONE_KIRCKIRP_stacked
        │   │   │   ├── test_KIRCKIRP_stacked_features_anticorrelated_scores_per_clusters.tsv
        │   │   │   ├── test_KIRCKIRP_stacked_features_scores_per_clusters.tsv
        │   │   │   ├── test_KIRCKIRP_stacked_full_labels.tsv
        │   │   │   ├── test_KIRCKIRP_stacked_KM_plot_boosting_full.pdf
        │   │   │   ├── test_KIRCKIRP_stacked_proba_KM_plot_boosting_full.pdf
        │   │   │   ├── test_KIRCKIRP_stacked_test_RNA_only_KM_plot_boosting_test.pdf
        │   │   │   ├── test_KIRCKIRP_stacked_test_RNA_only_proba_KM_plot_boosting_test.pdf
        │   │   │   ├── test_KIRCKIRP_stacked_test_RNA_only_supervised_test_kdeplot.html
        │   │   │   ├── test_KIRCKIRP_stacked_test_RNA_only_test_labels.tsv
        │   │   │   └── test_KIRCKIRP_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html
        │   │   ├── DONE_LUADLUST_stacked
        │   │   │   ├── test_LUADLUST_stacked_features_anticorrelated_scores_per_clusters.tsv
        │   │   │   ├── test_LUADLUST_stacked_features_scores_per_clusters.tsv
        │   │   │   ├── test_LUADLUST_stacked_full_labels.tsv
        │   │   │   ├── test_LUADLUST_stacked_KM_plot_boosting_full.pdf
        │   │   │   ├── test_LUADLUST_stacked_proba_KM_plot_boosting_full.pdf
        │   │   │   ├── test_LUADLUST_stacked_test_RNA_only_KM_plot_boosting_test.pdf
        │   │   │   ├── test_LUADLUST_stacked_test_RNA_only_proba_KM_plot_boosting_test.pdf
        │   │   │   ├── test_LUADLUST_stacked_test_RNA_only_supervised_test_kdeplot.html
        │   │   │   ├── test_LUADLUST_stacked_test_RNA_only_test_labels.tsv
        │   │   │   └── test_LUADLUST_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html
        │   │   ├── DONE_test_BRCA_stacked
        │   │   │   ├── test_BRCA_stacked_features_anticorrelated_scores_per_clusters.tsv
        │   │   │   ├── test_BRCA_stacked_features_scores_per_clusters.tsv
        │   │   │   ├── test_BRCA_stacked_full_labels.tsv
        │   │   │   ├── test_BRCA_stacked_KM_plot_boosting_full.pdf
        │   │   │   ├── test_BRCA_stacked_proba_KM_plot_boosting_full.pdf
        │   │   │   ├── test_BRCA_stacked_test_RNA_only_KM_plot_boosting_test.pdf
        │   │   │   ├── test_BRCA_stacked_test_RNA_only_proba_KM_plot_boosting_test.pdf
        │   │   │   ├── test_BRCA_stacked_test_RNA_only_supervised_test_kdeplot.html
        │   │   │   ├── test_BRCA_stacked_test_RNA_only_test_labels.tsv
        │   │   │   └── test_BRCA_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html
        │   │   ├── test_dummy
        │   │   │   ├── GE_dummy_encoder.h5
        │   │   │   ├── METH_dummy_encoder.h5
        │   │   │   ├── MIR_dummy_encoder.h5
        │   │   │   ├── test_dummy_dataset_dummy_KM_plot_test.pdf
        │   │   │   ├── test_dummy_dataset_dummy_test_labels.tsv
        │   │   │   ├── test_dummy_dataset_KM_plot_training_dataset.pdf
        │   │   │   └── test_dummy_dataset_training_set_labels.tsv
        │   │   ├── test_dummy_stacked
        │   │   │   ├── test_dummy_stacked_features_anticorrelated_scores_per_clusters.tsv
        │   │   │   ├── test_dummy_stacked_features_scores_per_clusters.tsv
        │   │   │   ├── test_dummy_stacked_full_labels.tsv
        │   │   │   ├── test_dummy_stacked_KM_plot_boosting_full.pdf
        │   │   │   ├── test_dummy_stacked_proba_KM_plot_boosting_full.pdf
        │   │   │   ├── test_dummy_stacked_TEST_DATA_1_KM_plot_boosting_test.pdf
        │   │   │   ├── test_dummy_stacked_TEST_DATA_1_proba_KM_plot_boosting_test.pdf
        │   │   │   ├── test_dummy_stacked_TEST_DATA_1_supervised_test_kdeplot.html
        │   │   │   ├── test_dummy_stacked_TEST_DATA_1_TEST_DATA_1_supervised_kdeplot.html
        │   │   │   └── test_dummy_stacked_TEST_DATA_1_test_labels.tsv
        │   │   └── test_hcc_stacked
        │   │       ├── test_hcc_stacked_features_anticorrelated_scores_per_clusters.tsv
        │   │       ├── test_hcc_stacked_features_scores_per_clusters.tsv
        │   │       ├── test_hcc_stacked_full_labels.tsv
        │   │       ├── test_hcc_stacked_KM_plot_boosting_full.pdf
        │   │       ├── test_hcc_stacked_proba_KM_plot_boosting_full.pdf
        │   │       ├── test_hcc_stacked_test_RNA_only_KM_plot_boosting_test.pdf
        │   │       ├── test_hcc_stacked_test_RNA_only_proba_KM_plot_boosting_test.pdf
        │   │       ├── test_hcc_stacked_test_RNA_only_supervised_test_kdeplot.html
        │   │       ├── test_hcc_stacked_test_RNA_only_test_labels.tsv
        │   │       └── test_hcc_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html
        │   └── simdeep
        │       ├── __init__.py
        │       ├── config.py
        │       ├── coxph_from_r.py
        │       ├── deepmodel_base.py
        │       ├── extract_data.py
        │       ├── plot_utils.py
        │       ├── simdeep_analysis.py
        │       ├── simdeep_boosting.py
        │       ├── simdeep_distributed.py
        │       ├── simdeep_multiple_dataset.py
        │       ├── simdeep_tuning.py
        │       ├── simdeep_utils.py
        │       ├── survival_model_utils.py
        │       └── survival_utils.py
        ├── mcat
        │   ├── 6_mcat.py
        │   ├── 6_mcat.sbatch
        │   ├── 7.1_mcat_gene_expression_and_methylation_early_stopping.py
        │   ├── 7.1_mcat_gene_expression_and_methylation_early_stopping.sbatch
        │   ├── 7.2_mcat_gene_expression_and_methylation_cross_validation.py
        │   ├── 7.2_mcat_gene_expression_and_methylation_cross_validation.sbatch
        │   ├── 8.1_smt_gene_expression_cross_validation.py
        │   ├── 8.1_smt_gene_expression_cross_validation.sbatch
        │   ├── 8.2_smt_methylation_cross_validation.py
        │   ├── 8.2_smt_methylation_cross_validation.sbatch
        │   ├── gene_expression_and_methylation_modules
        │   │   ├── dataset.py
        │   │   ├── mcat.py
        │   │   ├── testing.py
        │   │   ├── training.py
        │   │   └── validation.py
        │   ├── original_modules
        │   │   ├── blocks.py
        │   │   ├── dataset.py
        │   │   ├── fusion.py
        │   │   ├── loss.py
        │   │   ├── mcat.py
        │   │   ├── testing.py
        │   │   ├── training.py
        │   │   ├── utils.py
        │   │   └── validation.py
        │   └── single_modality_modules
        │       ├── dataset_gene_expression.py
        │       ├── dataset_methylation.py
        │       ├── smt.py
        │       ├── training.py
        │       └── validation.py
        └── methway_os
            ├── 9.1_methwayos_gene_expression_and_methylation_cross_validation.py
            ├── 9.1_methwayos_gene_expression_and_methylation_cross_validation.sbatch
            ├── gene_expression_and_methylation_modules
            │   ├── dataset.py
            │   ├── methway_os.py
            │   ├── testing.py
            │   ├── training.py
            │   └── validation.py
            └── original_modules
                ├── cross_attention.py
                ├── loss.py
                ├── methway_os.py
                └── utils.py
```

### 🌄 Project Index

<details open>
	<summary><b><code>METHWAYOS/</code></b></summary>
	<!-- __root__ Submodule -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ __root__</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/bash.sh'>bash.sh</a></b></td>
					<td style='padding: 8px;'>- Provides an automated process to prepare the computational environment for machine learning tasks by setting the appropriate CUDA version, activating the required Conda environment, and allocating necessary GPU and memory resources on the cluster<br>- Serves as the entry point for initializing reproducible and resource-efficient sessions within the broader project workflow, facilitating consistent development and experimentation across the codebase.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/environment_deepprog.yml'>environment_deepprog.yml</a></b></td>
					<td style='padding: 8px;'>- Certainly! To generate an effective summary, please provide the specific code file in question<br>- With the file and the given project structure, Ill craft a clear, high-level summary of its purpose and how it fits within the overall architecture.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/LICENSE'>LICENSE</a></b></td>
					<td style='padding: 8px;'>- Summary**The <code>LICENSE</code> file defines the legal terms under which the entire codebase can be used, modified, and distributed<br>- By including the GNU General Public License (GPL) v3.0, it ensures that the project remains open source and freely available to all users, while requiring that any derivative works also maintain these same freedoms<br>- This foundational document sets the permissions and obligations for contributors and users, supporting a collaborative and transparent software ecosystem throughout the project.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/requirements.txt'>requirements.txt</a></b></td>
					<td style='padding: 8px;'>- Dependency specification outlines all required third-party libraries and their versions necessary for building, running, and maintaining the entire project<br>- Encompassing components for asynchronous processing, scientific computing, machine learning, RESTful APIs, and data visualization, these requirements serve as the foundational backbone that ensures reproducibility, stability, and compatibility across the project’s diverse modules and functionalities.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- error Submodule -->
	<details>
		<summary><b>error</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ error</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/error\configuration_file_error.py'>configuration_file_error.py</a></b></td>
					<td style='padding: 8px;'>- Defines a specialized exception to handle errors related to misconfiguration or issues in configuration files within the project<br>- Centralizing configuration error handling enhances clarity and consistency across the codebase, allowing other modules to uniformly detect and respond to configuration problems while maintaining the separation of error types for more robust diagnostics and debugging throughout the application’s architecture.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/error\general_error.py'>general_error.py</a></b></td>
					<td style='padding: 8px;'>- Defines a unified mechanism for representing and communicating general errors throughout the codebase<br>- By encapsulating error messages in a dedicated exception class, it enables consistent error handling and clearer propagation of issues across various modules<br>- Integrating this approach streamlines debugging and supports maintainable application development, ensuring that unexpected situations are managed in a predictable and readable manner.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- config Submodule -->
	<details>
		<summary><b>config</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ config</b></code>
			<!-- files Submodule -->
			<details>
				<summary><b>files</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ config.files</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/config\files\mcat.yaml'>mcat.yaml</a></b></td>
							<td style='padding: 8px;'>- Centralizes configuration for the MCAT model experiments, defining hardware utilization, dataset sources, model parameters, and training routines<br>- Aligns experiment tracking with Weights & Biases and organizes paths for datasets, signatures, and outputs<br>- Enables consistent, reproducible runs across stages of data preprocessing, model training, validation, and evaluation, ensuring standardized execution aligned with the overall architecture for cancer genomics research.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/config\files\mcat_gene_expression_and_methylation.yaml'>mcat_gene_expression_and_methylation.yaml</a></b></td>
							<td style='padding: 8px;'>- Configuration orchestrates experiment reproducibility and model training for TCGA BRCA gene expression and methylation analysis<br>- Aligns dataset sources, model architecture, training hyperparameters, evaluation strategies, and experiment tracking to ensure consistency and facilitate cross-validation<br>- Integrates gene expression and methylation modalities, enabling systematic experimentation and streamlined workflow across the data pipeline, model checkpoints, and results management within the broader research framework.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/config\files\mcat_gene_expression_and_methylation_single_execution.yaml'>mcat_gene_expression_and_methylation_single_execution.yaml</a></b></td>
							<td style='padding: 8px;'>- Configuration orchestrates the experimental workflow for gene expression and methylation analysis by specifying data sources, preprocessing methods, model parameters, and training strategies<br>- It centralizes control over dataset selection, model fusion approaches, training schedules, and logging via Weights & Biases, enabling consistent, reproducible experimentation and integration within the broader architecture for evaluating MCAT-based multi-omics survival prediction on TCGA breast cancer data.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/config\files\smt_gene_expression.yaml'>smt_gene_expression.yaml</a></b></td>
							<td style='padding: 8px;'>- Defines the experimental configuration for running gene expression-based survival analysis with SMT models in the project<br>- Captures device, dataset sources, preprocessing options, model variants, training hyperparameters, and output logistics, ensuring standardized, reproducible setup for cross-validation on the TCGA-BRCA gene expression dataset<br>- Serves as a central control point for customizing and orchestrating experiments in alignment with the overall codebase architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/config\files\smt_methylation.yaml'>smt_methylation.yaml</a></b></td>
							<td style='padding: 8px;'>- Configuration of experiment parameters for training and evaluating a survival prediction model based on methylation data<br>- Defines device usage, dataset sources, preprocessing options, model variants, and training routines such as optimizer, learning rate, loss type, and early stopping<br>- Ensures experimental reproducibility, facilitates structured tracking via W&B, and enables flexible tuning for robust cross-validation within the broader architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/config\files\surv_path_gene_expression_and_methylation.yaml'>surv_path_gene_expression_and_methylation.yaml</a></b></td>
							<td style='padding: 8px;'>- Configuration defines experimental settings, model parameters, dataset locations, and training options for cross-validation studies of survival analysis on gene expression and methylation data<br>- Facilitates reproducibility and easy adjustment of workflows by centralizing control of device selection, data normalization, training schemes, and logging<br>- Integrates seamlessly with model training, evaluation pipelines, and experiment tracking infrastructure across the project.</td>
						</tr>
					</table>
				</blockquote>
			</details>
			<!-- methods Submodule -->
			<details>
				<summary><b>methods</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ config.methods</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/config\methods\configuration_loader.py'>configuration_loader.py</a></b></td>
							<td style='padding: 8px;'>- Configuration management is streamlined by enabling the secure and reliable loading of YAML-based settings into the application<br>- Serving as a foundational utility within the codebase, the logic ensures that configuration data is correctly parsed and validated, while also integrating error handling to maintain overall system stability<br>- This approach supports consistent, centralized configuration handling across the entire project architecture.</td>
						</tr>
					</table>
				</blockquote>
			</details>
			<!-- paths Submodule -->
			<details>
				<summary><b>paths</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ config.paths</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/config\paths\dataset_paths.yaml'>dataset_paths.yaml</a></b></td>
							<td style='padding: 8px;'>- Centralizes standardized paths to all gene expression, methylation, and overall survival datasets utilized across the project, enabling consistent and maintainable data access throughout the codebase<br>- Facilitates seamless integration of various datasets for different analytical tasks, including binary classification, multi-omics analysis, and external tool compatibility, ensuring robust data management and simplifying configuration for diverse machine learning workflows.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/config\paths\datastore_paths.yaml'>datastore_paths.yaml</a></b></td>
							<td style='padding: 8px;'>- Centralizes references to key data sources used throughout the project, organizing logical access to gene expression, methylation, overall survival, and whole slide image datasets<br>- Enables consistent and maintainable data retrieval across modules, supporting various downstream tasks such as classification, survival analysis, and multi-omics processing by standardizing datastore locations within the broader system architecture<br>- Facilitates reproducibility and scalability by unifying path management.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/config\paths\directories_paths.yaml'>directories_paths.yaml</a></b></td>
							<td style='padding: 8px;'>- Establishes standardized, centralized references for key dataset locations related to gene expression, methylation, and overall survival within the project<br>- Enables consistent data access across the codebase, supporting improved maintainability and reducing the likelihood of path misconfiguration<br>- Serves as a foundational configuration element for data-driven workflows handling biological and clinical research in the overall project architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/config\paths\json_paths.yaml'>json_paths.yaml</a></b></td>
							<td style='padding: 8px;'>- Centralize and organize references to key JSON resources related to gene expression, methylation, overall survival, and whole slide imaging data<br>- Enable consistent and maintainable access paths throughout the project, supporting core data ingestion and analysis workflows<br>- Facilitate seamless integration across modules by standardizing the location and retrieval of essential scientific datasets within the codebase architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/config\paths\table_paths.yaml'>table_paths.yaml</a></b></td>
							<td style='padding: 8px;'>- Defines centralized mappings to all major gene expression and methylation signature datasets used throughout the project, ensuring consistent access and organization of table paths across modules<br>- Serves as the authoritative reference for locating both raw and transformed signature data, supporting robust data loading, preprocessing, and analysis workflows fundamental to the projects genomics processing and biomarker discovery architecture.</td>
						</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<!-- json_dir Submodule -->
	<details>
		<summary><b>json_dir</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ json_dir</b></code>
			<!-- indexes Submodule -->
			<details>
				<summary><b>indexes</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ json_dir.indexes</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/json_dir\indexes\gene_expression_tss_names.json'>gene_expression_tss_names.json</a></b></td>
							<td style='padding: 8px;'>- Define the standardized column names used for gene expression transcription start site (TSS) datasets across the project<br>- By establishing these consistent data labels, the configuration ensures uniform indexing and seamless integration when processing, analyzing, or querying gene expression data within the architecture, facilitating reliable downstream workflows and data interoperability throughout the codebase.</td>
						</tr>
					</table>
				</blockquote>
			</details>
			<!-- methods Submodule -->
			<details>
				<summary><b>methods</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ json_dir.methods</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/json_dir\methods\json_loader.py'>json_loader.py</a></b></td>
							<td style='padding: 8px;'>- Enables reliable ingestion of JSON data into the application by loading and validating external JSON files, then providing their contents in a standardized format for further processing<br>- Plays a key role in abstracting file access and error management, ensuring that downstream components consistently receive structured data while gracefully handling missing or invalid inputs across the projects data-loading workflows.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/json_dir\methods\json_storer.py'>json_storer.py</a></b></td>
							<td style='padding: 8px;'>- Enable robust and consistent saving of data in JSON format, ensuring smooth data persistence across the application<br>- By centralizing the JSON write operation and unified error feedback, promote reliability and maintainability within the codebase<br>- Serve as a foundational utility for other modules that require structured data storage while supporting overall project stability and user feedback.</td>
						</tr>
					</table>
				</blockquote>
			</details>
			<!-- paths Submodule -->
			<details>
				<summary><b>paths</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ json_dir.paths</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/json_dir\paths\gene_expression.json'>gene_expression.json</a></b></td>
							<td style='padding: 8px;'>- Summary of <code>gene_expression.json</code>**The <code>gene_expression.json</code> file serves as a metadata descriptor for gene expression data files within the project<br>- It provides structured information linking specific gene expression datasets to related biological entities, such as aliquots and cases, and includes key attributes like data format, access level, and file details<br>- In the broader codebase architecture, this file enables consistent cataloging, discovery, and management of gene expression resources, supporting downstream analysis workflows and facilitating data integration across the project.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/json_dir\paths\methylation.json'>methylation.json</a></b></td>
							<td style='padding: 8px;'>- This file serves as a metadata descriptor for a methylation data file within the project<br>- Its primary purpose is to catalog and provide essential information—such as data format, access level, association with specific biological samples, and file identifiers—enabling the broader codebase to efficiently locate, validate, and integrate methylation data into larger workflows<br>- By organizing these metadata details, the file ensures that methylation datasets are discoverable, accessible, and correctly linked to their originating biological samples within the overall system architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/json_dir\paths\overall_survival.json'>overall_survival.json</a></b></td>
							<td style='padding: 8px;'>- Summary of <code>json_dir/paths/overall_survival.json</code> within the Project Architecture**This file serves as a structured metadata record, detailing the key attributes and associations for an overall survival clinical dataset within the project<br>- It maps the relationship between external clinical data files and specific study entities (such as patient cases), providing vital information needed for tracking data provenance, access, and integrity<br>- The presence of this metadata enables the codebase to efficiently locate, validate, and utilize clinical survival data, supporting downstream analyses and ensuring consistency across the larger biomedical data processing pipeline.</td>
						</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<!-- logs Submodule -->
	<details>
		<summary><b>logs</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ logs</b></code>
			<!-- methods Submodule -->
			<details>
				<summary><b>methods</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ logs.methods</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\methods\log_storer.py'>log_storer.py</a></b></td>
							<td style='padding: 8px;'>- Enables simultaneous logging of output to both the terminal and a designated file, ensuring that all messages are consistently captured in multiple locations<br>- Supports the projects broader logging strategy by enhancing traceability and accountability, which is particularly valuable for debugging and post-mortem analysis within a structured logging framework<br>- Contributes to robust system observability and aids in auditing application behavior.</td>
						</tr>
					</table>
				</blockquote>
			</details>
			<!-- slurm Submodule -->
			<details>
				<summary><b>slurm</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ logs.slurm</b></code>
					<!-- slurm_err Submodule -->
					<details>
						<summary><b>slurm_err</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ logs.slurm.slurm_err</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2518507.out'>lucas_job_2518507.out</a></b></td>
									<td style='padding: 8px;'>- Error log provides insight into the runtime behavior and potential issues encountered during execution of gene expression cross-validation workflows within the AIforBioinformatics project<br>- Highlights integration with experiment tracking via Weights & Biases and reveals a configuration-related issue affecting model optimization, offering essential feedback for troubleshooting and improving the training pipeline’s reliability and reproducibility.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2518670.out'>lucas_job_2518670.out</a></b></td>
									<td style='padding: 8px;'>- Logs and error output from a SLURM job document the process of running a gene expression and methylation cross-validation experiment tracked by Weights & Biases<br>- Captures initialization details, experiment metadata, and a critical runtime failure, aiding in both reproducibility and debugging within the broader AIforBioinformatics pipeline, specifically focusing on multimodal data analysis and workflow orchestration.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2518671.out'>lucas_job_2518671.out</a></b></td>
									<td style='padding: 8px;'>- Logging mechanism for SLURM-based deep learning experiments captures both the initialization of experiment tracking with Weights & Biases and the occurrence of a runtime error during gene expression cross-validation<br>- Provides essential run metadata and error tracebacks, enabling robust monitoring, auditability, and debugging in the broader bioinformatics model training pipeline<br>- Facilitates reproducibility and rapid issue identification across distributed training jobs.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2518672.out'>lucas_job_2518672.out</a></b></td>
									<td style='padding: 8px;'>- Error logging for a SLURM-managed machine learning experiment run, specifically detailing a runtime issue encountered during model training with PyTorch<br>- Captures the interaction with Weights & Biases for experiment tracking and provides visibility into failures in the training workflow, supporting reproducibility and streamlined debugging across the larger AIforBioinformatics project, which orchestrates cross-validation experiments on TCGA methylation data.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2518868.out'>lucas_job_2518868.out</a></b></td>
									<td style='padding: 8px;'>- Captures and reports the output and errors from a Slurm-managed machine learning experiment, specifically documenting the progress and logging details from a cross-validation run involving gene expression and methylation data<br>- Serves as both a run record for project monitoring via Weights & Biases and a diagnostic reference, highlighting critical issues such as disk quota limitations that may impact workflow reliability across the AIforBioinformatics project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2518869.out'>lucas_job_2518869.out</a></b></td>
									<td style='padding: 8px;'>- Execution record for a SLURM-scheduled gene expression cross-validation experiment, capturing the integration with Weights & Biases (wandb) for experiment tracking and highlighting a failure due to exceeded disk quota<br>- Serves as a diagnostic artifact within the project’s logging architecture, aiding in monitoring training runs, debugging resource issues, and ensuring reproducibility across distributed high-performance computing environments.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2518870.out'>lucas_job_2518870.out</a></b></td>
									<td style='padding: 8px;'>- Logs the operational output and error events from a cross-validation job focused on methylation data training within the broader AIforBioinformatics pipeline<br>- Captures both successful integration with experiment tracking tools and details any encountered runtime issues, such as storage limitations, helping users diagnose failures and monitor workflow execution as part of the project’s robust computational biology model training framework.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2518873.out'>lucas_job_2518873.out</a></b></td>
									<td style='padding: 8px;'>- Logs the progress and outcome of a machine learning experiment tracked with Weights & Biases within a SLURM-managed compute job<br>- Captures relevant experiment metadata, tracking information, and run URLs, as well as the eventual job cancellation event<br>- Supports monitoring, troubleshooting, and auditability within the broader AIforBioinformatics project, providing insights into both experiment tracking and job status during cross-validation on the TCGA-BRCA-ME dataset.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2519390.out'>lucas_job_2519390.out</a></b></td>
									<td style='padding: 8px;'>- Logs the status and progress of a machine learning experiment managed via SLURM and tracked with Weights & Biases, capturing authentication details, local run storage, synchronization status, and the eventual cancellation of the SLURM job<br>- Integrates WandB experiment management within the broader AIforBioinformatics project workflow, supporting experiment reproducibility, auditing, and troubleshooting as part of the cross-validation pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2519461.out'>lucas_job_2519461.out</a></b></td>
									<td style='padding: 8px;'>- Logs the execution results and error output of a Slurm-managed machine learning job that performs cross-validation on gene expression and methylation data<br>- Acts as a record for experiment management, WandB integration, and error tracking, revealing critical resource issues encountered during processing<br>- Supports reproducibility and debugging across the broader architecture for large-scale bioinformatics workflows within AIforBioinformatics.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2519471.out'>lucas_job_2519471.out</a></b></td>
									<td style='padding: 8px;'>Job execution tracking and reporting for a WandB-monitored SLURM job, capturing key information on SDK usage, user authentication, and run synchronization; facilitates experiment reproducibility and monitoring by linking to project dashboards and run details, while also documenting job cancellation events within the broader context of large-scale, bioinformatics-focused model training and evaluation workflows orchestrated on high-performance computing clusters.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2519472.out'>lucas_job_2519472.out</a></b></td>
									<td style='padding: 8px;'>- Log output documents the execution and tracking of a machine learning job managed through SLURM and monitored by Weights & Biases within the larger AIforBioinformatics project<br>- Provides a record of experiment metadata, workflow orchestration, and error handling, supporting reproducibility and auditability of bioinformatics research runs, while also indicating user session details and job termination information crucial for project lifecycle management.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2519475.out'>lucas_job_2519475.out</a></b></td>
									<td style='padding: 8px;'>- Logging output captures the interaction between the model training workflow and the external experiment tracking service, documenting the initialization and run management process on the platform<br>- Records session authentication, local and remote tracking links, and job cancellation due to cluster time limits, providing traceability and operational context within the broader experiment management and SLURM-based job scheduling architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2519476.out'>lucas_job_2519476.out</a></b></td>
									<td style='padding: 8px;'>- Log output captures the workflow execution and monitoring details for a specific SLURM-managed job, documenting initialization of experiment tracking with Weights & Biases, run metadata, and eventual job cancellation due to time constraints<br>- Serves as an audit trail within the project, aiding in diagnosing issues, verifying experiment synchronization, and facilitating reproducibility in large-scale bioinformatics research workflows managed by the broader codebase.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2519478.out'>lucas_job_2519478.out</a></b></td>
									<td style='padding: 8px;'>- Log output provides a detailed record of a machine learning experiments execution under SLURM job scheduling, including integration with the Weights & Biases platform for experiment tracking and monitoring<br>- Documents the start, identification, and early termination of a specific training job, supporting reproducibility and diagnostics within the broader workflow for bioinformatics research in the project architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2519479.out'>lucas_job_2519479.out</a></b></td>
									<td style='padding: 8px;'>- Provides a record of a machine learning experiments progress, metrics, and integration status with Weights & Biases within the project’s job execution pipeline<br>- Captures key performance outcomes for training and validation, ensures experiment reproducibility, and enables traceability by linking to external W&B dashboards<br>- Serves as an audit trail supporting model evaluation and collaborative experiment tracking in the overall research workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2520394.out'>lucas_job_2520394.out</a></b></td>
									<td style='padding: 8px;'>- Job execution log for a SLURM-managed task, demonstrating integration with Weights & Biases (wandb) for experiment tracking and providing transparency into job lifecycle events within the projects automated workflow<br>- Captures essential metadata including user session, run tracking URL, and exit status, thereby supporting reproducibility and offering insight into pipeline performance and resource management in the broader bioinformatics analysis framework.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2520395.out'>lucas_job_2520395.out</a></b></td>
									<td style='padding: 8px;'>- Logs progress and completion details of a machine learning experiment managed via the Weights & Biases platform<br>- Provides run metadata, performance metrics, and links to detailed reports for monitoring and analysis<br>- Supports experiment tracking within the larger bioinformatics pipeline, ensuring reproducibility and transparency for model development and evaluation across cross-validation runs in the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2520520.out'>lucas_job_2520520.out</a></b></td>
									<td style='padding: 8px;'>- Logs capture the execution details and monitoring outputs from a SLURM-managed computational job, including integration with Weights & Biases for experiment tracking and metadata on the jobs cancellation event<br>- Serve as an audit trail for the project’s training and evaluation processes, supporting reproducibility, debugging, and effective resource management within the larger bioinformatics experimentation and workflow pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2521025.out'>lucas_job_2521025.out</a></b></td>
									<td style='padding: 8px;'>- Logging and job tracking for the project are facilitated through integration with Weights & Biases (wandb) and the Slurm workload manager<br>- This output captures activity related to experiment monitoring, user authentication, run synchronization, and job status, providing transparency and traceability for machine learning experiments as part of the projects broader reproducibility and workflow management infrastructure.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2521026.out'>lucas_job_2521026.out</a></b></td>
									<td style='padding: 8px;'>- Log output documents the tracking and synchronization of a machine learning experiment using the Weights & Biases (wandb) tool within a SLURM-managed job<br>- It provides essential details for monitoring experiment status, including authentication, run metadata, and links to interactive dashboards, aiding in experiment reproducibility and collaboration across the project’s distributed AI and bioinformatics workflows.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2521027.out'>lucas_job_2521027.out</a></b></td>
									<td style='padding: 8px;'>- Job execution logging and experiment tracking are facilitated through integration with Weights & Biases (wandb), providing real-time monitoring, local run data storage, and centralized dashboard access<br>- The output documents the tracking of a machine learning experiment scheduled via SLURM, while also capturing the job cancellation event<br>- Such logs enhance reproducibility, debugging, and collaborative insight within the projects broader experimentation workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2521030.out'>lucas_job_2521030.out</a></b></td>
									<td style='padding: 8px;'>- Logs execution details and status updates for a machine learning job managed by SLURM, capturing integration with Weights & Biases for experiment tracking<br>- Documents user identity, run tracking URLs, local storage paths, and the job’s cancellation due to a time limit<br>- Supports traceability and accountability within the workflow, assisting in monitoring, debugging, and reproducibility across the AIforBioinformatics project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2521031.out'>lucas_job_2521031.out</a></b></td>
									<td style='padding: 8px;'>- Job output log captures the execution and monitoring details of a machine learning experiment run on a SLURM-managed cluster, including integration with Weights & Biases for experiment tracking<br>- Serves as an audit trail for job status, environment setup, and run cancellation, supporting reproducibility and facilitating debugging within the broader AIforBioinformatics project workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2521032.out'>lucas_job_2521032.out</a></b></td>
									<td style='padding: 8px;'>- Logs serve as a record of a WandB-tracked machine learning experiment executed via SLURM, detailing user authentication, run tracking information, and monitoring links<br>- They provide operational visibility into experiment progress within the larger bioinformatics workflow, capturing both WandB integration and job lifecycle events, such as cancellation due to time limits, crucial for auditing and troubleshooting runs in the overall project architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2521818.out'>lucas_job_2521818.out</a></b></td>
									<td style='padding: 8px;'>- Log output captures the workflow and status of a machine learning job managed by SLURM, showcasing experiment tracking via Weights & Biases integration<br>- It reflects both the successful initiation and monitoring of a research experiment, as well as system warnings and job termination due to exceeded time limits<br>- This record aids in diagnosing experiment runs and maintaining reproducibility in the overall AIforBioinformatics project lifecycle.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2522057.out'>lucas_job_2522057.out</a></b></td>
									<td style='padding: 8px;'>- Job execution log details the monitoring and synchronization of an experimental run via the Weights & Biases service, capturing both runtime tracking information and warnings generated during statistical computations<br>- As part of the overall project architecture, this output serves to provide real-time insights, diagnostics, and execution status for machine learning experiments launched through SLURM on the high-performance computing cluster.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2522213.out'>lucas_job_2522213.out</a></b></td>
									<td style='padding: 8px;'>- Indicates a failed execution attempt within the projects job scheduling and logging framework, specifically highlighting an issue with accessing a key script for gene expression and methylation cross-validation<br>- Serves as a diagnostic output for the workflow, helping users and maintainers quickly identify missing dependencies or misconfigurations in the orchestration of large-scale bioinformatics pipeline tasks.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2522215.out'>lucas_job_2522215.out</a></b></td>
									<td style='padding: 8px;'>- Logs the experiment tracking and monitoring details for a specific model training run within the project, capturing key performance metrics and results through the Weights & Biases (wandb) platform<br>- Supports reproducibility and transparency across the codebase by saving run metadata, training and validation performance, and providing convenient links to online dashboards for collaboration, analysis, and historical record-keeping of experiments.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2522217.out'>lucas_job_2522217.out</a></b></td>
									<td style='padding: 8px;'>- Captures and summarizes the outcome of a machine learning experiment run, highlighting key training and validation metrics as tracked by Weights & Biases (wandb)<br>- Facilitates monitoring and reproducibility by logging performance indicators and providing links to detailed experiment dashboards, supporting project transparency and enabling efficient collaboration within the AIforBioinformatics codebase’s experiment management and workflow tracking infrastructure.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2522328.out'>lucas_job_2522328.out</a></b></td>
									<td style='padding: 8px;'>- Logging and monitoring information for a SLURM-scheduled machine learning experiment is captured, including metadata tracked via Weights & Biases<br>- Records session authentication, run tracking URLs, and the status of the experiment, noting termination due to exceeding allocated runtime<br>- Supports reproducibility and experiment management by providing transparent visibility into resource usage and run status within the broader workflows of the AIforBioinformatics project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2522329.out'>lucas_job_2522329.out</a></b></td>
									<td style='padding: 8px;'>- Logs trace the execution of a SLURM-scheduled machine learning job, capturing its integration with Weights & Biases for experiment tracking and noting the job’s termination due to a time limit<br>- Serves as a record of job status, experiment metadata syncing, and SLURM resource management, supporting reproducibility and monitoring within the workflow of the AIforBioinformatics project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2522330.out'>lucas_job_2522330.out</a></b></td>
									<td style='padding: 8px;'>- Job output captures the logging and monitoring details of a WandB-tracked machine learning experiment orchestrated through a SLURM-managed job<br>- Demonstrates integration with experiment tracking tools, provides run visibility via WandB links, and records the job’s lifecycle, including cancellation due to time limits<br>- Facilitates reproducibility and effective oversight of computational experiments within the broader AIforBioinformatics project infrastructure.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2522588.out'>lucas_job_2522588.out</a></b></td>
									<td style='padding: 8px;'>- Logs serve as an audit trail for experiment tracking and monitoring within the broader bioinformatics project<br>- By leveraging the Weights & Biases (wandb) platform, they facilitate reproducibility, transparency, and remote accessibility for machine learning runs, ensuring that results and configurations tied to each experiment are systematically recorded and easily accessible to collaborators throughout the AIforBioinformatics workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2522589.out'>lucas_job_2522589.out</a></b></td>
									<td style='padding: 8px;'>- Log output documents the integration of experiment tracking into the workflow by utilizing the Weights & Biases (wandb) platform<br>- Captures the authentication status, run initialization, synchronization events, and links for real-time monitoring<br>- Supports overall project reproducibility and transparency by providing a record of runs, crucial for experiment management within the broader AIforBioinformatics codebase’s research and model validation processes.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2522590.out'>lucas_job_2522590.out</a></b></td>
									<td style='padding: 8px;'>- Facilitates experiment tracking and reproducibility by logging key metadata, run status, and links to interactive dashboards provided by the Weights & Biases (wandb) platform<br>- Integrates the project’s model training or evaluation pipelines with external monitoring tools, supporting oversight and collaboration within the broader AIforBioinformatics codebase<br>- Enhances visibility into experiment progress, results, and configuration for all stakeholders.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2524493.out'>lucas_job_2524493.out</a></b></td>
									<td style='padding: 8px;'>- Logs execution errors encountered during the cross-validation phase of gene expression and methylation analysis, providing insight into issues that arise within the pipeline<br>- Serves as a key feedback mechanism for debugging and quality assurance, ensuring the reliability and robustness of the broader bioinformatics workflow by documenting operational failures related to logging and file system paths.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2524497.out'>lucas_job_2524497.out</a></b></td>
									<td style='padding: 8px;'>- Job output log documents the real-time tracking and management of a machine learning experiment orchestrated via SLURM and monitored using Weights & Biases (wandb)<br>- Records status updates, logging activity, and notable errors encountered during execution, specifically noting issues with data synchronization and job cancellation<br>- Enables comprehensive auditing and troubleshooting within the broader experiment management workflow of the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2524499.out'>lucas_job_2524499.out</a></b></td>
									<td style='padding: 8px;'>- Summary of logs\slurm\slurm_err\lucas_job_2524499.out**This file serves as a log capturing the execution details and status updates for a batch job run on the SLURM cluster within the project<br>- Specifically, it records WandB (Weights & Biases) experiment tracking activity, confirming the job’s connection to experiment monitoring, local data saving, and links to the corresponding dashboard for run visualization<br>- The log is a valuable resource for diagnosing the progress and health of machine learning experiments, aiding in visibility and reproducibility within the broader research pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2524501.out'>lucas_job_2524501.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs\slurm\slurm_err\lucas_job_2524501.out serves as an execution log for a specific computational job managed on a SLURM-based high-performance computing cluster<br>- Within the broader architecture of the project, this log primarily documents the tracking and synchronization of experiment runs with Weights & Biases (wandb), a tool for monitoring machine learning experiments<br>- It provides transparency into experiment states, user credentials, and links to more detailed results, ensuring reproducibility and accountability for model training or evaluation tasks<br>- This file is essential for monitoring, debugging, and auditing the workflow outcomes within the project’s experimental pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2524502.out'>lucas_job_2524502.out</a></b></td>
									<td style='padding: 8px;'>- Log output captures the execution and monitoring details of a SLURM-scheduled experiment using the Weights & Biases (wandb) tracking system<br>- Documents the run state, syncing status, user context, and explicit job cancellation, supporting experiment traceability and troubleshooting within the broader AI for bioinformatics project workflow<br>- Enables clear auditing of job lifecycle and remote observability for distributed computation management.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2524503.out'>lucas_job_2524503.out</a></b></td>
									<td style='padding: 8px;'>- Job output and error tracking for machine learning cross-validation experiments, specifically capturing logging and failure details during model checkpoint loading and integration with Weights & Biases experiment tracking<br>- Enables monitoring of distributed training runs, surfacing issues such as missing model artifacts for faster troubleshooting and overall improvement of experiment reproducibility and transparency within the AI for Bioinformatics workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2524504.out'>lucas_job_2524504.out</a></b></td>
									<td style='padding: 8px;'>- Job error output serves as a record for tracking and diagnosing the results of a SLURM-managed experiment within the project, specifically logging the integration and status reporting from the Weights & Biases (wandb) experiment tracking tool alongside the job’s cancellation event<br>- Enables users to monitor experiment progress, verify successful connection to external logging platforms, and troubleshoot unexpected interruptions during large-scale or distributed model training runs.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2524505.out'>lucas_job_2524505.out</a></b></td>
									<td style='padding: 8px;'>- Execution trace and error log documentation for a machine learning model training job managed via SLURM and monitored with Weights & Biases<br>- Provides insight into run initialization, experiment tracking URLs, and key errors encountered—such as missing checkpoint files and data warnings—offering critical feedback for debugging and assessing workflow execution within the broader AIforBioinformatics project’s experiment pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2524813.out'>lucas_job_2524813.out</a></b></td>
									<td style='padding: 8px;'>- Log output provides a record of experiment tracking and synchronization activities performed by the Weights & Biases (W&B) tool during a specific model training job<br>- Captures metrics, run history, and links to detailed dashboards, enabling transparent monitoring and reproducibility<br>- Integrates seamlessly with the broader project to centralize experiment metadata, streamline collaboration, and facilitate evaluation of machine learning workflows within the bioinformatics research context.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2524890.out'>lucas_job_2524890.out</a></b></td>
									<td style='padding: 8px;'>- Logs operational details and status updates for a machine learning experiment executed on a SLURM-managed cluster, capturing integration with Weights & Biases (wandb) for experiment tracking<br>- Documents user authentication, data synchronization, run metadata, and the reason for job termination<br>- Supports monitoring, reproducibility, and debugging workflows across the project’s distributed, high-performance computing environment.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2524959.out'>lucas_job_2524959.out</a></b></td>
									<td style='padding: 8px;'>- Execution output and job status details are captured for a SLURM-managed training run, documenting both Weights & Biases experiment tracking integration and the job’s cancellation due to a time limit<br>- Serves as an audit trail within the logs subsystem, supporting reproducibility, monitoring, and diagnostics across the project’s distributed or scheduled machine learning workflows.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2524978.out'>lucas_job_2524978.out</a></b></td>
									<td style='padding: 8px;'>- Logs SLURM job activity and monitors experiment tracking through the Weights & Biases (wandb) platform, providing insights into job execution status, run metadata, and project links<br>- Facilitates traceability and debugging within the larger workflow by documenting both successful tracking integration and job termination events, ensuring reliable observability and historical record-keeping for AIforBioinformatics project experiments.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2524984.out'>lucas_job_2524984.out</a></b></td>
									<td style='padding: 8px;'>- Logs operational details and outcome metrics of a machine learning experiment tracked via the Weights & Biases (wandb) platform within the context of cross-validation on the TCGA-BRCA-GE dataset<br>- Provides a concise account of training and validation performance, supports experiment reproducibility, and enables seamless monitoring and analysis alongside other runs across the AIforBioinformatics project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2527208.out'>lucas_job_2527208.out</a></b></td>
									<td style='padding: 8px;'>- Job output log documents the execution and tracking of a computational experiment managed via SLURM and logged with Weights & Biases<br>- Captures workflow metadata, user authentication, storage of run artifacts, and links to project dashboards for monitoring and collaboration<br>- Records the job’s eventual cancellation, providing essential context for post-mortem analysis and supporting transparent experimentation within the broader research pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2527209.out'>lucas_job_2527209.out</a></b></td>
									<td style='padding: 8px;'>- Logs operational details and runtime errors encountered during gene expression and methylation model cross-validation within a high-performance computing cluster environment<br>- Captures system status, resource utilization feedback, and experiment tracking through Weights & Biases, offering critical insights for debugging and workflow optimization across the project’s orchestration and training phases<br>- Enables efficient identification of process failures and facilitates reproducibility in research experiments.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2527210.out'>lucas_job_2527210.out</a></b></td>
									<td style='padding: 8px;'>- Job output log captures the integration of experiment tracking through the Weights & Biases (wandb) platform during a model training run<br>- Documents authentication, run metadata storage, and provides direct links for monitoring experiment status and results<br>- Supports reproducibility and collaboration efforts by ensuring all relevant training metrics and artifacts are systematically tracked and accessible within the broader AIforBioinformatics project workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2527211.out'>lucas_job_2527211.out</a></b></td>
									<td style='padding: 8px;'>- Error and logging output captured during the execution of a SLURM job running multimodal gene expression and methylation cross-validation<br>- Provides high-level traceability for experiment configuration, integration with Weights & Biases for run tracking, and detailed failure context related to out-of-memory termination<br>- Assists in diagnosing resource or data handling issues within the broader AIforBioinformatics experimentation workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2527212.out'>lucas_job_2527212.out</a></b></td>
									<td style='padding: 8px;'>- Logs capture experiment tracking activity using the Weights & Biases platform, providing information about run initialization, user authentication, local data saving, and online synchronization<br>- Serve as an auditable record of metadata for specific machine learning experiments, supporting monitoring, reproducibility, and collaborative review within the larger AIforBioinformatics workflow, particularly for studies involving cross-validation with the TCGA-BRCA dataset.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2527231.out'>lucas_job_2527231.out</a></b></td>
									<td style='padding: 8px;'>- Capture of WandB experiment tracking output and SLURM job status, facilitating monitoring and reproducibility of machine learning runs within the project<br>- Provides essential run metadata, project links, and error notifications, enabling efficient debugging and experiment management<br>- Serves as an audit trail within the logging subsystem, integrating job execution details with remote experiment tracking for streamlined collaboration and analysis across the research workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2527233.out'>lucas_job_2527233.out</a></b></td>
									<td style='padding: 8px;'>- Logs output from a distributed training or analysis job executed via SLURM, capturing WandB experiment tracking details, run metadata, and job status notifications<br>- Supports reproducibility and troubleshooting by documenting experiment parameters, run links, and any interruptions or cancellations, thereby integrating job-level feedback into the larger machine learning experiment management and monitoring workflow across the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2527234.out'>lucas_job_2527234.out</a></b></td>
									<td style='padding: 8px;'>- Logs execution and monitoring details for a SLURM-managed machine learning job, including integration with Weights & Biases for experiment tracking and run management<br>- Captures run metadata, user authentication, local storage paths, and remote dashboard links for reproducibility and traceability<br>- Records the job’s cancellation event, aiding in post-mortem analysis and facilitating robust workflow management within the project’s computational pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2527235.out'>lucas_job_2527235.out</a></b></td>
									<td style='padding: 8px;'>- Logs the execution status and progress of a SLURM-managed job within the project, including integration and tracking details with Weights & Biases (wandb) for experiment monitoring<br>- Captures user session context, run identifiers, data storage locations, and remote dashboard links, while also noting the job’s cancellation<br>- Serves as an essential audit and diagnostic record within the project’s overall workflow and experiment management system.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2527236.out'>lucas_job_2527236.out</a></b></td>
									<td style='padding: 8px;'>- Provides a logging output capturing the tracking and synchronization details for a specific computational job managed by SLURM and monitored using Weights & Biases<br>- Facilitates auditability and experiment reproducibility by recording authentication status, run metadata, and execution events, while also documenting the job’s cancellation<br>- Supports transparency and traceability within the broader experiment tracking infrastructure of the codebase.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2527622.out'>lucas_job_2527622.out</a></b></td>
									<td style='padding: 8px;'>- Job log output documents the orchestration and status of a machine learning experiment tracked via Weights & Biases (wandb) within a Slurm-managed high-performance computing environment<br>- It provides visibility into user authentication, experiment run initialization, and synchronization, while also capturing system-level events such as job termination due to time limits, thereby supporting reproducibility and auditability across the broader project workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2527661.out'>lucas_job_2527661.out</a></b></td>
									<td style='padding: 8px;'>- Logs the execution details and system notifications of a machine learning job managed via SLURM, including integration with Weights & Biases for experiment tracking and monitoring<br>- Supports workflow observability by capturing authentication status, run metadata, and termination events, enabling users to audit, troubleshoot, and optimize distributed training runs within the broader bioinformatics project infrastructure.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2527662.out'>lucas_job_2527662.out</a></b></td>
									<td style='padding: 8px;'>- Captures the execution logs and monitoring information from a SLURM-managed compute job, providing visibility into experiment progress, user authentication, and run tracking through Weights & Biases integration<br>- Indicates the job’s status and termination reason, supporting auditability and troubleshooting within the larger workflow for managing and validating large-scale machine learning experiments in the project’s bioinformatics research pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2527663.out'>lucas_job_2527663.out</a></b></td>
									<td style='padding: 8px;'>- Job execution record captures the logging and monitoring output from a SLURM-managed run utilizing Weights & Biases for experiment tracking<br>- Documents user login, run synchronization, and run management details, while also noting termination due to time constraints<br>- Serves as an audit and status reference point within the project’s experiment tracking and infrastructure orchestration, supporting reproducibility and debugging across computational experiments.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2527664.out'>lucas_job_2527664.out</a></b></td>
									<td style='padding: 8px;'>- Logs capture the execution status of a machine learning workflow managed via SLURM and monitored using Weights & Biases<br>- They document project tracking details, user authentication, run-specific metadata, and report the termination of a scheduled job<br>- Serving as a vital audit and debugging resource, these logs support reproducibility and troubleshooting within the broader bioinformatics pipeline architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2527699.out'>lucas_job_2527699.out</a></b></td>
									<td style='padding: 8px;'>- Logs progress and metrics from a machine learning training job within the project, capturing key performance data such as loss and c-index scores for both training and validation<br>- Facilitates experiment tracking and reproducibility by syncing results with the Weights & Biases platform, supporting evaluation and comparison of model performance in the broader context of AI-driven bioinformatics research workflows.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2527700.out'>lucas_job_2527700.out</a></b></td>
									<td style='padding: 8px;'>- Logs execution and monitoring details for a compute job managed by SLURM, including integration with Weights & Biases for experiment tracking<br>- Captures environment context, user identity, job synchronization URLs, and job termination due to a time limit<br>- Facilitates experiment management and reproducibility within the larger AI for Bioinformatics pipeline, supporting traceability and oversight of computational experiments.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2527703.out'>lucas_job_2527703.out</a></b></td>
									<td style='padding: 8px;'>- Log output documents the execution and tracking of a machine learning experiment managed via the Weights & Biases platform, providing visibility into project status, run details, and experiment management<br>- Serves as a record for job monitoring within the SLURM job scheduling system, highlighting integration points and facilitating debugging, auditability, and reproducibility across the AIforBioinformatics projects workflow and research lifecycle.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2528093.out'>lucas_job_2528093.out</a></b></td>
									<td style='padding: 8px;'>- Logs job execution details for a SLURM-managed machine learning workflow, capturing environment settings, progress tracking via Weights & Biases, and termination status due to time constraints<br>- Enhances project observability and experiment reproducibility by documenting run metadata, user authentication, and traceability links, thereby supporting robust monitoring and auditing within the overall architecture of reproducible AI for bioinformatics research.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2528094.out'>lucas_job_2528094.out</a></b></td>
									<td style='padding: 8px;'>- Logs the workflow and status of a SLURM-managed machine learning experiment, highlighting integration with the Weights & Biases platform for experiment tracking and reproducibility<br>- Provides visibility into run metadata, logging configuration, and failure details, ensuring researchers can monitor progress, reference completed or failed jobs, and maintain accountability across distributed training tasks in the AIforBioinformatics project architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2528548.out'>lucas_job_2528548.out</a></b></td>
									<td style='padding: 8px;'>- Log output documents the status and key events of a SLURM-managed machine learning experiment, capturing integration with Weights & Biases for run tracking, authentication details, project and run URLs, and the eventual cancellation signal for the associated SLURM job<br>- Provides transparency and traceability for experiment management within the broader AIforBioinformatics workflow, aiding in debugging and reproducibility across the codebase.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2528549.out'>lucas_job_2528549.out</a></b></td>
									<td style='padding: 8px;'>- Job execution tracking and error reporting for a SLURM-managed experiment run, documenting WandB integration for monitoring and logging experiment progress, as well as capturing job cancellation status<br>- Supports experiment reproducibility, auditability, and troubleshooting within the project’s computational research workflows, ensuring that job lifecycle events and external tool interactions are visible for project stakeholders and contributors.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2528551.out'>lucas_job_2528551.out</a></b></td>
									<td style='padding: 8px;'>- Logs record the execution and performance tracking of a machine learning experiment managed via Weights & Biases, capturing run configuration, key training and validation metrics, and providing links for detailed experiment review<br>- Serve as an audit trail for experiment progress and results within the broader architecture, supporting reproducibility, monitoring, and collaboration across the bioinformatics model development workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2528553.out'>lucas_job_2528553.out</a></b></td>
									<td style='padding: 8px;'>- Provides a record of a machine learning experiments integration with Weights & Biases (wandb), capturing key run metrics and logging information for a cross-validation job<br>- Facilitates experiment tracking, reproducibility, and result sharing within the broader AI for Bioinformatics project, supporting streamlined model evaluation and collaborative analysis by centralizing progress and outcome data in the project’s workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2528555.out'>lucas_job_2528555.out</a></b></td>
									<td style='padding: 8px;'>- Monitors and documents the progress of a machine learning training run within the broader bioinformatics project<br>- Captures key training and validation metrics, facilitating experiment tracking and reproducibility<br>- Provides direct integration with Weights & Biases for visualization and centralized management of results, supporting effective model evaluation and iterative development workflows across the AIforBioinformatics codebase.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2528556.out'>lucas_job_2528556.out</a></b></td>
									<td style='padding: 8px;'>- Logs the execution details and error messages from a specific SLURM job that runs a machine learning cross-validation experiment on gene expression and methylation data<br>- Provides traceability for experiment tracking via Weights & Biases (wandb) integration and captures critical runtime diagnostics—such as segmentation faults—for debugging and reproducibility within the bioinformatics project’s computational workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2528558.out'>lucas_job_2528558.out</a></b></td>
									<td style='padding: 8px;'>- Log record documents the execution and subsequent cancellation of a SLURM job involved in machine learning experiments tracked with Weights & Biases<br>- It provides evidence of workflow monitoring, including run synchronization details and user context, which is vital for auditing, debugging, and reproducibility within the larger project focused on computational bioinformatics research and experiment management.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2528613.out'>lucas_job_2528613.out</a></b></td>
									<td style='padding: 8px;'>- Provides a log of a model training or experiment run managed via SLURM, highlighting integration with Weights & Biases (wandb) for experiment tracking and monitoring within the broader AIforBioinformatics project<br>- Captures metadata about the run, including links to the wandb dashboard for further analysis, and notes job termination due to SLURM time limits, supporting auditability and reproducibility efforts across the codebase.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2529616.out'>lucas_job_2529616.out</a></b></td>
									<td style='padding: 8px;'>- Logs the tracking and synchronization status of a machine learning experiment run using Weights & Biases within the project<br>- Serves as an audit trail for monitoring experiment progress, run metadata, and linking local computation with centralized dashboards<br>- Contributes to ensuring experiment reproducibility and traceability by recording essential high-level details about training executions in the context of the project’s larger workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2529618.out'>lucas_job_2529618.out</a></b></td>
									<td style='padding: 8px;'>- Logs progress and metadata for a specific SLURM job, reflecting integration with the Weights & Biases (wandb) experiment tracking platform<br>- Captures essential information about job execution, run synchronization, and provides direct access links to both the project and run dashboards<br>- Facilitates experiment monitoring and reproducibility within the broader bioinformatics workflow by consolidating status updates and traceability in training or evaluation processes.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2529716.out'>lucas_job_2529716.out</a></b></td>
									<td style='padding: 8px;'>- Operational logging and experiment tracking for a model training run are facilitated, recording key metrics such as training and validation performance via integration with the Weights & Biases platform<br>- Enables monitoring, synchronization, and reproducibility of machine learning experiments within the broader pipeline, providing visibility into run outcomes and serving as a reference point for future analysis or model comparison across the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2529717.out'>lucas_job_2529717.out</a></b></td>
									<td style='padding: 8px;'>- Log output captures the runtime behavior and execution status of a cross-validation job related to gene expression and methylation analysis within the TCGA-BRCA project<br>- Documents integration with Weights & Biases for experiment tracking, and provides evidence of an assertion failure during CUDA operations, which is crucial for diagnosing and resolving errors in the context of large-scale computational bioinformatics workflows.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2529845.out'>lucas_job_2529845.out</a></b></td>
									<td style='padding: 8px;'>- Log output captures the execution details of a machine learning experiment managed on a SLURM cluster, including integration with Weights & Biases (wandb) for experiment tracking and monitoring<br>- Provides visibility into authentication, run synchronization, and job metadata, while recording a runtime error encountered during the execution of a cross-validation workflow for gene expression and methylation analysis within the broader bioinformatics project framework.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2529886.out'>lucas_job_2529886.out</a></b></td>
									<td style='padding: 8px;'>- Reports output and activity logs generated during the execution of a batch job running gene expression and methylation cross-validation analysis within the project’s computational pipeline<br>- Captures interaction with the experiment tracking tool, WANDB, as well as system-level notifications and errors, offering transparency for successful monitoring, debugging, and reproducibility of high-performance computing runs across the bioinformatics workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2529895.out'>lucas_job_2529895.out</a></b></td>
									<td style='padding: 8px;'>- Job error log captures the execution details and failure context of a machine learning experiment, specifically during a cross-validation run that integrates gene expression and methylation data<br>- Provides insight into the workflow’s orchestration, external tool integration (such as experiment tracking with Weights & Biases), and error conditions encountered—critical for diagnosing issues and ensuring robust development within the AI for bioinformatics pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2529896.out'>lucas_job_2529896.out</a></b></td>
									<td style='padding: 8px;'>- Captures and documents the output from a specific model training job, highlighting experiment tracking activity via Weights & Biases within the broader AIforBioinformatics project<br>- Summarizes metrics, performance outcomes, and artifact syncing for the TCGA-BRCA-GE-ME-Cross-Validation experiment, supporting auditability and reproducibility of machine learning workflows across runs without exposing lower-level implementation specifics.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2530872.out'>lucas_job_2530872.out</a></b></td>
									<td style='padding: 8px;'>- Logs the outcome and system events during a model training run within the MCAT multimodal gene expression and methylation cross-validation workflow<br>- Provides insight into experiment tracking via Weights & Biases integration, captures a critical runtime error related to memory exhaustion during data loading, and serves as an audit trail for debugging and monitoring computational resource usage within the bioinformatics project pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2530876.out'>lucas_job_2530876.out</a></b></td>
									<td style='padding: 8px;'>- Production of job execution logs and system outputs for model training and evaluation tasks within a Slurm-managed cluster environment, capturing WandB experiment tracking activity and reporting runtime issues<br>- Supports project monitoring, debugging, and reproducibility by providing insight into the integration of workflow automation, experiment tracking, and error handling within the broader AI for Bioinformatics pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2530911.out'>lucas_job_2530911.out</a></b></td>
									<td style='padding: 8px;'>- Logs the monitoring and tracking of a machine learning training job managed on SLURM, detailing integration with Weights & Biases for experiment tracking<br>- Captures metrics such as training and validation loss and c-index, providing links to detailed run and project dashboards<br>- Supports transparency and reproducibility within the broader research workflow, ensuring systematic logging and structured progress tracking across experiments in the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2532004.out'>lucas_job_2532004.out</a></b></td>
									<td style='padding: 8px;'>- Provides a record of a specific SLURM jobs execution within the broader AIforBioinformatics project, capturing Weights & Biases (wandb) experiment tracking activity, user authentication, run synchronization status, and upload issues encountered during the job<br>- Supports debugging and reproducibility efforts by documenting operational details and workflow outcomes as part of the systems experiment management and job tracking infrastructure.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2532879.out'>lucas_job_2532879.out</a></b></td>
									<td style='padding: 8px;'>- Job output log provides real-time feedback and error reporting for machine learning experiments tracked via Weights & Biases within the project<br>- It highlights authentication, experiment tracking, and specifically surfaces a configuration error related to a parameter type mismatch<br>- Serves as a crucial tool for monitoring, debugging, and ensuring the robustness of automated workflows orchestrated through Slurm in the overall bioinformatics pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2532881.out'>lucas_job_2532881.out</a></b></td>
									<td style='padding: 8px;'>- Captures standard and error outputs generated during the execution of a specific SLURM job associated with cross-validating gene expression and methylation data using the MCAT workflow<br>- Provides traceability for experiment tracking with Weights & Biases, and surfaces a configuration parsing error that prevents successful run completion, enabling developers to diagnose and resolve issues within the broader AI for Bioinformatics project pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2532883.out'>lucas_job_2532883.out</a></b></td>
									<td style='padding: 8px;'>- Logs SLURM job execution details, including integration and tracking with the Weights & Biases platform, capturing experiment metadata and progress for a machine learning workflow within the AI for Bioinformatics project<br>- Facilitates experiment reproducibility, monitoring, and auditing, while also recording job termination due to a time limit, which is useful for resource management and debugging in distributed training or evaluation environments.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2532884.out'>lucas_job_2532884.out</a></b></td>
									<td style='padding: 8px;'>- Logging and monitoring of a SLURM jobs execution is captured, including integration with the Weights & Biases (wandb) platform for experiment tracking and synchronization<br>- Provides details on user authentication, tracking URLs, and storage locations for run data, while also recording the eventual cancellation of the job<br>- Supports traceability and troubleshooting within the broader workflow for machine learning experiments managed in the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2532886.out'>lucas_job_2532886.out</a></b></td>
									<td style='padding: 8px;'>- Logging output documents the tracking of a model training or analysis run within a SLURM-managed high-performance computing environment<br>- Captures WandB experiment management, including run activation, run location, and web links for monitoring progress, while also recording job termination due to time constraints<br>- Supports experiment reproducibility and traceability across the broader bioinformatics project, facilitating efficient workflow management and team collaboration.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2532922.out'>lucas_job_2532922.out</a></b></td>
									<td style='padding: 8px;'>- Logs the progress and outcomes of a machine learning training job run on a SLURM-managed cluster, including integration with Weights & Biases for experiment tracking<br>- Documents essential run metadata, performance metrics, and links to detailed dashboards, supporting experiment reproducibility and facilitating result analysis for training and validation phases within the broader TCGA-BRCA-GE-ME cross-validation workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2532945.out'>lucas_job_2532945.out</a></b></td>
									<td style='padding: 8px;'>- Log output in logs/slurm/slurm_err/lucas_job_2532945.out documents the status and results of a model training run managed and tracked via Weights & Biases (wandb)<br>- Provides a concise record of experiment metadata, performance metrics, artifact syncing, and links to interactive dashboards, supporting experiment traceability and reproducibility within the AIforBioinformatics project’s workflow and experiment management architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2533210.out'>lucas_job_2533210.out</a></b></td>
									<td style='padding: 8px;'>- Log output documents the execution and monitoring of a machine learning experiment using Weights & Biases (wandb) on a SLURM-managed cluster<br>- Captures run initialization, data syncing, and project tracking, while also noting termination due to exceeding the job’s allocated time<br>- Facilitates auditing, progress tracking, and troubleshooting within the broader bioinformatics workflow managed in the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2542055.out'>lucas_job_2542055.out</a></b></td>
									<td style='padding: 8px;'>- Captures and summarizes the execution details and WandB experiment tracking for a specific training job within the project<br>- Provides a record of model performance metrics, run identifiers, and links for monitoring and validation purposes<br>- Serves as an audit trail and progress indicator, supporting reproducibility and transparency in the research workflow and facilitating traceability across model development iterations.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2542056.out'>lucas_job_2542056.out</a></b></td>
									<td style='padding: 8px;'>- Provides a summary of a WandB-tracked machine learning experiment, including links to the project and run, as well as performance metrics such as training and validation loss and c-index<br>- Serves as an audit trail supporting reproducibility, progress monitoring, and result sharing within the broader AIforBioinformatics experimentation workflow, facilitating transparency and collaboration across model development stages.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2542057.out'>lucas_job_2542057.out</a></b></td>
									<td style='padding: 8px;'>- Captures and summarizes the output of a training run managed through the SLURM scheduler, providing high-level tracking of experiment metrics using the Weights & Biases (W&B) platform<br>- Enables project stakeholders to monitor model performance, training progress, and validation results, integrating seamlessly with the overall machine learning workflow for experiment reproducibility, monitoring, and collaboration across the research and engineering teams.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2542058.out'>lucas_job_2542058.out</a></b></td>
									<td style='padding: 8px;'>- Logs the execution details and status updates for a machine learning experiment tracked via Weights & Biases (wandb) on an HPC cluster managed by SLURM<br>- Captures user authentication, run initialization, local data storage paths, as well as the termination of the job due to exceeding the time limit, providing transparency and traceability for experiment monitoring within the larger AI for bioinformatics workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2542059.out'>lucas_job_2542059.out</a></b></td>
									<td style='padding: 8px;'>- Log output documents the activities and status of a computational job managed via SLURM, including integration with the Weights & Biases (wandb) experiment tracking platform<br>- Serves as an audit trail for monitoring experiment progress, tracking user credentials, capturing run metadata, and recording job termination events<br>- Facilitates debugging, reproducibility, and accountability within the broader machine learning workflow in the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2542060.out'>lucas_job_2542060.out</a></b></td>
									<td style='padding: 8px;'>- Job execution log documents the integration of experiment tracking using Weights & Biases (wandb), recording run metadata and progress for a machine learning workflow managed via the SLURM job scheduler<br>- Captures successful initialization, user context, and provides direct links to the tracked experiment, while also noting premature job termination due to a cluster-imposed time limit, aiding in debugging and workflow monitoring across the entire project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2542136.out'>lucas_job_2542136.out</a></b></td>
									<td style='padding: 8px;'>- Logs the progress and outcome of a model training job executed on a SLURM-managed cluster, including integration with Weights & Biases for experiment tracking<br>- Provides visibility into run status, user identity, configuration, and termination details, supporting reproducibility and auditability across the project’s distributed training workflows<br>- Enables centralized monitoring of resource usage and experimental results within the broader machine learning pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\lucas_job_2542142.out'>lucas_job_2542142.out</a></b></td>
									<td style='padding: 8px;'>- Log output documents the execution and tracking of a SLURM-managed machine learning experiment using Weights & Biases for monitoring<br>- Demonstrates integration with experiment management tools, records the workflow’s progress, and captures job cancellation due to time limits<br>- Serves as an operational record, supporting experiment reproducibility and troubleshooting within the broader bioinformatics research pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2542175.out'>MethWayOSMethWayOS_2542175.out</a></b></td>
									<td style='padding: 8px;'>- Captures diagnostic and tracking output for a computational experiment involving gene expression and methylation cross-validation within the broader AIforBioinformatics project<br>- Enables monitoring of workflow progress, resource utilization, and integration with experiment tracking tools such as Weights & Biases, while signaling operational issues like out-of-memory errors that may impact the reproducibility and reliability of large-scale biological data analyses.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2542176.out'>MethWayOS_2542176.out</a></b></td>
									<td style='padding: 8px;'>- Captures and documents runtime information, errors, and warnings generated during a SLURM-managed execution of gene expression and methylation cross-validation workflows<br>- Provides a historical trace of experiment runs, including integration with experiment tracking tools such as Weights & Biases, helping users monitor workflow status and troubleshoot out-of-memory or plotting issues within the broader AI-driven bioinformatics pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2542177.out'>MethWayOS_2542177.out</a></b></td>
									<td style='padding: 8px;'>- Log output captures the execution and failure status of a SLURM-scheduled experiment run, highlighting integration with Weights & Biases for experiment tracking, user authentication, and recording of metadata<br>- Provides insight into workflow monitoring and resource constraints, such as out-of-memory termination, supporting reproducibility and debugging for large-scale cross-validation experiments within the broader AI for Bioinformatics research framework.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2544870.out'>MethWayOS_2544870.out</a></b></td>
									<td style='padding: 8px;'>- Reports the execution progress and outcomes of a model training run, including integration with Weights & Biases for experiment tracking<br>- Summarizes key training and validation metrics, documents configuration and result uploads, and provides links for further analysis<br>- Plays a vital role in auditability and reproducibility by capturing the state, performance, and artifacts of a training experiment within the larger AIforBioinformatics platform.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2544876.out'>MethWayOS_2544876.out</a></b></td>
									<td style='padding: 8px;'>- Logs capture the execution details and monitoring data for a specific computational experiment run within the project, including integration with Weights & Biases for experiment tracking, resource usage feedback, and error messages arising from job execution on an HPC cluster using SLURM<br>- Serve as an audit trail for model training and evaluation workflows, helping users diagnose issues and optimize computational experiments in the larger bioinformatics pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2544877.out'>MethWayOS_2544877.out</a></b></td>
									<td style='padding: 8px;'>- Logs progress and metadata for a specific computational experiment, integrating with the Weights & Biases platform to facilitate experiment tracking and reproducibility within the project’s workflow<br>- Captures details of the run, user, and synchronization status before recording the SLURM job cancellation event, thereby supporting auditability and enabling seamless monitoring and recovery across distributed high-performance computing environments.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2544899.out'>MethWayOS_2544899.out</a></b></td>
									<td style='padding: 8px;'>- Slurm job output log captures the execution details and status for a survival pathway analysis involving CpG sites, including integration with the Weights & Biases (wandb) experiment tracking service<br>- Serves as an audit trail for workflow execution within the larger bioinformatics pipeline, facilitating reproducibility and debugging by documenting runtime warnings, resource constraints, and relevant experiment metadata associated with the project’s model training and evaluation steps.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\SurMethWayOS44900.out'>MethWayOS_2544900.out</a></b></td>
									<td style='padding: 8px;'>- Records execution logs and errors related to the gene expression and methylation cross-validation workflow within the projects computational pipeline<br>- Facilitates experiment tracking, debugging, and reproducibility by capturing outputs, warnings, and tracebacks—enabling effective monitoring of model training and evaluation processes critical to the project’s bioinformatics analysis of multi-omics data for survival prediction.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2544901.out'>MethWayOS_2544901.out</a></b></td>
									<td style='padding: 8px;'>- Error log highlights the monitoring and outcome tracking of a gene expression and methylation cross-validation experiment, with integration into the projects experiment tracking system<br>- Indicates that a failure occurred during the testing phase due to missing gene expression signature data, offering visibility into run status, dataset coverage, and crucial debugging information within the broader bioinformatics model validation workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2544990.out'>MethWayOS_2544990.out</a></b></td>
									<td style='padding: 8px;'>- Execution log captures the output and errors from a SLURM-scheduled run within the AIforBioinformatics project, documenting the WandB experiment tracking integration, progress of survival pathway analysis on CpG data, and job cancellation due to time limits<br>- Serves as a historical reference for debugging, resource management, and monitoring workflow progress in the distributed, high-performance computing environment underpinning the codebase.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2544991.out'>MethWayOS_2544991.out</a></b></td>
									<td style='padding: 8px;'>- Logs the progress and outcome of a model training run that integrates gene expression and methylation data, tracking metrics such as training and validation loss and concordance index using the W&B platform<br>- Serves as an execution trace for experiment runs within the broader bioinformatics pipeline, enabling monitoring, reproducibility, and comparison of survival prediction models across various data configurations.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2545213.out'>MethWayOS_2545213.out</a></b></td>
									<td style='padding: 8px;'>- Provides a detailed log of a specific model training and evaluation run, capturing experiment tracking, performance metrics, and potential warnings encountered during execution<br>- Serves as a valuable artifact for reproducibility, debugging, and performance analysis within the broader machine learning pipeline, supporting comprehensive experiment management and facilitating insights into the integration of gene expression and methylation modules for survival path analysis.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2545299.out'>MethWayOS_2545299.out</a></b></td>
									<td style='padding: 8px;'>- Captures and summarizes the output of a model training run, including integration with Weights & Biases (wandb) for experiment tracking and reporting<br>- Records training and validation performance metrics such as loss and concordance index, while also pointing to relevant dashboards and logs for further analysis<br>- Supports overall project transparency, reproducibility, and monitoring, which are essential for robust scientific workflow in the codebase.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2545300.out'>MethWayOS_2545300.out</a></b></td>
									<td style='padding: 8px;'>- Execution log captures the process and outcomes of a machine learning experiment using the Weights & Biases platform to track a survival pathway analysis involving CpG data<br>- Provides transparency into experiment progress, environment, and performance, facilitating reproducibility and debugging within the projects broader workflows for gene expression and methylation module analysis in bioinformatics research.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2545301.out'>MethWayOS_2545301.out</a></b></td>
									<td style='padding: 8px;'>- Job execution log capturing the progress and system messages during a model training or evaluation run tracked via Weights & Biases as part of a cross-validation workflow for molecular survival pathway analysis<br>- Provides insights into resource utilization, warnings, and errors, such as figure overflows and job cancellation due to time limits, supporting reproducibility and monitoring within the larger AIforBioinformatics project architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2545302.out'>MethWayOS_2545302.out</a></b></td>
									<td style='padding: 8px;'>- Log output documents the orchestration and monitoring of a distributed machine learning experiment, highlighting the integration with Weights & Biases for experiment tracking within a SLURM-managed high-performance computing environment<br>- Provides visibility into authentication, run synchronization, and job status updates, facilitating reproducibility and debugging throughout the experiment lifecycle as part of the larger bioinformatics pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2545321.out'>MethWayOS_2545321.out</a></b></td>
									<td style='padding: 8px;'>- Provides a log of a computational experiments execution managed via SLURM, capturing integration with Weights & Biases for experiment tracking on the TCGA-BRCA-GE-ME-Cross-Validation project<br>- Records runtime warnings, resource usage, and the termination event due to time constraints<br>- Enables reproducibility and monitoring within the broader AIforBioinformatics workflow by documenting operational details and linking results to broader project tracking infrastructure.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2545422.out'>MethWayOS_2545422.out</a></b></td>
									<td style='padding: 8px;'>- Logs the execution and progress of a model training run, highlighting the integration with the Weights & Biases (W&B) platform for experiment tracking and performance visualization<br>- Provides run-specific metadata, links to results dashboards, and summarizes key training metrics, supporting transparency and reproducibility for the broader AIforBioinformatics platform’s workflow, specifically within the context of gene expression and methylation module analysis.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2545423.out'>MethWayOS_2545423.out</a></b></td>
									<td style='padding: 8px;'>- Error and output logs in logs\slurm\slurm_err\MethWayOS_2545423.out capture the status and runtime feedback of a model training or evaluation job—specifically, the orchestration of gene expression and methylation module analysis using slurm and Weights & Biases for tracking<br>- Serve as a diagnostic record for monitoring experiment progress, integration with cloud-based experiment tracking, and identifying interruptions or resource limitations in batch executions.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2545424.out'>MethWayOS_2545424.out</a></b></td>
									<td style='padding: 8px;'>- Provides a detailed runtime log for a model training session, capturing experiment tracking, performance metrics, and validation summaries using Weights & Biases (wandb)<br>- Integrates into the broader codebase’s workflow for systematic monitoring, reproducibility, and evaluation of machine learning experiments, supporting end-to-end transparency and traceability within bioinformatics research focused on gene expression and methylation analysis.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2545467.out'>MethWayOS_2545467.out</a></b></td>
									<td style='padding: 8px;'>- Tracks and summarizes the results of a machine learning model training run focused on survival path analysis using gene expression and methylation data<br>- Provides insights into model performance metrics and experiment progress, while integrating with Weights & Biases for experiment tracking and visualization<br>- Plays a key role in monitoring and validating predictive models within the project’s bioinformatics workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_err\MethWayOS_2553735.out'>MethWayOS_2553735.out</a></b></td>
									<td style='padding: 8px;'>- Monitors and logs the execution and performance metrics of a model training run within the broader AI for Bioinformatics pipeline, capturing essential experiment details such as run summaries, loss values, and validation scores<br>- Supports experiment reproducibility and insight generation by syncing results to the W&B platform, enabling comprehensive tracking, comparison, and analysis alongside other bioinformatics experiments in the project.</td>
								</tr>
							</table>
						</blockquote>
					</details>
					<!-- slurm_out Submodule -->
					<details>
						<summary><b>slurm_out</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ logs.slurm.slurm_out</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2518507.out'>lucas_job_2518507.out</a></b></td>
									<td style='padding: 8px;'>- Documents the lifecycle and progress of a machine learning training job focused on gene expression data using a survival analysis model<br>- Captures environment setup, resource allocation, dataset preparation, and configuration details, along with indication of monitoring through external tools<br>- Serves as an audit trail for job execution and assists in tracing model training runs within the broader workflow of the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2518670.out'>lucas_job_2518670.out</a></b></td>
									<td style='padding: 8px;'>- Summary of Purpose**This file, <code>lucas_job_2518670.out</code>, serves as a log output that documents the execution of a specific machine learning job within the project<br>- It captures essential runtime information, including environment setup, resource allocation, dataset loading, and key milestones during the job’s lifecycle<br>- Positioned within the <code>logs/slurm/slurm_out/</code> directory, this output is integral for monitoring, auditing, and troubleshooting batch SLURM jobs<br>- Its primary role in the codebase is to provide transparency and traceability for experiments and computational tasks executed on the cluster, aiding both developers and researchers in validating that job runs proceed as intended.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2518671.out'>lucas_job_2518671.out</a></b></td>
									<td style='padding: 8px;'>- The file logs\slurm\slurm_out\lucas_job_2518671.out serves as an execution log, documenting the lifecycle and environment details of a specific SLURM-managed job within the project<br>- It provides a chronological account of key events such as environment setup, resource allocation (notably CUDA-enabled GPU usage), initialization steps, and dataset loading—with corresponding sample and class distribution information<br>- This log acts as an auditing and troubleshooting resource, offering visibility into both system configuration and high-level progress for experiments or data-processing tasks executed on the compute cluster<br>- Its role is critical for reproducibility, monitoring, and diagnosis within the broader codebase workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2518672.out'>lucas_job_2518672.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs/slurm/slurm_out/lucas_job_2518672.out serves as an execution log for a specific computational job within the project<br>- Its main purpose is to record key runtime information, such as environment setup, hardware utilization, progress updates, and data loading events<br>- This log supports transparency, debugging, and performance monitoring by providing insights into when and how the job was initiated, the computational resources used (including GPU details), and milestone events during processing<br>- Within the overall codebase architecture, this file acts as a traceable record, helping developers and researchers verify successful job execution and diagnose issues in systematic, reproducible experiments.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2518868.out'>lucas_job_2518868.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/lucas_job_2518868.out</code> serves as a runtime log for a specific computational job executed within the project’s larger workflow<br>- Its main purpose is to document the job’s lifecycle—including environment setup, hardware allocation, dataset loading, and initial processing steps—thus providing transparency, traceability, and diagnostic information<br>- This log is a crucial artifact for monitoring automated model training or data analysis pipelines, enabling users and developers to verify that resources were correctly provisioned and that core stages of the workflow executed as intended<br>- Within the greater codebase architecture, this log supports reproducibility, troubleshooting, and auditing of computational experiments.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2518869.out'>lucas_job_2518869.out</a></b></td>
									<td style='padding: 8px;'>- Summary**This file serves as an output log generated by a SLURM-managed job within the project<br>- It records key runtime events and environment details as the job executes, including resource utilization (such as GPU availability), environment activation, and dataset loading status<br>- By providing a detailed chronological trace of the jobs execution, the log facilitates monitoring, debugging, and post-run analysis, giving users and developers insights into job performance and data handling within the broader codebase workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2518870.out'>lucas_job_2518870.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/lucas_job_2518870.out</code> serves as an execution log for a computational job within the project<br>- It documents the environment setup, job status, hardware resources, and key initial steps of a run involving methylation data processing using CUDA acceleration<br>- This log is essential for monitoring, auditing, and troubleshooting jobs managed by a SLURM workload manager, providing transparency and traceability to the project’s data analysis and model training workflows.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2518873.out'>lucas_job_2518873.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/lucas_job_2518873.out</code> serves as a runtime log output for a scheduled computational job within the project<br>- It documents key events and environment details at the start of a machine learning workflow, confirming job initiation, resource allocation (such as CUDA usage and device information), and successful data loading<br>- This log is critical for monitoring, debugging, and auditing job execution, providing visibility into system setup and workflow progress without exposing internal code logic<br>- It helps ensure transparency and traceability as part of the projects broader pipeline infrastructure.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2519390.out'>lucas_job_2519390.out</a></b></td>
									<td style='padding: 8px;'>- Summary**This output log (<code>lucas_job_2519390.out</code>) serves as an execution record for a specific computational job within the project<br>- It documents the initialization and environment setup, resource allocation (such as GPU selection), and key data-loading steps, specifically confirming successful environment activation, hardware usage, and data preparation<br>- The log is essential for monitoring, debugging, and validating the correct operation of jobs in the overall workflow, providing transparency and traceability for the projects automated or batch-processed analyses.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2519461.out'>lucas_job_2519461.out</a></b></td>
									<td style='padding: 8px;'>- Execution log documents the end-to-end progress of a SLURM-scheduled machine learning experiment that integrates gene expression and methylation data to train and evaluate a survival analysis model using MCAT<br>- Captures environment setup, dataset preparation, partitioning, and model training phases, serving as a detailed trace for experiment reproducibility, resource utilization, and debugging within the broader workflow of large-scale biomedical data analysis.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2519471.out'>lucas_job_2519471.out</a></b></td>
									<td style='padding: 8px;'>- The <code>lucas_job_2519471.out</code> file serves as a log output capturing the execution details of a computational job within the project<br>- Specifically, it documents the environment activation, hardware utilization, dataset loading, and initialization of the MCAT" process<br>- This log is instrumental in monitoring and auditing the workflow by recording key milestones, such as resource allocation (e.g., GPU usage), data preparation steps, and job metadata<br>- Within the codebase architecture, this file supports transparency and reproducibility by providing a detailed chronological account of the job's runtime behavior, making it easier for users and maintainers to verify operations and diagnose issues.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2519472.out'>lucas_job_2519472.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The <code>lucas_job_2519472.out</code> file in the <code>logs/slurm/slurm_out</code> directory serves as an execution log for a scheduled MCAT (likely model training or analysis) job run on a high-performance computing cluster<br>- Its main purpose is to capture and document the environment setup, resource utilization, dataset loading, and initial configuration steps of the job<br>- This log provides critical runtime transparency, enabling users and developers to monitor job progress, verify system settings (such as CUDA availability and dataset specifications), and facilitate debugging as part of the larger project workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2519475.out'>lucas_job_2519475.out</a></b></td>
									<td style='padding: 8px;'>- Summary of <code>logs/slurm/slurm_out/lucas_job_2519475.out</code> in Project Context**This log file documents the execution of a specific computational job (Job 2519475) within the codebase, providing a chronological record of key events, environment setup, and dataset preprocessing steps<br>- Its main purpose is to capture runtime information for monitoring, debugging, and reproducibility<br>- The log confirms successful environment activation, resource allocation (including GPU availability), and the loading and processing of biological datasets<br>- Within the projects architecture, this output serves as a traceable artifact for tracking experiment status and results, supporting transparency and accountability in the project's workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2519476.out'>lucas_job_2519476.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/lucas_job_2519476.out</code> serves as an execution log for a computational job run within the projects workflow<br>- Its main purpose is to document the environment setup, resource allocation (including hardware details like GPU usage), and data-loading steps for a specific experiment or pipeline run<br>- This log provides transparency, traceability, and insight into how the job was processed on the cluster—including information about datasets used and any key runtime events<br>- Consequently, it supports reproducibility and debugging, offering a detailed account of a major step in the project’s data processing or analysis pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2519478.out'>lucas_job_2519478.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/lucas_job_2519478.out</code> serves as an execution log for a specific batch job within the projects workflow, captured under a SLURM-managed environment<br>- Its main purpose is to record the sequence of actions and key runtime events—including environment setup, hardware utilization, dataset loading, and job initialization—providing transparency and traceability for this compute task<br>- Within the overall codebase architecture, this log file is essential for monitoring, debugging, and auditing computational experiments, helping users and developers to verify job progress and troubleshoot any issues that may arise during automated runs on shared computing resources.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2519479.out'>lucas_job_2519479.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs/slurm/slurm_out/lucas_job_2519479.out serves as a log output for a specific computational job executed within the project’s workflow<br>- Its primary purpose is to provide a chronological record of the job’s execution, including environment setup, resource allocation, dataset loading, and key milestones during processing<br>- This log is essential for monitoring progress, troubleshooting, and auditing experiment runs, ensuring transparency and reproducibility within the project’s computational pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2520394.out'>lucas_job_2520394.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs\slurm\slurm_out\lucas_job_2520394.out serves as an execution log for a batch job managed by the SLURM workload scheduler within the project<br>- It captures the chronological progression and key milestones of a computational run—such as environment setup, resource allocation, data loading, and initialization events—offering transparency and traceability for model training or analysis workflows<br>- Positioned within the overall codebase architecture, this log is essential for monitoring computational experiments, diagnosing issues, and verifying that data and hardware resources are utilized as intended during large-scale or automated processing tasks.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2520395.out'>lucas_job_2520395.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs\slurm\slurm_out\lucas_job_2520395.out serves as a runtime log capturing the execution details of a specific batch job (ID 2520395) managed by the Slurm workload manager within the project<br>- Its main purpose is to document the chronological workflow for one experiment or computational run—covering environment setup (conda activation), resource allocation (CUDA device usage), data loading status, and the progression of analytical tasks<br>- This log provides transparency into the jobs environment and status, facilitating reproducibility, debugging, and performance monitoring within the broader project architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2521030.out'>lucas_job_2521030.out</a></b></td>
									<td style='padding: 8px;'>- Summary of logs\slurm\slurm_out\lucas_job_2521030.out**This file serves as a runtime log for a specific computational job within the broader project<br>- It documents the key stages and environmental context as the job progresses, such as resource allocation (CUDA device setup), data loading, and sample information<br>- These logs are crucial for monitoring, debugging, and auditing the execution of batch jobs within the codebases workflow, especially when leveraging high-performance computing resources<br>- The file helps ensure transparency and traceability of operations as part of the project's data processing and model training pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2521031.out'>lucas_job_2521031.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs/slurm/slurm_out/lucas_job_2521031.out serves as a runtime log capturing the execution details of a machine learning job (indexed as Job 2521031) within the codebase<br>- It documents key progress indicators such as environment setup, hardware utilization, dataset preparation, and preliminary job status<br>- This log is intended for monitoring, debugging, and auditing purposes, enabling developers and researchers to track workflow states, identify potential issues, and ensure reproducibility across compute clusters using the SLURM workload manager.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2521032.out'>lucas_job_2521032.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/lucas_job_2521032.out</code> serves as a detailed runtime log generated during the execution of a specific computational job (Job 2521032) within the projects workflow<br>- It documents the sequence of key events—such as environment setup, hardware configuration, and dataset preparation—providing traceability and transparency for experiments run on a SLURM-managed cluster<br>- Within the overall codebase architecture, this log file is instrumental for tracking experiment progress, diagnosing failures, and maintaining reproducibility by capturing the operational context and data processing milestones of each job execution.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2521818.out'>lucas_job_2521818.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The <code>logs/slurm/slurm_out/lucas_job_2521818.out</code> file serves as a comprehensive execution log for a specific batch job run on the projects SLURM-managed computation cluster<br>- It provides a chronological account of the environment setup, hardware utilization, dataset loading, and initial data processing steps for the MCAT pipeline<br>- This log is essential for monitoring resource allocation, debugging issues, and ensuring reproducibility across experiments<br>- Within the architecture, such logs enable users and developers to trace experiment histories, validate computational environments, and audit data processing consistency throughout the project lifecycle.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2522057.out'>lucas_job_2522057.out</a></b></td>
									<td style='padding: 8px;'>- Summary of <code>logs/slurm/slurm_out/lucas_job_2522057.out</code> in Project Architecture**This file captures the execution log for a computational job run within the project<br>- It records key environment details, resource allocation (such as GPU usage and host information), and confirmation of successful data loading and preprocessing steps (including gene expression and methylation data)<br>- Serving as an auditing and debugging resource, this log file enables team members to trace the workflows progress, verify that the computational pipeline initialized and ran as expected, and helps ensure reproducibility and transparency of analyses conducted on the project’s compute infrastructure.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2522213.out'>lucas_job_2522213.out</a></b></td>
									<td style='padding: 8px;'>- Logs the initiation and environment setup for a SLURM-managed job, confirming activation of the required Conda environment and successful job start<br>- Serves as an audit trail for job tracking and debugging within the larger workflow, enabling users and maintainers to verify the proper execution context and timing of computational tasks in the projects distributed or high-performance computing environment.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2522215.out'>lucas_job_2522215.out</a></b></td>
									<td style='padding: 8px;'>- Summary**This output log file captures the initialization and early execution stages of a computational job within the project, providing a transparent record of environment setup, resource allocation, and initial data loading steps<br>- Its primary purpose is to offer visibility into the runtime context (such as the computational device, active Conda environment, and dataset summary) for the job identified as <code>lucas_job_2522215</code><br>- By recording both the system settings and data preprocessing details, this log supports debugging, reproducibility, and workflow management, serving as an essential auditing and traceability artifact within the projects end-to-end processing pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2522217.out'>lucas_job_2522217.out</a></b></td>
									<td style='padding: 8px;'>- Managed via the SLURM workload manager<br>- This log records key stages such as environment activation, resource allocation (including GPU and CUDA usage), dataset loading, and sample preprocessing activities<br>- As part of the projects logging infrastructure, this output provides vital insights into job execution, resource utilization, and data pipeline checkpoints, supporting monitoring, debugging, and reproducibility within the overall codebase architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2522328.out'>lucas_job_2522328.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/lucas_job_2522328.out</code> serves as an execution log that documents the runtime status and key events of a computational job within the project<br>- It captures essential information such as environment setup, hardware utilization, dataset loading, and the progress of the job<br>- This log file is integral for tracking the success or failure of batch jobs, debugging issues, and auditing the project’s automated processing pipeline<br>- As part of the broader codebase, it supports reproducibility and operational transparency by providing a chronological record of job activities and system states.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2522329.out'>lucas_job_2522329.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs\slurm\slurm_out\lucas_job_2522329.out serves as an execution log for a computational job managed by SLURM within this project’s architecture<br>- It records key runtime events for a machine learning workflow, including environment setup, hardware availability, dataset loading, and initial processing statistics<br>- This log provides essential traceability and monitoring, enabling users to verify resource utilization, job status, and data integrity as part of reproducible and auditable scientific computing pipelines managed on a shared cluster environment.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2522330.out'>lucas_job_2522330.out</a></b></td>
									<td style='padding: 8px;'>- Succinct SummaryThis log file documents the execution of a machine learning job within the project’s workflow, capturing vital runtime information such as environment setup, hardware utilization, and data loading progress<br>- Its main purpose is to provide traceability and transparency for job runs, aiding in monitoring, debugging, and auditing computational experiments<br>- As part of the projects logging infrastructure, it facilitates reproducibility and performance analysis across the codebase by preserving key metadata and high-level status updates for each scheduled task.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2522588.out'>lucas_job_2522588.out</a></b></td>
									<td style='padding: 8px;'>- The file <code>logs/slurm/slurm_out/lucas_job_2522588.out</code> serves as a runtime log, capturing the execution details and environment setup for a specific batch job (ID: 2522588) within the project<br>- This log provides a sequential record of job initialization, resource allocation (including CUDA GPU use), and dataset loading steps, offering a transparent audit trail to monitor performance and troubleshoot issues<br>- Its role within the broader codebase architecture is to document the operational context and decisions made during automated computational runs, thus enabling reproducibility, accountability, and streamlined debugging in large-scale or collaborative computing environments using SLURM scheduling.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2522589.out'>lucas_job_2522589.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/lucas_job_2522589.out</code> serves as a runtime log output for a computational job executed within the projects workflow, specifically under a SLURM-managed high-performance computing environment<br>- Its primary purpose is to document the execution status and key environment details for the corresponding batch task—such as resource allocation, environment setup, data loading steps, and progress markers<br>- This log enables developers and researchers to monitor, audit, and debug the progress and proper functioning of large-scale computational experiments (in this case, MCAT analysis utilizing CUDA on genomics data), supporting reproducibility and transparency across the codebase.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2522590.out'>lucas_job_2522590.out</a></b></td>
									<td style='padding: 8px;'>- The file logs/slurm/slurm_out/lucas_job_2522590.out serves as a detailed execution log for a computational job (ID 2522590) within the project, providing a chronological record of the job’s environment setup, resource allocation, dataset loading, and initial progress<br>- Its main purpose is to facilitate transparent experiment tracking, debugging, and reproducibility by capturing key runtime information—such as activated environments, hardware utilization (CUDA device), and data processing status—for both users and maintainers<br>- This log acts as a critical auditing and diagnostic artifact within the overall architecture, supporting scientific rigor and operational reliability in the project’s workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2524499.out'>lucas_job_2524499.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The <code>logs/slurm/slurm_out/lucas_job_2524499.out</code> file serves as an execution log generated during a specific Slurm-scheduled job<br>- Its primary purpose is to provide a chronological trace of the job’s lifecycle—including environment activation, compute resource allocation, dataset loading, and key status updates<br>- Within the broader project architecture, this log is invaluable for monitoring experiment runs, diagnosing issues, and tracking resource utilization across the computational cluster<br>- It offers operational transparency that supports reproducibility and effective troubleshooting during large-scale data processing and model training workflows.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2524501.out'>lucas_job_2524501.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs\slurm\slurm_out\lucas_job_2524501.out serves as an execution log for a specific computational job (Job 2524501) processed within the project’s SLURM-managed high-performance environment<br>- This log documents the chronological progress and status of key runtime events, such as environment initialization, hardware utilization, dataset loading, and job start times<br>- As part of the project’s architecture, it provides transparency and traceability for computational experiments, enabling users and developers to monitor, audit, and debug pipeline runs without delving into implementation code.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2524502.out'>lucas_job_2524502.out</a></b></td>
									<td style='padding: 8px;'>- Slurm job output documents the end-to-end execution of a deep learning pipeline for survival analysis using gene expression and methylation data<br>- Captures the activation of the compute environment, dataset loading and preprocessing, model and training setup, and runtime status for MCAT model grid search<br>- Provides traceability, resource usage, and real-time feedback essential for debugging and validating large-scale experiments within the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2524503.out'>lucas_job_2524503.out</a></b></td>
									<td style='padding: 8px;'>- Within the project’s computational pipeline<br>- Its primary purpose is to provide traceability and transparency for the execution of data processing and model workflows—including environment setup, resource allocation (such as GPU and dataset details), and the status of key milestones<br>- This log is essential for debugging, performance monitoring, and auditing experiment runs, ultimately supporting reproducibility and efficient collaboration across the broader codebase.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2524504.out'>lucas_job_2524504.out</a></b></td>
									<td style='padding: 8px;'>- Logs the activation of the conda environment and the initiation of job 2524504 within the SLURM-managed job scheduling system<br>- Serves as a runtime checkpoint, confirming that required dependencies are loaded and the job has officially begun execution<br>- Provides essential traceability for debugging and monitoring in the context of automated, large-scale compute workflows managed by SLURM.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2524505.out'>lucas_job_2524505.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs\slurm\slurm_out\lucas_job_2524505.out serves as an execution log for a specific computational job (Job 2524505) managed via a SLURM workload scheduler<br>- Its main purpose is to document the step-by-step progress and key environment details of a machine learning run within the project<br>- This includes environment activation, dataset preparation, hardware resource allocation, and initiation of core analysis tasks (e.g., MCAT)<br>- In the context of the wider codebase architecture, this log provides transparency for job execution, aids in reproducibility, and offers a valuable record for monitoring, debugging, and auditing large-scale computational experiments typically managed in high-performance compute environments.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2524813.out'>lucas_job_2524813.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The <code>lucas_job_2524813.out</code> file serves as a log output capturing the lifecycle and progress of a computational job executed via the Slurm workload manager<br>- Its main purpose is to provide visibility into job execution, including environment setup, resource allocation, hardware utilization, and dataset loading status<br>- This log is essential for monitoring, debugging, and auditing the behavior of end-to-end analytic workflows within the broader project, ensuring transparency and facilitating efficient diagnosis of issues during large-scale data processing tasks.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2524890.out'>lucas_job_2524890.out</a></b></td>
									<td style='padding: 8px;'>- Summary of <code>logs/slurm/slurm_out/lucas_job_2524890.out</code> in Project Architecture**This file serves as an execution log capturing the lifecycle and runtime environment details of a specific computational job (<code>lucas_job_2524890</code>) run using the SLURM workload manager<br>- Within the broader project, such logs play a critical role in tracking experiment progress, resource usage (e.g., GPU availability and selection), dataset processing (such as gene expression and methylation data loading), and the configuration of monitoring tools (like Weights & Biases)<br>- By preserving detailed records of job execution and system context, this log file enhances traceability, reproducibility, and debugging capabilities across the entire data processing and model training pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2524959.out'>lucas_job_2524959.out</a></b></td>
									<td style='padding: 8px;'>- Summary**This file serves as an automated log output for a computational job executed within the project’s broader architecture<br>- Its primary purpose is to provide a detailed record of the environment setup, resource allocation, and data loading steps for a specific run (Job 2524959) of the MCAT workflow<br>- By capturing information such as environment activation, hardware utilization, dataset loading, and sample processing, this log file facilitates reproducibility, debugging, and monitoring within the project’s computational pipeline<br>- It helps users and developers track the computational context and execution status of individual jobs, supporting overall project transparency and traceability.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2524978.out'>lucas_job_2524978.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/lucas_job_2524978.out</code> serves as a log output for a specific execution of a SLURM-scheduled job within the projects workflow<br>- It provides a sequential record of key runtime events and system states, such as environment activation, hardware utilization, dataset loading, and job initialization<br>- This log is pivotal for monitoring the progress, verifying the computational environment, and diagnosing potential issues during scheduled batch runs<br>- Within the context of the entire codebase, this file is primarily used for transparency, traceability, and troubleshooting, enabling developers and researchers to audit and optimize the performance and correctness of automated experiments or data-processing jobs.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2524984.out'>lucas_job_2524984.out</a></b></td>
									<td style='padding: 8px;'>- Summary**This file serves as an execution log for a single SLURM-managed batch job within the project<br>- It documents key stages of the job lifecycle, such as environment activation, dataset initialization, hardware resource allocation, and the start of a major computational task (MCAT)<br>- By recording runtime details like host information, CUDA usage, and data-loading progress, the log enables users and maintainers to monitor, audit, and troubleshoot the workflow as part of the projects broader automated processing and experiment tracking infrastructure.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2527208.out'>lucas_job_2527208.out</a></b></td>
									<td style='padding: 8px;'>- Log output captures the activation of the conda environment and the initiation of a specific computational job, serving as a checkpoint in the job orchestration workflow<br>- Provides traceability for job execution status, aiding system monitoring and debugging efforts within the broader workflow management and resource scheduling infrastructure of the overall project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2527209.out'>lucas_job_2527209.out</a></b></td>
									<td style='padding: 8px;'>- Summary of logs\slurm\slurm_out\lucas_job_2527209.out**This log file serves as a runtime record for a specific computational job executed within the larger codebase<br>- It provides a chronological summary of key events and system status updates during the execution of a data processing and analysis workflow, including environment setup, hardware configuration (notably CUDA GPU usage), dataset loading, and initial data cleaning<br>- By capturing these details, the log facilitates monitoring, debugging, and reproducibility for users and developers, offering transparency into both the computational resources leveraged and the progress of critical pipeline stages within the project’s architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2527210.out'>lucas_job_2527210.out</a></b></td>
									<td style='padding: 8px;'>- This log file documents the execution of a specific computational job within the codebase, capturing essential runtime information such as environment setup, data loading steps, and hardware utilization<br>- It serves as a historical record for job 2527210, enabling users and developers to monitor progress, verify the correct initialization of resources (including dataset preparation and GPU allocation), and troubleshoot any issues that may arise during the workflow<br>- In the context of the overall project architecture, this file facilitates transparency and reproducibility for results derived from large-scale or resource-intensive analyses.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2527211.out'>lucas_job_2527211.out</a></b></td>
									<td style='padding: 8px;'>- This code file is a SLURM job output log that documents the execution of a machine learning experiment within the project<br>- Its primary purpose is to provide a detailed account of the jobs environment setup, resource allocation (such as GPU usage), data loading status, and experimental initialization<br>- As part of the broader codebase, this log file serves as an audit trail for computational runs, aiding in reproducibility, troubleshooting, and performance monitoring of experiments run on high-performance clusters.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2527212.out'>lucas_job_2527212.out</a></b></td>
									<td style='padding: 8px;'>- Summary of logs\slurm\slurm_out\lucas_job_2527212.out**This file serves as an execution log capturing the progress and critical runtime information for a specific computational job (Job 2527212) within the project<br>- It documents key stages—including environment setup, hardware utilization, dataset loading, and class distribution—providing traceability and operational transparency<br>- As part of the projects broader architecture, this log file supports debugging, monitoring, and reproducibility by detailing the job's configuration and status during a particular run on the compute cluster.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2527231.out'>lucas_job_2527231.out</a></b></td>
									<td style='padding: 8px;'>- This file serves as a log output generated from the execution of a specific job (identified as <code>lucas_job_2527231</code>) managed by the SLURM job scheduler<br>- Within the broader project architecture, it plays a crucial role in recording and tracking the runtime environment and any relevant information or diagnostics associated with the jobs execution<br>- This log assists developers and users by providing transparency into the job's lifecycle, aiding in debugging, auditing, and ensuring reproducibility across computational tasks orchestrated by SLURM.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2527233.out'>lucas_job_2527233.out</a></b></td>
									<td style='padding: 8px;'>- Summary of logs/slurm/slurm_out/lucas_job_2527233.out**This file serves as an output log for a computational job executed via the SLURM job scheduler<br>- Within the context of the projects architecture, it documents the key milestones and environment details of a specific workflow run, including system configuration, resource allocation, dataset loading, and process initiation<br>- This log is primarily used for tracking job execution status, monitoring resource usage, and facilitating reproducibility and debugging across the broader pipeline<br>- By providing a clear record of when and where processing tasks took place, along with hardware and data loading confirmations, this output file is essential for maintaining transparency, auditability, and operational oversight within the overall system.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2527234.out'>lucas_job_2527234.out</a></b></td>
									<td style='padding: 8px;'>- This file serves as an automated log output for a scheduled job run within the project, capturing key runtime events and environmental setup details<br>- Its main purpose is to provide traceability and insight into the execution of computational workflows—such as confirming environment activation, dataset loading, computational resource allocation, and job lifecycle milestones<br>- By documenting job progress and configuration, it helps users and developers monitor, audit, and debug large-scale analyses as part of the project’s broader data processing pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2527235.out'>lucas_job_2527235.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/lucas_job_2527235.out</code> serves as an execution log for a specific computational job within the project<br>- Its main purpose is to document the runtime environment, resource utilization (such as GPU availability and selected device), and the progress of various stages—such as environment activation, dataset loading, and initiation of key computational tasks<br>- This log is essential for monitoring, auditing, and debugging, providing transparency about how and where the job ran, which datasets were accessed, and the high-level status updates<br>- In the broader architecture, such logs help ensure reproducibility and facilitate efficient troubleshooting of large-scale, automated workflows managed by SLURM in the projects infrastructure.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2527236.out'>lucas_job_2527236.out</a></b></td>
									<td style='padding: 8px;'>- This file serves as an output log generated during the execution of an automated computational job within the project<br>- It documents key status updates, environment setup confirmation, resource allocation (such as GPU and dataset usage), and the progression of major workflow steps<br>- Primarily, this log enables users and developers to verify the successful start, resource assignment, and initial data preparation phases of the pipeline, supporting reproducibility, debugging, and monitoring within the overall project lifecycle.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2527622.out'>lucas_job_2527622.out</a></b></td>
									<td style='padding: 8px;'>- Summary**This output log file documents the automated execution and environment setup of a machine learning experiment managed via SLURM job scheduling<br>- It provides a high-level record of the run, including environment activation, hardware and software configuration, dataset preparation, and initial dataset statistics<br>- Serving as an auditable trail, this file supports reproducibility, monitoring, and debugging by preserving details critical to understanding experiment context and resource utilization within the broader project workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2527661.out'>lucas_job_2527661.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs/slurm/slurm_out/lucas_job_2527661.out serves as a runtime log output for a specific batch job executed within the projects computational pipeline<br>- Its main purpose is to document and report the progress, system environment, and key milestones during the execution of a machine learning or data analysis workflow—specifically highlighting environment setup, resource availability, dataset loading, and preliminary data integrity checks<br>- This log is essential for monitoring, troubleshooting, and auditing the workflow as it runs on a SLURM-managed compute cluster<br>- While not directly contributing to the codebase’s logic, this file plays a vital role in ensuring transparency and reproducibility in the project's end-to-end execution.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2527662.out'>lucas_job_2527662.out</a></b></td>
									<td style='padding: 8px;'>- Summary of logs\slurm\slurm_out\lucas_job_2527662.out**This file serves as a runtime log output for a specific batch job executed within the project, capturing a full record of the computational environment and execution steps<br>- Its main purpose is to document and verify the successful initiation and progression of a job—identified as MCAT"—including environment setup, hardware resources, data loading status, and key checkpoints<br>- This log is instrumental for auditability, troubleshooting, and reproducibility, providing users and developers with transparency into how and when critical tasks were performed in the workflow<br>- It complements the rest of the codebase by enabling robust monitoring of automated data processing or machine learning pipelines on high-performance computing infrastructure.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2527663.out'>lucas_job_2527663.out</a></b></td>
									<td style='padding: 8px;'>- This log file serves as an execution record for a machine learning experiment run within the project<br>- It captures key milestones such as environment activation, hardware resource allocation, dataset loading, and experiment metadata<br>- By documenting these checkpoints, the file provides transparency and traceability for the experiments lifecycle, supporting debugging and reproducibility within the broader data processing and modeling workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2527664.out'>lucas_job_2527664.out</a></b></td>
									<td style='padding: 8px;'>- Log output confirms successful activation of the Conda environment and initiation of job 2527664 within the SLURM workload management system<br>- Provides traceability and status verification for job submissions, assisting in monitoring and debugging workflows across the distributed computing infrastructure<br>- Facilitates coordination between environment setup and job execution throughout the project’s batch processing pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2527699.out'>lucas_job_2527699.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs/slurm/slurm_out/lucas_job_2527699.out serves as a runtime log capturing the execution details of a specific scheduled job within the project’s SLURM-managed workflow<br>- Its main purpose is to provide a chronological record of key events, environment setup, resource allocation (such as GPU usage), and dataset processing milestones encountered during the run<br>- This log is essential for monitoring, troubleshooting, and auditing experiment runs, offering insights into system configuration, data preparation, and initial pipeline status<br>- In the context of the broader codebase, such log files contribute to experiment transparency and reproducibility by documenting operational aspects that support both development and research workflows.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2527700.out'>lucas_job_2527700.out</a></b></td>
									<td style='padding: 8px;'>- Summary of logs/slurm/slurm_out/lucas_job_2527700.out**This file serves as an execution log for a batch job initiated on the projects HPC infrastructure, documenting key events and environment details from start to completion<br>- Its primary purpose is to provide transparency and traceability into the workflow’s runtime context, including environment activation, resource allocation (such as GPU usage), and dataset preparation steps<br>- By capturing these details, the log file supports debugging, experiment reproducibility, and performance analysis in the context of the overall data processing and model training pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2527703.out'>lucas_job_2527703.out</a></b></td>
									<td style='padding: 8px;'>- This log file chronologically documents the execution of a computational job within the project’s workflow, specifically capturing key runtime information such as environment setup, hardware utilization, dataset loading, and initial data preprocessing steps<br>- Serving as an audit trail, it is primarily used for monitoring, debugging, and verifying that crucial stages of the job—like resource allocation and data preparation—have completed successfully<br>- Its presence in the codebase ensures transparency and reproducibility by providing users and developers with a detailed record of each major process executed during this run.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2528093.out'>lucas_job_2528093.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs\slurm\slurm_out\lucas_job_2528093.out serves as a runtime log capturing the progress and system status of a specific batch job executed via SLURM, likely on an HPC or GPU-enabled server<br>- Its primary purpose within the architecture is to provide a reproducible record of key events—including environment activation, resource allocation (such as GPU usage), dataset loading steps, and preprocessing outcomes—for the MCAT workflow<br>- This log is crucial for monitoring, debugging, and auditing experiments, ensuring transparency and traceability for the larger projects model training or data analysis pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2528094.out'>lucas_job_2528094.out</a></b></td>
									<td style='padding: 8px;'>- The file logs\slurm\slurm_out\lucas_job_2528094.out serves as an execution log for a computational job within the projects workflow<br>- Its main purpose is to document the runtime environment, resource allocation, and key milestones of the job, such as environment activation, hardware utilization, dataset loading, and data preprocessing results<br>- This log provides essential transparency and traceability for users and developers, enabling efficient monitoring, debugging, and auditability of batch jobs executed in the system<br>- It plays a supportive yet vital role in the broader codebase by facilitating reproducibility and operational oversight.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2528551.out'>lucas_job_2528551.out</a></b></td>
									<td style='padding: 8px;'>- Summary of logs/slurm/slurm_out/lucas_job_2528551.out**This file serves as an execution log for a machine learning batch job managed by the SLURM scheduler within the project<br>- It provides a timestamped record of environment setup, hardware utilization, dataset preparation, and job initialization<br>- As part of the projects logging architecture, this output helps track experiment progress, resource usage (such as GPU allocation), and high-level steps during automated training or evaluation runs<br>- These logs are essential for monitoring, debugging, and reproducing computational workflows, supporting the overall transparency and reliability of large-scale or high-throughput experiments in the codebase.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2528553.out'>lucas_job_2528553.out</a></b></td>
									<td style='padding: 8px;'>- Summary of logs/slurm/slurm_out/lucas_job_2528553.out**This file serves as an execution log for a specific computational job (Job 2528553) run within the project<br>- It documents the environment setup, dataset loading, and hardware utilized during a workflow involving gene expression and methylation data analysis<br>- As part of the overall codebase, this log provides transparency and traceability for model runs, supporting reproducibility and debugging by capturing key milestones and resource usage throughout the data processing and analysis pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2528555.out'>lucas_job_2528555.out</a></b></td>
									<td style='padding: 8px;'>- The provided file, logs\slurm\slurm_out\lucas_job_2528555.out, serves as an execution log for a specific SLURM job within the project<br>- Its main purpose is to document the environment setup, resource allocation (such as GPU usage), dataset loading status, and overall workflow initiation for the corresponding job run<br>- This log is essential for tracking job progress, diagnosing issues, and verifying successful startup and data loading steps in the projects larger workflow<br>- By retaining these execution details, the codebase supports reproducibility, transparency, and troubleshooting in job management and result validation across the computational pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2528556.out'>lucas_job_2528556.out</a></b></td>
									<td style='padding: 8px;'>- Summary**This output log file documents the execution and key runtime events of a specific ML workflow within the projects pipeline, as orchestrated by a SLURM-managed job<br>- It serves as a historical record of environment setup, hardware allocation, data loading, and initial process milestones for a particular task run<br>- By capturing these details, the file provides vital traceability and operational transparency, supporting troubleshooting, performance assessment, and reproducibility across the whole codebase.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2528558.out'>lucas_job_2528558.out</a></b></td>
									<td style='padding: 8px;'>- The <code>lucas_job_2528558.out</code> file captures the runtime environment and execution log details for a specific batch job submitted to the SLURM job scheduler<br>- Within the context of the codebase, this log file serves as an audit trail, providing visibility into the start time, environment activation, resources used (such as GPU availability), and progress indicators for the job identified as <code>2528558</code><br>- This logging is crucial for debugging, monitoring, and verifying experiment reproducibility, helping users and maintainers track the operational flow and status of computational tasks across the projects larger automated workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2528613.out'>lucas_job_2528613.out</a></b></td>
									<td style='padding: 8px;'>- The file <code>logs/slurm/slurm_out/lucas_job_2528613.out</code> serves as a runtime log capturing the progress and environment details of a specific computational job (Job 2528613) executed via SLURM on the project’s high-performance computing cluster<br>- This log provides a chronological record of environment activation, dataset loading, hardware utilization (including GPU details), and the initialization of analytical tasks<br>- Within the broader architecture, this output is vital for monitoring job execution, auditing resource usage, and diagnosing issues, ensuring transparency and reproducibility for large-scale data processing tasks in the project workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2529616.out'>lucas_job_2529616.out</a></b></td>
									<td style='padding: 8px;'>- The file <code>logs/slurm/slurm_out/lucas_job_2529616.out</code> serves as a log output for a computational job executed via the SLURM workload manager<br>- Within the broader project architecture, this file documents the runtime environment, system resources, and key milestones in the execution of a data processing pipeline—specifically tracking the start, resource allocation (such as GPU usage), dataset loading stages, and sample filtering outcomes<br>- By capturing this information, the log is essential for monitoring, debugging, and auditing the job’s progression and resource utilization within the overall workflow, supporting reproducibility and operational transparency for the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2529618.out'>lucas_job_2529618.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/lucas_job_2529618.out</code> serves as an execution log for a specific batch job run within the project<br>- It captures the environment setup, dataset loading status, hardware utilization (including GPU details), and key operational milestones during the jobs lifecycle<br>- This log is instrumental for monitoring job progress, validating computational resources, and troubleshooting issues<br>- Within the overall architecture, such log files play a crucial role in ensuring reproducibility, auditing, and effective management of large-scale experiments or data processing tasks executed through the project's compute infrastructure.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2529716.out'>lucas_job_2529716.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs\slurm\slurm_out\lucas_job_2529716.out serves as an execution log for a specific computational job (ID 2529716) within the broader project<br>- Its primary purpose is to record the jobs runtime environment, resource allocation (e.g., hardware, available GPUs), and status updates during an analysis run, including data loading steps and sample counts<br>- This log is essential for monitoring, debugging, and auditing individual analyses, providing transparency into the workflow's operation on the SLURM-clustered infrastructure, and supporting reproducibility of results across the project’s pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2529717.out'>lucas_job_2529717.out</a></b></td>
									<td style='padding: 8px;'>- Logs comprehensive details of a specific machine learning training job, including resource allocation, dataset preparation, model configuration, and training progression<br>- Serves as an execution trace for a survival analysis task leveraging gene expression and methylation data<br>- Supports reproducibility, debugging, and performance monitoring within the project’s end-to-end experimentation workflow, offering visibility into each stage from environment activation to model optimization.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2529845.out'>lucas_job_2529845.out</a></b></td>
									<td style='padding: 8px;'>- Job runtime output details the complete execution lifecycle of a survival analysis experiment using gene expression and methylation data on a GPU-enabled node<br>- Logs trace dataset preparation, preprocessing, signature extraction, data partitioning, and the launch of a grid search with cross-validation for MCAT model training and evaluation<br>- Conveys experiment reproducibility, resource usage, hyperparameters, and progress for experiment tracking and auditing within the broader ML pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2529886.out'>lucas_job_2529886.out</a></b></td>
									<td style='padding: 8px;'>- Execution summary captures the initialization and orchestration of a comprehensive multi-omics survival analysis workflow within the codebase<br>- Logs detail environment setup, dataset preparation, signature extraction, and model hyperparameter tuning, culminating in cross-validation and training phases<br>- Acts as a central artifact for tracking computational provenance, model configuration choices, and resource utilization, supporting both experimentation transparency and reproducibility across the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2529895.out'>lucas_job_2529895.out</a></b></td>
									<td style='padding: 8px;'>- Summary**This file serves as an automatically generated output log that records the initialization and execution details of a computational job (Job 2529895) within the projects workflow<br>- It captures the activation of the working environment, job start time, system resources utilized (such as GPU and dataset information), and preliminary dataset processing steps<br>- The information provided by this log is crucial for monitoring and debugging the job's progress and resource allocation, ensuring transparency and traceability within the overall architecture of the project’s batch processing and model training pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2529896.out'>lucas_job_2529896.out</a></b></td>
									<td style='padding: 8px;'>- SummaryThe <code>lucas_job_2529896.out</code> file serves as an execution log for a computational job related to the <strong>MethWayOS</strong> workflow within the project<br>- Its main purpose is to document the progression and status of a specific experiment or analysis run—capturing key events such as environment activation, hardware utilization (CUDA device setup), data loading (gene expression, methylation), and data preprocessing outcomes (e.g., the number of valid samples)<br>- This log provides valuable audit trails and feedback for users and developers, facilitating reproducibility, debugging, and performance monitoring as part of the broader projects pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2530872.out'>lucas_job_2530872.out</a></b></td>
									<td style='padding: 8px;'>Certainly! Please provide the code file or its contents so I can deliver an accurate summary referencing your project structure and context.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2530876.out'>lucas_job_2530876.out</a></b></td>
									<td style='padding: 8px;'>- The file at <code>logs/slurm/slurm_out/lucas_job_25308</code> serves as a log output for a specific job (lucas_job_25308") managed by the SLURM workload manager<br>- Within the context of the project’s codebase, this log file captures runtime information, outputs, and potential errors generated during the execution of that SLURM job<br>- It is primarily used for monitoring, debugging, and auditing computational tasks performed on a cluster, and does not contain source code or business logic<br>- Instead, it supports the overall project infrastructure by providing essential transparency and traceability for batch processing workflows executed on SLURM-managed resources.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2530911.out'>lucas_job_2530911.out</a></b></td>
									<td style='padding: 8px;'>- SummaryThe <code>logs/slurm/slurm_out/lucas_job_2530911.out</code> file documents the execution log of a specific batch job within the project, providing a detailed trace of the runtime environment, resource allocation, and data loading steps<br>- Serving as a record of both system setup and initial dataset processing, this log enables users and developers to track the job’s configuration, monitor hardware utilization (such as GPU availability), and verify that data preprocessing stages were completed successfully<br>- Within the broader codebase architecture, these logs are essential for debugging, auditing, and optimizing high-performance computational workflows, particularly in environments managed by SLURM and oriented towards large-scale, reproducible experiments.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2532004.out'>lucas_job_2532004.out</a></b></td>
									<td style='padding: 8px;'>- Certainly! However, it appears that you haven’t provided the code file in question or any further context about the project<br>- Please supply the code file or additional project details so I can deliver an accurate, succinct summary tailored to the code’s purpose within your codebase’s architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2532879.out'>lucas_job_2532879.out</a></b></td>
									<td style='padding: 8px;'>The code file or its name-Additional data about the project (such as the project structure, high-level description, or key modules)Once you provide this, I’ll deliver a succinct, purpose-focused summary for the code file in context with your codebase architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2532881.out'>lucas_job_2532881.out</a></b></td>
									<td style='padding: 8px;'>- Captures the execution trace of a machine learning workflow focused on integrating gene expression and methylation data for a cancer survival prediction task<br>- Serves as a log output within the broader architecture, detailing environment setup, data processing, risk stratification, and experiment tracking<br>- Provides essential runtime insights for experiment reproducibility, resource utilization, data integrity, and validation of end-to-end pipeline operation within the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2532883.out'>lucas_job_2532883.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/lucas_job_2532883.out</code> serves as an output log for a specific batch job executed within the projects SLURM-based high-performance computing (HPC) environment<br>- Its main purpose is to document the runtime environment setup, resource allocation, and progress of a critical model training or analysis session—namely, the MCAT process—on biological datasets such as gene expression and methylation data<br>- By capturing job status, hardware context (like CUDA GPU availability), dataset loading steps, and initial configuration details, this log provides essential visibility into the execution and reproducibility of large-scale computational experiments within the project’s data processing and machine learning pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2532884.out'>lucas_job_2532884.out</a></b></td>
									<td style='padding: 8px;'>- Log output documents the successful activation of the required Conda environment and the initiation of job 2532884 within the SLURM workload manager<br>- Serving as an execution checkpoint, it provides essential traceability for job scheduling and environment setup, supporting the overall systems ability to monitor, debug, and validate computational workflows within the broader pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2532886.out'>lucas_job_2532886.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/lucas_job_2532886.out</code> serves as an execution log, documenting the lifecycle and key runtime events of a computational job within the project’s workflow<br>- This log captures high-level milestones such as environment setup, resource allocation (including hardware details), dataset loading, and initial data validation steps<br>- Its purpose is to provide an accessible record of the job’s execution context and progress, enabling transparency, monitoring, and troubleshooting in the broader context of automated, batch-processed data analysis pipelines orchestrated via SLURM.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2532922.out'>lucas_job_2532922.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs\slurm\slurm_out\lucas_job_2532922.out serves as an automatically generated log capturing the lifecycle events, environment setup, and key progress checkpoints for a specific SLURM-managed compute job within the project<br>- It summarizes the successful activation of the required execution environment (conda), the allocation and identification of available hardware resources (CUDA/RTX GPU), and the correct loading and filtering of essential biological datasets (Gene Expression, Methylation)<br>- This log provides end-users and developers with immediate visibility into the status, reproducibility, and resource context of the associated MCAT workflow, facilitating robust job monitoring and debugging without delving into low-level implementation specifics.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2532945.out'>lucas_job_2532945.out</a></b></td>
									<td style='padding: 8px;'>- Summary of lucas_job_2532945.out in Project Architecture**This file serves as an execution log for a computational job initiated via the SLURM workload manager<br>- It documents the start, environment activation, hardware setup (including CUDA GPU usage), and initial data loading steps for the job<br>- Within the overall codebase, this log provides a transparent record of the runtime environment and data pipeline status for experiment tracking, debugging, and reproducibility—ensuring all key stages and system settings are auditable for future reference and reporting.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2533210.out'>lucas_job_2533210.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/lucas_job_2533210.out</code> serves as an execution log for a batch job within the project<br>- It records the progress and environment details of a specific computational task related to the MCAT process, including conda environment activation, hardware usage (notably CUDA GPU), data loading steps, and dataset statistics<br>- This log provides transparency and traceability for the jobs run, helping users monitor, debug, and audit task execution as part of the project's overall workflow, especially when leveraging high-performance computing resources managed via SLURM.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2542136.out'>lucas_job_2542136.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The provided file, <code>lucas_job_2542136.out</code>, is an automated log output from a batch job executed within the project’s high-performance computing environment<br>- Its primary purpose is to record the initialization and runtime status of a modelling workflow, capturing essential details such as environment setup, computational resources, dataset loading, and data integrity checks<br>- This log serves as a transparent, timestamped account of the jobs context and early progress, aiding developers and researchers in monitoring, debugging, and auditing workflow executions within the broader project architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\lucas_job_2542142.out'>lucas_job_2542142.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/lucas_job_2542142.out</code> serves as an execution log for a scheduled computational job within the project<br>- It documents the environment activation, resource allocation, and dataset loading processes, providing traceable records of a run for the <code>MethWayOS</code> module on a high-performance computing cluster (using SLURM job scheduling)<br>- This log helps developers and researchers monitor job status, track resource usage, and verify data preprocessing steps for reproducibility and debugging within the broader analytics pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MCAT_2542055_0.6243.out'>MCAT_2542055_0.6243.out</a></b></td>
									<td style='padding: 8px;'>- Summary of logs/slurm/slurm_out/MCAT_2542055_0.6243.out in Project Architecture**This log file documents the execution of a computational job within the project, specifically capturing the environment setup and successful initiation of a machine learning workflow under the MCAT" module<br>- It records important run-time details such as resource allocation (e.g., GPU availability), dataset loading progress, and preprocessing actions (like the removal of incomplete samples)<br>- Serving primarily as an operational record, this file enables monitoring, reproducibility, and debugging for experiments managed through a SLURM job scheduler in the overall codebase.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MCAT_2542056_0.5666.out'>MCAT_2542056_0.5666.out</a></b></td>
									<td style='padding: 8px;'>- SummaryThe provided file, <code>logs/slurm/slurm_out/MCAT_2542056_0.5666.out</code>, serves as an automated runtime log capturing the progress and key checkpoints of an MCAT (Multi-omics Clinical Analysis Tool) computational job executed on a managed cluster<br>- This log records environment activation, job and hardware initialization, external tool configuration (such as Weights & Biases for experiment tracking), data loading steps, and dataset statistics<br>- Within the broader codebase architecture, this file is integral for monitoring experiments, diagnosing failures, and auditing resource usage, enabling developers and researchers to verify that complex cloud-based analyses execute as intended without delving into implementation specifics.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MCAT_2542057_0.5315.out'>MCAT_2542057_0.5315.out</a></b></td>
									<td style='padding: 8px;'>- Summary**This output log captures the execution trace of a computational job within the project, illustrating the successful initialization and runtime environment of a specific experiment<br>- The log verifies critical aspects such as environment activation, hardware resource detection (notably CUDA GPU usage), dataset loading (including gene expression and methylation data), and sample preprocessing stages<br>- Serving as a key record, it enables users and developers to validate experiment reproducibility, monitor resource utilization, and troubleshoot the experimental pipeline within the broader codebase.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\SurvPath_2542058_0.6065.out'>SurvPath_2542058_0.6065.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs\slurm\slurm_out\SurvPath_2542058_0.6065.out serves as a runtime log output for a scheduled SLURM job executing the SurvPath pipeline within the project<br>- Its primary purpose is to document the environment setup, hardware utilization, and initial dataset loading steps for a specific job instance<br>- This log allows users and developers to monitor the successful launch and configuration of a SurvPath experiment, as well as to track data preprocessing and resource allocation status<br>- This traceability is essential for debugging and auditing within the larger context of reproducible computational workflows managed by the project’s job orchestration and logging infrastructure.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\SurvPath_2542060_0.6091.out'>SurvPath_2542060_0.6091.out</a></b></td>
									<td style='padding: 8px;'>- Summary of Purpose**This code file serves as an execution log generated during a model training or experiment run within the SurvPath project<br>- It records the activation of the runtime environment, the initialization of hardware resources (such as GPU allocation), and the successful loading and preprocessing of key biological datasets (gene expression and methylation data)<br>- By capturing these steps, the log provides traceability and confirmation that the experimental pipeline started correctly, the computational environment was appropriately configured, and the data was prepared as intended<br>- This information is essential for troubleshooting, reproducibility, and monitoring within the broader project workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2542175_0.6982.out'>MethWayOS_2542175_0.6982.out</a></b></td>
									<td style='padding: 8px;'>- This file serves as an execution log for a computational job run within the project, capturing the progress and environment status of a MethWayOS analysis utilizing both gene expression and methylation data<br>- It documents the initiation of a specific job on the cluster, providing essential checkpoints such as environment activation, resource allocation, dataset loading, and the state of sample preprocessing<br>- This log is valuable for tracing, auditing, and diagnosing the workflow of the MethWayOS module, supporting reproducibility and transparency within the projects end-to-end data processing and machine learning pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2542176_0.6947.out'>MethWayOS_2542176_0.6947.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/MethWayOS_2542176_0.6947.out</code> serves as a runtime log output for a specific SLURM-managed job within the project<br>- Its primary purpose is to record and report key execution events, environment information, software environment activation, resource allocation, and dataset loading statuses for a MethWayOS analysis run<br>- This log enables developers and researchers to monitor, audit, and debug the jobs lifecycle in the context of the larger codebase, which likely involves large-scale, GPU-accelerated genomic or methylation data processing workflows<br>- It plays a crucial role in operational transparency and reproducibility, assisting users in tracing experiment outcomes and identifying issues without delving into implementation details.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2542177_0.6261.out'>MethWayOS_2542177_0.6261.out</a></b></td>
									<td style='padding: 8px;'>- File Purpose & Role in Project Architecture**The file <code>logs/slurm/slurm_out/MethWayOS_2542177_0.6261.out</code> serves as a runtime log output generated by a SLURM-managed job execution within the projects workflow<br>- Its main role is to provide a detailed, timestamped record of the MethWayOS model's execution, capturing key environment details, resource allocation (such as CUDA device usage), dataset loading steps, and high-level milestones during the job<br>- This log is essential for monitoring, debugging, and auditing the behavior of computational experiments, especially when running resource-intensive tasks on cluster hardware<br>- Within the architecture, such log files enable reproducibility and traceability of model runs, supporting both developers and researchers in ensuring and validating the integrity of automated training and evaluation pipelines.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2544870_0.6317.out'>MethWayOS_2544870_0.6317.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/MethWayOS_2544870_0.6317.out</code> serves as a runtime log capturing the execution lifecycle of a computational job within the project<br>- Specifically, it documents the environment setup, hardware utilization, and data loading steps for a MethWayOS analysis task, including job initiation, resource allocation, and preprocessing progress<br>- As part of the projects logging subsystem, this log provides visibility into reproducibility, debugging, and performance monitoring—helping users and developers track the status and outcomes of individual jobs in a high-performance or distributed compute environment.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2544876_0.6211.out'>MethWayOS_2544876_0.6211.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/MethWayOS_2544876_0.6211.out</code> serves as an execution log for a specific run of the MethWayOS analysis within the project<br>- It documents the start of a SLURM-managed batch job, the activation of the necessary computing environment, allocation of GPU resources, and successful loading and preprocessing of the gene expression and methylation datasets<br>- This log is vital for monitoring and auditing experiment progress and data integrity, providing contextual insights into resource usage and dataset handling within the overall analysis pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2544899_0.6587.out'>MethWayOS_2544899_0.6587.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/MethWayOS_2544899_0.6587.out</code> serves as a job log output capturing the execution details and progress of the MethWayOS pipeline within the broader project<br>- Positioned within the logging directory structure, this file documents runtime environment setup, hardware utilization, data loading steps, and preliminary filtering processes for a specific computational experiment<br>- Its primary role is to provide traceability and transparency for resource usage, dataset status, and initial pipeline milestones, assisting users and developers in monitoring, validating, and debugging the MethWayOS job as part of the projects end-to-end data processing and analysis workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2544900.out'>MethWayOS_2544900.out</a></b></td>
									<td style='padding: 8px;'>- Summary of Purpose**The <code>MethWayOS_2544900.out</code> file serves as a comprehensive log capturing the execution details of a MethWayOS analysis job within the project<br>- Its primary role is to document the job’s lifecycle—including environment setup, hardware usage, dataset loading, and the initiation of model training or analysis—providing both transparency and traceability for computational experiments<br>- This logging is crucial for monitoring progress, diagnosing issues, and reproducibility of results across the codebase, especially in high-performance or distributed computing environments where multiple jobs and resources are managed concurrently.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2544901.out'>MethWayOS_2544901.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/MethWayOS_2544901.out</code> documents the execution log of a Slurm-managed computational job integral to the MethWayOS projects workflow<br>- Specifically, it provides a chronological record of resource allocation, environment setup, dataset loading, and hardware utilization for an analysis run focusing on gene expression and methylation data<br>- This log file is essential for monitoring job progress, troubleshooting, and validating successful execution steps within the broader automated data processing and modeling pipeline facilitated by the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2544990_0.5743.out'>MethWayOS_2544990_0.5743.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The provided log file, <code>MethWayOS_2544990_0.5743.out</code>, serves as an execution record for a computational job run by the MethWayOS pipeline within the projects workflow<br>- Its primary purpose is to capture job initialization, computing environment details, data loading status, and system resource utilization (such as GPU availability) during this specific run<br>- This log file is valuable for tracking the success and reproducibility of the pipeline's operations, facilitating debugging, and providing traceability within the overall data analysis architecture<br>- It complements the codebase by documenting runtime progress and outcomes, thus supporting transparent and auditable computational processes.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2544991_0.5841.out'>MethWayOS_2544991_0.5841.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs\slurm\slurm_out\MethWayOS_2544991_0.5841.out serves as a detailed runtime log for a computational experiment executed within the project<br>- Its primary purpose is to capture the execution context, resource allocation, and step-by-step progress of a survival analysis pipeline utilizing gene expression and methylation data<br>- By recording environment setup, hardware usage, data loading, and sample filtering, this log provides transparency and traceability for the experiment, supporting reproducibility and aiding in debugging within the larger workflow managed by SLURM and Conda on high-performance computing resources.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2545213_0.6026.out'>MethWayOS_2545213_0.6026.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/MethWayOS_2545213_0.6026.out</code> serves as a runtime log capturing the execution details of a specific computational job within the project<br>- Its primary role is to document the key steps taken during a model training or analysis workflow, including environment activation, hardware resource allocation, dataset loading, and resource usage (e.g., GPU selection)<br>- This log provides transparency and traceability for successful operations and potential issues, aiding in both debugging and reproducibility within the projects broader data analysis and machine learning pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2545299_0.6230.out'>MethWayOS_2545299_0.6230.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs\slurm\slurm_out\MethWayOS_2545299_0.6230.out serves as a runtime log output for the MethWayOS job executed within the projects workflow<br>- It provides a chronological record of the environment setup, resource allocation (e.g., GPU usage), and successful data loading processes, confirming that gene expression and methylation data have been integrated and preprocessed for downstream analysis<br>- This log is essential for validating and auditing the execution of computational experiments, offering transparency and traceability within the larger system pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2545300_0.5813.out'>MethWayOS_2545300_0.5813.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs\slurm\slurm_out\MethWayOS_2545300_0.5813.out serves as a runtime log for a specific execution of the MethWayOS pipeline within the codebase<br>- Its primary purpose is to chronicle essential events and environment details—including job initialization, hardware setup, dataset loading, and sample filtering—for an individual Slurm-managed computation<br>- This log is crucial for users and developers, as it documents reproducibility information, system environment specifics, and workflow progress, enabling effective monitoring and troubleshooting of long-running or resource-intensive pipeline tasks without delving into implementation details.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2545301_0.5883.out'>MethWayOS_2545301_0.5883.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs\slurm\slurm_out\MethWayOS_2545301_0.5883.out serves as a runtime log output for a specific execution of the MethWayOS pipeline within the codebase<br>- Its main purpose is to chronicle the environment setup, resource allocation, data loading, and initial processing steps during a scheduled job, providing transparency and traceability for computational experiments run via the SLURM workload manager<br>- This log helps users monitor and debug pipeline runs, verify resource usage (such as CUDA-enabled GPUs), and confirm successful data preprocessing stages—supporting overall experiment management and reproducibility within the project’s broader architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2545321_0.6211.out'>MethWayOS_2545321_0.6211.out</a></b></td>
									<td style='padding: 8px;'>- Summary**This code file serves as a runtime log generated by a scheduled SLURM job responsible for executing a MethWayOS analysis<br>- Its main purpose is to provide traceable and timestamped feedback about the job’s progress, environment setup, hardware utilization (notably CUDA GPU usage), dataset loading status, and preprocessing steps<br>- Within the overall project architecture, this log file acts as an audit record for monitoring job executions, diagnosing issues, and validating computational reproducibility within the data processing and analysis pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2545422_0.6142.out'>MethWayOS_2545422_0.6142.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file logs/slurm/slurm_out/MethWayOS_2545422_0.6142.out serves as an execution log for a batch job run within the projects computational pipeline<br>- It documents the initialization, runtime environment, dataset loading, and preliminary status updates for a specific model training or inferencing task related to the MethWayOS workflow<br>- This log is essential for tracking job execution details, diagnosing issues, and verifying that each stage in the pipeline has started and completed as expected<br>- Its role within the codebase is to provide transparent, chronological insights into automated batch processing, complementing other system outputs, and facilitating smooth large-scale analysis and troubleshooting.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2545423_0.5681.out'>MethWayOS_2545423_0.5681.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/MethWayOS_2545423_0.5681.out</code> serves as an execution log capturing the runtime environment, status updates, and progress messages for a batch job associated with the MethWayOS pipeline<br>- Its primary role within the codebase is to provide traceability and transparency for long-running, resource-intensive analyses conducted on clustered systems (using SLURM workload manager)<br>- This log is essential for monitoring, debugging, and auditing the MethWayOS job by detailing environment activation, hardware utilization, data loading steps, and sample filtering outcomes without exposing low-level code or algorithm specifics.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2545424_0.6099.out'>MethWayOS_2545424_0.6099.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The <code>logs/slurm/slurm_out/MethWayOS_2545424_0.6099.out</code> file serves as an execution log for a specific MethWayOS batch job run on the projects cluster infrastructure<br>- Within the context of the codebase, this log captures the environment setup, system resources allocation (including GPU availability and usage), data loading status, and the commencement of a survival analysis workflow<br>- It provides critical visibility into the operational lifecycle of a MethWayOS experiment, supporting monitoring, debugging, and auditability of computational runs without exposing implementation-level details.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2545467.out'>MethWayOS_2545467.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/MethWayOS_2545467.out</code> documents the execution progress and runtime environment for a specific SLURM job associated with the MethWayOS pipeline<br>- It serves as a log output, confirming successful initialization of required environments, hardware utilization (including CUDA-enabled GPUs), data loading steps (such as gene expression and methylation datasets), and sample preprocessing results<br>- Within the broader architecture, this log provides essential traceability and diagnostic feedback for monitoring computational jobs, aiding in troubleshooting, performance tracking, and reproducibility across the data analysis workflows managed by the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/logs\slurm\slurm_out\MethWayOS_2553735.out'>MethWayOS_2553735.out</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>logs/slurm/slurm_out/MethWayOS_2553735.out</code> is an output log generated during the execution of a machine learning workflow within the project<br>- This log captures the initialization and key runtime events of a job tasked with processing gene expression and methylation data, including dataset loading, resource allocation, and environment setup<br>- Its main purpose is to provide a traceable record of the job’s progress and execution environment, supporting monitoring, debugging, and reproducibility across the codebase.</td>
								</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<!-- src Submodule -->
	<details>
		<summary><b>src</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ src</b></code>
			<!-- binary_classification Submodule -->
			<details>
				<summary><b>binary_classification</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ src.binary_classification</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\2.1_ge_&_os_sklearn.py'>2.1_ge_&_os_sklearn.py</a></b></td>
							<td style='padding: 8px;'>- Orchestrates the complete binary classification pipeline for gene expression datasets, encompassing data acquisition, exploratory analysis, preprocessing, feature selection, splitting, model selection, hyperparameter tuning via grid search, cross-validation, model training, and evaluation<br>- Acts as the main workflow driver for the machine learning process, ensuring standardized execution and logging across each stage of the pipeline within the larger project architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\2.1_ge_&_os_sklearn.sbatch'>2.1_ge_&_os_sklearn.sbatch</a></b></td>
							<td style='padding: 8px;'>- Orchestrates the execution of a binary classification experiment on a SLURM-managed high-performance computing cluster<br>- Ensures proper configuration of computational resources, environment setup, and logging while launching the core classification workflow<br>- Integrates seamlessly into the project’s broader architecture by automating and standardizing reproducible large-scale experiments with GPU acceleration in the context of the AI for Bioinformatics initiative.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\2.2_methylation_&_os_sklearn.py'>2.2_methylation_&_os_sklearn.py</a></b></td>
							<td style='padding: 8px;'>- End-to-end pipeline orchestration for binary classification using DNA methylation data, driving the process from dataset acquisition through exploratory analysis, feature engineering, model selection, hyperparameter tuning, validation, training, and final evaluation<br>- Serves as the central script coordinating modular components, enabling reproducible, systematic experimentation and model assessment within the broader machine learning workflow for genomic data in the project.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\2.2_methylation_&_os_sklearn.sbatch'>2.2_methylation_&_os_sklearn.sbatch</a></b></td>
							<td style='padding: 8px;'>- Automates the execution of a binary classification experiment related to methylation and overall survival analysis within a high-performance computing environment<br>- Integrates with SLURM to allocate computing resources, set up the necessary software environment, and launch the relevant Python workflow<br>- Ensures reproducibility and efficient resource management, supporting scalable experimentation as part of the projects bioinformatics research pipeline.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\3.1_ge_&_os_gpu.py'>3.1_ge_&_os_gpu.py</a></b></td>
							<td style='padding: 8px;'>- Coordinates the end-to-end binary classification pipeline for gene expression data, orchestrating dataset acquisition, exploratory analysis, preprocessing, feature selection, and splitting<br>- Executes feature selection trials, converts data for GPU processing, performs hyperparameter tuning using grid search, and trains and tests deep learning models<br>- Centralizes experiment logging, enabling systematic evaluation of model performance across varying feature dimensions within the broader project workflow.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\3.1_ge_&_os_gpu.sbatch'>3.1_ge_&_os_gpu.sbatch</a></b></td>
							<td style='padding: 8px;'>- Automates the scheduling and execution of a binary classification GPU-based training job within a high-performance computing environment<br>- Integrates core project dependencies by configuring environment modules, activating the appropriate Conda environment, and ensuring correct project path setup<br>- Enables reproducible and efficient submission of deep learning experiments as part of the project’s larger binary classification workflow, supporting scalable experimentation and reliable resource utilization across heterogeneous GPU clusters.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\3.2_methylation_&_os_gpu.py'>3.2_methylation_&_os_gpu.py</a></b></td>
							<td style='padding: 8px;'>- Facilitates the end-to-end pipeline for binary classification on methylation data, orchestrating data acquisition, exploratory analysis, preprocessing, feature selection, model training, and evaluation<br>- Integrates both scikit-learn and PyTorch stages, leveraging automated feature selection and hyperparameter optimization<br>- Serves as the main entry point to evaluate model performance across varying feature counts, with comprehensive logging for experiment tracking within the larger classification framework.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\3.2_methylation_&_os_gpu.sbatch'>3.2_methylation_&_os_gpu.sbatch</a></b></td>
							<td style='padding: 8px;'>- Orchestrates the automated execution of GPU-accelerated methylation and overall survival analysis as part of the binary classification pipeline, leveraging SLURM job scheduling on a high-performance computing cluster<br>- Integrates environment preparation, resource allocation, and logging, ensuring reproducible results and smooth workflow integration within the projects broader AI-driven bioinformatics framework for studying genomic data.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\3.3_ge_&_methylation_&_os_gpu.py'>3.3_ge_&_methylation_&_os_gpu.py</a></b></td>
							<td style='padding: 8px;'>- Orchestrates the full binary classification workflow for gene expression and methylation data, automating data acquisition, exploratory analysis, preprocessing, feature selection, train/test splitting, and model training and evaluation on a GPU<br>- Integrates grid search for hyperparameter optimization, iteratively assessing different feature counts to identify optimal model configurations, and logs all outcomes, serving as a core pipeline for robust, reproducible experiments across omics datasets.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\3.3_ge_&_methylation_&_os_gpu.sbatch'>3.3_ge_&_methylation_&_os_gpu.sbatch</a></b></td>
							<td style='padding: 8px;'>- Orchestrates the scheduling and execution of a GPU-accelerated binary classification task by leveraging gene expression, methylation, and overall survival data<br>- Integrates with the projects SLURM workflow, ensuring the appropriate computational resources and environment are configured<br>- Supports scalable, reproducible experimentation within the broader AIforBioinformatics pipeline, underpinning model training and evaluation phases central to bioinformatics research objectives.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\4.1_ge_&_os_gpu_v1.py'>4.1_ge_&_os_gpu_v1.py</a></b></td>
							<td style='padding: 8px;'>- Orchestrates a complete GPU-accelerated binary classification pipeline for gene expression data, encompassing data acquisition, preprocessing, feature selection, and exploratory analysis for both training and testing sets<br>- Progresses through model preparation, hyperparameter optimization via grid search, and robust training using k-fold cross-validation with voting, culminating in standardized evaluation and logging of model performance for reproducible machine learning experimentation within the project’s architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\4.1_ge_&_os_gpu_v1.sbatch'>4.1_ge_&_os_gpu_v1.sbatch</a></b></td>
							<td style='padding: 8px;'>- Facilitates the automated execution of a GPU-accelerated binary classification task within a high-performance computing environment<br>- Integrates with SLURM to manage compute resources, environment setup, and logging, ensuring reproducibility and efficient utilization of available hardware<br>- Supports the overall architecture by orchestrating model training or inference workflows for large-scale bioinformatics experiments aligned with the projects objectives.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\4.2_methylation_&_os_gpu_v1.py'>4.2_methylation_&_os_gpu_v1.py</a></b></td>
							<td style='padding: 8px;'>- Orchestrates the end-to-end pipeline for binary classification using DNA methylation data, encompassing dataset acquisition, preprocessing, feature selection, data transformation from Scikit-learn to PyTorch, hyperparameter tuning with grid search, model training, and comprehensive evaluation<br>- Integrates robust logging and configuration management, serving as the primary workflow driver for methylation-based outcome prediction within the overall project architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\4.2_methylation_&_os_gpu_v1.sbatch'>4.2_methylation_&_os_gpu_v1.sbatch</a></b></td>
							<td style='padding: 8px;'>- Automates and manages the execution of a GPU-accelerated binary classification experiment focused on methylation data within a high-performance computing environment<br>- Integrates resource allocation, environment setup, and logging, ensuring reproducible and efficient job runs<br>- Serves as a critical entry point for orchestrating large-scale, resource-intensive model training within the project’s overarching machine learning pipeline.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\4.3_ge_&_methylation_&_os_gpu_v1.py'>4.3_ge_&_methylation_&_os_gpu_v1.py</a></b></td>
							<td style='padding: 8px;'>- Orchestrates the end-to-end workflow for binary classification using gene expression and methylation data, including data loading, preprocessing, feature selection, and conversion to PyTorch tensors<br>- Executes hyperparameter tuning, model training with k-fold cross-validation, and rigorous evaluation, while systematically logging results<br>- Acts as the main entry point integrating data science and machine learning pipelines within the projects binary classification architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\4.3_ge_&_methylation_&_os_gpu_v1.sbatch'>4.3_ge_&_methylation_&_os_gpu_v1.sbatch</a></b></td>
							<td style='padding: 8px;'>- Automates the execution of a GPU-accelerated binary classification experiment that integrates gene expression, methylation, and overall survival data<br>- Manages resource allocation and environment setup on an HPC cluster, ensuring consistent and reproducible runs<br>- Serves as a bridge between the project’s machine learning workflows and the cluster’s job scheduling system, enabling scalable experimentation within the project’s data analysis pipeline.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\5_ge_&_methylation_statistics_&_os_gpu.py'>5_ge_&_methylation_statistics_&_os_gpu.py</a></b></td>
							<td style='padding: 8px;'>- Orchestrates the entire workflow for binary classification using gene expression and methylation data by managing data acquisition, preprocessing, exploratory analysis, feature selection, hyperparameter tuning, model training, and evaluation<br>- Integrates both scikit-learn and PyTorch pipelines, leveraging configuration-driven paths and logging, to enable reproducible, end-to-end experiments within the broader machine learning architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\5_ge_&_methylation_statistics_&_os_gpu.sbatch'>5_ge_&_methylation_statistics_&_os_gpu.sbatch</a></b></td>
							<td style='padding: 8px;'>- Automates the execution of a GPU-accelerated analysis pipeline for binary classification of gene expression and methylation data, integrating with the projects machine learning environment and resource allocation on a high-performance computing cluster<br>- Supports robust statistical evaluation and operational tracking, ensuring reproducible, large-scale experiments and seamless integration with the broader AIforBioinformatics codebase for bioinformatics research and outcome analysis.</td>
						</tr>
					</table>
					<!-- functions_sklearn Submodule -->
					<details>
						<summary><b>functions_sklearn</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.binary_classification.functions_sklearn</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_sklearn\f10_testing.py'>f10_testing.py</a></b></td>
									<td style='padding: 8px;'>- Facilitates evaluation of multiple binary classification models by generating and displaying key performance metrics, such as accuracy, precision, recall, and F1-score, on a provided test dataset<br>- Supports standardized assessment and benchmarking of model effectiveness within the broader architecture, streamlining model comparison and validation as part of the projects machine learning workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_sklearn\f1_dataset_acquisition.py'>f1_dataset_acquisition.py</a></b></td>
									<td style='padding: 8px;'>- Facilitates the retrieval of datasets for binary classification tasks by leveraging a centralized CSV loading utility<br>- Integrates seamlessly with the projects overall data ingestion pipeline, ensuring that datasets and their corresponding columns are efficiently prepared for downstream processing and model training within the binary classification module<br>- Contributes to modularity and reusability by abstracting dataset access behind a simple interface.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_sklearn\f2_exploratory_data_analysis.py'>f2_exploratory_data_analysis.py</a></b></td>
									<td style='padding: 8px;'>- Facilitates initial data exploration within the binary classification workflow by providing both summary statistics and visualizations of the dataset, with a focus on label distribution<br>- Supports rapid understanding of data characteristics for analysts and modelers working in the src/binary_classification/functions_sklearn module, serving as a crucial first step before downstream preprocessing and modeling activities in the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_sklearn\f3_features_preprocessing.py'>f3_features_preprocessing.py</a></b></td>
									<td style='padding: 8px;'>- Feature preprocessing for binary classification workflow is enabled by standardizing selected columns, handling missing and duplicate values, and mapping target labels into binary classes based on configurable thresholds<br>- Integrates clean data preparation to ensure consistency and quality prior to model training or evaluation, fitting into the broader pipeline for reliable machine learning outcomes across the project’s binary classification tasks.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_sklearn\f4_features_selection.py'>f4_features_selection.py</a></b></td>
									<td style='padding: 8px;'>- Feature selection for binary classification is streamlined by combining Principal Component Analysis (PCA) for dimensionality reduction with a correlation filter to retain features most relevant to the target variable<br>- This process ensures that subsequent modeling within the codebase leverages a compact and informative set of features, optimizing performance while reducing noise and redundancy in the input data.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_sklearn\f5_dataset_splitting.py'>f5_dataset_splitting.py</a></b></td>
									<td style='padding: 8px;'>- Dataset partitioning is streamlined by separating input data into distinct training and testing sets, ensuring proper evaluation of model performance<br>- As part of the binary classification pipeline, this functionality prepares datasets for downstream model development and validation, maintaining consistency and reproducibility across experiments by leveraging configurable shuffling and random seed options within the projects scikit-learn integration.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_sklearn\f6_models.py'>f6_models.py</a></b></td>
									<td style='padding: 8px;'>- Provides centralized access to a curated selection of machine learning models, their display names, and associated hyperparameter grids, all tailored for binary classification tasks<br>- By serving as part of the model selection and configuration layer within the codebase, it streamlines experimentation, enabling efficient benchmarking and optimization of different classifiers during the pipelines training and evaluation phases.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_sklearn\f7_grid_search.py'>f7_grid_search.py</a></b></td>
									<td style='padding: 8px;'>- Automates the process of hyperparameter tuning for multiple machine learning models within the binary classification pipeline<br>- Integrates model selection and evaluation by systematically applying grid search to identify optimal hyperparameters, enhancing model performance and consistency<br>- Facilitates experimentation and comparison across different algorithms, supporting robust decision-making within the broader supervised learning workflow of the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_sklearn\f8_cross_validation_model_assessment.py'>f8_cross_validation_model_assessment.py</a></b></td>
									<td style='padding: 8px;'>- Cross-validation model assessment orchestrates the evaluation of several key classification models—Decision Tree, Multi-Layer Perceptron, Support Vector Classifier, and a Bagging-based Random Forest—using optimal hyperparameters and weighted metrics<br>- Serving as a central validation point, it provides standardized performance comparisons and outputs configured estimators, supporting informed model selection within the broader binary classification workflow of the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_sklearn\f9_training.py'>f9_training.py</a></b></td>
									<td style='padding: 8px;'>- Facilitates the training of multiple scikit-learn models on a given dataset while providing runtime metrics for each model<br>- Plays a key role in the binary classification workflow by automating the model fitting process and returning trained estimators, enabling efficient experimentation and comparison of algorithm performance within the broader machine learning pipeline of the project.</td>
								</tr>
							</table>
						</blockquote>
					</details>
					<!-- functions_torch Submodule -->
					<details>
						<summary><b>functions_torch</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.binary_classification.functions_torch</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_torch\f10_training_bagging_ensemble.py'>f10_training_bagging_ensemble.py</a></b></td>
									<td style='padding: 8px;'>- Orchestrates the training of a bagging ensemble of neural network models for binary classification, leveraging data sampling, model diversity, and early stopping to boost predictive robustness<br>- Integrates with the broader architecture by enabling efficient and scalable creation of multiple models, ultimately supporting improved generalization and stability for downstream prediction tasks across the project’s binary classification workflows.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_torch\f10_training_kfold_voting.py'>f10_training_kfold_voting.py</a></b></td>
									<td style='padding: 8px;'>- Facilitates robust model training for binary classification by orchestrating k-fold cross-validation with an ensemble approach<br>- Integrates hyperparameter tuning, early stopping, and dynamic learning rate adjustment to optimize multilayer perceptron performance<br>- Serves as a core component for evaluating and producing reliable, generalizable models within the projects binary classification workflow, enhancing overall predictive accuracy and model robustness.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_torch\f10_training_single_model.py'>f10_training_single_model.py</a></b></td>
									<td style='padding: 8px;'>- Coordinates the end-to-end training workflow for a binary classification neural network using PyTorch, handling cross-validation, early stopping, class imbalance, and hyperparameter configurations<br>- Integrates with the broader codebase to ensure that the best-performing model is selected and fully trained for downstream use, forming the critical backbone for reliable and well-validated model development in the binary classification component.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_torch\f11_testing_kfold_voting.py'>f11_testing_kfold_voting.py</a></b></td>
									<td style='padding: 8px;'>- Enables robust evaluation of multiple neural network models in the binary classification pipeline by aggregating predictions through majority voting and calculating key performance metrics, including accuracy, loss, precision, recall, and F1 score<br>- Facilitates objective model comparison and selection across k-fold splits, serving as a critical component of the validation and testing workflow within the projects PyTorch-based architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_torch\f11_testing_single_model.py'>f11_testing_single_model.py</a></b></td>
									<td style='padding: 8px;'>- Evaluates the performance of a trained binary classification model on test data by generating key metrics such as loss, accuracy, precision, recall, and F1 score<br>- Facilitates objective assessment and comparison of model effectiveness within the broader machine learning pipeline, ensuring trustworthy evaluation before deployment or integration with other system components in the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_torch\f1_dataset_acquisition_and_splitting.py'>f1_dataset_acquisition_and_splitting.py</a></b></td>
									<td style='padding: 8px;'>- Dataset ingestion and preparation for binary classification are streamlined by loading data from CSV, applying label thresholds to define class boundaries, and splitting the data into stratified training and testing sets<br>- This functionality ensures consistent and reproducible dataset management across experiments, forming a foundational step in the pipeline for model training and evaluation within the project’s binary classification module.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_torch\f4_features_selection_training_set.py'>f4_features_selection_training_set.py</a></b></td>
									<td style='padding: 8px;'>- Feature selection logic refines input datasets for binary classification by isolating the most relevant predictors based on variance and correlation criteria<br>- By systematically filtering out redundant and irrelevant features, it ensures that downstream Torch-based model training operates on a streamlined, information-rich feature set, ultimately enhancing model performance and reducing computational overhead within the broader machine learning pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_torch\f5_features_selection_testing_set.py'>f5_features_selection_testing_set.py</a></b></td>
									<td style='padding: 8px;'>- Feature selection functionality isolates the most relevant input variables from a preprocessed dataset to streamline and optimize the testing phase of the binary classification pipeline<br>- By filtering the dataset to retain only the top-determined features and preserving target labels, the process ensures that subsequent evaluation steps operate on a reduced, more informative feature set that aligns with prior selection decisions in the overall project workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_torch\f6_sklearn_to_torch.py'>f6_sklearn_to_torch.py</a></b></td>
									<td style='padding: 8px;'>- Facilitates seamless conversion of data from pandas DataFrames to PyTorch tensors for both training and testing datasets, while automatically managing device allocation to either GPU or CPU<br>- Serves as a core utility enabling compatibility between sklearn-style data preprocessing and neural network models in PyTorch, supporting an efficient transition within the project’s binary classification pipeline architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_torch\f7_hyperparameters.py'>f7_hyperparameters.py</a></b></td>
									<td style='padding: 8px;'>- Defines centralized sets of hyperparameter options tailored for different neural network architectures within the binary classification module<br>- Supports consistent and flexible configuration selection for model training and optimization in PyTorch-based workflows<br>- Facilitates systematic experimentation and hyperparameter tuning across varying network depths, enabling the broader codebase to standardize model training pipelines and streamline comparative evaluations of network performance.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_torch\f8_grid_search.py'>f8_grid_search.py</a></b></td>
									<td style='padding: 8px;'>- Hyperparameter optimization engine orchestrates comprehensive grid search and cross-validation to identify the most effective neural network configuration for binary classification tasks<br>- By systematically testing parameter combinations and evaluating performance metrics, it ensures optimal model accuracy and reliability<br>- Integral to the model training workflow, it connects model definition, training, and testing modules to drive high-quality results across the project’s binary classification pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\functions_torch\f9_mlp_models.py'>f9_mlp_models.py</a></b></td>
									<td style='padding: 8px;'>- Defines a flexible multi-layer perceptron (MLP) component tailored for binary classification workflows, supporting configurable hidden layer architectures to accommodate varying model complexities<br>- Serves as a core neural network building block within the binary classification module, enabling the broader project to leverage customizable deep learning models for processing and classifying input data as part of the end-to-end machine learning pipeline.</td>
								</tr>
							</table>
						</blockquote>
					</details>
					<!-- model_weights Submodule -->
					<details>
						<summary><b>model_weights</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.binary_classification.model_weights</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\binary_classification\model_weights\Methylation & OS Binary Classification.pth'>Methylation & OS Binary Classification.pth</a></b></td>
									<td style='padding: 8px;'>- Summary**The file <code>src/binary_classification/model_weights/Methylation & OS Binary Classification.pth</code> contains pre-trained model weights essential for the binary classification component of the project<br>- It enables the core functionality of predicting outcomes related to methylation and overall survival (OS) without requiring retraining from scratch<br>- Integrated into the broader codebase, this file allows rapid deployment and evaluation of the projects classification models, underscoring its role in delivering accurate and efficient predictions within the system’s architecture.</td>
								</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
			<!-- dataset_creators Submodule -->
			<details>
				<summary><b>dataset_creators</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ src.dataset_creators</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\dataset_creators\1.1_gene_expression_&_overall_survival.py'>1.1_gene_expression_&_overall_survival.py</a></b></td>
							<td style='padding: 8px;'>- Generation of a machine learning-ready dataset by integrating gene expression feature vectors with overall survival labels for each patient, leveraging parallel processing for efficiency<br>- Aligns gene identifiers, processes raw JSON datastores, and outputs standardized CSV files for downstream analysis, serving as the foundational data preparation step within the projects broader genomic and survival outcome pipeline architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\dataset_creators\1.2_methylation_&_overall_survival.py'>1.2_methylation_&_overall_survival.py</a></b></td>
							<td style='padding: 8px;'>- Generation of a comprehensive dataset that merges gene-associated methylation features with overall survival outcomes for each patient, facilitating downstream analysis and model development<br>- Operates as a key pipeline component, efficiently aligning and labeling patient records, storing the final results in CSV format, and ensuring robust progress tracking and logging within the broader data processing architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\dataset_creators\1.3.1_gene_expression_&_methylation_&_overall_survival.py'>1.3.1_gene_expression_&_methylation_&_overall_survival.py</a></b></td>
							<td style='padding: 8px;'>- Generates a comprehensive dataset by integrating gene expression, methylation, and overall survival information from multiple sources<br>- Aligns patient records across these domains, constructs feature vectors combining genetic and epigenetic data, and assigns survival outcomes as labels<br>- Facilitates robust downstream analysis or model training by storing the resulting dataset in a standardized CSV format, supporting reproducibility and efficient data handling within the project architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\dataset_creators\1.3.2_gene_expression_&_methylation_&_overall_survival.py'>1.3.2_gene_expression_&_methylation_&_overall_survival.py</a></b></td>
							<td style='padding: 8px;'>- Generates a comprehensive dataset by integrating gene expression, methylation, and overall survival data for each patient<br>- Serves as a core data preparation step, aligning multiple modalities into unified feature vectors and labels for downstream analysis or model training<br>- Supports parallel processing for efficiency and ensures results are stored in a standardized CSV format, supporting reproducibility across the broader research pipeline.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\dataset_creators\1.4.1_mcat_gene_expression_&_overall_survival_for_wsi.py'>1.4.1_mcat_gene_expression_&_overall_survival_for_wsi.py</a></b></td>
							<td style='padding: 8px;'>- Generates a comprehensive dataset by integrating gene expression profiles, whole slide image metadata, and overall survival information for patients<br>- Enables efficient preparation of feature vectors and survival labels by aggregating and aligning data across multiple sources, facilitating downstream analysis or machine learning tasks within the project’s broader workflow for biomedical research and predictive modeling.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\dataset_creators\1.4.2_mcat_gene_expression_&_overall_survival_for_methylation.py'>1.4.2_mcat_gene_expression_&_overall_survival_for_methylation.py</a></b></td>
							<td style='padding: 8px;'>- Generation of a comprehensive dataset that integrates gene expression profiles with patient overall survival data, preparing it for downstream analysis related to methylation studies<br>- Patient records are matched and processed in parallel to efficiently extract relevant features and survival labels, producing a well-structured CSV output that serves as a foundational component for predictive modeling and further research within the project’s data pipeline.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\dataset_creators\1.5_mcat_methylation_&_overall_survival.py'>1.5_mcat_methylation_&_overall_survival.py</a></b></td>
							<td style='padding: 8px;'>- Dataset generation pipeline merges patient DNA methylation profiles with corresponding overall survival data, aligning and structuring these attributes into a unified tabular dataset<br>- Enables downstream machine learning or statistical modeling by exporting a comprehensive CSV that combines molecular features with survival outcomes, enhancing reproducibility and efficiency within the broader biomedical data preparation workflow for analysis across the projects research and modeling components.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\dataset_creators\1.6_deep_prog_gene_expression_&_overall_survival.py'>1.6_deep_prog_gene_expression_&_overall_survival.py</a></b></td>
							<td style='padding: 8px;'>- Generation of standardized DeepProg-compatible gene expression and overall survival datasets, enabling downstream analysis and modeling within the larger pipeline<br>- Ingests raw MCAT-formatted methylation data, applies preprocessing and normalization steps, and outputs clean, analysis-ready files while logging all operations<br>- Plays a critical role in ensuring data consistency for subsequent modules that utilize standardized input within the projects broader data processing architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\dataset_creators\1.7_deep_prog_methylation.py'>1.7_deep_prog_methylation.py</a></b></td>
							<td style='padding: 8px;'>- Facilitates the preprocessing and alignment of DNA methylation, gene expression, and survival datasets for downstream analysis in the DeepProg pipeline<br>- Transforms raw methylation input into standardized gene-level features, ensures sample consistency across all modalities, and manages standardized output storage, providing a harmonized data foundation crucial for integrative multi-omics analysis and robust machine learning workflows within the overall project architecture.</td>
						</tr>
					</table>
				</blockquote>
			</details>
			<!-- data_extractors Submodule -->
			<details>
				<summary><b>data_extractors</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ src.data_extractors</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\data_extractors\0.0_case_id_extractor.py'>0.0_case_id_extractor.py</a></b></td>
							<td style='padding: 8px;'>- Extraction and validation of case identifiers across multiple cancer-related datasets form the core purpose within the data pipeline<br>- Facilitates streamlined gathering and basic integrity checks of case IDs from gene expression, methylation, and survival JSON sources, adapting to cancer type variations<br>- Outputs relevant summaries and potential data inconsistencies, supporting downstream analytics and ensuring data quality throughout the larger project architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\data_extractors\0.1_gene_expression_extractor.py'>0.1_gene_expression_extractor.py</a></b></td>
							<td style='padding: 8px;'>- Gene expression data extraction and transformation for downstream analysis is orchestrated in alignment with the project’s modular data pipeline<br>- Input across multiple sources—including survival metadata, genomic annotations, and raw gene expression files—is systematically filtered, formatted, and synchronized by case identifiers<br>- Final structured outputs are stored in standardized JSON formats, supporting consistent, automated integration with broader analytical workflows throughout the codebase.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\data_extractors\0.2.1_methylation_extractor.py'>0.2.1_methylation_extractor.py</a></b></td>
							<td style='padding: 8px;'>- Automates extraction, filtering, and formatting of DNA methylation data by integrating survival and methylation datasets, cross-referencing relevant patient IDs, and generating a processed JSON datastore for downstream analysis<br>- Serves as a critical data preprocessing step within the architecture, transforming raw methylation files into structured outputs that are ready for machine learning workflows and further statistical evaluation across the pipeline.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\data_extractors\0.2.2_cgp950_methylation_islands_counter_per_gene.py'>0.2.2_cgp950_methylation_islands_counter_per_gene.py</a></b></td>
							<td style='padding: 8px;'>- Quantifies and sorts methylation island data associated with specific genes for downstream genomic analysis<br>- Integrates configuration, input, and output paths, processes JSON-based datasets, and logs statistical summaries on methylation island counts per gene<br>- Ensures organized and consistently ordered data storage, facilitating reliable data extraction workflows and supporting robust, reproducible research across the broader bioinformatics data processing architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\data_extractors\0.2.3.1_methylation_vectors_for_each_gene.py'>0.2.3.1_methylation_vectors_for_each_gene.py</a></b></td>
							<td style='padding: 8px;'>- Generation of gene-centric methylation feature vectors by aggregating site-specific methylation data around each gene’s transcription start site, facilitating downstream analysis and integration with gene expression profiles<br>- Enables structured storage of extracted vectors and associated genomic regions, serving as a data preparation step in the overall pipeline for studying relationships between methylation patterns and gene expression across samples.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\data_extractors\0.2.3.2_methylation_vectors_for_each_gene.py'>0.2.3.2_methylation_vectors_for_each_gene.py</a></b></td>
							<td style='padding: 8px;'>- Gene-level methylation profiles are generated by extracting and organizing methylation island data relative to gene promoters, considering strand orientation and transcription start sites<br>- The resulting vectors are saved alongside summaries mapping genes to methylation islands, supporting downstream epigenetic analysis and integration with gene expression data within the broader pipeline for genomic and transcriptomic data processing.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\data_extractors\0.2.4.1_gene_associated_methylation_extractor.py'>0.2.4.1_gene_associated_methylation_extractor.py</a></b></td>
							<td style='padding: 8px;'>- Extraction and preparation of gene-associated methylation data are facilitated by aggregating, filtering, and standardizing methylation and survival datasets for a specified cancer type<br>- Ensures only relevant and high-quality patient data are included, aligns input sources, and outputs a harmonized dataset suitable for further downstream analysis throughout the project’s data processing pipeline.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\data_extractors\0.2.4.2_gene_associated_methylation_statistics_extractor.py'>0.2.4.2_gene_associated_methylation_statistics_extractor.py</a></b></td>
							<td style='padding: 8px;'>- Gene-associated methylation statistical extraction orchestrates the aggregation and computation of gene-specific methylation metrics across patient datasets<br>- It integrates raw methylation data with gene mappings and patient survival information, efficiently distilling key statistics per gene for downstream analysis<br>- Results are filtered for consistency and stored for further use, supporting comprehensive exploration of gene-methylation relationships within the broader data pipeline.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\data_extractors\0.3_overall_survival_extractor.py'>0.3_overall_survival_extractor.py</a></b></td>
							<td style='padding: 8px;'>- Overall survival data for a specific cancer type is extracted, filtered, and consolidated from input JSON and XML datasets<br>- The process identifies relevant cases, distinguishes between alive and deceased patients based on defined criteria, ensures deduplication by case ID, and outputs the curated dataset to a standardized JSON datastore<br>- The approach supports longitudinal cancer research workflows within the broader data extraction pipeline.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\data_extractors\0.4_whole_slide_image_extractor.py'>0.4_whole_slide_image_extractor.py</a></b></td>
							<td style='padding: 8px;'>- Extraction and deduplication of whole slide image data are automated by reading input JSON, identifying unique cases, and generating a consolidated, patient-level datastore<br>- Logging and configuration management maintain reproducibility and traceability<br>- This step streamlines and standardizes the preparation of whole slide image references for downstream processes within the data pipeline, ensuring reliable input consistency for broader project workflows.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\data_extractors\0.5.1_mcat_genes_signatures.py'>0.5.1_mcat_genes_signatures.py</a></b></td>
							<td style='padding: 8px;'>- Gene signature extraction and transformation is orchestrated here to standardize and deduplicate gene expression data for downstream analysis<br>- Within the broader data pipeline, the module interfaces with established configuration and logging systems to process raw gene CSV files as defined in YAML configurations, ensuring that transformed, unique gene signature datasets are reliably generated and stored for further bioinformatics workflows.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\data_extractors\0.5.2_mcat_methylation_signatures.py'>0.5.2_mcat_methylation_signatures.py</a></b></td>
							<td style='padding: 8px;'>- Extraction and organization of methylation site signatures from diverse genomic datasets form the core function here, bridging configuration, data loading, and transformation<br>- It ensures that gene-to-CpG mappings, as well as functional group associations, are compiled and stored in standardized CSV outputs<br>- This empowers downstream modules across the codebase with consistent, ready-to-use methylation signature datasets for analysis and modeling.</td>
						</tr>
					</table>
				</blockquote>
			</details>
			<!-- deep_prog Submodule -->
			<details>
				<summary><b>deep_prog</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ src.deep_prog</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\10.1_simple_deepprog_model.py'>10.1_simple_deepprog_model.py</a></b></td>
							<td style='padding: 8px;'>- Demonstrates the end-to-end workflow of training and evaluating a deep learning-based classification model using example data defined in the projects configuration<br>- Serves as a practical entry point for executing data loading, model fitting, prediction, and results saving within the DeepProg framework, showcasing how to utilize the core functionalities of the broader codebase for both development and testing purposes.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\10.2_ensemble_deepprog_model.py'>10.2_ensemble_deepprog_model.py</a></b></td>
							<td style='padding: 8px;'>- Orchestrates the end-to-end training, evaluation, and testing workflow for a stacked ensemble model within the DeepProg framework, leveraging distributed computing to process multi-omic survival data<br>- Coordinates model fitting, performance evaluation, feature importance analysis, and result visualization, providing automated handling of both training and test datasets to robustly assess and report predictive and biological significance across omics data types.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\10.3_hcc_ensemble_deepprog_model.py'>10.3_hcc_ensemble_deepprog_model.py</a></b></td>
							<td style='padding: 8px;'>- Orchestrates the full pipeline for training, evaluating, and testing an ensemble deep learning model on multi-omics cancer datasets within the DeepProg framework<br>- Enables distributed training, performance benchmarking, feature importance analysis, and visualization for survival prediction tasks<br>- Serves as a comprehensive workflow driver, integrating data preparation, model fitting, internal validation, and final predictions to support reproducible research and robust model assessment.</td>
						</tr>
					</table>
					<!-- simdeep Submodule -->
					<details>
						<summary><b>simdeep</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.deep_prog.simdeep</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\simdeep\config.py'>config.py</a></b></td>
									<td style='padding: 8px;'>- Provides centralized configuration for the SimDeep module, specifying key parameters for clustering, survival analysis, feature selection, normalization, data paths, and machine learning workflows<br>- Establishes default values and structures for model training, evaluation, and file management, ensuring orchestrated, reproducible experiments across multi-omics survival analysis pipelines and downstream modules in the overall project architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\simdeep\coxph_from_r.py'>coxph_from_r.py</a></b></td>
									<td style='padding: 8px;'>- This file serves as a compatibility and integration layer that enables the project to leverage both Python and R-based survival analysis tools<br>- Specifically, it focuses on facilitating the use of the Cox Proportional Hazards (CoxPH) model and related survival analysis functionality—drawing from Python libraries (like lifelines) and, when available, R packages accessed via rpy2<br>- This dual approach ensures robustness and flexibility in performing survival analysis throughout the codebase, supporting advanced statistical modeling workflows required by the project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\simdeep\deepmodel_base.py'>deepmodel_base.py</a></b></td>
									<td style='padding: 8px;'>- Provides the foundational interface and workflow for managing deep learning autoencoders within the SimDeep framework, handling dataset loading, model construction, training, and encoder management<br>- Serves as the central component for encoding multi-omics data, facilitating streamlined integration with data preprocessing, model persistence, and embedding extraction, thereby supporting the broader architecture’s modular and extensible deep learning capabilities for biological data analysis.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\simdeep\extract_data.py'>extract_data.py</a></b></td>
									<td style='padding: 8px;'>- Summary of <code>src/deep_prog/simdeep/extract_data.py</code>**This file serves as a critical component of the data preprocessing pipeline within the project<br>- Its primary role is to prepare, normalize, and filter multi-omic datasets—bringing together various input sources and ensuring the data is clean, consistent, and ready for downstream survival analysis or model training<br>- It orchestrates the loading of data, applies configurable normalization techniques, and manages the alignment and integration of sample and feature sets across different modalities<br>- By centralizing these data extraction and transformation steps, the module helps maintain reproducibility and compatibility throughout the projects machine learning workflows.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\simdeep\plot_utils.py'>plot_utils.py</a></b></td>
									<td style='padding: 8px;'>- Visualization utilities enable interactive, color-encoded plots of clustered sample data, primarily for analyzing and interpreting dimensionality-reduced biological or clinical datasets<br>- Enhances interpretability through overlaying kernel density estimates and embedding sample-specific metadata in HTML tooltips, producing visualizations that can be directly embedded in web-based reports<br>- Supports intuitive cluster differentiation and facilitates in-depth exploration of survival and classification outcomes within the broader analysis pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\simdeep\simdeep_analysis.py'>simdeep_analysis.py</a></b></td>
									<td style='padding: 8px;'>- This file defines the primary DeepProg class responsible for orchestrating a single instance of the projects core modeling workflow<br>- As a central interface in the codebase, it brings together configuration, clustering, survival analysis, and model management components to analyze high-dimensional multi-omics data<br>- Serving as the entry point for a full DeepProg analysis run, it integrates various modules to perform data-driven patient stratification and predictive modeling, aligning with the project's overarching goal of robust and automated survival outcome prediction.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\simdeep\simdeep_boosting.py'>simdeep_boosting.py</a></b></td>
									<td style='padding: 8px;'>- This file serves as a central orchestrator for the project’s machine learning boosting workflow<br>- It integrates essential components responsible for data loading, preprocessing, survival analysis, clustering, and evaluation<br>- By coordinating configuration settings and resource management, the file enables robust ensemble modeling and rigorous validation across multiple data splits<br>- Within the overall codebase, this module is key for running comprehensive experiments that leverage boosting strategies to improve predictive performance on complex biomedical datasets.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\simdeep\simdeep_distributed.py'>simdeep_distributed.py</a></b></td>
									<td style='padding: 8px;'>- Enables distributed execution of SimDeeps core analysis workflow by leveraging Ray for parallelism across computational resources<br>- Integrates configuration parameters and extends SimDeep functionality to support scalable, multi-node processing of omics data analysis tasks<br>- Facilitates efficient, large-scale model training and evaluation, fitting seamlessly within the projects broader architecture for robust and high-throughput data-driven biological research.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\simdeep\simdeep_multiple_dataset.py'>simdeep_multiple_dataset.py</a></b></td>
									<td style='padding: 8px;'>- Extends core analysis functionality to support scenarios involving multiple datasets, enabling unified processing and integration within the broader SimDeep framework<br>- Serves as an architectural bridge for handling complex multi-dataset workflows, ensuring that advanced analysis techniques can be consistently applied across diverse data sources<br>- Contributes to modularity and scalability by building on foundational capabilities established in the main SimDeep component.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\simdeep\simdeep_tuning.py'>simdeep_tuning.py</a></b></td>
									<td style='padding: 8px;'>- Orchestrates hyperparameter optimization for SimDeep models by integrating automated search, experiment management, and model evaluation across datasets<br>- Enables robust tuning workflows by coordinating training, model assessment, and result tracking, ensuring optimal parameter selection<br>- Plays a central role in automating the search for high-performing SimDeep model configurations within the overall system architecture, enhancing reproducibility and scalability of experiments.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\simdeep\simdeep_utils.py'>simdeep_utils.py</a></b></td>
									<td style='padding: 8px;'>- Utility functions in simdeep_utils.py facilitate core model management in the SimDeep project by handling the saving and loading of model states, enforcing configuration consistency, and validating metadata and feature selection options<br>- Additionally, label file parsing ensures accurate data ingestion for downstream tasks<br>- These capabilities collectively support reproducibility, maintainability, and reliable execution within the broader SimDeep deep-learning pipeline architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\simdeep\survival_model_utils.py'>survival_model_utils.py</a></b></td>
									<td style='padding: 8px;'>- Enables survival analysis modeling and clustering within the broader DeepProg framework, specifically facilitating the extraction of significant prognostic features and stratification of patient samples according to predicted survival risk<br>- By integrating both Python-based and R-backed Cox proportional hazards models, it provides flexible prediction, probability estimation, and feature selection capabilities essential for downstream survival prediction and patient subgroup identification workflows across multi-omic datasets.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\simdeep\survival_utils.py'>survival_utils.py</a></b></td>
									<td style='padding: 8px;'>- Provides essential utilities for survival analysis and data preprocessing across the project, including loading and validating survival datasets, converting metadata, feature normalization, dimensionality reduction, and statistical evaluation<br>- Enables seamless integration, transformation, and quality assurance of input data for downstream survival modeling, clustering, and feature selection—supporting robust, reproducible computational biology workflows within the larger codebase architecture.</td>
								</tr>
							</table>
						</blockquote>
					</details>
					<!-- results Submodule -->
					<details>
						<summary><b>results</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.deep_prog.results</b></code>
							<!-- DONE_20_test_KIRCKIRP_stacked Submodule -->
							<details>
								<summary><b>DONE_20_test_KIRCKIRP_stacked</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ src.deep_prog.results.DONE_20_test_KIRCKIRP_stacked</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\results\DONE_20_test_KIRCKIRP_stacked\test_KIRCKIRP_stacked_test_RNA_only_supervised_test_kdeplot.html'>test_KIRCKIRP_stacked_test_RNA_only_supervised_test_kdeplot.html</a></b></td>
											<td style='padding: 8px;'>- Summary**The file <code>test_KIRCKIRP_stacked_test_RNA_only_supervised_test_kdeplot.html</code> provides a visual summary of the test results for a specific machine learning experiment within the project<br>- It presents the output for the RNA only" supervised model evaluated on the KIRCKIRP dataset, using a stacked approach<br>- This HTML file serves as a report, allowing users to easily review and interpret the performance of the model through graphical and tabular data (such as KDE plots and metrics)<br>- As part of the <code>results</code> directory, it helps stakeholders understand and compare experimental outcomes without requiring direct code interaction or technical expertise.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\results\DONE_20_test_KIRCKIRP_stacked\test_KIRCKIRP_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html'>test_KIRCKIRP_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html</a></b></td>
											<td style='padding: 8px;'>- Summary**This file serves as an automatically generated visualization artifact that presents the results of a specific supervised analysis (RNA-only modality) within the project’s deep learning pipeline for multi-omics data integration<br>- Used primarily for review and interpretation, the HTML report provides a user-friendly, graphical summary—specifically using KDE plots—of predictive model outcomes pertaining to the KIRCKIRP dataset<br>- By making these insights accessible in a browsable format, this report complements the core analytical workflows, supporting result validation and communication across the broader project architecture.</td>
										</tr>
									</table>
								</blockquote>
							</details>
							<!-- DONE_25_9_test_KIRCKIRP_stacked Submodule -->
							<details>
								<summary><b>DONE_25_9_test_KIRCKIRP_stacked</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ src.deep_prog.results.DONE_25_9_test_KIRCKIRP_stacked</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\results\DONE_25_9_test_KIRCKIRP_stacked\test_KIRCKIRP_stacked_test_RNA_only_supervised_test_kdeplot.html'>test_KIRCKIRP_stacked_test_RNA_only_supervised_test_kdeplot.html</a></b></td>
											<td style='padding: 8px;'>- Summary**The file test_KIRCKIRP_stacked_test_RNA_only_supervised_test_kdeplot.html is an automatically generated HTML report that provides a visual summary of machine learning analysis results for the KIRCKIRP dataset using RNA data under a supervised learning approach<br>- It is part of the results output within the project<br>- This report serves as a key artifact for users and stakeholders to review and interpret the outcomes of the model evaluation, offering accessible insights into performance and data distributions<br>- Within the overall codebase architecture, such result files facilitate transparency, reproducibility, and decision-making by presenting complex computational findings in a comprehensible format.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\results\DONE_25_9_test_KIRCKIRP_stacked\test_KIRCKIRP_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html'>test_KIRCKIRP_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html</a></b></td>
											<td style='padding: 8px;'>- Summary**The file <code>test_KIRCKIRP_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html</code> provides a visual summary—specifically, a KDE (Kernel Density Estimate) plot—of the model’s predictions or data distributions from a supervised analysis focused on RNA data for the KIRCKIRP dataset<br>- As part of the <code>results</code> directory within the project, this HTML file enables users and stakeholders to quickly assess, interpret, and share model performance or feature distributions after a given experiment<br>- It serves as a reporting asset that supports the overall project goal of transparent and reproducible analysis in deep prognostic modeling workflows.</td>
										</tr>
									</table>
								</blockquote>
							</details>
							<!-- DONE_GBMLGG_stacked Submodule -->
							<details>
								<summary><b>DONE_GBMLGG_stacked</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ src.deep_prog.results.DONE_GBMLGG_stacked</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\results\DONE_GBMLGG_stacked\test_GBMLGG_stacked_test_RNA_only_supervised_test_kdeplot.html'>test_GBMLGG_stacked_test_RNA_only_supervised_test_kdeplot.html</a></b></td>
											<td style='padding: 8px;'>- Summary**This file serves as an automatically generated HTML report visualizing the results of a supervised machine learning model trained on RNA data for the GBMLGG dataset within the project<br>- It provides interactive plots and data tables—specifically a KDE (Kernel Density Estimation) plot—that help users interpret the models performance and distribution of predictions<br>- As part of the broader codebase, this result file enables stakeholders and researchers to quickly assess experimental outcomes, facilitating model evaluation and comparison in the deep_prog framework.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\results\DONE_GBMLGG_stacked\test_GBMLGG_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html'>test_GBMLGG_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html</a></b></td>
											<td style='padding: 8px;'>- The file <code>test_GBMLGG_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html</code>, located within the <code>src/deep_prog/results/DONE_GBMLGG_stacked/</code> directory, serves as a visual results artifact for the project<br>- Its main purpose is to present a kernel density estimation (KDE) plot that summarizes the performance or distribution of test results for the GBMLGG_stacked" model variant, specifically using RNA-only supervised data<br>- This HTML report provides an accessible, styled summary for users and stakeholders to quickly assess outcomes relevant to this experiment, aligning with the codebase's broader goal of delivering interpretable and shareable insights from machine learning model evaluations.</td>
										</tr>
									</table>
								</blockquote>
							</details>
							<!-- DONE_KIRCKIRP_stacked Submodule -->
							<details>
								<summary><b>DONE_KIRCKIRP_stacked</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ src.deep_prog.results.DONE_KIRCKIRP_stacked</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\results\DONE_KIRCKIRP_stacked\test_KIRCKIRP_stacked_test_RNA_only_supervised_test_kdeplot.html'>test_KIRCKIRP_stacked_test_RNA_only_supervised_test_kdeplot.html</a></b></td>
											<td style='padding: 8px;'>- Summary**This file is an HTML report generated as part of the projects results output<br>- It visually presents kernel density estimation (KDE) plots and tabular data summarizing the performance of a supervised machine learning model trained and tested on RNA-only data for the KIRCKIRP dataset, using a stacked ensemble approach<br>- Located within the results directory for completed experiments, this report serves as a convenient, shareable artifact, allowing users and stakeholders to easily review and interpret the model's outcomes<br>- It plays a vital role in communicating experiment findings within the overall workflow of the codebase, bridging the gap between technical output and actionable insights.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\results\DONE_KIRCKIRP_stacked\test_KIRCKIRP_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html'>test_KIRCKIRP_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html</a></b></td>
											<td style='padding: 8px;'>- Summary**This file provides a visual summary of model test results generated by the DeepProg framework for the KIRCKIRP dataset using only RNA data in a supervised learning scenario<br>- Integrated within the results reporting architecture, this HTML file allows users to interactively review and interpret the performance metrics, likely through visualizations such as KDE plots and statistical tables<br>- It serves as a user-friendly, sharable report that supports model validation and comparison, making it easier to communicate outcomes to both technical and non-technical audiences within the broader DeepProg analysis pipeline.</td>
										</tr>
									</table>
								</blockquote>
							</details>
							<!-- DONE_LUADLUST_stacked Submodule -->
							<details>
								<summary><b>DONE_LUADLUST_stacked</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ src.deep_prog.results.DONE_LUADLUST_stacked</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\results\DONE_LUADLUST_stacked\test_LUADLUST_stacked_test_RNA_only_supervised_test_kdeplot.html'>test_LUADLUST_stacked_test_RNA_only_supervised_test_kdeplot.html</a></b></td>
											<td style='padding: 8px;'>- This file provides a stylized HTML visualization of RNA-based supervised test results for the LUADLUST dataset, generated as part of the projects stacked analysis pipeline<br>- Its main purpose is to present key output metrics or distributions in a readable, accessible format to facilitate interpretation and reporting of model performance<br>- As an output artifact in the results directory, it serves as a final, human-friendly summary for stakeholders to review the effectiveness of the RNA-only supervised analysis within the broader deep_prog workflow.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\results\DONE_LUADLUST_stacked\test_LUADLUST_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html'>test_LUADLUST_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html</a></b></td>
											<td style='padding: 8px;'>- Summary**This file is a generated HTML report that visually presents the results of a supervised machine learning analysis performed on RNA-only test data for the LUADLUST dataset using the projects stacked modeling pipeline<br>- It provides stakeholders with an accessible summary of model performance and outcomes—serving as a deliverable for interpreting and communicating key findings from the computational workflow, without requiring direct interaction with code or raw data<br>- Within the broader codebase, this report acts as an end-point artifact, supporting result validation, review, and collaboration among users and researchers.</td>
										</tr>
									</table>
								</blockquote>
							</details>
							<!-- DONE_test_BRCA_stacked Submodule -->
							<details>
								<summary><b>DONE_test_BRCA_stacked</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ src.deep_prog.results.DONE_test_BRCA_stacked</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\results\DONE_test_BRCA_stacked\test_BRCA_stacked_test_RNA_only_supervised_test_kdeplot.html'>test_BRCA_stacked_test_RNA_only_supervised_test_kdeplot.html</a></b></td>
											<td style='padding: 8px;'>- Summary**The <code>test_BRCA_stacked_test_RNA_only_supervised_test_kdeplot.html</code> file provides a visual representation of the test results for a supervised learning model applied to RNA-only data from the BRCA dataset<br>- Within the broader codebase, this HTML file serves as an output artifact—enabling users to intuitively examine model performance, trends, and distributions via interactive plots<br>- Its main role is to facilitate result interpretation and model assessment, supporting both technical and non-technical stakeholders in understanding the outcomes generated by the analytic pipeline.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\results\DONE_test_BRCA_stacked\test_BRCA_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html'>test_BRCA_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html</a></b></td>
											<td style='padding: 8px;'>- Summary**This file is a generated HTML report that presents the results of a supervised RNA-only analysis for the BRCA dataset within the DeepProg project<br>- It provides a visual summary—likely showcasing statistical or classification outcomes such as distribution plots—from one of the model evaluation steps<br>- Its primary role in the codebase is to offer an easily accessible, human-readable overview of the models performance, supporting data interpretation and validation for researchers and users working with the DeepProg pipeline<br>- This file is part of the output artifacts that help users assess and communicate the results of their analyses.</td>
										</tr>
									</table>
								</blockquote>
							</details>
							<!-- test_dummy_stacked Submodule -->
							<details>
								<summary><b>test_dummy_stacked</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ src.deep_prog.results.test_dummy_stacked</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\results\test_dummy_stacked\test_dummy_stacked_TEST_DATA_1_supervised_test_kdeplot.html'>test_dummy_stacked_TEST_DATA_1_supervised_test_kdeplot.html</a></b></td>
											<td style='padding: 8px;'>- Summary**The file <code>test_dummy_stacked_TEST_DATA_1_supervised_test_kdeplot.html</code> serves as a visual results artifact within the project’s reporting infrastructure<br>- Specifically, it provides an HTML-based density plot visualization (KDE plot) of supervised testing results from a dummy stacked model on the first test dataset<br>- This output enables users and stakeholders to intuitively assess and interpret model performance as part of the overall experimental workflow, supporting transparent evaluation and streamlined analysis within the projects results subsystem.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\results\test_dummy_stacked\test_dummy_stacked_TEST_DATA_1_TEST_DATA_1_supervised_kdeplot.html'>test_dummy_stacked_TEST_DATA_1_TEST_DATA_1_supervised_kdeplot.html</a></b></td>
											<td style='padding: 8px;'>- Summary**The file <code>test_dummy_stacked_TEST_DATA_1_TEST_DATA_1_supervised_kdeplot.html</code> serves as a visual output within the project, providing an interactive report for reviewing model results<br>- Specifically, it presents a kernel density estimation (KDE) plot that visualizes the distribution of predictions or features for the “dummy stacked” model when applied to the test data<br>- This HTML report enables users to interpret and assess the performance or patterns of the supervised learning pipeline through direct, data-driven visual feedback, supporting streamlined result interpretation within the codebases experiment tracking and analysis workflow.</td>
										</tr>
									</table>
								</blockquote>
							</details>
							<!-- test_hcc_stacked Submodule -->
							<details>
								<summary><b>test_hcc_stacked</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ src.deep_prog.results.test_hcc_stacked</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\results\test_hcc_stacked\test_hcc_stacked_test_RNA_only_supervised_test_kdeplot.html'>test_hcc_stacked_test_RNA_only_supervised_test_kdeplot.html</a></b></td>
											<td style='padding: 8px;'>- The file <code>test_hcc_stacked_test_RNA_only_supervised_test_kdeplot.html</code> serves as a results artifact, providing a visually formatted HTML report—likely a kernel density estimation plot—that showcases the performance or distribution summary of the RNA only supervised" model evaluated on the test_hcc_stacked dataset<br>- Nested within the <code>results</code> directory, this artifact enables users and stakeholders of the codebase to easily review and interpret model outcomes, supporting overall project objectives of transparent, reproducible, and accessible analysis reporting in the context of deep learning-based prognostic modeling.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\deep_prog\results\test_hcc_stacked\test_hcc_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html'>test_hcc_stacked_test_RNA_only_test_RNA_only_supervised_kdeplot.html</a></b></td>
											<td style='padding: 8px;'>- Summary**This file serves as a visual output report within the codebase, specifically providing a styled HTML representation of results generated from the projects analysis pipeline<br>- Situated in the results directory for a test case involving HCC (hepatocellular carcinoma) and RNA-only data, it presents supervised analysis findings—likely in the form of tables or plots (such as KDE plots)—to facilitate interpretation and communication of model performance or data insights<br>- The file plays a critical role in making complex computational results accessible, supporting users and stakeholders in evaluating the effectiveness and relevance of the project's analytical workflows.</td>
										</tr>
									</table>
								</blockquote>
							</details>
						</blockquote>
					</details>
				</blockquote>
			</details>
			<!-- mcat Submodule -->
			<details>
				<summary><b>mcat</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ src.mcat</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\6_mcat.py'>6_mcat.py</a></b></td>
							<td style='padding: 8px;'>- Orchestrates the setup, configuration, training, validation, and testing processes for the Multimodal Co-Attention Transformer (MCAT) within the broader pipeline, integrating experiment logging, data management, model initialization, and device assignment<br>- Acts as the primary workflow entry point that manages key project components and experiment lifecycle, facilitating reproducible experimentation and robust performance monitoring for multimodal survival analysis tasks.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\6_mcat.sbatch'>6_mcat.sbatch</a></b></td>
							<td style='padding: 8px;'>- Orchestrates and manages the scheduled execution of the MCAT computational workflow within a high-performance computing environment<br>- Ensures proper resource allocation, environment setup, and logging, thereby facilitating reproducible and efficient runs of MCAT analyses<br>- Plays a critical role in automating large-scale bioinformatics experiments within the AIforBioinformatics project by integrating seamlessly with SLURM workload management for scalable and reliable performance.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\7.1_mcat_gene_expression_and_methylation_early_stopping.py'>7.1_mcat_gene_expression_and_methylation_early_stopping.py</a></b></td>
							<td style='padding: 8px;'>- Orchestrates the end-to-end training and evaluation of the Multimodal Co-Attention Transformer for integrating gene expression and methylation data in survival analysis tasks<br>- Manages data preparation, model configuration, training with early stopping, logging with Weights & Biases, checkpointing of best models, and final validation, serving as the main entry point for automated and reproducible multimodal experiments within the project architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\7.1_mcat_gene_expression_and_methylation_early_stopping.sbatch'>7.1_mcat_gene_expression_and_methylation_early_stopping.sbatch</a></b></td>
							<td style='padding: 8px;'>- Orchestrates the execution of a gene expression and methylation analysis workflow on a high-performance computing cluster, ensuring appropriate resource allocation, environment setup, and integration with the broader bioinformatics project<br>- Enables automated and reproducible runs of the main analysis script, supporting efficient experimentation and early stopping strategies within the context of multi-omics data processing as part of the overall MCAT pipeline.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\7.2_mcat_gene_expression_and_methylation_cross_validation.py'>7.2_mcat_gene_expression_and_methylation_cross_validation.py</a></b></td>
							<td style='padding: 8px;'>- The file <code>7.2_mcat_gene_expression_and_methylation_cross_validation.py</code> serves as the main orchestrator for running cross-validation experiments on multimodal biomedical data, specifically integrating gene expression and DNA methylation profiles<br>- It leverages the projects core modules for model training, validation, and testing—coordinating data handling, experiment setup, logging, and configuration management<br>- Positioned within the overall architecture, this script provides a reproducible and automated workflow for evaluating the Multimodal Co-Attention Transformer (MCAT) model's performance in survival prediction tasks, forming a critical component for model benchmarking and experimental reproducibility within the project.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\7.2_mcat_gene_expression_and_methylation_cross_validation.sbatch'>7.2_mcat_gene_expression_and_methylation_cross_validation.sbatch</a></b></td>
							<td style='padding: 8px;'>- Automates the execution of gene expression and methylation cross-validation experiments within the broader MCAT pipeline<br>- Integrates environment setup, resource allocation, and Python script execution on a high-performance computing cluster, ensuring consistency and scalability across experiment runs<br>- Supports reproducible research by standardizing the way cross-validation tasks are launched in the context of the projects analysis workflows.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\8.1_smt_gene_expression_cross_validation.py'>8.1_smt_gene_expression_cross_validation.py</a></b></td>
							<td style='padding: 8px;'>- Code Purpose SummaryThe <code>src/mcat/8.1_smt_gene_expression_cross_validation.py</code> script orchestrates the cross-validation workflow for evaluating gene expression models within the projects architecture<br>- Its primary role is to integrate configuration, dataset loading, model instantiation, and logging functionalities to systematically assess the performance and robustness of single-modality transformer models on gene expression data<br>- By coordinating data preparation, reproducibility settings, and training/validation routines, this script enables reliable benchmarking and model selection as part of the broader machine learning pipeline for gene expression analysis.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\8.1_smt_gene_expression_cross_validation.sbatch'>8.1_smt_gene_expression_cross_validation.sbatch</a></b></td>
							<td style='padding: 8px;'>- Orchestrates the execution of gene expression cross-validation experiments using GPU resources within a high-performance computing environment<br>- Integrates job scheduling, environment setup, and resource allocation to automate running the main Python workflow for cross-validation on RNA-Seq data, supporting robust and reproducible results as part of the broader AI-driven bioinformatics analysis framework<br>- Enables efficient utilization of computational resources and streamlined experiment management within the project.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\8.2_smt_methylation_cross_validation.py'>8.2_smt_methylation_cross_validation.py</a></b></td>
							<td style='padding: 8px;'>- Summary of <code>src\mcat\8.2_smt_methylation_cross_validation.py</code>**This file orchestrates the cross-validation process for methylation data using a Single Modal Transformer (SMT) model within the broader MCAT project<br>- It coordinates the setup and execution of training and validation routines, leveraging custom loss functions and dataset handling modules<br>- Serving as a central driver script, it ensures deterministic experiment reproducibility and standardized logging, thereby supporting robust evaluation and benchmarking of the SMT model on methylation datasets as part of the projects modular machine learning pipeline.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\8.2_smt_methylation_cross_validation.sbatch'>8.2_smt_methylation_cross_validation.sbatch</a></b></td>
							<td style='padding: 8px;'>- Automates the submission and execution of a cross-validation experiment for SMT CpG methylation analysis within a high-performance computing environment<br>- Integrates environment setup, resource allocation, and logging, ensuring the underlying Python workflow runs efficiently on compatible GPUs<br>- Supports scalable, reproducible machine learning experiments as part of the broader AIforBioinformatics project’s methylation data analysis pipeline.</td>
						</tr>
					</table>
					<!-- gene_expression_and_methylation_modules Submodule -->
					<details>
						<summary><b>gene_expression_and_methylation_modules</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.mcat.gene_expression_and_methylation_modules</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\gene_expression_and_methylation_modules\dataset.py'>dataset.py</a></b></td>
									<td style='padding: 8px;'>- Summary of <code>src/mcat/gene_expression_and_methylation_modules/dataset.py</code>**This file defines the main dataset management logic for the project’s gene expression and DNA methylation analysis modules<br>- Serving as a central interface for loading, organizing, and pre-processing multi-modal biological data, it enables seamless integration of gene expression and methylation datasets for further training and evaluation tasks<br>- Within the context of the overall codebase, this module underpins the data handling infrastructure, ensuring that downstream modeling and analytical components receive well-structured and consistent input data, thereby facilitating reproducible and scalable experiments.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\gene_expression_and_methylation_modules\mcat.py'>mcat.py</a></b></td>
									<td style='padding: 8px;'>- Defines the core multimodal neural network model responsible for integrating gene expression and DNA methylation data within the overall architecture<br>- Implements specialized encoders, co-attention mechanisms, and fusion strategies to jointly process both omics modalities, enabling nuanced survival prediction outputs with interpretable attention maps<br>- Serves as the foundational component orchestrating the multi-pathway feature extraction and prediction within the gene expression and methylation modules of the system.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\gene_expression_and_methylation_modules\testing.py'>testing.py</a></b></td>
									<td style='padding: 8px;'>- Coordinates the evaluation phase of the gene expression and methylation analysis pipeline by running inference on test data, tracking key survival and model output metrics, and optionally saving visualization and attention matrices for interpretability<br>- Integrates model outputs with patient and experiment metadata, enabling performance assessment and qualitative insights, thereby supporting the overall validation and interpretability objectives within the projects modular architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\gene_expression_and_methylation_modules\training.py'>training.py</a></b></td>
									<td style='padding: 8px;'>- Coordinates the training process for gene expression and methylation survival models by managing data flow, loss calculation, regularization, and optimization steps across training epochs<br>- Tracks key performance metrics—such as loss and concordance index—while supporting logging and hyperparameter tuning<br>- Supports integration with experiment tracking tools and underpins the model refinement workflow within the context of the project’s molecular data analysis pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\gene_expression_and_methylation_modules\validation.py'>validation.py</a></b></td>
									<td style='padding: 8px;'>- Validates the machine learning models performance on gene expression and methylation data by evaluating loss and survival prediction accuracy during training and testing<br>- Integrates with the wider architecture to provide feedback on model generalization, log key metrics, and support experiment tracking<br>- Serves as a critical checkpoint for monitoring model quality in the context of survival analysis using molecular datasets.</td>
								</tr>
							</table>
						</blockquote>
					</details>
					<!-- original_modules Submodule -->
					<details>
						<summary><b>original_modules</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.mcat.original_modules</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\original_modules\blocks.py'>blocks.py</a></b></td>
									<td style='padding: 8px;'>- Implements the core attention and gating mechanisms that underpin modular fusion and context modeling, enabling advanced multi-modal data integration within the architecture<br>- Provides flexible attention blocks, including gated and contextual variations, to facilitate nuanced feature interaction between heterogeneous data sources<br>- Serves as the foundational layer for intelligent information exchange, supporting the projects broader objective of robust representation learning across multiple data modalities.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\original_modules\dataset.py'>dataset.py</a></b></td>
									<td style='padding: 8px;'>- Multimodal data integration and management serve as the foundation for the dataset component of the project, enabling seamless loading, preprocessing, and access to both omics data and image-derived features for survival analysis<br>- By standardizing data handling, filtering, and partitioning, this module ensures consistent input preparation across training and evaluation workflows, supporting reproducibility and flexibility in model development and experimentation.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\original_modules\fusion.py'>fusion.py</a></b></td>
									<td style='padding: 8px;'>- Fusion strategies for multi-modal or multi-feature inputs are provided, enabling flexible integration of different data streams within the models architecture<br>- By supporting concatenation, gated concatenation, and bilinear fusion approaches, these modules facilitate the combination of learned representations, enhancing the expressive power and adaptability of downstream tasks throughout the codebases modular deep learning pipelines.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\original_modules\loss.py'>loss.py</a></b></td>
									<td style='padding: 8px;'>- Defines a suite of specialized loss functions tailored for survival analysis tasks within the project, supporting various approaches including cross-entropy, negative log-likelihood, Cox proportional hazards, and Tobit loss formulations<br>- Provides core capabilities for optimizing deep learning models with respect to censored data, a central requirement for accurate survival modeling throughout the codebase’s machine learning pipelines.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\original_modules\mcat.py'>mcat.py</a></b></td>
									<td style='padding: 8px;'>- Defines the central multimodal transformer model that integrates whole slide imaging (WSI) data and multiple omics data sources through co-attention, transformer encoders, and configurable fusion strategies<br>- Enables joint representation learning and survival prediction, serving as the primary architecture for multimodal data analysis in the project<br>- Supports configurable model sizes and fusion types, ensuring adaptability for diverse biomedical applications within the codebase.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\original_modules\testing.py'>testing.py</a></b></td>
									<td style='padding: 8px;'>- Facilitates the evaluation of trained models on test datasets by executing inference, logging key analytics, and optionally persisting attention mechanisms for interpretability<br>- Integrates with the broader architecture to validate model performance across patient data using survival analysis metrics, supporting robust model assessment and reproducibility within the MCAT pipeline for multi-omics and image-based cancer outcome prediction.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\original_modules\training.py'>training.py</a></b></td>
									<td style='padding: 8px;'>- Centralizes the core model training loop, orchestrating forward and backward passes, loss computation, metric evaluation, dynamic learning rate scheduling, and periodic checkpointing for survival prediction tasks<br>- Enables efficient tracking of training progress and performance by integrating logging and model persistence, forming a foundational component that directly interacts with both data loaders and model architecture, supporting robust experimentation and reproducibility across the pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\original_modules\utils.py'>utils.py</a></b></td>
									<td style='padding: 8px;'>- Provides essential utility functions that support data handling and model regularization within the larger multi-omics analysis framework<br>- Facilitates extraction of omics data dimensions from HDF5 datasets, retrieves RNA-seq and CNV sizes, offers L1 and L2 regularization for model optimization, and initializes neural network weights<br>- Serves as a foundational component, enabling robust preprocessing and model setup across the codebase.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\original_modules\validation.py'>validation.py</a></b></td>
									<td style='padding: 8px;'>- Validation function orchestrates model evaluation on the validation dataset by computing performance metrics and tracking key statistics, including loss and concordance index, across epochs<br>- Integrates with logging systems for experiment tracking and supports flexible loss and regularization schemes<br>- Serves as a critical checkpoint within the training loop, providing quantitative feedback to guide model tuning and monitor generalization throughout the entire training process.</td>
								</tr>
							</table>
						</blockquote>
					</details>
					<!-- single_modality_modules Submodule -->
					<details>
						<summary><b>single_modality_modules</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.mcat.single_modality_modules</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\single_modality_modules\dataset_gene_expression.py'>dataset_gene_expression.py</a></b></td>
									<td style='padding: 8px;'>- Provides a specialized dataset interface for gene expression data, enabling structured loading, preprocessing, stratification, and batching for machine learning workflows within the broader project<br>- Facilitates the transformation of raw RNA-seq and clinical data into appropriately partitioned and standardized datasets, supporting both basic and signature-based analyses, and integrates seamlessly with the pipeline’s modular data handling and model training components.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\single_modality_modules\dataset_methylation.py'>dataset_methylation.py</a></b></td>
									<td style='padding: 8px;'>- Methylation data ingestion and management for the overall platform, enabling efficient loading, preprocessing, signature extraction, and stratified dataset splitting for machine learning workflows<br>- Serves as a core module connecting raw methylation datasets to training and evaluation routines, with integrated support for normalization, signature-based feature selection, and k-fold cross-validation to facilitate reproducible and scalable model development.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\single_modality_modules\smt.py'>smt.py</a></b></td>
									<td style='padding: 8px;'>- Implements the SingleModalTransformer, a neural network module dedicated to processing and classifying single-modality omics data for survival analysis in the broader MCAT framework<br>- Facilitates feature encoding, transformer-based set modeling, and global attention pooling to generate predictions and attention scores, serving as a core building block for handling single-source data within the architecture’s multi-omics analysis pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\single_modality_modules\training.py'>training.py</a></b></td>
									<td style='padding: 8px;'>- Orchestrates the training phase for single-modality models within the MCAT framework, managing forward and backward passes, loss and regularization calculations, and key training metrics<br>- Captures survival analysis-specific metrics such as concordance index, integrates logging with Weights & Biases, and supports flexible configuration for loss functions and optimizers, ensuring robust, configurable model optimization as part of the broader training workflow.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\mcat\single_modality_modules\validation.py'>validation.py</a></b></td>
									<td style='padding: 8px;'>- Validation logic orchestrates the assessment of the models performance during each epoch by computing loss and concordance index metrics on a validation dataset<br>- It integrates seamlessly with the training workflow, supporting multiple loss configurations and optional regularization, while logging relevant evaluation statistics<br>- This step ensures reliable monitoring and tuning of single-modality modules in the projects overall survival analysis pipeline.</td>
								</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
			<!-- methway_os Submodule -->
			<details>
				<summary><b>surv_path</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ src.surv_path</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\surv_path\9.1_methwayos_gene_expression_and_methylation_cross_validation.py'>9.1_methwayos_gene_expression_and_methylation_cross_validation.py</a></b></td>
							<td style='padding: 8px;'>- File Summary**This file orchestrates the cross-validation workflow for evaluating survival path models that integrate gene expression and DNA methylation data within the project<br>- Positioned as a primary entry point for model assessment, it configures experimental settings, manages data loading, and invokes modular routines for training, validation, and testing<br>- By coordinating these processes, the file enables robust, reproducible benchmarking of multimodal survival analysis approaches, supporting the broader codebase’s goal of advancing predictive modeling in bioinformatics.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\surv_path\9.1_methwayos_gene_expression_and_methylation_cross_validation.sbatch'>9.1_methwayos_gene_expression_and_methylation_cross_validation.sbatch</a></b></td>
							<td style='padding: 8px;'>- Orchestrates the cross-validation workflow for gene expression and methylation survival analysis by scheduling and configuring a GPU-enabled job in a high-performance computing environment<br>- Ensures the proper computational resources, project paths, and machine learning environment are set up, integrating seamlessly within the broader bioinformatics pipeline to facilitate reproducible and scalable model evaluation tasks.</td>
						</tr>
					</table>
					<!-- gene_expression_and_methylation_modules Submodule -->
					<details>
						<summary><b>gene_expression_and_methylation_modules</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.surv_path.gene_expression_and_methylation_modules</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\surv_path\gene_expression_and_methylation_modules\dataset.py'>dataset.py</a></b></td>
									<td style='padding: 8px;'>- Summary of <code>src/surv_path/gene_expression_and_methylation_modules/dataset.py</code>**This file serves as a foundational component of the project’s data handling pipeline, focusing on the integration and preparation of gene expression and methylation datasets<br>- It defines data structures and loading mechanisms that enable efficient access, preprocessing, and partitioning of multimodal biological data<br>- By standardizing how these large-scale datasets are loaded, validated, and made available for downstream processes, this module ensures consistency and reliability across the project’s workflow—from initial data ingestion to model training and evaluation<br>- Its central role supports the broader system architecture by enabling flexible, configurable, and reproducible experiments with complex biomedical data.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\surv_path\gene_expression_and_methylation_modules\surv_path.py'>surv_path.py</a></b></td>
									<td style='padding: 8px;'>- Implements the core deep learning module for MethWayOS, enabling joint modeling of gene expression and methylation data for survival prediction<br>- Serves as the central point for integrating and transforming multi-omics features via attention and feedforward layers, ultimately producing survival-related probabilities and interpretable attention scores for downstream tasks in the broader multi-omics survival analysis pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\surv_path\gene_expression_and_methylation_modules\testing.py'>testing.py</a></b></td>
									<td style='padding: 8px;'>- Enables evaluation of model performance on gene expression and DNA methylation data for survival analysis by orchestrating model inference, risk prediction, and attention score extraction across test batches<br>- Supports generating detailed visualizations and saving attention matrices for interpretability and further analysis, playing a crucial role in model validation and interpretability workflows within the projects modular data pipeline.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\surv_path\gene_expression_and_methylation_modules\training.py'>training.py</a></b></td>
									<td style='padding: 8px;'>- Coordinates the supervised training workflow for gene expression and methylation-based survival models, managing batch-wise data processing, dynamic loss and regularization selection, gradient updates, and progress logging<br>- Supports critical evaluation metrics and experiment tracking, ensuring the model learns relevant survival patterns from multi-omic features within the broader gene expression and methylation module, integral to the projects survival analysis architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\surv_path\gene_expression_and_methylation_modules\validation.py'>validation.py</a></b></td>
									<td style='padding: 8px;'>- Provides the validation routine for evaluating the performance of gene expression and methylation survival models during training and testing phases<br>- Tracks relevant metrics such as validation loss and concordance index, enabling effective model selection and monitoring<br>- Integrates with experiment tracking tools and fits seamlessly within the broader pipeline for assessing model generalization on unseen biomedical data.</td>
								</tr>
							</table>
						</blockquote>
					</details>
					<!-- original_modules Submodule -->
					<details>
						<summary><b>original_modules</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.surv_path.original_modules</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\surv_path\original_modules\cross_attention.py'>cross_attention.py</a></b></td>
									<td style='padding: 8px;'>- Enables advanced attention mechanisms for integrating and modeling relationships between pathway and histology data within the architecture<br>- Facilitates both cross-attention between modalities and self-attention among pathways, supporting nuanced feature interactions crucial for downstream predictive tasks<br>- Plays a central role in enriching contextual representations, aiding the model in capturing complex multi-modal dependencies required for effective survival analysis or biomedical predictive modeling.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\surv_path\original_modules\loss.py'>loss.py</a></b></td>
									<td style='padding: 8px;'>- Defines the core loss functions used for training and evaluating survival analysis models across the project, supporting various survival learning objectives and handling censored data<br>- Enables flexible model optimization by providing specialized losses for tasks like Cox regression, cross-entropy survival, and attention regularization<br>- Acts as a foundation for ensuring accurate risk estimation and event prediction within the projects broader machine learning architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\surv_path\original_modules\surv_path.py'>surv_path.py</a></b></td>
									<td style='padding: 8px;'>- Summary**The <code>src/surv_path/original_modules/surv_path.py</code> file defines fundamental neural network building blocks—specifically, a self-normalizing multilayer reception block and related utilities—that support the construction of attention-based models within the codebase<br>- This module is pivotal in enabling robust deep learning architectures for the larger project, providing reusable components that facilitate advanced feature extraction and integration, particularly for tasks involving cross-attention and survival analysis<br>- By encapsulating these standard components, the file helps ensure consistency and modularity across the overall neural network architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='C:\Users\giova\Downloads\Altro\MethWayOS/blob/master/src\surv_path\original_modules\utils.py'>utils.py</a></b></td>
									<td style='padding: 8px;'>- Utility functions in this module enable extraction of feature sizes from multi-omics and genomics datasets, and provide essential helpers for neural network training such as L1/L2 regularization and weight initialization<br>- Serving as foundational support within the project, these utilities streamline data handling and model preparation, promoting consistency and reproducibility during the development and evaluation of survival analysis models leveraging omics data.</td>
								</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
		</blockquote>
	</details>
</details>

---

## ✨ Credits

- [MCAT](https://github.com/mahmoodlab/MCAT)
- [SurvPath](https://github.com/mahmoodlab/SurvPath)
