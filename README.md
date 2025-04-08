# RaVSNet

This is the official code of RaVSNet: Pretraining-based Relevance-aware Visit Similarity Network for Drug Recommendation.

## Code Base Structure

- `data/`
  - `input/` 
    - `drug-atc.csv`, `ndc2atc_level4.csv`, `ndc2rxnorm_mapping.txt`: mapping files for drug code transformation
    - `idx2drug.pkl`: Drug ID (we use ATC-4 level code to represent drug ID) to drug SMILES string dictionary
    - `drug-DDI.csv`: this a large file, containing the drug DDI information, coded by CID. The file could be downloaded from https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing
    - `symptoms_list.pkl`: For extracting symptoms from discharge summary
      
  - `output/`
    - `voc_final.pkl`: diag/prod/med index to code dictionary
    - `ddi_A_final.pkl`: ddi adjacency matrix
    - `ddi_matrix_H.pkl`: H mask structure (This file is created by ddi_mask_H.py)
    - `mask_single_records_final.pkl`:  The records of patients with only one visit
    - `records_final.pkl`: The final EHR records of each patient
Due to policy reasons, we are unable to provide processed data. Users are asked to process it themselves according to the instructions in the next section
      
  - `graphs/`
    - `causal_graph.pkl`: casual graphs extracted from visit records
    - `Diag_Med_causal_effect.pkl`,`Proc_Med_casual_effect.pkl`,`Sym_Med_casual_effect.pkl`: causal effects between diag/proc/sym and med
    
  - `processing.ipynb`: The python script responsible for generating `voc_final.pkl`, `mask_single_records_final.pkl`, `records_final.pkl`, and `ddi_A_final.pkl`   

- `src/` folder contains all the source code.
  - `modules/`: Code for model definition.
  - `util.py`: Code for metric calculations and some data preparation.
  - `training.py`: Code for the functions used in training and evaluation.
  - `main.py`: Train or evaluate our Model.
 



## Step 1: Package Dependency

Please install the environment according to the following version

```bash
python == 3.8.16
torch == 2.0.1
dill == 0.3.6
pandas==1.5.3
numpy==1.23.5
scipy==1.10.0
torch-geometric==2.3.1
cdt==0.6.0
dowhy==0.10.1
statsmodels==0.14.0
```
## Step 2：Data Processing

1.MIMIC-III: Due to the privacy of medical data, we cannot directly provide source data. You must apply for permission at https://physionet.org/content/mimiciii/1.4/ and download the data set after passing the review. And go into the folder and unzip four main files (PROCEDURES_ICD.csv.gz, PRESCRIPTIONS.csv.gz, DIAGNOSES_ICD.csv.gz, NOTEEVENTS.csv.gz) into /data/inputs/.

2.MIMIC-IV: Same as MIMIC-III dataset, you must apply for permission at https://physionet.org/content/mimiciv/2.2/ and download the data set after passing the review. And go into the folder and unzip four main files (procedures_icd.csv.gz, prescriptions.csv.gz, diagnoses_icd.csv.gz, discharge.csv.gz) into /data/inputs/.

3.processing the data to get a complete records.

## Step 3：Run the Code
First, you need to extract the causal diagram from the patient's visit record based on the following code:
```bash
python src/module/causal_construction.py
```
Then, you can train and test the model according to the following code:
```bash
python src/main.py
```

## Citation & Acknowledgement
We are grateful to everyone who contributed to this project.

If the code and the paper are useful for you, it is appreciable to cite our paper.