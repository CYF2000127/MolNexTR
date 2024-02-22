# MolNexTR
This is the offical code of following paper "MolNexTR: A Generalized Deep Learning Model for Molecular Image Recognition".

## Highlights
<p align="justify">
In this work, We propose MolNexTR, a novel graph generation model. The model follows the encoder-decoder architecture, takes three-channel molecular images as input, outputs molecular graph structure prediction, and can be easily converted to SMILES. We aim to enhance the robustness and generalization of the molecular structure recognition model by enhancing the feature extraction ability of the model and the augmentation strategy, to deal with any molecular images that may appear in the real literature.

[comment]: <> ()
![visualization](figure/arch.png)
<div align="center">
Overview of our MolNexTR model.
</div> 

## Using the code
Please clone the following repositories:
```
git clone https://github.com/CYF2000127/MolNexTR
```

| Model/Modification       | Hard Match Precision | Hard Match Recall | Hard Match F1 | Soft Match Precision | Soft Match Recall | Soft Match F1 |
|--------------------------|----------------------|-------------------|---------------|----------------------|-------------------|---------------|
| ReactionDataExtractor    | 4.1                  | 1.3               | 1.9           | 19.4                 | 5.9               | 9.0           |
| OChemR                   | 4.4                  | 2.8               | 3.4           | 12.4                 | 7.9               | 9.6           |
| RxnScribe                | 72.3                 | 66.2              | 69.1          | 83.8                 | 76.5              | 80.0          |
| - No pretraining         | 66.4                 | 59.4              | 62.7          | 80.4                 | 71.3              | 75.5          |
| - No compositional augmentation | 67.1         | 60.7              | 63.8          | 78.2                 | 70.2              | 74.0          |
| - Random reaction order  | 72.0                 | 64.2              | 67.9          | 83.9                 | 74.3              | 78.8          |
| - No postprocessing      | 70.8                 | 66.0              | 68.3          | 82.1                 | 76.4              | 79.1          |
| Ours                     | 59.24                | 60.32             | 59.78         | 74.30                | 74.96             | 74.63         |
| new                      | 60.82                | 61.32             | 61.07         | 74.68                | 75.35             | 75.01         |


## Experiments

### Requirement
```
pip install -r requirements.txt
```

### Data preparation
For training and inference, please download the following datasets to your own path.
#### Training datasets
1. **Synthetic:**  [PubChem](https://www.dropbox.com/s/mxvm5i8139y5cvk/pubchem.zip?dl=0)
2. **Realistic:**  [USPTO](https://www.dropbox.com/s/3podz99nuwagudy/uspto_mol.zip?dl=0)

#### Testing datasets
1. **Synthetic:**  [Indigo, ChemDraw](https://huggingface.co/datasets/CYF200127/MolNexTR/blob/main/synthetic.zip)
2. **Realistic:**  [CLEF, UOB, USPTO, Staker, ACS](https://huggingface.co/datasets/CYF200127/MolNexTR/blob/main/real.zip) 
3. **Perturbed by img transform:** [CLEF, UOB, USPTO, Staker, ACS](https://huggingface.co/datasets/CYF200127/MolNexTR/blob/main/perturb_by_imgtransform.zip)
4. **Perturbed by curved arrows:** [CLEF, UOB, USPTO, Staker, ACS](https://huggingface.co/datasets/CYF200127/MolNexTR/blob/main/perturb_by_arrows.zip)


### Training
Run the following command:
```
sh ./exps/train.sh
```
The default batch size was set to 256. And it takes about 20 hours to train with 10 NVIDIA RTX 3090 GPUs. 

### Inference
Run the following command:
```
sh ./exps/eval.sh
```
The default batch size was set to 32 with a single NVIDIA RTX 3090 GPU.

### Visualization
Use [`prediction.ipynb`](prediction.ipynb) for single or batched prediction and visualization.

We also show some qualitative results of our method below:

![visualization](figure/vs1.png)
<div align="center">
Qualitative results of our method on ACS.

![visualization](figure/vs1.png)
Qualitative results of our method on some hand-drawn molecules images.
</div> 

## Acknowledgment 
This code is based on [MolScribe](https://github.com/thomas0809/MolScribe), thanks for their excellent work!
