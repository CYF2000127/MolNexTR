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

### Using the code
Please clone the following repositories:
```
git clone https://github.com/CYF2000127/MolNexTR
```

### Requirement
```
pip install -r requirements.txt
```

### Data preparation
#### Download
1. **Abdominal CT**  [Synapse Multi-atlas Abdominal Segmentation dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
2. **Abdominal MRI**  [Combined Healthy Abdominal Organ Segmentation dataset](https://chaos.grand-challenge.org/)  
3. **Cardiac MRI** [Multi-sequence Cardiac MRI Segmentation dataset (bSSFP fold)](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mscmrseg/) 

#### Pre-processing
Please refer to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation.git)

### Training
1. Download pre-trained [ResNet-101 weights](https://download.pytorch.org/models/resnet101-63fe2227.pth) and put into your own backbone folder.
2. Run the following command for Abdominal CT/MRI:
```
sh ./exps/train_Abd.sh
```
Run the following command for Cardiac MRI:
```
sh ./exps/train_CMR.sh
```

### Inference
Run `./exp/validation.sh`

### Visualization
[comment]: <> ()
![visualization](figures/visual.png)
<div align="center">
Qualitative results of our method on Abd-CT and Abd-MRI.
</div> 

## Acknowledgment 
This code is based on [Q-Net](https://github.com/zjlab-ammi/q-net), [PFENet](https://github.com/dvlab-research/PFENet), thanks for their excellent work!
