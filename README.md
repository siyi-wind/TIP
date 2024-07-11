<div align="center">

<h1><a href="https://arxiv.org/abs/2407.07582">TIP: Tabular-Image Pre-training for Multimodal Classification with Incomplete Data (ECCV 2024)</a></h1>

**[Siyi Du](https://scholar.google.com.hk/citations?user=wZ4M4ecAAAAJ&hl=en&oi=ao), [Shaoming Zheng](https://scholar.google.com/citations?user=84zgYXEAAAAJ&hl=en&oi=ao),
[Yinsong Wang](https://orcid.org/0009-0008-7288-4227), [Wenjia Bai](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=IA1QFM4AAAAJ&sortby=pubdate), [Declan P. O'Regan](https://scholar.google.com/citations?user=85u-LbAAAAAJ&hl=en&oi=ao), and [Chen Qin](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=mTWrOqHOqjoC&pagesize=80&sortby=pubdate)** 

![](https://komarev.com/ghpvc/?username=siyi-windTIP&label=visitors)
![GitHub stars](https://badgen.net/github/stars/siyi-wind/TIP)
[![](https://img.shields.io/badge/license-Apache--2.0-blue)](#License)
[![](https://img.shields.io/badge/arXiv-2407.07582-b31b1b.svg)](https://arxiv.org/abs/2407.07582)

</div>

![TIP](./Images/model.jpg)
<p align="center">Model architecture and algorithm of TIP: (a) Model overview with its image encoder, tabular encoder, and multimodal interaction module, which are pre-trained using 3 SSL losses: $\mathcal{L}_{itc}$, $\mathcal{L}_{itm}$, and $\mathcal{L}_{mtr}$. (b) Model details for (b-1) $\mathcal{L}_{itm}$ and $\mathcal{L}_{mtr}$ calculation and (b-2) tabular embedding with missing data. (c) Pre-training algorithm.</p>

This is an official PyTorch implementation for [TIP: Tabular-Image Pre-training for Multimodal Classification with Incomplete Data, ECCV 2024][1]. We built the code based on [paulhager/MMCL-Tabular-Imaging](https://github.com/paulhager/MMCL-Tabular-Imaging). 

Concact: s.du23@imperial.ac.uk (Siyi Du)

Share us a :star: if this repository does help. 

## Updates
[**11/07/2024**] The arXiv paper is released. 

[**08/07/2024**] The code is released.

## Contents
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Training & Testing](#training--testing)
- [Checkpoints](#checkpoints)
- [Lisence & Citation](#lisence--citation)
- [Acknowledgements](#acknowledgements)

## Requirements
This code is implemented using Python 3.9.15, PyTorch 1.11.0, PyTorch-lighting 1.6.4, CUDA 11.3.1, and CuDNN 8.

```sh
conda create -n tip python=3.9
conda activate tip  # activate the environment and install all dependencies
cd TIP/
conda env create --file environment.yaml
```

## Data
Download DVM data from [here][2]

Apply for the UKBB data [here][3]

### Preparation
1. Execute [data/create_dvm_dataset.ipynb](./data/create_dvm_dataset.ipynb) to get train, val, test datasets.
2. Execute [data/image2numpy.ipynb](./data/image2numpy.py) to convert jpg images to numpy format for faster reading during training. 
3. Execute [data/create_missing_mask.ipynb](./data/create_missing_mask.ipynb) to create missing masks (RVM, RFM, MIFM, LIFM) for incomplete data fine-tuning experiments.

## Training

### Pre-training & Fine-tuning
```sh
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_TIP exp_name=pretrain
```

### Fine-tuning
```sh
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_TIP exp_name=finetune pretrain=False evaluate=True checkpoint={YOUR_PRETRAINED_CKPT_PATH}
```

### Fine-tuning with incomplete data
```sh
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_TIP exp_name=missing pretrain=False evaluate=True checkpoint={YOUR_PRETRAINED_CKPT_PATH} missing_tabular=True missing_strategy=value missing_rate=0.3
```

## Checkpoints
### Pre-trained Checkpoints
Datasets | DVM | Cardiac 
--- | :---: | :---: 
Checkpoints | [Download](https://drive.google.com/file/d/1FPUfO-XNwlYb_YklIdi8vOHr5GjpcJvY/view?usp=sharing)| [Download](https://drive.google.com/file/d/1AKUq64WXn3j6-IhoUwarRuZ2PVDgNg_g/view?usp=sharing) 

### Fine-tuned Checkpoints

Task | Linear-probing | Fully fine-tuning 
--- | :---: | :---: 
Car model prediction (DVM) | [Download](https://drive.google.com/drive/folders/1trw5GJ9zUU_pMDyxQ86RMzFq-c3OTsfT?usp=sharing)| [Download](https://drive.google.com/drive/folders/1xvlwANfW3vCCQtOKJEgEKJirnXBJpaQM?usp=sharing) 
CAD classification (Cardiac) | [Download](https://drive.google.com/drive/folders/1ZcNgw3iqbCw6MCRsotQEAkQAaajCkIid?usp=sharing)| [Download](https://drive.google.com/drive/folders/1ZC7f_CsP_ycqxxb0119a_mynoU5tw8Zx?usp=sharing) 
Infarction classification (Cardiac) | [Download](https://drive.google.com/drive/folders/1z-f7rUr2DWkLgQNw9p5k0vnjafHAthLg?usp=sharing)| [Download](https://drive.google.com/drive/folders/1lv94dYWdfXKuCvsxHYEq6Jgv9-JPXmsb?usp=sharing) 

## Lisence & Citation
This repository is licensed under the Apache License, Version 2.

If you use this code in your research, please consider citing:

```text
@inproceedings{du2024tip,
  title={{TIP}: Tabular-Image Pre-training for Multimodal Classification with Incomplete Data},
  author={Du, Siyi and Zheng, Shaoming and Wang, Yinsong and Bai, Wenjia and O'Regan, Declan P. and Qin, Chen},
  booktitle={18th European Conference on Computer Vision (ECCV 2024)},
```

## Acknowledgements
We would like to thank the following repositories for their great works:
* [MMCL](https://github.com/paulhager/MMCL-Tabular-Imaging)
* [BLIP](https://github.com/salesforce/BLIP)


[1]: https://github.com/siyi-wind
[2]: https://deepvisualmarketing.github.io/
[3]: https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access
