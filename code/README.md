# UniGM
Codes of UniGM

## Pretrained Models

| Pretrained Models | Reference |
| ------ | ------ |
| MGSSL	 | https://github.com/zaixizhang/MGSSL/tree/main/motif_based_pretrain/saved_model |
| GraphMVP | https://github.com/chao1224/GraphMVP |
| SimGRACE | https://github.com/junxia97/SimGRACE |
| GraphCL | https://github.com/Shen-Lab/GraphCL/tree/master/transferLearning_MoleculeNet_PPI |

## Requirements

* Python 3.7.4
* PyTorch 1.7.0
* torch_geometric 1.5.0
* tqdm
* einops
* requests

## Quick Start

* Download and prepare the pretrained models in `saved_model/`.
* Download the datasets of downstream tasks from [chem data](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) (2.5GB),unzip it, and put it under `chem/`.
* Finetuning with UniGM
```
python fuse.py --dataset DOWNSTREAM_DATASET --device DEVICE --freeze MODE
```
This will finetune pre-trained models with mode specified in `MODE` using dataset `DOWNSTREAM_DATASET` on device `DEVICE`.

