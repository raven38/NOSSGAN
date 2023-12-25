# NOSSGAN
This repository contains the code for the paper Soft Curriculum for Learning Conditional GANs with Noisy-Labeled and Uncurated Unlabeled Data (WACV2024)

[Paper](https://openaccess.thecvf.com/content/WACV2024/html/Katsumata_Soft_Curriculum_for_Learning_Conditional_GANs_With_Noisy-Labeled_and_Uncurated_WACV_2024_paper.html)

This repo is implemented upon the [StudioGAN repo](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN).

The main dependencies are:
- Python 3.9.7 or later
- PyTorch 1.10 or later
- TensorFlow 2.4.1 with GPU support 

## Setup enviroment

```
conda env create a100_env.yaml
conda activate a100
```


### Make ImageNet


Manual download of the ImageNet dataset (for evaluation and training). 
Please follow the instructions https://www.tensorflow.org/datasets/catalog/imagenet2012

Put the training and validation set of the ImageNet dataset on `./code/data/ILSVRC2012/{train|valid}`.

Convert folder dataset to hdf5 dataset
```bash
python3 src/main.py -t -e -l -s -iv -sync_bn -stat_otf -mpc --eval_type valid -c src/configs/ILSVRC2012/BigGAN256.json
python3 src/make_nosssimagenet.py -c src/configs/NOSSILSVRC2012_200_10_05_010/DiffAugGAN256.json --src data/ILSVRC2012 --dst data/NOSSILSVRC2012 --subset_class 200 --ratio 0.05 --noise_rate 0.1 --usage 0.1
```

### Training 

```bash
python3 src/main.py -t -e -s -iv -l -sync_bn -stat_otf -mpc --eval_type valid -c src/configs/NOSSILSVRC2012_200_10_05_010/DiffAugNOSSGAN256_l875-CR.json
```
