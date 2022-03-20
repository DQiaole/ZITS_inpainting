# Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding

Codes and trained models for our CVPR2022 paper ZITS_inpainting will be released here.

## Overview

## News


## Preparation


## Training

python TSR_train.py --nodes 1 --gpus 1 --GPU_ids '0' --AMP

python TSR_train.py --nodes 1 --gpus 1 --GPU_ids '0' --AMP --MaP True

python FTR_train.py --nodes 1 --gpus 1 --GPU_ids '2' --path ./ckpt/lama --config_file ./config_list/config_LAMA.yml --lama
python FTR_train.py --nodes 1 --gpus 2 --GPU_ids '0,1' --path ./ckpt/places2 --config_file ./config_list/config_LAMA_MPE_places2.yml --DDP
python FTR_train.py --nodes 1 --gpus 2 --GPU_ids '0,1' --path ./ckpt/places2 --config_file ./config_list/config_LAMA_MPE_HR_places2.yml --DDP

## More 1K Results

![](./imgs/supp_highres.jpg)

## Acknowledgments

* Some test images from [LaMa](https://github.com/saic-mdal/lama)

## Cite

If you found our program helpful, please consider citing:

```
@inproceedings{dong2022incremental,
      title={Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding}, 
      author={Qiaole Dong and Chenjie Cao and Yanwei Fu},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2022}
}
```