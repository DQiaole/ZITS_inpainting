# Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding
by [Qiaole Dong*](https://github.com/DQiaole),
[Chenjie Cao*](https://github.com/ewrfcas),
[Yanwei Fu](http://yanweifu.github.io/)

[Paper and Supplemental Material (arXiv)](https://arxiv.org/abs/2203.00867)

[![LICENSE](https://img.shields.io/github/license/DQiaole/ZITS_inpainting)](https://github.com/DQiaole/ZITS_inpainting/blob/main/LICENSE)

## Pipeline

![](./imgs/overview.jpg)
The overview of our ZITS. At first, the TSR model is used to restore structures with low resolutions. Then the simple CNN based 
upsampler is leveraged to upsample edge and line maps. Moreover, the upsampled sketch space is encoded by the SFE model, and added
to the FTR through ZeroRA to restore the textures. The top left corner show details about the transformer block. The input feature are
learned through row-wise and column-wise attentions respectively, then encoded by a standard attention module.

## TO DO

- [x] Releasing inference codes.
- [x] Releasing pre-trained moodel.
- [ ] Releasing training codes.

## Preparation

1. Preparing the environment:

    as there is some bug when using GP loss with DDP, we recommend installing Apex without CUDA extensions via
    ```
    conda create -n train_env python=3.6
    conda activate train_env
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" ./
    ```
    then complete the environment via
    ```
    pip install -r train_requirement.txt
    ```
2. For training, [MST](https://github.com/ewrfcas/MST_inpainting) provide irregular and segmentation masks ([download](https://drive.google.com/drive/folders/1eU6VaTWGdgCXXWueCXilt6oxHdONgUgf?usp=sharing)) with different masking rates. And you should define the mask file list before the training as in [MST](https://github.com/ewrfcas/MST_inpainting).  
3. Download the pretrained masked wireframe detection model: [LSM-HAWP](https://drive.google.com/drive/folders/1yg4Nc20D34sON0Ni_IOezjJCFHXKGWUW?usp=sharing) ([MST ICCV2021](https://github.com/ewrfcas/MST_inpainting) retrained from [HAWP CVPR2020](https://github.com/cherubicXN/hawp)).
4. Prepare the wireframs:
    
    as the MST train the LSM-HAWP using Pytorch 1.3.1 and it encounter problem when inference in Pytorch 1.9.1, we should
    prepare a new environment for wireframs inference specifically
    ```
    conda create -n wireframs_inference_env python=3.6
    conda activate wireframs_inference_env
    pip install -r wireframs_inference_requirement.txt
    ``` 
   then extracting the wireframs use following code
    ```
    CUDA_VISIBLE_DEVICES=0 python lsm_hawp_inference.py --ckpt_path <best_lsm_hawp.pth> --input_path <input image path> --output_path <output image path>
    ```
5. Download the pretrained models for perceptual loss,
 provided by [LaMa](https://github.com/saic-mdal/lama):
    ```
    mkdir -p ade20k/ade20k-resnet50dilated-ppm_deepsup/
    wget -P ade20k/ade20k-resnet50dilated-ppm_deepsup/ http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth
    ```
   
## Eval
For eval, you only need to complete steps 1, 3 and 4 above.

Download the pretrained models on Places2 [here](https://drive.google.com/drive/folders/1Dg_6ZCAi0U3HzrYgXwr9nSaOLnPsf9n-?usp=sharing) to the './ckpt' fold.
Then modify the config file according to you image, mask and wireframes path.

Test on 256 images:
```
python FTR_inference.py --path ./ckpt/zits_places2 --config_file ./config_list/config_ZITS_places2.yml --GPU_ids '0'
```
Test on 512 images:
```
python FTR_inference.py --path ./ckpt/zits_places2_hr --config_file ./config_list/config_ZITS_HR_places2.yml --GPU_ids '0'
```

## Training

:warning: Warning: The training codes is not fully tested yet after refactoring

#### Training TSR
```
python TSR_train.py --name [exp_name] --data_path [training_data_path] \
 --train_line_path [training_wireframes_path] \
 --mask_path ['irregular_mask_list.txt', 'coco_mask_list.txt'] \
 --train_epoch [epochs] --validation_path [validation_data_path] \
 --val_line_path [validation_wireframes_path] \
 --valid_mask_path [validation_mask] --nodes 1 --gpus 1 --GPU_ids '0' --AMP
```

```
python TSR_train.py --name [exp_name] --data_path [training_data_path] \
 --train_line_path [training_wireframes_path] \
 --mask_path ['irregular_mask_list.txt', 'coco_mask_list.txt'] \
 --train_epoch [epochs] --validation_path [validation_data_path] \
 --val_line_path [validation_wireframes_path] \
 --valid_mask_path [validation_mask] --nodes 1 \
 --gpus 1 --GPU_ids '0' --AMP --MaP
```

#### Training LaMa First

```
python FTR_train.py --nodes 1 --gpus 1 --GPU_ids '0' --path ./ckpt/lama \
--config_file ./config_list/config_LAMA.yml --lama
```

#### Training FTR

256:
```
python FTR_train.py --nodes 1 --gpus 2 --GPU_ids '0,1' --path ./ckpt/places2 \
--config_file ./config_list/config_LAMA_MPE_places2.yml \--DDP
```

256~512:
```
python FTR_train.py --nodes 1 --gpus 2 --GPU_ids '0,1' --path ./ckpt/places2 \
--config_file ./config_list/config_LAMA_MPE_HR_places2.yml --DDP
```
## More 1K Results

![](./imgs/supp_highres.jpg)

## Acknowledgments

* This repo is built upon [MST](https://github.com/ewrfcas/MST_inpainting), [ICT](https://github.com/raywzy/ICT) and [LaMa](https://github.com/saic-mdal/lama).

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