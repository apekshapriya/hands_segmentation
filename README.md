# Hands Segmentation for cooking activities

This code base is cloned from this repo https://github.com/guglielmocamporese/hands-segmentation-pytorch and is adapted to train on EPICK-KITCHENS-100 dataset.



### Updates for this repo:


1. Added data-preprocessor for conversion of EPIC-KITCHEN format to HAND_SEGMENTATION format
2. Added a dataloader for EPICK-KITCHENS dataset 
3. Added training config for finetuning the pretrained checkpoint


## Installation:

### clone the repo
```python
$ git clone https://github.com/apekshapriya/hands_segmentation.git 

$ cd HANDS_SEGMENTATION 
```

### Create a new environment through venv and activate it

``` python
$ python3 -m venv venv_hs
$ source venv/bin/activate
``` 

### Further, all the dependencies are listed in requirements.txt and can be installed by:

``` python
$ python3 -m pip install pip setuptools wheel

$ python3 -m pip install -e .
```

## Dataset:
 
The pretrained model taken from the referred repo was trained on following dataset:

    EgoYouTubeHands
    EgoHands
    HandOverFace
    GTEA

The new dataset having numerous cooking activities images was added to finetune the pretrained checkpoint:
    
    EPIC-KITCHEN-100 



Being the very huge dataset, some samples were randomly chosen from the dataset and the annotations were converted to masked images.

To create the dataset:

    Choose the samples from https://epic-kitchens.github.io/2023 and keep in the following structure:
    
    dataset:
        annotations:
            train
                P01_01.json
                P01_02.json
                P02_01.json
                ...........
                ...........
                P37_01.json

        rgb_images:
            train
                P01
                    P01_01
                        P01_01_frame1.jpg
                        .................
                        .................
                        P01_01_frame10.jpg
                    P01_02
                        P01_02_frame1.jpg
                        .................
                        .................
                        P01_02_frame10.jpg
                P02
                    P02_01
                        P02_01_frame1.jpg
                        .................
                        .................
                        P02_01_frame10.jpg
                    P02_02
                        P02_02_frame1.jpg
                        .................
                        .................
                        P02_02_frame10.jpg
            
            


    and run the below command to create the masks and images.

    cd hand_segmenter
    gen_coco_format.py  --epick_visor_store ../dataset  --num 1  --copy_img  --split train --mode handonly  --unzip_img



### Note: 
Very few samples were taken because of the limited resources and time. If we want to better the model furthermore, the dataset can be used wisely.

## Model

The finetuned model's checkpoint can be downloaded from here:


The model is based on architecture of DeepLabV3 with resnet50 backbone model originaly trained on COCO dataset for seemantic segmentation. This model is chosen after thorough research and comparison of other pretrained models and repositories.

The training logs for the best model is given in the log dir


## To get the Predictions from the model on test images:

#### Update the following args in config.py

    mode="predict"
    data_base_path="../test_samples"
    predictions_path ="../test_output"
    model_checkpoint= "../logs/checkpoints/epoch=0-step=1067.ckpt"

#### Run the following command

```python
$ cd hand_segmenter
$ python main.py 

```

## Finetuning the model

#### Update the following args in config.py

    mode="train"
    data_base_path="../datasets"
    model_checkpoint= "../logs/checkpoints/epoch=0-step=1067.ckpt"
    model_pretrained=True

    Update other hyperparameters if required

#### Run the following command:

```python
$ python main.py 

The config file is given to configure paths and models hyperparameters

The results of the two test folders are given here:

link

The sample output from the best checkpoint are shown below:

![](test_output/sample_video_img_0011.png?raw=true)


![](test_output/sample_video_img_0012.png?raw=true)

![](test_output /sample_video_img_0132.png?raw=true)




### Note:

The model faces problem when objects of similar color shade to that of human hands come into picture.
