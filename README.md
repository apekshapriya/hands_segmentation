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

$ cd hands_segmentation/
```

### Create a new environment through venv and activate it

``` python
$ python3 -m venv venv_hs
$ source venv_hs/bin/activate
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

The new dataset EPIC-KITCHEN-100 having numerous cooking activities images is added to finetune the pretrained checkpoint:
    
 

Being very huge dataset, some samples a rarendomly chosen from the dataset and the annotations are converted to masked images.

To create the dataset:

    Choose and dowload the samples from https://epic-kitchens.github.io/2023 and keep in the following structure:
    
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
            
            


    Now run the below command to create the masks and images.

    $ cd hand_segmenter

```python

$ python convert_visor_to_hands_seg_format.py  --epick_visor_store ../dataset  --num 1  --copy_img  --split train --mode handonly  --unzip_img
    
```
This would create the folder structure as:
    
    data:
        epick_dataset
            img 
                frame1.jpg
                .........
                frame10.jpg
            masks
                frame1.jpg
                ..........
                frame10.jpg


#### Note: 

Very few samples were taken because of the limited resources and time. If we want to better the model furthermore, the dataset can be used wisely.

## Model


The finetuned model's checkpoint can be downloaded from [here](https://drive.google.com/drive/folders/1JOtbVFlDaT3o7zouKz47j-fND0DmQvcz?usp=sharing).


The model is based on architecture of DeepLabV3 with resnet50 backbone originaly trained on COCO dataset for seemantic segmentation. This model is chosen after thorough research and comparison of other pretrained models and repositories.


#### Note
The config file is given to configure paths and models hyperparameters for prediction, training and testing.

## Predictions from the model on test images:

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

The pretrained model is taken from the referred repo and then is fined tuned on EpicKitchen dataset

#### Update the following args in config.py

    mode="train"
    data_base_path="../datasets"
    model_checkpoint= "<any pretrained checkpoint path>"
    model_pretrained=True

    Update other hyperparameters if required

#### Run the following command:

```python
$ python main.py 
```


The sample output from the best checkpoint are shown below:



![sample_output1_vid1](https://github.com/apekshapriya/hands_segmentation/blob/master/test_output/sample_video1_img_0011.png)


![smple_output2_vid1](https://github.com/apekshapriya/hands_segmentation/blob/master/test_output/sample_video1_img_0013.png)

![sample_output3_vid1](https://github.com/apekshapriya/hands_segmentation/blob/master/test_output/sample_video1_img_0132.png)

![sample_output1_vid2](https://github.com/apekshapriya/hands_segmentation/blob/master/test_output/sample_video2_img_0153.png)


![smple_output2_vid2](https://github.com/apekshapriya/hands_segmentation/blob/master/test_output/sample_video2_img_0267.png)

![sample_output3_vid3](https://github.com/apekshapriya/hands_segmentation/blob/master/test_output/sample_video2_img_0352.png)




#### Note:

We can see that model is able to pick up the hand patterns but faces problem when objects of similar color shade to that of human hands come into picture. Also fails to pick-up when a totally different shade to that of human hand, like gloves.


## Experiment Tracking and metrics:

All the training logs and metrics of the best model are logged in tensorboard and can be accessed from [here](https://drive.google.com/drive/folders/1JOtbVFlDaT3o7zouKz47j-fND0DmQvcz?usp=sharing).



## Results for tthe wo test videos

Results of hand-segmentation for two videos are share [here](https://drive.google.com/drive/folders/1V-m2WzqGvPoUXh4KOF_k05fxlT-07oCM?usp=sharing).