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

