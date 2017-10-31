# SegUNet(UNet + SegNet)
This model is a new model combining the architecture of Unet and Segnet with keras.

**Aarchitecture**

This architecture is encoder-decoder model(29 conv2D layers).
- Skip connection(UNet) and indeces pooling(SegNet) are incorporated to propagate the spatial information of the image.


<div align="center">
<img src=https://user-images.githubusercontent.com/27678705/32180433-4ca5d2be-bdd5-11e7-83d1-0459131076d1.png title="archiecture" width="200px">
</div>


- original

<div align="center">
<img src=https://user-images.githubusercontent.com/27678705/32210409-ceb03454-be50-11e7-9410-cbca1ed3fc91.png title="archiecture" width="200px">
</div>


- ground truth

<div align="center">
<img src=https://user-images.githubusercontent.com/27678705/32210411-cfdb9d46-be50-11e7-8de1-f0a8d1350e3b.png title="archiecture" width="200px">
</div>


- predict

<div align="center">
<img src=https://user-images.githubusercontent.com/27678705/32210412-d12067d6-be50-11e7-9f86-4fb3d7e4d778.png title="archiecture" width="200px">
</div>


**DEMO**

## train

`python SegUNet.py [--options](Please refer to this code)`
