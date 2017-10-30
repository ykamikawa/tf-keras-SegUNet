# SegUNet(UNet + SegNet)
This model is a new model combining the architecture of Unet and Segnet with keras.

**Aarchitecture**

This architecture is encoder-decoder model(29 conv2D layers).
- Skip connection(UNet) and indeces pooling(SegNet) are incorporated to propagate the spatial information of the image.


<div align="center">
<img src=https://user-images.githubusercontent.com/27678705/32180433-4ca5d2be-bdd5-11e7-83d1-0459131076d1.png title="archiecture" width="200px">
</div>


**DEMO**

## train

`python SegUNet.py [--options](Please refer to this code)`
