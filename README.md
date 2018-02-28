# Seg-UNet(SegNet + UNet)
SegUNet is a model of semantic segmentation based on SegNet and UNet(these model are based on Fully Convolutional Network).
Architecture dedicated to restoring pixel position information.
This architecture is good at fine edge restoration etc.

This repository contains the implementation of learning and testing in keras and tensorflow.

## Architecture
- SegNet

  <div align="center">
    <img src="https://user-images.githubusercontent.com/27678705/33705132-74380e72-db72-11e7-8931-33dfd6d3ff0c.png" alt="SegNet image" width="400">
  </div>

  - indoces pooling

      <div align="center">
        <img src="https://user-images.githubusercontent.com/27678705/33705169-9271acc2-db72-11e7-8ff4-7566e82cd3e7.png" alt="indicespoolingimage">
      </div>

- UNet

  - skip connection

      <div align="center">
        <img src="https://user-images.githubusercontent.com/27678705/33705059-37221a28-db72-11e7-8315-5db47b515440.png" alt="UNet image" width="400">
      </div>

This architecture is encoder-decoder model(29 conv2D layers).
- Skip connection(UNet) and indeces pooling(SegNet) are incorporated to propagate the spatial information of the image.


<div align="center">
<img src=https://user-images.githubusercontent.com/27678705/32180433-4ca5d2be-bdd5-11e7-83d1-0459131076d1.png title="archiecture" width="500px">
</div>

## Usage

### train

- Segmentation involveing multiple categories

  `python train.py --options`

- Segmentation of mask image

  `python train_mask.py --options`

  - options

    - image dir
    - mask image dir
    - batchsize, nb_epochs, epoch_per_steps, input_configs
    - class weights
    - device num

## DEMO
- dataset

  - LIP(Look into person)

      ![demo1](https://user-images.githubusercontent.com/27678705/33703457-8a504fdc-db6b-11e7-8922-db3c61294b18.png)

## Author

**ykamikawa**
