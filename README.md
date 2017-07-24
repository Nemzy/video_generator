# video_generator
This is implementation of convolutional variational autoencoder in TensorFlow library and it was used for video generation.

Also this is my code for the Siraj Raval's coding challenge "How to Generate Images - Intro to Deep Learning #14".

You can find more about it [here](https://www.youtube.com/watch?v=3-UDwk1U77s&t=2s)

## Overview
For more info about video generation see jupyter notebook and for implementation details of conv-variational autoencoder see **conv_vae.py**.
Dataset was created by using ffmpeg tool.

## Dependencies
* TensorFlow r1.0
* pillow
* matplotlib
* numpy
* sklearn

All dependencies can be installed by pip.

## Dataset creation steps
* Resize video: ffmpeg -i input.mp4 -s 64x64 -c:a copy output.mp4
* Create frames: ffmpeg -i output.mp4 -r NUM-OF-FRAMES-PER-SEC -f image2 SOME/PATH/%05d.png

## Credits
Credits go to these guys:

* Kevin Frans [url](http://kvfrans.com/variational-autoencoders-explained/)

* Arthur Juliani [url](https://medium.com/@awjuliani/introducing-neural-dream-videos-5d517b3cc804)