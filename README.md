# Whale Sound Detection
Transfer Learning Xception was used for recognizing and classifying whale voices using spectrograms and time series of signal complexity.

<p align="center">
	<img src="https://github.com/NasonovIvan/NN-whale-recognition/blob/main/images/marinexplore_kaggle.png" width="350">
</p>

### Description:
In this project I analyze [The Marinexplore and Cornell University Whale Detection Challenge](https://www.kaggle.com/c/whale-detection-challenge), where participants were tasked with developing an algorithm to correctly classify audio clips containing sounds from the North Atlantic right whale.

In my work, I concentrated on the analysis of spectrogram images by the [Xception](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf) neural network.

### Data:
The Kaggle training set includes approximately 30,000 labeled audio files. Each file encodes a two second monophonic audio clip in AIFF format with a 2000 Hz sampling rate. Based on these files, I obtained spectrogram images, as in the example below:
<p align="center">
	<img src="https://github.com/NasonovIvan/NN-whale-recognition/blob/main/datasets/pngs_from_wavs/trian4.png" width="350">
</p>

### Neural Network:
Xception is a convolutional neural network that is 71 layers deep. I loaded a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories. As a result, the network has learned rich feature representations for a wide range of images, and it is what I used for recognizing and classifying whale voices.

<p align="center">
	<img src="https://github.com/NasonovIvan/NN-whale-recognition/blob/main/images/depthwise.png" width="350">
</p>

### Results:
Xception accuracy is 97.33% on the test set. The best [score](https://www.kaggle.com/competitions/whale-detection-challenge/leaderboard) in Kaggle competition is 98.384%.
I checked the implementation of this network for recognizing noised data and the result was 89.3%.

### References:
- [Xception: Deep Learning with Depthwise Separable Convolutions](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf)