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
	<img src="https://github.com/NasonovIvan/NN-whale-recognition/blob/main/images/train4.png" width="350">
</p>

### Xception:
Xception is a convolutional neural network that is 71 layers deep. I loaded a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories. As a result, the network has learned rich feature representations for a wide range of images, and it is what I used for recognizing and classifying whale voices.

<p align="center">
	<img src="https://github.com/NasonovIvan/NN-whale-recognition/blob/main/images/depthwise.png" width="350">
</p>

### Inception
An Inception-style neural network was created using the Keras framework. The network is designed for 1D data and can be used for tasks such as sequence classification or regression. We use it for training on the Complexity-Entropy series, derived from spectrograms.

The network consists of several convolutional layers followed by batch normalization, ReLU activation, and dropout regularization. The architecture employs the concept of Inception modules to capture features at different scales. Finally, a global average pooling layer is applied to summarize the features, and a fully connected layer with softmax activation is used for classification with `num_classes` output classes.

#### Network Architecture
- **Input Layer:** The input layer takes the input shape as its argument, defining the shape of the input data.

- **Convolutional Layers:** Three consecutive convolutional layers are applied, each with 128 filters and a kernel size of 3. Padding is set to "same" to preserve the spatial dimensions of the input.

- **Batch Normalization:** Batch normalization is applied after each convolutional layer to normalize the activations and improve network performance and training stability.

- **ReLU Activation:** Rectified Linear Unit (ReLU) activation is used after each batch normalization layer to introduce non-linearity into the network.

- **Dropout Regularization:** Dropout regularization is applied after each ReLU activation layer with a dropout rate of 0.2. Dropout randomly sets a fraction of the input units to 0 during training, which helps prevent overfitting.

- **Global Average Pooling:** Global Average Pooling is performed to reduce the spatial dimensions of the feature maps to a vector representation by taking the average of each feature map.

- **Output Layer:** The output layer is a fully connected layer with softmax activation, producing the final probabilities for the classification task. The number of units in this layer corresponds to `num_classes`.

### Results:
Xception accuracy is over 90% on the test set. On the other hand, Inception's accuracy is 83%. The best [score](https://www.kaggle.com/competitions/whale-detection-challenge/leaderboard) in Kaggle competition is 98.384%.
I checked the implementation of this network for recognizing noised data and the result was 89.3% for Xception.

### References:
- [Xception: Deep Learning with Depthwise Separable Convolutions](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf)
- [InceptionTime: Finding AlexNet for time series classification](https://link.springer.com/article/10.1007/s10618-020-00710-y)
- [Characterizing Time Series via Complexity-Entropy Curves](https://arxiv.org/abs/1705.04779)
