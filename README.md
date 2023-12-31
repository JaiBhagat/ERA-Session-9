## Architecture Description

The architecture of the network is specified as C1C2C3C40, where there are no MaxPooling layers but instead, three consecutive convolutional layers are used. The last convolutional layer has a stride of 2, reducing the spatial dimensions of the feature maps.

The requirements for the architecture are as follows:
- Total Receptive Field (RF) must be more than 44.
- One of the layers must use Depthwise Separable Convolution.
- One of the layers must use Dilated Convolution.
- Global Average Pooling (GAP) is compulsory.
- An optional fully connected (FC) layer is added after GAP to target the desired number of classes.

The albumentation library is used for data augmentation, including horizontal flip, shift-scale-rotate, and coarse dropout. The objective is to achieve 85% accuracy with no specific limitations on the number of epochs. The total number of parameters in the model should be less than 200,000.

## Depthwise Separable Convolution

Depthwise Separable Convolution is a technique used to improve efficiency and reduce computational cost in convolutional neural networks. It involves two separate convolutional operations: depthwise convolution and pointwise convolution.

- Depthwise Convolution: Applies a separate filter to each input channel independently, capturing spatial information within each channel.
- Pointwise Convolution: Applies a 1x1 convolution to mix and transform the features obtained from the depthwise convolution, combining the channels into a new set.

By separating the spatial and channel dimensions, depthwise separable convolution reduces parameters and computational complexity, enhancing the network's efficiency.

## Dilated Kernels

Dilated Kernels, or atrous convolution, increase the receptive field of convolutional layers without significantly increasing parameters or computations.

- Gaps are introduced between kernel values, allowing for a larger stride and expanding the receptive field.
- Adjusting the dilation rate controls the level of context captured, from local details to global information.

Dilated convolutions efficiently capture spatial dependencies over large areas, making them valuable for tasks like image segmentation and scene understanding.

## Importance of Depthwise Separable Convolution and Dilated Kernels

- Efficiency: Depthwise separable convolutions reduce parameters and computations, making models more efficient, particularly in resource-constrained environments.
- Increased Receptive Field: Dilated convolutions capture larger spatial context without adding many parameters, useful for understanding global patterns and long-range dependencies.
- Performance: Models utilizing depthwise separable convolutions and dilated kernels achieve comparable or better performance while using fewer resources.

Incorporating depthwise separable convolutions and dilated kernels in the given architecture allows for an efficient and effective model that meets the specified objectives within the desired parameter limit.

## RF Calculation for the Network

To calculate the receptive field (RF) of the model, we need to determine the effective receptive field of each layer and track how it changes through the network.

The receptive field (RF) of a convolutional layer is the area in the input that influences a particular neuron in that layer.

Let's calculate the RF for each layer in the given model:

- Convolution Block 1:
  - depthwise_separable_conv:
    - Kernel Size: 3x3
    - Padding: 1
    - Dilation: 1
    - RF calculation: (previous RF) + (2 * (padding + (dilation * (kernel size - 1)))) = (1 * 1) + (2 * (1 + (1 * (3 - 1)))) = 3x3

  - Convolution layers:
    - Kernel Size: 3x3
    - Padding: 1
    - Dilation: 1
    - RF calculation: (previous RF) + (2 * (padding + (dilation * (kernel size - 1)))) = (3 * 3) + (2 * (1 + (1 * (3 - 1)))) = 3x3

  - Convolution with Stride 2:
    - Kernel Size: 3x3
    - Stride: 2
    - Dilation: 2
    - RF calculation: (previous RF) + (2 * (dilation * (kernel size - 1))) = (3 * 3) + (2 * (2 * (3 - 1))) = 9x9 (increased by stride and dilation)

- Convolution Block 2:
  - depthwise_separable_conv with Dilation 2:
    - Kernel Size: 3x3
    - Padding: 1
    - Dilation: 2
    - RF calculation: (previous RF) + (2 * (padding + (dilation * (kernel size - 1)))) = (9 * 9) + (2 * (1 + (2 * (3 - 1)))) = 21x21 (increased by dilation)

  - Convolution layer with Padding 0:
    - Kernel Size: 3x3
    - Padding: 0
    - RF calculation: (previous RF) + (2 * (padding + (dilation * (kernel size - 1)))) = (21 * 21) + (2 * (0 + (1 * (3 - 1)))) = 29x29 (increased by kernel size)

  - Convolution with Stride 2:
    - Kernel Size: 3x3
    - Stride: 2
    - Dilation: 2
    - RF calculation: (previous RF) + (2 * (dilation * (kernel size - 1))) = (29 * 29) + (2 * (2 * (3 - 1))) = 61x61 (increased by stride and dilation)

- Convolution Block 3:
  - depthwise_separable_conv with Dilation 1:
    - Kernel Size: 3x3
    - Padding: 1
    - Dilation: 1
    - RF calculation: (previous RF) + (2 * (padding + (dilation * (kernel size - 1)))) = (61 * 61) + (2 * (1 + (1 * (3 - 1)))) = 61x61

  - Convolution layer with Padding 0 and Dilation 1:
    - Kernel Size: 3x3
    - Padding: 0
    - Dilation: 1
    - RF calculation: (previous RF) + (2 * (padding + (dilation * (kernel size - 1)))) = (61 * 61) + (2 * (0 + (1 * (3 - 1)))) = 69x69 (increased by kernel size)

  - Convolution with Stride 2 and Padding 1:
    - Kernel Size: 1x1
    - Stride: 2
    - Padding: 1
    - RF calculation: (previous RF) + (2 * padding) + (kernel size - 1) = (69 * 69) + (2 * 1) + (1 - 1) = 133x133 (increased by stride and padding)

- Global Average Pooling (GAP):
  - No change in RF

Therefore, the total RF of the network is 133x133.

Depthwise Separable Convolution:
- Depthwise separable convolution affects the RF by performing separate convolutions on each input channel independently. This preserves the spatial dimensions while capturing information within each channel separately.

Dilated Kernels:
- Dilated convolutions affect the RF by introducing gaps between the kernel values, effectively increasing the receptive field. By adjusting the dilation rate, the model captures larger spatial context without significantly increasing parameters or computations.

Both depthwise separable convolution and dilated kernels enhance the model's ability to capture spatial information and context, allowing for more effective and efficient feature learning.

## Albumentations 
Albumentations is an open-source library for image augmentation in machine learning and computer vision tasks. It provides a wide range of image transformation techniques that can be applied to enhance the training data and improve the model's generalization capabilities.

In our code, Albumentations is used for data augmentation during the training phase. The `train_transforms_albu` object defines a sequence of transformations to be applied to the training images. Let's discuss each transformation:

- Normalize:
  - Applies normalization to the image using specified mean and standard deviation values.
  - Mean: (0.4914, 0.4822, 0.4465)
  - Standard Deviation: (0.2023, 0.1994, 0.2010)

- HorizontalFlip:
  - Flips the image horizontally with a probability of 0.5.
  - This transformation helps the model learn to generalize and handle mirrored or horizontally flipped objects.

- RandomBrightnessContrast:
  - Randomly adjusts the brightness and contrast of the image.
  - Probability: 0.2
  - This variation in brightness and contrast helps the model become more robust to different lighting conditions.

- RandomContrast:
  - Randomly adjusts the contrast of the image.
  - Limit: 0.2
  - This transformation further enhances the contrast of the images.

- ShiftScaleRotate:
  - Randomly applies a combination of translation, scaling, and rotation to the image.
  - Shift Limit: 0.0625 (fraction of the image size)
  - Scale Limit: 0.1 (fraction of the image size)
  - Rotate Limit: 45 degrees
  - Probability: 0.2
  - This transformation simulates variations in the viewpoint or perspective of the objects in the images.

- CoarseDropout:
  - Randomly masks out rectangular regions of the image, creating dropout-like effects.
  - Max Holes: 1
  - Max Height: 16 pixels
  - Fill Value: (0.4914, 0.4822, 0.4465) (using the mean values of the dataset)
  - Mask Fill Value: None
  - This transformation helps the model become more robust to missing or occluded parts of the images.

- ToTensorV2:
  - Converts the image and its corresponding labels to PyTorch tensors.

These transformations collectively introduce various variations and distortions to the training images, helping the model learn robust and generalized representations.

## Results 

EPOCH: 77
Loss=0.7087705731391907 Batch_id=97 Accuracy=79.77: 100%|██████████| 98/98 [00:13<00:00,  7.49it/s]
Test set: Average loss: 0.4410, Accuracy: 8527/10000 (85.27%)




