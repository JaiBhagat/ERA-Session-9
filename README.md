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
