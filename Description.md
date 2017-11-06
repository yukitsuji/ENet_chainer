## DeepLab v1, v2
<b>Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs: https://arxiv.org/pdf/1606.00915.pdf</b>

### Problems
- **overly reduced feature resolution** <br/>
Repeated combination of max-pooling and downsampling such as convolution with stride over 2 reduces spatial resolution. This impede small objects prediction.<br/>
This cause the probl
- **multi-scale objects in a image**<br/>
Capture both local and global context is difficult.
- **reduced localization of object boundaries accuracy due to model invariance**

### Points
- **Atrous Convolution**<br/>
Atrous Convolution enlarge the field of view of filters(receptive field) without the loss of feature resolution and increasing the number of parameters or the amount of computation. This can resolve the problem `reduced feature resolution`.
- **Atrous Spatial Pyramid Pooling(ASPP)**
ASPP can extract multi-scale features by using technique inspired by R-CNN spatial pyramid pooling method. ASPP uses multiple parallel atrous convolutional layers with difference sampling rates to the feature map. The feature extracted for each sampling rate are furthrer processed by two 1 by 1 convolutional layers in each branches, and fused to generate the final result. This can resolve the problem `multi-scale objects in a image`.
- **Upsampling by bilinear interpolation**<br/>
This paper employs bilinear interpolation to upsample by a factor of 8 the score map to reach the original image resolution.
Unlike the deconvolution, there is no need to require learning any extra parameters, leading to faster model training. This paper said bilinear interpolation is sufficient in this setting because the class score maps are quite smooth.
- **DenseCRF**<br/>
Traditionally conditional random fields(CRFs) have been employed to smooth noisy segmentation maps. This paper calls this short-range CRFS. But deep based models are quite smooth and the goal of the model is to recover thin-structure. DenseCRF can resolve `reduced localization of object boundaries`. But I think DenseCRF has heavy computational cost.
- **Poly learning policy**<br/>
Please read papers.

## DeepLab v3
<b>Rethinking Atrous Convoluion for Semantic Image Segmentation: https://arxiv.org/pdf/1706.05587.pdf</b>

### This paper propose four categories to handle multi-scale objects
- **Image Pyramid**<br/>
Using some scaled images.
- **Encoder-Decoder structure**
- **Extra modules are cascaded on top of the original network for capturing long range information**<br/>
DenseCRF is employed to encode pixel-level pairwise similarities. While several extra convolutional layers in cascade to gradually capture long range context.
- **Spatial Pyramid Pooling(SPP)**<br/>
SPP probes an incoming feature map with filters or pooling operations at multiple rates, and capturing objects at multiple scales.

### Problems
- **overly reduced feature resolution** <br/>
Repeated combination of max-pooling and downsampling such as convolution with stride over 2 reduces spatial resolution.<br/>
This cause the probl
- **multi-scale objects in a image**<br/>
Capture both local and global context is difficult.

### Points
- **Improved ASPP**<br/>
Include global context calculated by global average pooling to the last feature map.
- **Cascaded Atrous Block**<br/>
Reference: https://arxiv.org/pdf/1702.08502.pdf

## ENet
- **Symmetric Convolution**
- **SqueezeNet based bottleneck module**
- **Early downsampling**
- **Poly learning rate**
- **Atrous Convolution**
- **Upsampling with max pooling's indexes**

## SegNet
- **Symmetry Encoder-Decoder Model**

## PSPNet
### Points
- **Pyramid Pooling Modules**
- **Auxiliary Loss**

## UNet
