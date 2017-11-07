# ENet_chainer
Implementation of ENet by chainer

```
# Download ENet caffemodel and convert it to chainer's weight format
# by using https://github.com/TimoSaemann/ENet/tree/master/enet_weights_zoo
cd converter && ./enet_weight_download.sh
python caffe_to_chainer.py experiments/paper_enc_dec.yml
python create_class_weight.py

# Training by cityscapes
python train.py experiments/paper_enc.yml
```

# Implementation
- Spatial Dropout using cupy
- Baseline, model architecture
- Evaluate by citydataset
- Compare SegNet

# Requirement
- Chainer2
- Cupy
- Chainercv

# TODO
- Class Weights
- Create decoder module
- Visualize output of cityscapes
- Convert caffemodel to chainer's model format
- Create merge function between convolution and batch normalization
