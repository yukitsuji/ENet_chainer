# ENet_chainer
Implementation of ENet by chainer

```
# Download ENet caffemodel and convert it to chainer's weight format
# by using https://github.com/TimoSaemann/ENet/tree/master/enet_weights_zoo
python caffe_to_chainer.py

# Training by cityscapes
python train.py experiments/paper_base.yml
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
