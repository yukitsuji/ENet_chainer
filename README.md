# ENet_chainer
Implementation of ENet by chainer  
ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation [[link](https://arxiv.org/pdf/1606.02147.pdf)]

```
# Test using caffemodel
## Download ENet caffemodel and convert it to chainer's weight format
## by using https://github.com/TimoSaemann/ENet/tree/master/enet_weights_zoo
cd converter && ./enet_weight_download.sh
# TODO: python caffe_to_chainer.py experiments/paper_enc_dec.yml
# TODO: python test.py ./experiments/test_caffemodel.yml

######## Training by cityscapes ########
# Calculate class balancing
python calculate_class_weight.py [mean or loss] --base_dir data_dir --result name --source ./pretrained_model/data.txt --num_classes 19 --dataset [cityscapes or camvid]

# Training encoder by cityscapes
・Single GPU
python train.py experiments/enc_paper.yml
・Multi GPUs
python train.py experiments/enc_paper.multi.yml

# Training decoder by cityscapes
python train.py experiments/enc_dec_paper.yml
```

# Implementation
- Spatial Dropout using cupy
- Baseline, model architecture
- Evaluate by citydataset
- Calculate class weights for training model
- Poly leraning rate policy

# Requirement
- Python3
- Chainer3
- Cupy
- Chainercv
- OpenCV

# TODO
- Visualize output of cityscapes
- Convert caffemodel to chainer's model format
- Create merge function between convolution and batch normalization
