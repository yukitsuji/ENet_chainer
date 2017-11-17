# ENet_chainer
Implementation of ENet by chainer  
ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation [[link](https://arxiv.org/pdf/1606.02147.pdf)]

```
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

######## Evaluate by cityscapes ########
python test.py experiments/test_enc.yml

######## Visualize by cityscapes ########
python demo.py experiments/test_enc.yml --img_path img.png
```
| Implementation | Global accuracy | Class accuracy | mean IoU   |
|:--------------:|:---------------:|:--------------:|:----------:|
| Chainer      | 92.59 %          | **71.49 %**     | **59.1 %** |

# Implementation
- Spatial Dropout using cupy
- Baseline, model architecture
- Evaluate by citydataset
- Visualize output of cityscapes
- Calculate class weights for training model
- Poly leraning rate policy

# Requirement
- Python3
- Chainer3
- Cupy
- Chainercv
- OpenCV

# TODO
- Convert caffemodel to chainer's model format
- Create merge function between convolution and batch normalization
