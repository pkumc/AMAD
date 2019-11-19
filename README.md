# anomaly detection 
The source code for our anomaly detection project. Including our model and seven baseline models.

## How to use the code

### Dataset
[label feature1 feature2 feature3 ...] 

label 1 表示normal data. 0表示为anomaly data.

特别注意： feature 的ID 是要从1 开始。 0是占位符dummy node. 为了Padding 用的,具体表现为当前feature为空。所有的features的ID 统一存放在一个大的ID list里面。normal data 写在dataset 的前面。写完normal data 之后，最后再加上anomaly data.

example:

1 2 5 0

1 3,4,2 6 8

1 1,2 6 7,9

1 1,2 6 10,11

1 1,2 6 7,8,9

0 1,2 0 8,9

0 1,2 5 8,9

Including sythetic data, public data and Alimama data

Numeric data format: used for the case when model can't deal with high dimensional categorical data

example:

1 0.5,0.4,0.2

1 0.2,0.4,0.3

1 0.1,0.3,0.2

1 0.5,0.2,0.2

0 0.1,0.2,0.3

0 0.2,0.4,0.5

### Run the baseline code

There are seven baseline algorithms in folder ./baseline/. There are two basic models (NCF-IsoForest, NCF-OC-SVM), three GAN based model (ALOCC, EfficientGAN, GANomaly), and two deep models (RDA,OCNN). we use ALOCC.py as the example to run the code: 

```
$ python ALOCC.py --input synthetic.txt --instance-output instance.txt --block-output block.txt --batch-size 10 --block-size 10 --hidden-dim 128 --epoch 10 
```

### Parameter tuning introduction

In each baseline, there are some parameters to be tuned. Some of them are common in all baselines, while rest of them are unique in each baseline. For details of each parameter explanation, please use `--help` to discover more.

#### For common parameters: 
* `--input`: the input dataset
* `--instance-output`: the instance score output 
* `--block-output`: the block score output 
* `--batch-size` : the number of instances in a batch to be trained
* `--block-size`: the number of instances in a block
* `--hidden_dim`: hidden layer dimension
* `--epoch`: the number of epochs in the training process
* `--learning-rate`: learning rate during the training process
* `--block-ratio`: when the ratio of instances predicted as anomaly data in a block is reached to this, we label the block as an anomaly block.
* `--threshold-scale`: The average training loss multiplies the threshold scale to form a threshold, which is used to predict whether the testing instance is anomaly or not.
* `--alpha`,`--beta`,`--gamma`: weights on different losses to form the final loss
* `--categorical` and `--no-categorical`: whether the input data is categoriacal data or numeric data.

#### For unique parameters:

##### OCNN:
* `--v`: weight parameter when calculating final loss

##### RDA:
* `--hidden-dim-list`, a list of hidden layer dimensions. e.g. [128,256,128] 


In the dataset (e.g. synthetic.txt), it includes both training and testing data. And the precision, recall, F1-macro, F1-micro score are calculated as evaluation metrics. The result is directly printed out in the terminal.

## Citations

if you use the code, please cite:

## License
The code is released under GNU license. 


## Contributor

* **Zheng Gao** - [gao27@indiana.edu](gao27@indiana.edu) <br />


