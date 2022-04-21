# Implementation of GreedyHash(NeurIPS2018) Based on PaddlePaddle

This is an unofficial repo based on PaddlePaddle of NeurIPS2018:
[Greedy Hash: Towards Fast Optimization for Accurate Hash Coding in CNN](https://papers.nips.cc/paper/7360-greedy-hash-towards-fast-optimization-for-accurate-hash-coding-in-cnn.pdf)

English | [简体中文](./README.md)
   
   * [paddle_greedyhash](#基于-paddlepaddle-实现-greedyhashneurips2018)
      * [1 Introduction](#1-introduction)
      * [2 Accuracy](#2-accuracy)
      * [3 Dataset](#3-dataset)
      * [4 Environment](#4-environment)
      * [5 Quick Start](#5-quick-start)
         * [step1: git and download](#step1-git-and-download)
         * [step2: change arguments](#step2-change-arguments)
         * [step3: eval](#step3-eval)
         * [step4: train](#step4-train)
         * [step5: predict](#step5-predict)
      * [6 TIPC](#6-tipc)
      * [7 Code Structure and Description](#7-code-structure-and-description)
      * [8 Model info](#8-model-info)
      * [9 Citation](#9-citation)

- Paper：[Greedy Hash: Towards Fast Optimization for Accurate Hash Coding in CNN](https://papers.nips.cc/paper/7360-greedy-hash-towards-fast-optimization-for-accurate-hash-coding-in-cnn.pdf).

- Official repo（PyTorch）[GreedyHash](https://github.com/ssppp/GreedyHash).

- Unofficial repo（PyTorch）[DeepHash-pytorch](https://github.com/swuxyj/DeepHash-pytorch).

## 1 Introduction

To tackle the NP hard problem in **Deep Hashing**, GreedyHash adopts the greedy principle by iteratively updating the network toward the probable optimal discrete solution in each iteration. A hash coding layer is designed to implement their approach which strictly uses the sign function in forward propagation to maintain the discrete constraints, while in back propagation the gradients are transmitted intactly to the front layer to avoid the vanishing gradients. In addition to the theoretical derivation, GreedyHash provide a new perspective to visualize and understand the effectiveness and efficiency of this algorithm.

The algorithm has been summarized in:

<p align="center">
<img src="./resources/algorithm.png" alt="drawing" width="90%" height="90%"/>
    <h4 align="center"> Algorithm of GreedyHash</h4>
</p>

## 2 Accuracy

|      | Framework | 12bits | 24bits | 32bits | 48bits|
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
| Paper Results  | PyTorch |  0.774  |  0.795  |  0.810  |  0.822  |
| Reproduce  | PyTorch |  0.789  |  0.799  |  0.813  |**0.824**|
| Reproduce  | PaddlePaddle  |**0.798**|**0.809**|**0.817**|  0.819(**0.824**)  |

- It is worthnoting that owing to the previous PyTorch utilized by official repo [GreedyHash/cifar1.py](https://github.com/ssppp/GreedyHash/blob/master/cifar1.py)  the code to process dataset cannot run.
Therefore, this part of codes are copied from [DeepHash-pytorch](https://github.com/swuxyj/DeepHash-pytorch) so that it is possible to re-run the code based on PyTorch, the results of which are also listed in the table.
The modified codes and logs of training process are placed in [pytorch_greedyhash/main.py](pytorch_greedyhash/main.py) and [pytorch_greedyhash/logs](pytorch_greedyhash/logs), respectively.
Some biases are allowed as I forgot to set the random seed.

- The results of running 12/24/32/48 bits of this project (based on PaddlePaddle) are listed in the table above, and the model parameters and training logs are placed in the [output](output) folder.
Since the random seed is fixed during training, it is theoretically reproducible.
However, after repeating running the same codes several times, I found that the results still fluctuated. For instance, the model of **48bits** had a result with **0.824** once, indicating that the randomness of the algorithm still exists.. The corresponding logs and weights are put in [output/bit48_alone](output/ bit48_alone).

## 3 Dataset
cifar-1, which is called CIFAR-10 (I) in the paper.

- CIFAR-10: 60,000 32×32 color images in 10 classes。

- CIFAR-10 (I): 1000 images (100 images per class) are
selected as the query set, and the remaining 59,000 images are used as database. Besides, 5,000
images (500 images per class) are randomly sampled from the database as the training set. Related codes can be seen in [utils/datasets.py](utils/datasets.py)。

## 4 Environment

My Environment：

- Python: 3.7.11
- [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html): 2.2.2
- Hardware: NVIDIA 2080Ti * 1

    <small>*p.s. CIFAR-10 is so small that one GPU is enough to handle the training.*</small>

## 5 Quick start

### step1: git and download

```
git clone https://github.com/hatimwen/paddle_greedyhash.git
cd paddle_greedyhash
```

- Due to the large size of the trained weights, they are put in BaiduNetdisk ([code: tl1i](https://pan.baidu.com/s/1-90a8HEEHM4zmqk5T6DCrQ)).

- NOTE that you should rearrange them following [6 Code Structure and Description](#6-code-structure-and-description).

### step2: change arguments

Please change the scripts you want to run in [scripts](./scripts/) according to the practical needs.

### step3: eval

- **Note**：Remember to download weights from [BaiduNetdisk](https://pan.baidu.com/s/1-90a8HEEHM4zmqk5T6DCrQ).

```
sh scripts/test.sh
```

### step4: train

```
sh scripts/train.sh
```

### step5: predict

```
python predict.py \
--bit 48 \
--pic_id 1949
```

<p align="center">
<img src="./resources/cifar10_1949.jpg"/>
    <h4 align="center">Picture（class：飞机 airplane， id: 0）</h4>
</p>

Output Results:

```
----- Pretrained: Load model state from output/bit_48.pdparams
----- Predicted Class_ID: 0, Prob: 0.9965014457702637, Real Label_ID: 0
----- Predicted Class_NAME: 飞机 airplane, Real Class_NAME: 飞机 airplane
```

Clearly, the output is is in line with expectations.

## 6 TIPC

- TIPC configs for 12/24/32/48 bits are put separately in [test_tipc/configs](test_tipc/configs/). For convenience, [scripts/tipc.sh](scripts/tipc.sh) is a shell script used to run all these scripts.

- Detailed logs running TIPC are put in [test_tipc/output](test_tipc/output/).

- Please refer to [test_tipc/README.md](test_tipc/README.md) for the specific introduction of TIPC.
## 7 Code Structure and Description

```
|-- paddle_greedyhash
    |-- deploy
        |-- inference_python
            |-- infer.py            # TIPC inference
            |-- README.md           # Intro of TIPC inference
    |-- output              # logs and weights
        |-- bit48_alone         # logs and weights of the best bit48(bit48_alone)
            |-- bit_48.pdparams     # model weights for bit48_alone
            |-- log_48.txt          # logs of bit48_alone
        |-- bit_12.pdparams     # model weights for 12bits
        |-- bit_24.pdparams     # model weights for 24bits
        |-- bit_32.pdparams     # model weights for 32bits
        |-- bit_48.pdparams     # model weights for 48bits
        |-- log_eval.txt        # log of evaluation(including bit48_alone)
        |-- log_train.txt       # log of training 12/24/32/48 bits, except for bit48_alone
    |-- models
        |-- __init__.py
        |-- alexnet.py      # definition of AlexNet
        |-- greedyhash.py   # definition of GreedyHash
    |-- test_tipc               # TIPC
    |-- utils
        |-- datasets.py         # dataset, dataloader, transforms
        |-- lr_scheduler.py     # scheduler of learning rate
        |-- tools.py            # mAP, acc, set random seed
    |-- eval.py             # code of evaluation
    |-- export_model.py     # code of transferring dynamic models to static ones
    |-- predict.py          # demo of predicting
    |-- train.py            # code of training
    |-- README.md
    |-- pytorch_greedyhash
        |-- datasets.py         # PyTorch definition of dataset, dataloader, transforms
        |-- cal_map.py          # PyTorch mAP
        |-- main.py             # PyTorch codes of both training and test
        |-- output              # PyTorch logs
```

## 8 Model info

| Info | Description |
| --- | --- |
| Author | Hatimwen |
| Email | hatimwen@163.com |
| Date | 2022.04 |
| Version | PaddlePaddle 2.2.2 |
| Field | Deep Hashing |
| Supported Devices | GPU、CPU |
| Download | [Modes(code: tl1i)](https://pan.baidu.com/s/1-90a8HEEHM4zmqk5T6DCrQ)  |
| AI Studio | [AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/1945560)|
| License | [Apache 2.0 license](LICENCE)|

## 9 Citation

```
@article{su2018greedy,
  title={Greedy hash: Towards fast optimization for accurate hash coding in cnn},
  author={Su, Shupeng and Zhang, Chao and Han, Kai and Tian, Yonghong},
  year={2018},
  journal={Advances in Neural Information Processing Systems},
  volume={31},
  year={2018}
}
```

- [PaddlePaddle](https://github.com/paddlepaddle/paddle)

Last but not least, thank PaddlePaddle very much for its holding [飞桨论文复现挑战赛（第六期）](https://aistudio.baidu.com/aistudio/competition/detail/205/0/introduction), which helps me learn a lot.
