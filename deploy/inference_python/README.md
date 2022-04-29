# Inference 推理


# 目录

- [1. 简介](#1-简介)
- [2. 推理过程](#2-推理过程)
    - [2.1 准备推理环境](#21-准备推理环境)
    - [2.2 模型动转静导出](#22-模型动转静导出)
    - [2.3 模型推理](#23-模型推理)


## 1. 简介

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用MKLDNN、CUDNN、TensorRT进行预测加速，从而实现更优的推理性能。

本文档主要基于Paddle Inference的 **greedyhash_48** 模型推理。

更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/inference_cn.html)。


## 2. 推理过程

### 2.1 准备推理环境

安装好PaddlePaddle即可体验Paddle Inference部署能力。


### 2.2 模型动转静导出

按照下面的步骤完成`greedyhash_48`模型的动转静导出。

- 【如果已下载请忽略】从 [BaiduNetdisk 提取码: tl1i](https://pan.baidu.com/s/1-90a8HEEHM4zmqk5T6DCrQ) 中下载 greedyhash_48 模型预训练权重 并放置在 `output/bit_48.pdparams` 路径下。

- 生成推理模型：

    ```bash
    python export_model.py \
    --save-inference-dir="./tipc_output/" \
    --bit=48
    ```

最终在`./tipc_output/`文件夹下会生成下面的3个文件。

```
tipc_output
     |----inference_48.pdiparams     : 模型参数文件
     |----inference_48.pdmodel       : 模型结构文件
     |----inference_48.pdiparams.info: 模型参数信息文件
```

### 2.3 模型推理


```bash
python deploy/inference_python/infer.py \
--model-dir="./tipc_output/" \
--bit=48 \
--img-path="./resources/cifar10_1949.jpg"
```

对于下面的图像进行预测

<p align="center">
<img src="../../resources/cifar10_1949.jpg"/>
    <h4 align="center">验证图片（resources/cifar10_1949.jpg）</h4>
</p>

在终端中输出结果如下。

```
image_name: ./resources/cifar10_1949.jpg, class_id: 0, prob: 0.997
```

表示预测的类别ID是`0`，置信度为`0.997`，该结果与基于训练引擎的结果一致。

