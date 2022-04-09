import math
import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn.initializer import Uniform

from .alexnet import alexnet

class Hash(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input):
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class GreedyHash(nn.Layer):
    def __init__(self, hash_bit, num_class=1000, pretrained=True):
        super(GreedyHash, self).__init__()

        model_alexnet = alexnet(pretrained=pretrained)
        self.features = model_alexnet.features
        
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias

        self.classifier_plus = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(),
            nn.Dropout(),
            cl2,
            nn.ReLU(),
            nn.Linear(4096, hash_bit,
                weight_attr=ParamAttr(initializer=self.uniform_init(4096)),),
        )
        self.fc = nn.Linear(hash_bit, num_class,
                            weight_attr=ParamAttr(initializer=self.uniform_init(hash_bit)),
                            bias_attr=False)

    def uniform_init(self, num):
        stdv = 1.0 / math.sqrt(num)
        return Uniform(-stdv, stdv)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape([x.shape[0], 256 * 6 * 6])
        x = self.classifier_plus(x)
        code = Hash.apply(x)
        output = self.fc(code)
        return output, x, code

