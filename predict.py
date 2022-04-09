import os
import numpy as np
from PIL import Image
import argparse
import paddle
import paddle.nn.functional as F
from paddle.vision import transforms
from paddle.vision import datasets as dsets

import warnings
warnings.filterwarnings('ignore')

from models import GreedyHash

ckp_list = {
    12: "output/bit_12.pdparams",
    24: "output/bit_24.pdparams",
    32: "output/bit_32.pdparams",
    48: "output/bit_48.pdparams"
}

cifar10_id2name = {
    0: "飞机 airplane",
    1: "汽车 automobile",
    2: "鸟类 bird",
    3: "猫 cat",
    4: "鹿 deer",
    5: "狗 dog",
    6: "蛙类 frog",
    7: "马 horse",
    8: "船 ship",
    9: "卡车 truck"
}

def get_arguments():
    parser = argparse.ArgumentParser(description='GreedyHash')
    # normal settings
    parser.add_argument('--model', type=str, default="GreedyHash")

    # data settings
    parser.add_argument('--n_class', type=int, default=10)
    parser.add_argument('--crop_size', type=int, default=224)

    # Some configs you should change:
    parser.add_argument('--bit', type=int, default=48,
                help="choose the model of certain bit type", choices=[12, 24, 32, 48])
    parser.add_argument('--pic_id', type=int, default=1949,
                help="choose one picture from Cifar10")
    arguments = parser.parse_args()
    return arguments

def get_pic(idx=np.random.randint(0, 500), save_path="resources"):
    cifar10 = dsets.Cifar10(mode='test', transform=None)
    image, label = cifar10.data[idx]
    image = np.reshape(image, [3, 32, 32])
    image = image.transpose([1, 2, 0])
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    save_path = os.path.join(save_path, "cifar10_{}.jpg".format(idx))
    if not os.path.isfile(save_path):
        print(f"----- Image to predict has been saved in {save_path}")
        image.save(save_path)
    return image, label

@paddle.no_grad()
def main(config):
    # define model
    model = GreedyHash(config.bit, config.n_class)
    model.eval()

    # load weights
    ckp = ckp_list[config.bit]
    assert os.path.isfile(ckp), "{} doesn't exist!".format(ckp)
    model_state = paddle.load(ckp)
    model.set_dict(model_state)
    print(f"----- Pretrained: Load model state from {ckp}")

    # define transforms
    eval_transforms = transforms.Compose([
        transforms.Resize(config.crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image, label_id = get_pic(idx=config.pic_id, save_path="resources")
    img = eval_transforms(image)
    img = img.expand([1] + img.shape)

    output, _, _ = model(img)
    output = F.softmax(output).numpy()[0]
    class_id = output.argmax()
    prob = output[class_id]
    return class_id, prob, label_id

if __name__ == "__main__":
    config = get_arguments()
    class_id, prob, label_id = main(config)
    class_name = cifar10_id2name[class_id]
    real_name = cifar10_id2name[label_id]
    print(f"----- Predicted Class_ID: {class_id}, Prob: {prob}, Real Label_ID: {label_id}")
    print(f"----- Predicted Class_NAME: {class_name}, Real Class_NAME: {real_name}")
