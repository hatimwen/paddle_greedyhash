import os, sys
os.environ['CUDA_VISIBLE_DEVICES']="0"
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torchvision
import logging
import warnings
warnings.filterwarnings('ignore')

from .cal_map import calculate_map, compress
from .dataset import cifar_dataset

num_epochs = 50
batch_size = 32
epoch_lr_decrease = 30
learning_rate = 0.001
encode_length = 32
num_classes = 10
dataset = "cifar10-1"
mode = "train"
log_path = "pytorch_greedyhash/logs"

train_loader, test_loader, database_loader, num_train, num_test, num_dataset = cifar_dataset(batch_size=batch_size, dataset=dataset)

def get_logger(filename, logger_name=None):
    """set logging file and format
    Args:
        filename: str, full path of the logger file to write
        logger_name: str, the logger name, e.g., 'master_logger', 'local_logger'
    Return:
        logger: python logger
    """
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m%d %I:%M:%S %p")
    # different name is needed when creating multiple logger in one process
    logger = logging.getLogger(logger_name)
    fh = logging.FileHandler(os.path.join(filename))
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
    return logger

log_path = '{}/{}-{}'.format(log_path, mode, time.strftime('%Y%m%d-%H-%M-%S'))
if not os.path.exists(log_path):
    os.makedirs(log_path, exist_ok=True)
master_logger = get_logger(
        filename=os.path.join(log_path, 'log.txt'),
        logger_name='master_logger')

# new layer
class hash(Function):
    @staticmethod
    def forward(ctx, input):
        #ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        #input,  = ctx.saved_tensors
        #grad_output = grad_output.data

        return grad_output

def hash_layer(input):
    return hash.apply(input)

class CNN(nn.Module):
    def __init__(self, encode_length, num_classes):
        super(CNN, self).__init__()
        self.alex = torchvision.models.alexnet(pretrained=True)
        self.alex.classifier = nn.Sequential(*list(self.alex.classifier.children())[:6])
        self.fc_plus = nn.Linear(4096, encode_length)
        self.fc = nn.Linear(encode_length, num_classes, bias=False)

    def forward(self, x):
        x = self.alex.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.alex.classifier(x)
        x = self.fc_plus(x)
        code = hash_layer(x)
        output = self.fc(code)

        return output, x, code


cnn = CNN(encode_length=encode_length, num_classes=num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // epoch_lr_decrease))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train_eval():
    Best_mAP = 0.

    # Train the Model
    for epoch in range(num_epochs):
        cnn.cuda().train()
        train_loss = 0
        lr = adjust_learning_rate(optimizer, epoch)
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs, feature, _ = cnn(images)
            loss1 = criterion(outputs, labels)
            loss2 = torch.mean(torch.abs(torch.pow(torch.abs(feature) - Variable(torch.ones(feature.size()).cuda()), 3)))
            loss = loss1 + 0.1 * loss2
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_loader)
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        master_logger.info("{}[{:2d}/{:2d}][{}] bit:{:d}, lr:{:.9f}, dataset:{}, train loss:{:.3f}".format(
                "GreedyHash", epoch + 1, num_epochs, current_time, encode_length, lr, dataset, train_loss))

        # Test the Model
        cnn.eval()  # Change model to 'eval' mode
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = Variable(images.cuda(), volatile=True)
            outputs, _, _ = cnn(images)
            _, predicted = torch.max(outputs.cpu().data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        acc = 1.0 * correct / total

        retrievalB, retrievalL, queryB, queryL = compress(database_loader, test_loader, cnn)

        # print('---calculate map---')
        mAP = calculate_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL)
        # print(result)

        if mAP > Best_mAP:
            Best_mAP = mAP
            torch.save(cnn.state_dict(), 'pytorch_greedyhash/temp.pkl')

        master_logger.info("{} epoch:{}, bit:{}, dataset:{}, MAP:{:.3f}, Best MAP: {:.3f}, Acc: {:.3f}".format(
                "GreedyHash", epoch + 1, encode_length, dataset, mAP, Best_mAP, acc))
