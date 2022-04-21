#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import logging
import argparse
import paddle
import paddle.nn as nn

import warnings
warnings.filterwarnings('ignore')

from utils.datasets import get_data, get_dataloader
from utils.tools import set_random_seed, compress, calculate_map, calculate_acc
from utils.lr_scheduler import DecreaseLRScheduler
from models import GreedyHash

def get_arguments():
    parser = argparse.ArgumentParser(description='GreedyHash')
    # normal settings
    parser.add_argument('--model', type=str, default="GreedyHash")
    parser.add_argument('--seed', type=int, default=2000, help="NOTE: IMPORTANT TO REPRODUCE THE RESULTS!")
    parser.add_argument('--bit', type=int, default=12,
                help="choose the model of certain bit type", choices=[12, 24, 32, 48])

    # data settings
    parser.add_argument('--dataset', type=str, default="cifar10-1")
    parser.add_argument('--n_class', type=int, default=10)
    parser.add_argument('--topK', type=int, default=-1)
    parser.add_argument('--crop_size', type=int, default=224)

    # training settings
    parser.add_argument('-bs', '--batch-size', type=int, default=32, help='batch_size')
    parser.add_argument('-ee', '--eval_epoch', type=int, default=2, help="After each eval_epoch, one eval process is performed")
    parser.add_argument('--alpha', type=float, default=0.1, help="Determines the tradeoff between losses")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-lr_de', '--epoch_lr_decrease', type=int, default=30)
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-op', '--optimizer', type=str, default="SGD")
    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--save_path', type=str, default="checkpoints/")
    parser.add_argument('--log_path', type=str, default="logs/")
    parser.add_argument('--output-dir', type=str, default="checkpoints/", help='output_dir')
    arguments = parser.parse_args()
    return arguments

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

def val(model, test_loader, database_loader):
    acc = calculate_acc(test_loader, model)
    retrievalB, retrievalL, queryB, queryL = compress(database_loader, test_loader, model)
    mAP = calculate_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL)
    return mAP, acc

def train_val(model,
              config,
              bit,
              master_logger,
              train_loader,
              test_loader,
              database_loader):
    master_logger.info(f'{config}')
    scheduler = DecreaseLRScheduler(learning_rate=config.learning_rate,
                                    start_lr=config.learning_rate,
                                    epoch_lr_decrease=config.epoch_lr_decrease)
    if config.optimizer == "SGD":
        optimizer = paddle.optimizer.Momentum(
            parameters=model.parameters(),
            learning_rate=scheduler,
            weight_decay=config.weight_decay,
            momentum=config.momentum)
    elif config.optimizer == "AdamW":
        optimizer = paddle.optimizer.AdamW(
            parameters=model.parameters(),
            learning_rate=scheduler,
            weight_decay=config.weight_decay)
    else:
        raise NotImplementedError(f"Unsupported Optimizer: {config.optimizer}.")

    criterion = nn.CrossEntropyLoss()

    Best_mAP = 0
    for epoch in range(config.epoch):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            outputs, feature, _  = model(images)

            loss1 = criterion(outputs, labels)
            loss2 = (feature.abs() - 1).pow(3).abs().mean()
            loss = loss1 + config.alpha * loss2
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
        train_loss = train_loss / len(train_loader)

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        master_logger.info("{}[{:2d}/{:2d}][{}] bit:{:d}, lr:{:.9f}, dataset:{}, train loss:{:.3f}".format(
            config.model, epoch + 1, config.epoch, current_time, bit, optimizer.get_lr(), config.dataset, train_loss))
        scheduler.step()

        if (epoch + 1) % config.eval_epoch == 0 or epoch + 1==config.epoch:
            model.eval()
            mAP, acc = val(model, test_loader, database_loader)
            if mAP > Best_mAP:
                Best_mAP = mAP

                if config.output_dir is not None:
                    if not os.path.exists(config.output_dir):
                        os.makedirs(config.output_dir)
                    save_path = os.path.join(config.output_dir, "model_best_{}".format(bit))
                    master_logger.info(f"save in {save_path}")
                    paddle.save(optimizer.state_dict(), save_path + ".pdopt")
                    paddle.save(model.state_dict(), save_path + ".pdparams")
            master_logger.info("{} epoch:{}, bit:{}, dataset:{}, MAP:{:.3f}, Best MAP: {:.3f}, Acc: {:.3f}".format(
                config.model, epoch + 1, bit, config.dataset, mAP, Best_mAP, acc))

def main():
    config = get_arguments()
    set_random_seed(config.seed)
    mode = "train"

    log_path = '{}/{}-{}'.format(config.log_path, mode, time.strftime('%Y%m%d-%H-%M-%S'))
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    master_logger = get_logger(
            filename=os.path.join(log_path, 'log.txt'),
            logger_name='master_logger')

    train_dataset, test_dataset, database_dataset = get_data(config)
    config.num_train = len(train_dataset)
    train_loader = get_dataloader(config=config,
                                  dataset=train_dataset,
                                  mode='train')
    test_loader = get_dataloader(config=config,
                                 dataset=test_dataset,
                                 mode='test')
    database_loader = get_dataloader(config=config,
                                     dataset=database_dataset,
                                     mode='test')
    model = GreedyHash(config.bit, config.n_class)
    train_val(model,
              config,
              config.bit,
              master_logger,
              train_loader,
              test_loader,
              database_loader)

if __name__ == "__main__":
    main()