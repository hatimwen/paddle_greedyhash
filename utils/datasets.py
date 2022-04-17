import numpy as np
from PIL import Image
from paddle.vision import transforms
from paddle.vision import datasets as dsets
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler

class MyCIFAR10(dsets.Cifar10):
    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = np.reshape(image, [3, 32, 32])
        image = image.transpose([1, 2, 0])

        if self.backend == 'pil':
            image = Image.fromarray(image.astype('uint8'))
        if self.transform is not None:
            image = self.transform(image)

        # label = np.eye(10, dtype=np.float32)[np.array(label)] # one-hot
        label = np.array(label)
        if self.backend == 'pil':
            return image, label
        return image.astype(self.dtype), label

def get_dataloader(config, dataset, mode='train', multi_process=False, drop_last=False):
    """Get dataloader with config, dataset, mode as input, allows multiGPU settings.

        Multi-GPU loader is implements as distributedBatchSampler.

    Args:
        config: configurations set in main.py
        dataset: paddle.io.dataset object
        mode: train/val
        multi_process: if True, use DistributedBatchSampler to support multi-processing
    Returns:
        dataloader: paddle.io.DataLoader object.
    """

    batch_size = config.batch_size

    if multi_process is True:
        sampler = DistributedBatchSampler(dataset,
                                          batch_size=batch_size,
                                          shuffle=(mode == 'train'),
                                          drop_last=drop_last,
                                          )
        dataloader = DataLoader(dataset,
                                batch_sampler=sampler,
                                num_workers=4)
    else:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=4,
                                shuffle=(mode == 'train'),
                                drop_last=drop_last)
    return dataloader

def get_data(config):
    # NOTE: for now, only [cifar, cifar-1, cifar-2] is supported.
    assert "cifar" in config.dataset, "{} is not supported now. (Supported list: cifar, cifar-1, cifar-2)".format(config.dataset)
    train_size = 500
    test_size = 100

    if config.dataset == "cifar10-2":
        train_size = 5000
        test_size = 1000

    transform = transforms.Compose([
        transforms.Resize(config.crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # cifar_dataset_root = 'dataset/cifar/cifar-10-python.tar.gz'
    # Dataset
    train_dataset = MyCIFAR10(
        # data_file=cifar_dataset_root,
                              mode='train',
                              transform=transform)

    test_dataset = MyCIFAR10(
        # data_file=cifar_dataset_root,
                             mode='test',
                             transform=transform)

    database_dataset = MyCIFAR10(
        # data_file=cifar_dataset_root,
                                 mode='test',
                                 transform=transform)

    X = np.concatenate((np.array(train_dataset.data)[:, 0], np.array(test_dataset.data)[:, 0]))
    L = np.concatenate((np.array(train_dataset.data)[:, 1], np.array(test_dataset.data)[:, 1]))

    first = True
    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N) # NOTE: Here, random factor affects the results!
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if config.dataset == "cifar10":
        # test:1000, train:5000, database:54000
        pass
    elif config.dataset == "cifar10-1":
        # test:1000, train:5000, database:59000
        database_index = np.concatenate((train_index, database_index))
    elif config.dataset == "cifar10-2":
        # test:10000, train:50000, database:50000
        database_index = train_index

    train_dataset_image = X[train_index]
    train_dataset_label = L[train_index]
    train_dataset.data = []
    for i in range(len(train_dataset_image)):
        train_dataset.data.append((train_dataset_image[i], train_dataset_label[i]))

    test_dataset_image = X[test_index]
    test_dataset_label = L[test_index]
    test_dataset.data = []
    for i in range(len(test_dataset_image)):
        test_dataset.data.append((test_dataset_image[i], test_dataset_label[i]))

    database_dataset_image = X[database_index]
    database_dataset_label = L[database_index]
    database_dataset.data = []
    for i in range(len(database_dataset_image)):
        database_dataset.data.append((database_dataset_image[i], database_dataset_label[i]))

    return train_dataset, test_dataset, database_dataset