import numpy as np
import random
import paddle
from tqdm import tqdm

def set_random_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def compress(train, test, model, classes=10):
    retrievalB = list([])
    retrievalL = list([])
    desc = "--- Compressing(train) "
    for batch_step, (data, target) in enumerate(tqdm(train, desc)):
        _,_, code = model(data)
        retrievalB.extend(code.cpu().numpy())
        retrievalL.extend(target)

    queryB = list([])
    queryL = list([])
    desc = "--- Compressing(test) "
    for batch_step, (data, target) in enumerate(tqdm(test, desc)):
        _,_, code = model(data)
        queryB.extend(code.cpu().numpy())
        queryL.extend(target)

    retrievalB = np.array(retrievalB)
    retrievalL = np.eye(classes)[np.array(retrievalL)]

    queryB = np.array(queryB)
    queryL = np.eye(classes)[np.array(queryL)]
    return retrievalB, retrievalL.squeeze(1), queryB, queryL.squeeze(1)


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.shape[1] # max inner product value
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def calculate_map(qB, rB, queryL, retrievalL):
    """
       :param qB: {-1,+1}^{mxq} query bits
       :param rB: {-1,+1}^{nxq} retrieval bits
       :param queryL: {0,1}^{mxl} query label
       :param retrievalL: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = queryL.shape[0]
    map = 0
    desc = "--- Calculating mAP "
    for iter in tqdm(range(num_query), desc):
        # gnd : check if exists any retrieval items with same label
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        # tsum number of items with same label
        tsum = np.sum(gnd).astype(int)
        if tsum == 0:
            continue
        # sort gnd by hamming dist
        hamm = calculate_hamming(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum) # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map


def calculate_top_map(qB, rB, queryL, retrievalL, topk):
    """
    :param qB: {-1,+1}^{mxq} query bits
    :param rB: {-1,+1}^{nxq} retrieval bits
    :param queryL: {0,1}^{mxl} query label
    :param retrievalL: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = queryL.shape[0]
    topkmap = 0
    desc = "--- Calculating top mAP "
    for iter in tqdm(range(num_query), desc):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


def calculate_acc(test_loader, model):
    correct = 0
    total = 0
    desc = "--- Calculating Acc "
    for images, labels in tqdm(test_loader, desc):
        outputs, _, _ = model(images)
        predicted = outputs.argmax(axis=1)
        total += labels.shape[0]
        correct += (predicted == labels).sum()
    acc = 100.0 * correct.cpu().numpy() / total
    return acc[0]