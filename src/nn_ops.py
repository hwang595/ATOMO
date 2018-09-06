from __future__ import print_function

import time

import numpy as np
import torch
from torch.autograd import Variable

from model_ops.lenet import LeNet
from model_ops.resnet import *
from model_ops.resnet_split import *
from model_ops.alexnet import *

import numpy.linalg as LA


def nuclear_indicator(grad, s):
    m, n  = grad.shape
    return np.sum(s)*np.sqrt(m+n)


def l1_indicator(grad):
    return np.linalg.norm(grad.reshape(-1), 1)


def _resize_to_2d(x):
    """
    x.shape > 2
    If x.shape = (a, b, *c), assumed that each one of (a, b) pairs has relevant information in c.
    """
    shape = x.shape
    if x.ndim == 1:
        n = x.shape[0]
        return x.reshape((n//2, 2))
    if all([s == 1 for s in shape[2:]]):
        return x.reshape((shape[0], shape[1]))
    # each of (a, b) has related features
    x = x.reshape((shape[0], shape[1], -1))
    # stack those related features into a tall matrix
    x_tmp = x.reshape((shape[0]*shape[1], -1))
    tmp_shape = x_tmp.shape
    return x_tmp.reshape((int(tmp_shape[0]/2), int(tmp_shape[1]*2)))


def _sample_svd(s, rank=0):
    if s[0] < 1e-6:
        return [0], np.array([1.0])
    probs = s / s[0] if rank == 0 else rank * s / s.sum()
    for i, p in enumerate(probs):
        if p > 1:
            probs[i]=1
    sampled_idx = []
    sample_probs = []
    for i, p in enumerate(probs):
        #if np.random.rand() < p:
        # random sampling from bernulli distribution
        if np.random.binomial(1, p):
            sampled_idx += [i]
            sample_probs += [p]
    rank_hat = len(sampled_idx)
    if rank_hat == 0:  # or (rank != 0 and np.abs(rank_hat - rank) >= 3):
        return _sample_svd(s, rank=rank)
    return np.array(sampled_idx, dtype=int), np.array(sample_probs)


def svd_encode(grad, **kwargs):
    orig_size = list(grad.shape)
    ndims = grad.ndim
    reshaped_flag = False
    if ndims != 2:
        grad = _resize_to_2d(grad)
        shape = list(grad.shape)
        ndims = len(shape)
        reshaped_flag = True

    if ndims == 2:
        u, s, vT = LA.svd(grad, full_matrices=False)

        nuclear_ind = nuclear_indicator(grad, s)
        l1_ind = l1_indicator(grad)
        print("Step: {}, Nuclear Indicator: {}, L1 Indicator: {}".format(
            kwargs['step'], nuclear_ind, l1_ind))


'''this is a trial example, we use MNIST on LeNet for simple test here'''
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class NN_Trainer(object):
    def __init__(self, **kwargs):
        self.batch_size = kwargs['batch_size']
        self.lr = kwargs['learning_rate']
        self.max_epochs = kwargs['max_epochs']
        self.momentum = kwargs['momentum']
        self.network_config = kwargs['network']

    def build_model(self):
        # build network
        if self.network_config == "LeNet":
            self.network=LeNet()
        elif self.network_config == "ResNet18":
            self.network=ResNet18()
        elif self.network_config == "ResNet34":
            self.network=ResNet34()
        elif self.network_config == "AlexNet":
            self.network=alexnet()
        # set up optimizer
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_and_validate(self, train_loader, test_loader):
        # iterate of epochs
        for i in range(self.max_epochs):
            # change back to training mode
            self.network.train()      
            for batch_idx, (data, y_batch) in enumerate(train_loader):
                iter_start_time = time.time()
                data, target = Variable(data), Variable(y_batch)
                self.optimizer.zero_grad()
                ################# backward on normal model ############################

                logits = self.network(data)
                loss = self.criterion(logits, target)
                loss.backward()
                #######################################################################

                ################ backward on splitted model ###########################
                '''
                logits = self.network(data)
                logits_1 = Variable(logits.data, requires_grad=True)
                loss = self.criterion(logits_1, target)
                loss.backward()
                self.network.backward_single(logits_1.grad)
                '''
                #######################################################################
                tmp_time_0 = time.time()
                
                #for param in self.network.parameters():
                #    grads = param.grad.data.numpy().astype(np.float64)
                #    svd_encode(grads, step=batch_idx)

                duration_backward = time.time()-tmp_time_0

                tmp_time_1 = time.time()
                self.optimizer.step()
                duration_update = time.time()-tmp_time_1

                # calculate training accuracy
                prec1, prec5 = accuracy(logits.data, y_batch, topk=(1, 5))
                # load the training info
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Prec@1: {}  Prec@5: {}  Time Cost: {}'.format(
                    i, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0], 
                    prec1.numpy()[0], 
                    prec5.numpy()[0], time.time()-iter_start_time))
            # we evaluate the model performance on end of each epoch
            self.validate(test_loader)

    def validate(self, test_loader):
        self.network.eval()
        test_loss = 0
        correct = 0
        prec1_counter_ = prec5_counter_ = batch_counter_ = 0
        for data, y_batch in test_loader:
            data, target = Variable(data, volatile=True), Variable(y_batch)
            output = self.network(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            #pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            prec1_tmp, prec5_tmp = accuracy(output.data, y_batch, topk=(1, 5))
            prec1_counter_ += prec1_tmp.numpy()[0]
            prec5_counter_ += prec5_tmp.numpy()[0]
            batch_counter_ += 1
        prec1 = prec1_counter_ / batch_counter_
        prec5 = prec5_counter_ / batch_counter_
        test_loss /= len(test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Prec@1: {} Prec@5: {}'.format(test_loss, prec1, prec5))
