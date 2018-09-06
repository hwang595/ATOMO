from __future__ import print_function
from mpi4py import MPI
import numpy as np

from nn_ops import NN_Trainer

from model_ops.lenet import LeNet, LeNetSplit
from model_ops.resnet import *
from model_ops.resnet_split import *
from model_ops.vgg import *
from model_ops.alexnet import *
from model_ops.fc_nn import FC_NN, FC_NN_Split
from model_ops.densenet import DenseNet

from utils import compress
import codings

import torch
from torch.autograd import Variable

import time
from datetime import datetime
import copy
from sys import getsizeof
import pickle

STEP_START_ = 1
# use compression tool to make it run faster
_FAKE_SGD = True
TAG_LIST_ = [i*30 for i in range(50000)]

def prepare_grad_list(params):
    grad_list = []
    for param_idx, param in enumerate(params):
        # get gradient from layers here
        # in this version we fetch weights at once
        # remember to change type here, which is essential
        grads = param.grad.data.numpy().astype(np.float32)
        grad_list.append((param_idx, grads))
    return grad_list

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

def _4d_to_2d(tensor):
    return tensor.view(tensor.size()[0], tensor.size()[1]*tensor.size()[2]*tensor.size()[3])

def _construct_grad_packet(module):
    '''
    input: weight (\in R^{m*n}) and bias (\in R^{m*n})
    output: grad packet (\in R^{m*(n+1)})
    '''
    ndims = len(module.weight.grad.data.size())
    if ndims == 4:
        tmp_grad = _4d_to_2d(module.weight.grad.data)
        if module.bias is None:
            return tmp_grad
        else:
            return torch.cat((tmp_grad, module.bias.grad.data), 1)
    elif ndims == 2:
        if module.bias is None:
            return module.weight.grad.data
        else:
            return torch.cat((module.weight.grad.data, module.bias.grad.data), 1)
    elif ndims == 1:
        if module.bias is None:
            return module.weight.grad.data
        else:
            return torch.cat((module.bias.grad.data, module.weight.grad.data), 0)        


class ModelBuffer(object):
    def __init__(self, network):
        """
        this class is used to save model weights received from parameter server
        current step for each layer of model will also be updated here to make sure
        the model is always up-to-date
        """
        self.recv_buf = []
        self.layer_cur_step = []
        for param_idx, param in enumerate(network.parameters()):
            self.recv_buf.append(np.zeros(param.size()))
            self.layer_cur_step.append(0)


class DistributedWorker(NN_Trainer):
    def __init__(self, comm, **kwargs):
        self.comm = comm   # get MPI communicator object
        self.world_size = comm.Get_size() # total number of processes
        self.rank = comm.Get_rank() # rank of this Worker
        #self.status = MPI.Status()
        self.cur_step = 0
        self.next_step = 0 # we will fetch this one from parameter server

        self.batch_size = kwargs['batch_size']
        self.max_epochs = kwargs['max_epochs']
        self.momentum = kwargs['momentum']
        self.lr = kwargs['learning_rate']
        self.network_config = kwargs['network']
        self._max_steps = kwargs['max_steps']
        self.comm_type = kwargs['comm_method']
        self._compress = kwargs['compress']
        self._enable_gpu = kwargs['enable_gpu']
        self._eval_batch_size = 100
        self._eval_freq = kwargs['eval_freq']
        self._train_dir = kwargs['train_dir']
        # encode related
        self._svd_rank = kwargs['svd_rank']
        self._quantization_level = kwargs['quantization_level']
        self._bucket_size = kwargs['bucket_size']

        # this one is going to be used to avoid fetch the weights for multiple times
        self._layer_cur_step = []
        self._code = kwargs['code']
        if kwargs['code'] == 'sgd':
            if not _FAKE_SGD:
                self._coder = codings.svd.SVD(compress=False)
            else:
                self._coder = codings.lossless_compress.LosslessCompress()
        elif kwargs['code'] == 'svd':
            print("train.py, svd_rank =", self._svd_rank)
            self._coder = codings.svd.SVD(random_sample=True, 
                rank=self._svd_rank, compress=True)
        else:
            raise ValueError('args.code not recognized')

    def build_model(self, num_classes=10):
        # build network
        if self.network_config == "LeNet":
            self.network=LeNet()
        elif self.network_config == "ResNet18":
            self.network=ResNet18(num_classes=num_classes)
        elif self.network_config == "ResNet34":
            self.network=ResNet34(num_classes=num_classes)
        elif self.network_config == "FC":
            self.network=FC_NN()
        elif self.network_config == "DenseNet":
            self.network=DenseNet(growthRate=40, depth=190, reduction=0.5,
                            bottleneck=True, nClasses=10)
        elif self.network_config == "VGG11":
            self.network=vgg11_bn(num_classes)
        elif self.network_config == "AlexNet":
            self.network=alexnet(num_classes=10)

        # set up optimizer
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()
        # assign a buffer for receiving models from parameter server
        self.init_recv_buf()
        # enable GPU here
        if self._enable_gpu:
            self.network.cuda()

    def train(self, train_loader, test_loader):
        # the first step we need to do here is to sync fetch the inital worl_step from the parameter server
        # we still need to make sure the value we fetched from parameter server is 1
        self.sync_fetch_step()
        # do some sync check here
        assert(self.update_step())
        assert(self.cur_step == STEP_START_)

        # number of batches in one epoch
        num_batch_per_epoch = len(train_loader.dataset) / self.batch_size
        batch_idx = -1
        epoch_idx = 0
        epoch_avg_loss = 0
        iteration_last_step=0
        iter_start_time=0

        first = True

        print("Worker {}: starting training".format(self.rank))
        # start the training process
        # start the training process
        for num_epoch in range(self.max_epochs):
            for batch_idx, (train_image_batch, train_label_batch) in enumerate(train_loader):
                # worker exit task
                if self.cur_step == self._max_steps:
                    break
                if self._enable_gpu:
                    X_batch, y_batch = Variable(train_image_batch.cuda()), Variable(train_label_batch.cuda())
                else:
                    X_batch, y_batch = Variable(train_image_batch), Variable(train_label_batch)
                while True:
                    # the worker shouldn't know the current global step
                    # except received the message from parameter server
                    self.async_fetch_step()

                    # the only way every worker know which step they're currently on is to check the cur step variable
                    updated = self.update_step()

                    if (not updated) and (not first):
                        # wait here unitl enter next step
                        continue

                    # the real start point of this iteration
                    iteration_last_step = time.time() - iter_start_time
                    iter_start_time = time.time()
                    first = False
                    print("Rank of this node: {}, Current step: {}".format(self.rank, self.cur_step))

                    # TODO(hwang): return layer request here and do weight before the forward step begins, rather than implement
                    # the wait() in the fetch function
                    fetch_weight_start_time = time.time()

                    self.async_fetch_weights_bcast()

                    fetch_weight_duration = time.time() - fetch_weight_start_time

                    # switch to training mode
                    self.network.train()
                    # manage batch index manually
                    self.optimizer.zero_grad()

                    # forward step
                    comp_start = time.time()
                    logits = self.network(X_batch)
                    loss = self.criterion(logits, y_batch)

                    epoch_avg_loss += loss.data[0]

                    # backward step
                    backward_start_time = time.time()
                    loss.backward()
                    comp_dur = time.time() - comp_start

                    # gradient encoding step
                    encode_start = time.time()
                    msgs,_msg_counter = self._encode()
                    encode_dur = time.time() - encode_start

                    # communication step
                    comm_start = time.time()
                    self._send_grads(msgs)
                    comm_dur = time.time()-comm_start

                    prec1, prec5 = accuracy(logits.data, y_batch.data, topk=(1, 5))
                    # on the end of a certain iteration
                    print('Worker: {}, Step: {}, Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.4f}, Time Cost: {:.4f}, Comp: {:.4f}, Encode: {: .4f}, Comm: {: .4f}, Msg(MB): {: .4f}, Prec@1: {: .4f}, Prec@5: {: .4f}'.format(
                        self.rank, self.cur_step, num_epoch, batch_idx * self.batch_size, len(train_loader.dataset), 
                        (100. * (batch_idx * self.batch_size) / len(train_loader.dataset)), loss.data[0], time.time()-iter_start_time, 
                        comp_dur, encode_dur, comm_dur, _msg_counter/(1024.0**2), prec1.numpy()[0], prec5.numpy()[0]))
                    # break here to fetch data then enter fetching step loop again
                    if self.cur_step%self._eval_freq==0:
                        self._evaluate_model(test_loader)
                    break

    def init_recv_buf(self):
        self.model_recv_buf = ModelBuffer(self.network)

    def sync_fetch_step(self):
        '''fetch the first step from the parameter server'''
        self.next_step = self.comm.recv(source=0, tag=10)

    def async_fetch_step(self):
        req = self.comm.irecv(source=0, tag=10)
        self.next_step = req.wait()
    
    def async_fetch_weights_bcast(self):
        layers_to_update = []
        for layer_idx, layer in enumerate(self.model_recv_buf.recv_buf):
            if self.model_recv_buf.layer_cur_step[layer_idx] < self.cur_step:
                layers_to_update.append(layer_idx)
                self.comm.Bcast([self.model_recv_buf.recv_buf[layer_idx], MPI.DOUBLE], root=0)
        weights_to_update = []
        for req_idx, layer_idx in enumerate(layers_to_update):
            weights = self.model_recv_buf.recv_buf[req_idx]
            weights_to_update.append(weights)
            # we also need to update the layer cur step here:
            self.model_recv_buf.layer_cur_step[req_idx] = self.cur_step
        self.model_update(weights_to_update)
    
    def update_step(self):
        '''update local (global) step on worker'''
        changed = (self.cur_step != self.next_step)
        self.cur_step = self.next_step
        return changed

    def model_update(self, weights_to_update):
        """write model fetched from parameter server to local model"""
        new_state_dict = {}
        model_counter_ = 0
        for param_idx,(key_name, param) in enumerate(self.network.state_dict().items()):
            # handle the case that `running_mean` and `running_var` contained in `BatchNorm` layer
            if "running_mean" in key_name or "running_var" in key_name:
                tmp_dict={key_name: param}
            else:
                assert param.size() == weights_to_update[model_counter_].shape
                if self._enable_gpu:
                    tmp_dict = {key_name: torch.from_numpy(weights_to_update[model_counter_]).cuda()}
                else:
                    tmp_dict = {key_name: torch.from_numpy(weights_to_update[model_counter_])}
                model_counter_ += 1
            new_state_dict.update(tmp_dict)
        self.network.load_state_dict(new_state_dict)

    def _encode(self):
        msgs = []
        _msg_counter = 0
        def __count_msg_sizes(msg):
            return len(msg)
        for p_index, p in enumerate(self.network.parameters()):
            grad = p.grad.data.numpy().astype(np.float32)
            coded = self._coder.encode(grad)
            pickled = pickle.dumps(coded)
            byte_code = bytearray(pickled)
            _msg_counter+=__count_msg_sizes(byte_code)
            msgs.append(byte_code)
        return msgs, _msg_counter

    def _send_grads(self, msgs):
        req_send_check = []
        for msg_index, m in enumerate(msgs):
            req_isend = self.comm.isend(m, dest=0, tag=88+msg_index)
            req_isend.wait()
        [req_send_check[i].wait() for i in range(len(req_send_check))]

    def _generate_model_path(self):
        return self._train_dir+"model_step_"+str(self.cur_step)

    def _save_model(self, file_path):
        with open(file_path, "wb") as f_:
            torch.save(self.network.state_dict(), f_)

    def _evaluate_model(self, test_loader):
        self.network.eval()
        test_loss = 0
        correct = 0
        prec1_counter_ = prec5_counter_ = batch_counter_ = 0
        for data, y_batch in test_loader:
            if self._enable_gpu:
                data, target = Variable(data.cuda(), volatile=True), Variable(y_batch.cuda())
            else:
                data, target = Variable(data, volatile=True), Variable(y_batch)
            
            output = self.network(data)
            test_loss += F.nll_loss(F.log_softmax(output), target, size_average=False).data[0]

            prec1_tmp, prec5_tmp = accuracy(output.data, target.data, topk=(1, 5))
            if self._enable_gpu:
                prec1_counter_ += prec1_tmp.cpu().numpy()[0]
                prec5_counter_ += prec5_tmp.cpu().numpy()[0]
            else:
                prec1_counter_ += prec1_tmp.numpy()[0]
                prec5_counter_ += prec5_tmp.numpy()[0]
            batch_counter_ += 1
        prec1 = prec1_counter_ / batch_counter_
        prec5 = prec5_counter_ / batch_counter_
        test_loss /= len(test_loader.dataset)
        print('Test set: Step: {}, Average loss: {:.4f}, Prec@1: {} Prec@5: {}'.format(self.cur_step, 
                                                                            test_loss, prec1, prec5))

if __name__ == "__main__":
    # this is only a simple test case
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    worker_fc_nn = WorkerFC_NN(comm=comm, world_size=world_size, rank=rank)
    print("I am worker: {} in all {} workers".format(worker_fc_nn.rank, worker_fc_nn.world_size))