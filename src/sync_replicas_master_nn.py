from __future__ import print_function
import time
import copy
from sys import getsizeof
import warnings
import pickle

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

from svd_compression import decode
from optim.adam import Adam
from optim.sgd import SGD
from utils import decompress

import torch
import codings

STEP_START_ = 1
# use compression tool to make it run faster
_FAKE_SGD = True

def update_params_dist_version(param, avg_grad, learning_rate):
    '''
    update the network layer by layer
    '''
    assert param.shape == avg_grad.shape
    param -= learning_rate * avg_grad
    return param


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


class GradientAccumulator(object):
    '''
    a simple class to implement gradient aggregator like the `Conditional Accumulators` in tensorflow
    '''
    def __init__(self, module, num_worker, mode='None'):
        super(GradientAccumulator, self).__init__()
        # we will update this counter dynamically during the training process
        # the length of this counter should be number of fc layers in the network
        # we used list to contain gradients of layers
        self.gradient_aggregate_counter = []
        self.model_index_range = []
        self.gradient_aggregator = []
        
        for param_idx, param in enumerate(module.parameters()):
            tmp_aggregator = []
            for worker_idx in range(num_worker):
                _shape = param.size()
                if len(_shape) == 1:
                    tmp_aggregator.append(bytearray(getsizeof(np.zeros((_shape[0]*2,)))*4))
                else:
                    tmp_aggregator.append(bytearray(getsizeof(np.zeros(_shape))*4))
            # initialize the gradient aggragator
            self.gradient_aggregator.append(tmp_aggregator)
            self.gradient_aggregate_counter.append(0)
            self.model_index_range.append(param_idx)

    def meset_everything(self):
        self._meset_grad_counter()
        self._meset_grad_aggregator()

    def _meset_grad_counter(self):
        self.gradient_aggregate_counter = [0 for _ in self.gradient_aggregate_counter]

    def _meset_grad_aggregator(self):
        pass


class SyncReplicasMaster_NN(NN_Trainer):
    def __init__(self, comm, **kwargs):
        '''master node here, no rank needed since the rank will always be 0 for master node'''
        self.comm = comm   # get MPI communicator object
        self.world_size = comm.Get_size() # total number of processes
        self.cur_step = STEP_START_
        # initial learning rate:
        self.lr = kwargs['learning_rate']
        # we use this parameter to shrink the step size per epoch
        self._lr_shrinkage = kwargs['lr_shrinkage']
        self._base_lr =  kwargs['learning_rate']
        # TODO(hwang): change this to a more sophisticated version later
        self.shrinkage_freq = 50
        self.shrink_counter = 0

        self.momentum = kwargs['momentum']
        self.network_config = kwargs['network']
        self.comm_type = kwargs['comm_method']

        self._num_grad_to_collect = self.world_size - 1
        # used to aggregate tmp gradients, the length is the same as # of fc layer 
        self._grad_aggregate_buffer = []
        self._model_shapes = []
        self._first_grad_received = False
        self._eval_freq = kwargs['eval_freq']
        self._train_dir = kwargs['train_dir']
        self._max_steps = kwargs['max_steps']
        self._compress = kwargs['compress']
        
        self._enable_gpu = kwargs['enable_gpu']
        self._num_aggregate = kwargs['num_aggregate']

        self._svd_rank = kwargs['svd_rank']
        self._quantization_level = kwargs['quantization_level']
        self._bucket_size = kwargs['bucket_size']
        self._r = self._svd_rank

        ############ will be deprecated soon #############################
        self._eval_batch_size = 1000

        if kwargs['code'] == 'sgd':
            if not _FAKE_SGD:
                self._coder = codings.svd.SVD(compress=False)
            else:
                self._coder = codings.lossless_compress.LosslessCompress()
        elif kwargs['code'] == 'svd':
            print("train.py, svd_rank =", self._svd_rank)
            self._coder = codings.svd.SVD(random_sample=False, rank=self._svd_rank,
                                   compress=True)
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

        # TODO(hwang): make sure this is useful
        self.optimizer = SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        # assign a gradient accumulator to collect gradients from workers
        self.grad_accumulator = GradientAccumulator(self.network, self.world_size-1, self._compress)
        self.init_model_shapes()
        # enable GPU here
        if self._enable_gpu:
            self.network.cuda()

    def train(self):
        # the first step we need to do here is to sync fetch the inital worl_step from the parameter server
        # we still need to make sure the value we fetched from parameter server is 1
        # please note that step is start from one here
        self.async_bcast_step()

        # fake test here:
        for i in range(1, self._max_steps+1):
            # switch back to training mode
            self.network.train()
            self._first_grad_received = False
            enough_gradients_received = False

            print("Master node is entering step: {}".format(i))

            self.async_bcast_step()

            self.async_bcast_layer_weights_bcast()
            
            # set the gradient fetch step and gather the request
            gradient_fetch_requests=self.async_fetch_gradient_start()

            coded_msgs = {}
            # wait for enough gradients to be aggregated:
            gather_start_time = time.time()
            while not enough_gradients_received:
                status = MPI.Status()
                source, code = MPI.Request.waitany(requests=gradient_fetch_requests, status=status)
                layer_index = status.tag-88
                if layer_index not in coded_msgs.keys():
                    coded_msgs[layer_index] = [code]
                else:
                    coded_msgs[layer_index].append(code)
                    
                self.grad_accumulator.gradient_aggregate_counter[layer_index] += 1
                
                enough_gradients_received = True
                for j in self.grad_accumulator.gradient_aggregate_counter:
                    enough_gradients_received = enough_gradients_received and (j >= self._num_grad_to_collect)
            gather_duration = time.time() - gather_start_time

            decode_start = time.time()
            self._decode(coded_msgs)
            decode_dur = time.time() - decode_start
            # update `state_dict` in pytorch modules
            print("Master: Step: {}, Decode Cost: {}, Cur lr {}, Gather: {}".format(self.cur_step, decode_dur, self.lr, gather_duration))
            self._model_update()

            # reset essential elements
            self.meset_grad_buffer()
            self.grad_accumulator.meset_everything()

            self.cur_step += 1
            if self.cur_step % self.shrinkage_freq == 0:
                self.shrink_counter += 1
                self.lr = self._base_lr * self._lr_shrinkage ** self.shrink_counter

    def _model_update(self):
        # gradient shipped from workers are averaged and update the model
        self._grad_aggregate_buffer = map(lambda x: x / float(self._num_grad_to_collect), self._grad_aggregate_buffer)
        self.optimizer.step(grads=self._grad_aggregate_buffer)   

    def init_model_shapes(self):
        for p_index, p in enumerate(self.network.parameters()):
            self._model_shapes.append(p.size())
            self._grad_aggregate_buffer.append(np.zeros(p.size()))

    def async_bcast_step(self):
        req_list = []
        for i in range(self.world_size):
            if i != 0:
                req_list.append(self.comm.isend(self.cur_step, dest=i, tag=10))
        for i in range(len(req_list)):
            req_list[i].wait()

    def async_bcast_layer_weights_async(self):
        request_layers = []
        for layer_idx, layer in enumerate(self.network.parameters()):
            request_workers = []
            layer_to_send = layer.data.numpy().astype(np.float32)
            for i in range(self.world_size):
                if i != 0:
                    req = self.comm.Isend([layer_to_send, MPI.DOUBLE], dest=i, tag=11+layer_idx)
                    request_workers.append(req)

            request_layers.append(request_workers)
        # TODO(hwang): check to see if these `wait` calls are necessary here
        for req_l in request_layers:
            for req_worker in req_l:
                req_worker.wait()

    def async_bcast_layer_weights_bcast(self):
        request_layers = []
        for layer_idx, layer in enumerate(self.network.parameters()):
            request_workers = []
            if self._enable_gpu:
                # copy data to CPU then do the communicaiton staff
                layer_to_send = layer.data.cpu().numpy().astype(np.float32)
            else:
                layer_to_send = layer.data.numpy().astype(np.float32)
            self.comm.Bcast([layer_to_send, MPI.DOUBLE], root=0)

    def async_fetch_gradient_start(self):
        '''
        make gradient fetch requests and return the request list
        '''
        gradient_fetch_requests = []
        for module_idx, module in enumerate(self.network.parameters()):
            for k in range(self._num_grad_to_collect):
                req = self.comm.irecv(self.grad_accumulator.gradient_aggregator[module_idx][k], source=k+1, tag=88+module_idx)
                gradient_fetch_requests.append(req)
        return gradient_fetch_requests

    def aggregate_gradient(self, gradient, layer_idx):
        '''
        keep in mind the gradient here is wrapped gradient, which means it contains `W` and `b`
        '''
        self._grad_aggregate_buffer[layer_idx] += gradient.numpy().astype(np.float32)

    def model_update(self, tmp_module):
        """write model fetched from parameter server to local model"""
        new_state_dict = {}
        model_counter_ = 0
        for param_idx,(key_name, param) in enumerate(self.network.state_dict().items()):
            # handle the case that `running_mean` and `running_var` contained in `BatchNorm` layer
            if "running_mean" in key_name or "running_var" in key_name:
                tmp_dict = {key_name : param}
            else:
                assert param.size() == tmp_module[model_counter_].shape
                tmp_dict = {key_name: torch.from_numpy(tmp_module[model_counter_])}
                model_counter_+=1
            new_state_dict.update(tmp_dict)
        self.network.load_state_dict(new_state_dict)

    def meset_grad_buffer(self):
        for i in range(len(self._grad_aggregate_buffer)):
            self._grad_aggregate_buffer[i][0] = np.zeros(self._grad_aggregate_buffer[i][0].shape)
            if len(self._grad_aggregate_buffer[i])==2:
                self._grad_aggregate_buffer[i][1] = np.zeros(self._grad_aggregate_buffer[i][1].shape)

    def _decode(self, coded_msgs):
        # k: `layer_index` v: coded gradients
        for index, (k, v) in enumerate(coded_msgs.iteritems()):
            for code in v:
                code = pickle.loads(code)
                grad=self._coder.decode(code)
                try:
                    assert (grad.shape == self._model_shapes[k])
                except AssertionError:
                    warnings.warn("shape dosen't match, should really be careful")
                self.aggregate_gradient(gradient=grad, layer_idx=k)

    def _generate_model_path(self):
        return self._train_dir+"model_step_"+str(self.cur_step)

    def _save_model(self, file_path):
        with open(file_path, "wb") as f_:
            torch.save(self.network.state_dict(), f_)

    def _evaluate_model(self, validation_loader):
        self.network.eval()
        prec1_counter_ = prec5_counter_ = batch_counter_ = 0
        # which indicate an epoch based validation is done
        while validation_loader.dataset.epochs_completed <= self._epoch_counter:
            eval_image_batch, eval_label_batch = validation_loader.next_batch(batch_size=self._eval_batch_size)
            X_batch, y_batch = Variable(eval_image_batch.float()), Variable(eval_label_batch.long())
            output = self.network(X_batch)
            prec1_tmp, prec5_tmp = accuracy(output.data, eval_label_batch.long(), topk=(1, 5))
            prec1_counter_ += prec1_tmp
            prec5_counter_ += prec5_tmp
            batch_counter_ += 1
        prec1 = prec1_counter_ / batch_counter_
        prec5 = prec5_counter_ / batch_counter_
        self._epoch_counter = validation_loader.dataset.epochs_completed
        if self._enable_gpu:
            prec1 = prec1.cpu().numpy()[0]
            prec5 = prec5.cpu().numpy()[0]
        else:
            prec1 = prec1.numpy()[0]
            prec5 = prec5.numpy()[0]
        print('Testset Performance: Cur Step:{} Prec@1: {} Prec@5: {}'.format(self.cur_step, prec1, prec5))