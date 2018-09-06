import torch
from torch import nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
from torch.autograd import Variable

from mpi4py import MPI

# we use fc nn here for our simple case
class FC_NN(nn.Module):
    def __init__(self):
        super(FC_NN, self).__init__()
        self.fc1 = nn.Linear(784, 800)
        self.fc2 = nn.Linear(800, 500)
        self.fc3 = nn.Linear(500, 10)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    def name(self):
        return 'fc_nn'

# we use fc nn here for our simple case
class FC_NN_Split(nn.Module):
    def __init__(self):
        super(FC_NN_Split, self).__init__()
        self.fc1 = nn.Linear(784, 800)
        self.fc2 = nn.Linear(800, 500)
        self.fc3 = nn.Linear(500, 10)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # helper
        self.full_modules = [self.fc1, self.fc2, self.fc3]
        self._init_channel_index = len(self.full_modules)*2
    def forward(self, x):
        '''
        split layers
        '''
        self.output = []
        self.input = []
        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.fc1(x)
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.relu(x)
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.fc2(x)
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)        
        x = self.relu(x)
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x) 
        x = self.fc3(x)
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x) 
        x = self.sigmoid(x)
        self.output.append(x)
        return x
    @property
    def fetch_init_channel_index(self):
        return self._init_channel_index
    def backward_normal(self, g, communicator, req_send_check, cur_step):
        mod_avail_index = len(self.full_modules)-1
        #channel_index = len(self.full_modules)*2-2
        channel_index = self._init_channel_index - 2
        mod_counters_ = [0]*len(self.full_modules)
        for i, output in reversed(list(enumerate(self.output))):
            req_send_check[-1].wait()
            if i == (len(self.output) - 1):
                # for last node, use g
                output.backward(g)
                # get gradient here after some sanity checks:
                '''
                tmp_grad = self.full_modules[mod_avail_index].weight.grad
                if not pd.isnull(tmp_grad):
                    grads = tmp_grad.data.numpy().astype(np.float64)
                    req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                    req_send_check.append(req_isend)
                    # update counters
                    mod_avail_index-=1
                    channel_index-=1
                else:
                    continue
                '''
            else:
                output.backward(self.input[i+1].grad.data)
                tmp_grad_weight = self.full_modules[mod_avail_index].weight.grad
                tmp_grad_bias = self.full_modules[mod_avail_index].bias.grad
                # specific for this fc nn setting
                if mod_avail_index == len(self.full_modules)-1:
                    if not pd.isnull(tmp_grad_weight):
                        grads = tmp_grad_weight.data.numpy().astype(np.float64)
                        req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                        req_send_check.append(req_isend)
                        # update counters
                        mod_avail_index-=1
                        channel_index-=1
                    else:
                        continue
                else:
                    if not pd.isnull(tmp_grad_weight) and not pd.isnull(tmp_grad_bias):
                        # we always send bias first
                        if mod_counters_[mod_avail_index] == 0:
                            grads = tmp_grad_bias.data.numpy().astype(np.float64)
                            req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                            req_send_check.append(req_isend)
                            channel_index-=1
                            mod_counters_[mod_avail_index]+=1
                        elif mod_counters_[mod_avail_index] == 1:
                            grads = tmp_grad_weight.data.numpy().astype(np.float64)
                            req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                            req_send_check.append(req_isend)
                            channel_index-=1
                            mod_counters_[mod_avail_index]+=1
                            # update counters
                            mod_avail_index-=1
                    else:
                        continue
        if mod_counters_[0] == 1:
            req_send_check[-1].wait()
            grads = tmp_grad_weight.data.numpy().astype(np.float64)
            req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
            req_send_check.append(req_isend)
#        if cur_step >= 2:
#            exit()
        return req_send_check
    @property
    def name(self):
        return 'fc_nn'