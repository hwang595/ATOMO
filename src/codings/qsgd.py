from functools import reduce
import numpy as np
import numpy.linalg as LA
from scipy import stats
import torch
import time
import math
from .coding import Coding

import torch.nn.functional as F


class QSGD(Coding):
    def __init__(self, scheme='qsgd', bucket_size=512, *args, **kwargs):
        self.scheme = scheme
        self._quantization_level = kwargs['quantization_level']
        self._bucket_size=bucket_size

    def encode(self, v, **kwargs):
        if isinstance(v, (torch.Tensor, torch.cuda.FloatTensor)):
            w = v.cpu().numpy().flat[:]
        elif isinstance(v, np.ndarray):
            w = v.flat[:]
        else:
            raise ValueError("Object passed to encode not ndarray or torch.Tensor")

        if 'neo_bucket_size' in kwargs.keys():
            bucket_size = min(self._bucket_size, kwargs['neo_bucket_size'])
        else:
            bucket_size = self._bucket_size
        # Apply bucketing
        if bucket_size != 0:
            code_buckets = []
            shape = v.shape
            neo_kwargs = {'neo_bucket_size': 0}
            buckets = np.split(w, (w.shape[0] + bucket_size - 1) / bucket_size)
            for bucket in buckets:
                code = self.encode(bucket, **neo_kwargs)
                code_buckets.append(code)
            return {'code_buckets': code_buckets, 'shape': shape}

        if self.scheme == 'qsgd':
            norm = LA.norm(v)
        elif self.scheme == 'terngrad':
            norm = np.linalg.norm(w, ord=np.inf)
            limit = grad_clip_limit(w, clip_factor=2.5)
            w = np.clip(w, -limit, limit)

        s = (1 << self._quantization_level) - 1
        shape = v.shape

        num_int_each_64_bits = int(64 / (2 + self._quantization_level)) # number of element stored / 64-bits
        num_section = num_int_each_64_bits
        len_each_section = int((w.shape[0] + num_section - 1) / num_section) # number of 64-bits to ship w vector
        w = np.pad(w, (0, len_each_section * num_section - w.shape[0]), mode='constant') # pad w to length of total elements

        sign_array = np.sign(w)
        sign_array += 1 # -1, 0, 1 to 0, 1, 2
        sign_array = sign_array.astype('uint64')
        normalization_array = np.abs(w) / norm * s

        truncated_array = normalization_array.astype(int) # l <= \frac{s \|w\|_i}{\|w\|_2} <= l+1
        prob_array = normalization_array - truncated_array # \frac{s \|w\|_i}{\|w\|_2} - l i.e. p function p(a, s) = as - l
        dice_array = np.random.rand(len(prob_array))
        xi_array = truncated_array + (dice_array > prob_array) # l+1 or l
        xi_array = xi_array.astype('uint64')

        xi_array = xi_array.reshape((num_section, len_each_section))
        sign_array = sign_array.reshape((num_section, len_each_section))

        neo_array = np.zeros(len_each_section)
        neo_array = neo_array.astype('uint64')

        for i in range(num_int_each_64_bits):
            xi = xi_array[i]
            sign = sign_array[i]
            neo_array <<= (2 + self._quantization_level)
            neo_array = neo_array | (sign << self._quantization_level | xi)

        code = {'neo': neo_array, 'norm': norm, 'quantization_level': self._quantization_level,
                'len_each_section': len_each_section, 'num_int_each_64_bits': num_int_each_64_bits,
                'shape': shape}

        if kwargs.pop('timings', False):
            data = {}
            return code, data
        return code

    def decode(self, code, cuda=False, implementation='numpy', codes=[], **kwargs):
        """
        Decode the coding.
        ## NumPy
         'comm_wait': 0.0728750228881836,
         'decode_time': 0.1349341869354248,
         'example_to_gpu': 0.0006515979766845703,
         'grad_compute_time': 0.5815503597259521,
         'grad_forward_pass': 0.23496603965759277,
         'grad_variance_increase': 31.754316389320049,
         'iallgather_prepare_time': 0.017401456832885742,
         'isend_time': 0.029105424880981445,
        ## PT GPU
        """
        if self.scheme == 'terngrad' and len(codes) > 0:
            code['norm'] = self._get_max_norm(codes)

        if implementation == 'numpy':
            if 'neo_bucket_size' in kwargs.keys():
                bucket_size = min(self._bucket_size, kwargs['neo_bucket_size'])
            else:
                bucket_size = self._bucket_size
            # Decode from bucketing
            if bucket_size != 0:
                v_list = []
                neo_kwargs = {'neo_bucket_size': 0}
                for code_bucket in code['code_buckets']:
                    v = self.decode(code=code_bucket, cuda=cuda, implementation=implementation, codes=codes, **neo_kwargs)
                    v_list.extend(v)
                v = np.array(v_list)
                v = v.reshape(code['shape'])
            else:
                norm = code['norm']
                s = (1 << self._quantization_level) - 1

                real_size = np.prod(code['shape'])

                neo_array = code['neo'].astype('uint64')
                num_int_each_64_bits = code['num_int_each_64_bits']
                num_section = num_int_each_64_bits
                len_each_section = code['len_each_section']
                xi_array = np.ones((num_section, len_each_section))
                sign_array = np.ones((num_section, len_each_section))
                mask_for_xi = (1 << self._quantization_level) - 1
                mask_for_sign = 3 << self._quantization_level
                for i in range(num_int_each_64_bits)[::-1]:
                    sign_array[i] = (neo_array & mask_for_sign) >> self._quantization_level
                    xi_array[i] = neo_array & mask_for_xi
                    neo_array >>= (2 + self._quantization_level)

                xi_array = xi_array.reshape(-1).astype('uint64')
                sign_array = sign_array.reshape(-1).astype('int8')
                sign_array -= 1
                v = sign_array * xi_array * norm / s

                v = v[:real_size]
                v = v.reshape(code['shape'])
        else:
            raise ValueError('Whoops, implementation')
        v = torch.Tensor(v)
        if cuda:
            v = v.cuda()
        return v

    def _get_max_norm(self, codes):
        scalars = [code['norm'] for code in codes]
        return max(scalars)

    def encode_cuda(self, v, **kwargs):
        if isinstance(v, torch.cuda.FloatTensor):
            w = v.view(-1)
        else:
            raise ValueError("Object passed wasn't set on GUDA, please check CUDA availability!")

        #norm = LA.norm(v)
        norm = torch.norm(w)

        s = (1 << self._quantization_level) - 1
        shape = v.size()

        num_int_each_64_bits = int(64 / (2 + self._quantization_level)) # number of element stored / 64-bits
        num_section = num_int_each_64_bits
        len_each_section = int((w.shape[0] + num_section - 1) / num_section) # number of 64-bits to ship w vector

        w = F.pad(w, (0, len_each_section * num_section - w.size()[0]), 'constant', 0)

        sign_array = torch.sign(w)

        sign_array += 1 # -1, 0, 1 to 0, 1, 2
        #sign_array = sign_array.astype('uint64')
        sign_array = sign_array.to(dtype=torch.int64)

        normalization_array = torch.abs(w) / norm * s

        #truncated_array = normalization_array.astype(int) 
        truncated_array = normalization_array.to(dtype=torch.int) # l <= \frac{s \|w\|_i}{\|w\|_2} <= l+1
        
        prob_array = normalization_array - truncated_array.float() # \frac{s \|w\|_i}{\|w\|_2} - l i.e. p function p(a, s) = as - l
        
        dice_array = torch.rand(len(prob_array)).to(torch.device("cuda"))
        
        xi_array = truncated_array + (dice_array > prob_array).to(dtype=torch.int) # l+1 or l
        xi_array = xi_array.to(dtype=torch.int64)
        xi_array = xi_array.view((num_section, len_each_section))

        sign_array = sign_array.view((num_section, len_each_section))

        neo_array = torch.zeros(len_each_section).to(dtype=torch.int64).to(torch.device("cuda"))

        for i in range(num_int_each_64_bits):
            xi = xi_array[i]
            sign = sign_array[i]
            neo_array *= 2**(2 + self._quantization_level)
            sign *= 2**self._quantization_level
            sign += xi
            neo_array += sign.to(dtype=torch.int64)           

        code = {'neo': neo_array, 'norm': norm, 'quantization_level': self._quantization_level,
                'len_each_section': len_each_section, 'num_int_each_64_bits': num_int_each_64_bits,
                'shape': shape}
        return code


def grad_clip_limit(grad, clip_factor=2.5):
    """ Get the scalers."""
    if clip_factor > 1.0e-5:
        return clip_factor * np.std(grad.flat[:])
    return np.max(np.abs(grad.flat[:]))


if __name__ == "__main__":
    a_cpu = torch.randn(20)
    a_cuda = a_cpu.to(torch.device("cuda"))

    kwargs = {'quantization_level':8}
    coder = QSGD(bucket_size=0, **kwargs)
    print(a_cuda.dtype)
    code_cpu = coder.encode(a_cpu)
    code_cuda = coder.encode_cuda(a_cuda)
    print("CPU compression: {}, Type: {}".format(code_cpu['neo'], code_cpu['neo'].dtype))
    print("")
    print("CUDA compression: {}, Type: {}".format(code_cuda['neo'], code_cuda['neo'].dtype))