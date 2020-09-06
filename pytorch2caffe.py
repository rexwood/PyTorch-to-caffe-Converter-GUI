"""
project name:           pytorch2caffe converter
function:               convert deep learning network(pytorch-->caffe) without offset.
supported layers:       conv2d, conv2d_transpose, linear, max_pool2d, avgpool_2d, adaptive_avgpool2d,
                        relu, leaky_relu, batch_norm, dropout, softmax,
supported operations:   view, flatten, cat and common torch operations.
description:            This is the 1st version of the converter. Haven't test the converting offset yet.
references:             https://github.com/xxradon/PytorchToCaffe
                        https://github.com/hahnyuan/nn_tools
package version:        pytorch - 1.4.0
                        caffe - 1.0: https://github.com/BVLC/caffe
tested examples:        alexnet, inception, resnet,darknet53, mobilenetV2, vggnet
"""

import torch
import torch.nn.functional as F
import numpy as np
from Caffe import caffe_net
from Caffe.caffe_net import Layer_param
from torch.nn.modules.utils import _pair

NET_INITED = False
WARNING_STRINGS = ''
RP_TRANSFERRING_FLAG = False  # this flag prevents transferring Rp function in Rp function.


class Blob_LOG:
    def __init__(self):
        self.data = {}

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)


class TransLog(object):
    def __init__(self):
        self.layers = {}  # name of the layers
        self.detail_layers = {}  # detail of the net, #of each kind of layer
        self.detail_blobs = {}
        self._blobs = Blob_LOG()  # Record the blobs
        self._blob_data = []
        self.cnet = caffe_net.Caffemodel('')  # the caffenet
        self.debug = False
        self.pytorch_layer_name = None

    def init(self, inputs):
        '''

        :param inputs: is a list of input variables
        '''
        self.layers['data'] = 'data'
        self.add_blobs(inputs, 'data', False)

    def add_blobs(self, blobs, name='blob', with_num=True):
        rst = []
        for blob in blobs:
            self._blob_data.append(blob)
            blob_id = int(id(blob))
            # record blobs details
            if name not in self.detail_blobs.keys():
                self.detail_blobs[name] = 0
            self.detail_blobs[name] += 1

            if with_num:
                rst.append('{}{}'.format(name, self.detail_blobs[name]))
            else:
                rst.append('{}'.format(name))

            if self.debug:
                print("{}:{} was added to blobs".format(blob_id, rst[-1]))
            print('Add blob {} : {}'.format(rst[-1].center(21), blob.size()))
            self._blobs[blob_id] = rst[-1]
        return rst

    def add_layer(self, name='layer'):
        if name in self.layers:
            return self.layers[name]
        if name not in self.detail_layers.keys():
            self.detail_layers[name] = 0
        self.detail_layers[name] += 1
        name = '{}{}'.format(name, self.detail_layers[name])
        self.layers[name] = name  # layer type and number
        if self.debug:
            print("{} was added to layers".format(self.layers[name]))
        return self.layers[name]

    def blobs(self, var):
        var = id(var)
        if self.debug:
            print("{}:{} getting".format(var, self._blobs[var]))

        try:
            return self._blobs[var]
        except:
            print("Warning: cannot found blob {}".format(var))
            return None


log = TransLog()
layer_names = {}


# core replacement function
class Rp(object):
    def __init__(self, raw, replace, **kwargs):
        self.obj = replace
        self.raw = raw

    def __call__(self, *args, **kwargs):
        print("call RP function:", str(self.raw))
        global RP_TRANSFERRING_FLAG
        if RP_TRANSFERRING_FLAG:
            return self.raw(*args, **kwargs)
        RP_TRANSFERRING_FLAG = True
        '''
        if not NET_INITED:
            return self.raw(*args, **kwargs)
        '''
        '''
        for stack in traceback.walk_stack(None):
            if 'self' in stack[0].f_locals:
                layer = stack[0].f_locals['self']
                # print("layer in stack[0]:", layer)
                if layer in layer_names:
                    log.pytorch_layer_name = layer_names[layer]
                    print("Processing Layer:" + layer_names[layer])
                    break
        '''
        out = self.obj(self.raw, *args, **kwargs)
        RP_TRANSFERRING_FLAG = False
        outlist.append(out)
        return out


# replacement functions for layers:


def _conv2d(raw, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    print("converting conv layer,bottom:", log.blobs(input))
    x = raw(input, weight, bias, stride, padding, dilation, groups)
    name = log.add_layer(name='conv')
    log.add_blobs([x], name='conv_blob')
    layer = Layer_param(name=name, type="Convolution",
                        bottom=[log.blobs(input)], top=[log.blobs(x)])
    layer.conv_param(x.size()[1], weight.size()[2:], stride=_pair(stride),
                     pad=_pair(padding), dilation=_pair(dilation), bias_term=bias is not None, groups=groups)
    # 传输权值
    if bias is not None:
        layer.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
    else:
        layer.param.convolution_param.bias_term = False
        layer.add_data(weight.cpu().data.numpy())
    log.cnet.add_layer(layer)  # add to our caffe net
    return x


def _conv_transpose2d(raw, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    print("converting deconv layer, bottom:", log.blobs(input))
    x = raw(input, weight, bias, stride, padding, output_padding, groups, dilation)
    name = log.add_layer(name='conv_transpose')
    log.add_blobs([x], name='conv_transpose_blob')
    layer = Layer_param(name=name, type="Deconvolution",
                        bottom=[log.blobs(input)], top=[log.blobs(x)])
    layer.conv_param(x.size()[1], weight.size()[2:], stride=_pair(stride),
                     pad=_pair(padding), dilation=_pair(dilation), bias_term=bias is not None, groups=groups)

    if bias is not None:
        layer.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
    else:
        layer.param.convolution_param.bias_term = False
        layer.add_data(weight.cpu().data.numpy())
    log.cnet.add_layer(layer)
    return x


def _linear(raw, input, weight, bias=None):
    print("converting linear layer, bottom:", log.blobs(input))
    x = raw(input, weight, bias)
    name = log.add_layer(name='fc')
    log.add_blobs([x], name='fc_blob')
    layer = Layer_param(name=name, type='InnerProduct',
                        bottom=[log.blobs(input)], top=[log.blobs(x)])
    print(x.size())
    layer.fc_param(x.size()[-1],
                   has_bias=bias is not None)

    if bias is not None:
        layer.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
    else:
        layer.add_data(weight.cpu().data.numpy())  # with torch.no_grad():
    log.cnet.add_layer(layer)
    return x


def _pool(type, raw, input, x, kernel_size, stride, padding, ceil_mode):
    print("converting pooling layer, bottom:", log.blobs(input))
    name = log.add_layer(name='{}_pool'.format(type))
    log.add_blobs([x], name='{}_pool_blob'.format(type))
    layer = Layer_param(name=name, type='Pooling',
                        bottom=[log.blobs(input)], top=[log.blobs(x)])
    layer.pool_param(kernel_size=kernel_size, stride=kernel_size if stride is None else stride,
                     pad=padding, type=type.upper(), ceil_mode=ceil_mode)
    log.cnet.add_layer(layer)

    # processing ceil mode:


def _max_pool2d(raw, input, kernel_size, stride=None, padding=0, dilation=1,
                return_indices=False, ceil_mode=False):
    x = raw(input, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    _pool('max', raw, input, x, kernel_size, stride, padding, ceil_mode)
    return x


def _avg_pool2d(raw, input, kernel_size, stride=None, padding=0,
                ceil_mode=False, count_include_pad=True):
    x = raw(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    _pool('ave', raw, input, x, kernel_size, stride, padding, ceil_mode)
    return x


def _adaptive_avg_pool2d(raw, input, output_size):
    print("this is adaptive pooling layer")
    x = raw(input, output_size)
    if isinstance(output_size, int):
        out_dim = output_size
    else:
        out_dim = output_size[0]
    tmp = max(input.shape[2], input.shape[3])
    stride = tmp // out_dim
    kernel_size = tmp - (out_dim - 1) * stride
    _pool('ave', raw, input, x, kernel_size, stride, 0, False)
    return x


def _relu(raw, input, inplace=False):
    print("converting relu layer, bottom:", log.blobs(input))
    x = raw(input, False)
    name = log.add_layer(name='relu')
    log.add_blobs([x], name='relu_blob')
    layer = Layer_param(name=name, type='ReLU',
                        bottom=[log.blobs(input)], top=[log.blobs(x)])
    log.cnet.add_layer(layer)
    return x


def _leaky_relu(raw, input, negative_slope=0.01, inplace=False):
    print("converting leaky_relu, bottom:", log.blobs(input))
    x = raw(input, negative_slope)
    name = log.add_layer(name='leaky_relu')
    log.add_blobs([x], name='leaky_relu_blob')
    layer = caffe_net.Layer_param(name=name, type='ReLU',
                                  bottom=[log.blobs(input)], top=[log.blobs(x)])
    layer.param.relu_param.negative_slope = negative_slope
    log.cnet.add_layer(layer)
    return x


def _hardtanh(raw, input, min_val, max_val, inplace):
    print("converting relu6 layer buttom:", log.blobs(input))
    x = raw(input, min_val, max_val)
    name = log.add_layer(name='relu6')
    log.add_blobs([x], name='relu6_blob')
    layer = Layer_param(name=name, type='ReLU6',
                        bottom=[log.blobs(input)], top=[log.blobs(x)])
    log.cnet.add_layer(layer)
    return x


def _threshold(raw, input, threshold, value, inplace=False):
    # for threshold or relu
    if threshold == 0 and value == 0:
        x = raw(input, threshold, value, False)
        name = log.add_layer(name='relu')
        log.add_blobs([x], name='relu_blob')
        layer = caffe_net.Layer_param(name=name, type='ReLU',
                                      bottom=[log.blobs(input)], top=[log.blobs(x)])
        log.cnet.add_layer(layer)
        return x
    if value != 0:
        raise NotImplemented("value !=0 not implemented in caffe")
    x = raw(input, input, threshold, value, False)
    bottom_blobs = [log.blobs(input)]
    layer_name = log.add_layer(name='threshold')
    top_blobs = log.add_blobs([x], name='threshold_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Threshold',
                                  bottom=bottom_blobs, top=top_blobs)
    layer.param.threshold_param.threshold = threshold
    log.cnet.add_layer(layer)
    return x


def _batch_norm(raw, input, running_mean, running_var, weight=None, bias=None,
                training=False, momentum=0.1, eps=1e-5):
    # because the runing_mean and runing_var will be changed after the _batch_norm operation,
    # we first save the parameters

    x = raw(input, running_mean, running_var, weight, bias,
            training, momentum, eps)
    bottom_blobs = [log.blobs(input)]
    layer_name1 = log.add_layer(name='batch_norm')
    top_blobs = log.add_blobs([x], name='batch_norm_blob')
    layer1 = caffe_net.Layer_param(name=layer_name1, type='BatchNorm',
                                   bottom=bottom_blobs, top=top_blobs)
    if running_mean is None or running_var is None:
        # not use global_stats, normalization is performed over the current mini-batch
        layer1.batch_norm_param(use_global_stats=0, eps=eps)
    else:
        layer1.batch_norm_param(use_global_stats=1, eps=eps)
        running_mean_clone = running_mean.clone()
        running_var_clone = running_var.clone()
        layer1.add_data(running_mean_clone.cpu().numpy(), running_var_clone.cpu().numpy(), np.array([1.0]))
    log.cnet.add_layer(layer1)
    if weight is not None and bias is not None:
        layer_name2 = log.add_layer(name='bn_scale')
        layer2 = caffe_net.Layer_param(name=layer_name2, type='Scale',
                                       bottom=top_blobs, top=top_blobs)
        layer2.param.scale_param.bias_term = True
        layer2.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
        log.cnet.add_layer(layer2)
    print("out batch_norm")
    return x


def _dropout(raw, input, p=0.5, training=True, inplace=False):
    print("converting dropout layer, bottom:", log.blobs(input))
    x = raw(input, p, training, inplace)
    bottom_blobs = [log.blobs(input)]
    layer_name = log.add_layer(name='dropout')
    top_blobs = log.add_blobs([x], name=bottom_blobs[0], with_num=False)
    layer = caffe_net.Layer_param(name=layer_name, type='Dropout',
                                  bottom=bottom_blobs, top=top_blobs)
    layer.param.dropout_param.dropout_ratio = p
    layer.param.include.extend([caffe_net.pb.NetStateRule(phase=0)])  # 1 for test, 0 for train
    log.cnet.add_layer(layer)
    return x


def get_softmax_dim(name, ndim, stacklevel):
    if ndim == 0 or ndim == 1 or ndim == 3:
        ret = 0
    else:
        ret = 1
    return ret


def _softmax(raw, input, dim=None, _stacklevel=3):
    # for F.softmax
    x = raw(input, dim=dim)
    if dim is None:
        dim = get_softmax_dim('softmax', input.dim(), _stacklevel)

    print("converting softmax layer:")
    name = log.add_layer(name='softmax')
    log.add_blobs([x], name='softmax_blob')
    layer = Layer_param(name=name, type='Softmax',
                        bottom=[log.blobs(input)], top=[log.blobs(x)])
    layer.param.softmax_param.axis = dim
    log.cnet.add_layer(layer)
    return x


# for torch.Tensor operations:

def _view(input, *args):
    x = raw_tensor_op['view'](input, *args)
    if not NET_INITED:
        return x
    name = log.add_layer(name='view')
    top_blobs = log.add_blobs([x], name='view_blob')
    layer = Layer_param(name=name, type='Reshape',
                        bottom=[log.blobs(input)], top=top_blobs)
    dims = list(args)
    dims[0] = 0  # the first dim should be batch_size
    layer.param.reshape_param.shape.CopyFrom(caffe_net.pb.BlobShape(dim=dims))
    log.cnet.add_layer(layer)
    return x


def _flatten(input, *args, **kwargs):
    print("converting flatten layer, bottom:", log.blobs(input))
    print("input size:", input.size())
    x = raw_tensor_op['flatten'](input, *args)
    if not NET_INITED:
        return x
    print("_flatten size:", x.size())
    print("args", list(args))
    print("kargs", kwargs)
    print("lailailaibababa")
    name = log.add_layer(name='flatten')
    top_blobs = log.add_blobs([x], name='flatten_blob')
    layer = Layer_param(name=name, type='Flatten',
                        bottom=[log.blobs(input)], top=top_blobs)
    arg = list(args)
    # to do add param
    log.cnet.add_layer(layer)
    return x


def _mean(input, *args, **kwargs):
    x = raw_tensor_op['mean'](input, *args, **kwargs)
    if not NET_INITED:
        return x
    layer_name = log.add_layer(name='mean')
    top_blobs = log.add_blobs([x], name='mean_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Reduction',
                                  bottom=[log.blobs(input)], top=top_blobs)
    if len(args) == 1:
        dim = args[0]
    elif 'dim' in kwargs:
        dim = kwargs['dim']
    else:
        raise NotImplementedError('mean operation must specify a dim')
    if dim != len(input.size()) - 1:
        raise NotImplementedError('mean in Caffe Reduction Layer: only reduction along ALL "tail" axes is supported')
    if kwargs.get('keepdim'):
        raise NotImplementedError('mean operation must keep_dim=False')
    layer.param.reduction_param.operation = 4
    layer.param.reduction_param.axis = dim
    log.cnet.add_layer(layer)
    return x


def _sum(input, *args, **kwargs):
    x = raw_tensor_op['sum'](input, *args, **kwargs)
    if not NET_INITED:
        return x
    layer_name = log.add_layer(name='sum')
    top_blobs = log.add_blobs([x], name='sum_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Reduction',
                                  bottom=[log.blobs(input)], top=top_blobs)
    if len(args) == 1:
        dim = args[0]
    elif 'dim' in kwargs:
        dim = kwargs['dim']
    else:
        raise NotImplementedError('sum operation must specify a dim')
    if dim != len(input.size()) - 1:
        raise NotImplementedError('sum in Caffe Reduction Layer: only reduction along ALL "tail" axes is supported')
    if kwargs.get('keepdim'):
        raise NotImplementedError('sum operation must keep_dim=False')
    layer.param.reduction_param.operation = 1  # operation 1 for sum
    layer.param.reduction_param.axis = dim
    log.cnet.add_layer(layer)
    return x


def _add(input, *args):
    return ___add__(input, *args)


def _sub(input, *args):
    return ___sub__(input, *args)


def _mul(input, *args):
    return ___mul__(input, *args)


def _div(input, *args):
    return ___div__(input, *args)


def _pow(input, *args):
    return ___pow__(input, *args)


def _sqrt(input, *args):
    x = raw_tensor_op['sqrt'](input, *args)
    if not NET_INITED:
        return x
    layer_name = log.add_layer(name='sqrt')
    top_blobs = log.add_blobs([x], name='sqrt_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Power',
                                  bottom=[log.blobs(input)], top=top_blobs)
    layer.param.power_param.power = 0.5
    log.cnet.add_layer(layer)
    return x


def ___add__(input, *args):
    x = raw_tensor_op['__add__'](input, *args)
    if not NET_INITED:
        return x
    layer_name = log.add_layer(name='add')
    top_blobs = log.add_blobs([x], name='add_blob')
    if not isinstance(args[0], torch.Tensor):
        layer = caffe_net.Layer_param(name=layer_name, type='Power',
                                      bottom=[log.blobs(input)], top=top_blobs)
        layer.param.power_param.shift = args[0]
    else:
        layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                      bottom=[log.blobs(input), log.blobs(args[0])], top=top_blobs)
        layer.param.eltwise_param.operation = 1  # sum is 1
    log.cnet.add_layer(layer)
    return x


def ___iadd__(input, *args):
    x = raw_tensor_op['__iadd__'](input, *args)
    if not NET_INITED:
        return x
    x = x.clone()
    layer_name = log.add_layer(name='add')
    top_blobs = log.add_blobs([x], name='add_blob')
    if not isinstance(args[0], torch.Tensor):
        layer = caffe_net.Layer_param(name=layer_name, type='Power',
                                      bottom=[log.blobs(input)], top=top_blobs)
        layer.param.power_param.shift = args[0]
    else:
        layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                      bottom=[log.blobs(input), log.blobs(args[0])], top=top_blobs)
        layer.param.eltwise_param.operation = 1  # sum is 1
    log.cnet.add_layer(layer)
    return x


def ___sub__(input, *args):
    x = raw_tensor_op['__sub__'](input, *args)
    if not NET_INITED:
        return x
    layer_name = log.add_layer(name='sub')
    top_blobs = log.add_blobs([x], name='sub_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[log.blobs(input), log.blobs(args[0])], top=top_blobs)
    layer.param.eltwise_param.operation = 1  # sum is 1
    layer.param.eltwise_param.coeff.extend([1., -1.])
    log.cnet.add_layer(layer)
    return x


def ___isub__(input, *args):
    x = raw_tensor_op['__isub__'](input, *args)
    if not NET_INITED:
        return x
    x = x.clone()
    layer_name = log.add_layer(name='sub')
    top_blobs = log.add_blobs([x], name='sub_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[log.blobs(input), log.blobs(args[0])], top=top_blobs)
    layer.param.eltwise_param.operation = 1  # sum is 1
    log.cnet.add_layer(layer)
    return x


def ___mul__(input, *args):
    x = raw_tensor_op['__mul__'](input, *args)
    if not NET_INITED:
        return x
    layer_name = log.add_layer(name='mul')
    top_blobs = log.add_blobs([x], name='mul_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[log.blobs(input), log.blobs(args[0])], top=top_blobs)
    layer.param.eltwise_param.operation = 0  # product is 1
    log.cnet.add_layer(layer)
    return x


def ___imul__(input, *args):
    x = raw_tensor_op['__imul__'](input, *args)
    if not NET_INITED:
        return x
    x = x.clone()
    layer_name = log.add_layer(name='mul')
    top_blobs = log.add_blobs([x], name='mul_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[log.blobs(input), log.blobs(args[0])], top=top_blobs)
    layer.param.eltwise_param.operation = 0  # product is 1
    layer.param.eltwise_param.coeff.extend([1., -1.])
    log.cnet.add_layer(layer)
    return x


def ___div__(input, *args):
    x = raw_tensor_op['__div__'](input, *args)
    if not NET_INITED:
        return x
    if not isinstance(args[0], torch.Tensor):
        layer_name = log.add_layer(name='div')
        top_blobs = log.add_blobs([x], name='div_blob')
        layer = caffe_net.Layer_param(name=layer_name, type='Power',
                                      bottom=[log.blobs(input)], top=top_blobs)
        layer.param.power_param.scale = 1 / args[0]
        log.cnet.add_layer(layer)
    else:
        pre_layer_name = log.add_layer(name='pre_div')
        pre_div_blobs = log.add_blobs([x], name='pre_div_blob')
        pre_layer = caffe_net.Layer_param(name=pre_layer_name, type='Power',
                                          bottom=[log.blobs(input)], top=pre_div_blobs)
        pre_layer.param.power_param.power = -1
        pre_layer.param.power_param.shift = 1e-6
        log.cnet.add_layer(pre_layer)
        layer_name = log.add_layer(name='div')
        top_blobs = log.add_blobs([x], name='div_blob')
        layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                      bottom=[pre_div_blobs[0], log.blobs(args[0])], top=top_blobs)
        layer.param.eltwise_param.operation = 0  # product is 1
        log.cnet.add_layer(layer)
    return x


def ___truediv__(input, *args): return ___div__(input, *args)


def ___pow__(input, *args):
    x = raw_tensor_op['__pow__'](input, *args)
    if not NET_INITED:
        return x
    if not isinstance(args[0], int):
        raise NotImplementedError('power only support int now in nn_tools')
    layer_name = log.add_layer(name='power')
    top_blobs = log.add_blobs([x], name='power_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Power',
                                  bottom=[log.blobs(input)], top=top_blobs)
    layer.param.power_param.power = args[0]  # product is 1
    log.cnet.add_layer(layer)
    return x


# torch functions:
def torch_max(raw, *args):
    assert NotImplementedError
    x = raw(*args)
    if len(args) == 1:
        # TODO max in one tensor
        assert NotImplementedError
    else:
        if isinstance(x, tuple):
            x = x[0]
        bottom_blobs = []
        for arg in args:
            bottom_blobs.append(log.blobs(arg))
        layer_name = log.add_layer(name='max')
        top_blobs = log.add_blobs([x], name='max_blob')
        layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                      bottom=bottom_blobs, top=top_blobs)
        layer.param.eltwise_param.operation = 2
        log.cnet.add_layer(layer)
    return x


def torch_cat(raw, inputs, dimension=0):
    x = raw(inputs, dimension)
    if not NET_INITED:
        return x
    bottom_blobs = []
    for input in inputs:
        bottom_blobs.append(log.blobs(input))
    layer_name = log.add_layer(name='cat')
    top_blobs = log.add_blobs([x], name='cat_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Concat',
                                  bottom=bottom_blobs, top=top_blobs)
    layer.param.concat_param.axis = dimension
    log.cnet.add_layer(layer)
    return x


def torch_split(raw, tensor, split_size, dim=0):
    # split in pytorch is slice in caffe
    x = raw(tensor, split_size, dim)
    layer_name = log.add_layer('split')
    top_blobs = log.add_blobs(x, name='split_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Slice',
                                  bottom=[log.blobs(tensor)], top=top_blobs)
    slice_num = int(np.floor(tensor.size()[dim] / split_size))
    slice_param = caffe_net.pb.SliceParameter(axis=dim, slice_point=[split_size * i for i in range(1, slice_num)])
    layer.param.slice_param.CopyFrom(slice_param)
    log.cnet.add_layer(layer)
    return x


def torch_add(raw, *args):
    x = raw(*args)
    if not NET_INITED:
        return x
    layer_name = log.add_layer(name='add')
    top_blobs = log.add_blobs([x], name='add_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[log.blobs(input), log.blobs(args[0])], top=top_blobs)
    layer.param.eltwise_param.operation = 1  # sum is 1
    log.cnet.add_layer(layer)
    return x


def torch_sub(raw, *args):
    return ___sub__(*args)


def torch_mul(raw, *args):
    return ___mul__(*args)


def torch_div(raw, *args):
    return ___div__(*args)


def torch_pow(raw, *args):
    x = raw(*args)
    if not NET_INITED:
        return x
    if not isinstance(args[0], int):
        raise NotImplementedError('power only support int now in nn_tools')
    layer_name = log.add_layer(name='power')
    top_blobs = log.add_blobs([x], name='power_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Power',
                                  bottom=[log.blobs(input)], top=top_blobs)
    layer.param.power_param.power = args[0]  # product is 1
    log.cnet.add_layer(layer)
    return x


def torch_sqrt(raw, *args):
    x = raw(*args)
    if not NET_INITED:
        return x
    if not isinstance(args[0], int):
        raise NotImplementedError('sqrt only support int now in nn_tools')
    layer_name = log.add_layer(name='sqrt')
    top_blobs = log.add_blobs([x], name='sqrt_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Power',
                                  bottom=[log.blobs(input)], top=top_blobs)
    layer.param.power_param.power = 0.5
    log.cnet.add_layer(layer)
    return x


def torch_flatten(raw, input, *args, **kwargs):
    print("converting flatten layer, bottom:", log.blobs(input))
    print("input size:", input.size())
    x = raw(input, *args)
    print("_flatten size:", x.size())
    print("args", list(args))
    print("kargs", kwargs)
    if not NET_INITED:
        return x
    print("lailailaibababa")
    name = log.add_layer(name='flatten')
    top_blobs = log.add_blobs([x], name='flatten_blob')
    layer = Layer_param(name=name, type='Flatten',
                        bottom=[log.blobs(input)], top=top_blobs)
    arg = list(args)
    # to do add param
    log.cnet.add_layer(layer)
    return x


def op_placeholder(raw, *args, **kwargs):
    output = raw(*args, **kwargs)
    bottom_blobs = []
    warning_string = "======\nCRITICAL WARN: layer {} cannot be transfer, " \
                     "because it cannot be implemented with original version of caffe\n" \
                     "Nn_tools place a placeholder with Python type layer in caffe. \n======".format(
        str(raw).split(' ')[1])
    # print(warning_string)
    global WARNING_STRINGS
    print('askfdjlakskdfjlaskdjflkdsjaflksjdflaksjdfl')
    WARNING_STRINGS += warning_string
    for arg in args:
        if isinstance(arg, torch.Tensor):
            try:
                bottom_blobs.append(log.blobs(arg))
            except:
                print("WARN: at op_placehoder, tensor {} is not in the graph".format(arg))
    output_blobs = []
    if isinstance(output, tuple):
        for out in output:
            output_blobs.append(out)
    else:
        output_blobs.append(output)
    top_blobs = log.add_blobs(output_blobs, name='op_placehoder_blob')
    layer_name = log.add_layer(name='op_placehoder')
    layer = caffe_net.Layer_param(name=layer_name, type='Python',
                                  bottom=bottom_blobs, top=top_blobs)
    log.cnet.add_layer(layer)
    return output


outlist = []

F_supported = [
    'conv2d',
    'conv_transpose2d',
    'linear',
    'relu',
    'hardtanh',  # relu6
    'leaky_relu',
    'threshold',
    'max_pool2d',
    'avg_pool2d',
    'adaptive_avg_pool2d',
    'dropout',
    'batch_norm',
    'softmax',

]

torch_op_supported = [
    'split',
    'max',
    'cat',
    'add',
    # 'sub',
    # 'mul',
    # 'div',
    # 'pow',
    # 'sqrt',
    'flatten',
]

tensor_op_supported = [
    'view',
    'flatten',
    'add',
    'sub',
    'mul',
    'div',
    'pow',
    'sqrt',
    'sum',
    'mean',
    '__add__',
    '__iadd__',
    '__sub__',
    '__isub__',
    '__mul__',
    '__imul__',
    '__div__',
    '__truediv__',
    '__pow__',
]

# replace in raw function with the new one
# layers:
for op_name in F.__dict__:
    if op_name in F_supported:
        raw_func = getattr(F, op_name)
        transfer_func = globals()['_' + op_name]
        op_wrapper = Rp(raw_func, transfer_func)
        setattr(F, op_name, op_wrapper)
    else:
        if op_name[0] == '_' or op_name in ['division,', 'warnings', 'math', 'torch', 'utils', 'vision', 'Col2Im',
                                            'Im2Col', 'grad', 'weak_script', 'List']:
            continue
        setattr(F, op_name, Rp(getattr(F, op_name), op_placeholder))

# ops in torch.Tensor:
raw_tensor_op = {}
for op_name in tensor_op_supported:
    raw_op = getattr(torch.Tensor, op_name)
    raw_tensor_op[op_name] = raw_op
    setattr(torch.Tensor, op_name, globals()['_' + op_name])

# ops in torch:
for op_name in torch_op_supported:
    raw_op = getattr(torch, op_name)
    op_wrapper = Rp(raw_op, globals()['torch_' + op_name])
    setattr(torch, op_name, op_wrapper)


def trans_net(net, input_var, name='transferedPytorchModel'):
    print("Starting the transform. This will take a while")
    print("type of input:", type(input_var))
    log.init([input_var])
    log.cnet.net.name = name
    # log.cnet.net.input.extend([log.blobs(input_var)])
    # log.cnet.net.input_dim.extend(input_var.size())
    layer = caffe_net.Layer_param(name='data', type='Input', top=['data'])
    layer.input_param(input_var.data.numpy().shape)
    log.cnet.add_layer(layer)
    global NET_INITED
    NET_INITED = True

    for name, layer in net.named_modules():
        layer_names[layer] = name

    out = net.forward(input_var)
    RP_TRANSFERRING_FLAG = True

    print("Transform completed")
    print(WARNING_STRINGS)
    signal = True
    if WARNING_STRINGS != '':
        signal = False
    return signal


def save_prototxt(save_name):
    log.cnet.save_prototxt(save_name)


def save_caffemodel(save_name):
    log.cnet.save(save_name)
