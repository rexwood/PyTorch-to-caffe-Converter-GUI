import caffe
from numpy import random, ascontiguousarray, float64
from numpy.linalg import norm
import torch


def verify(name, model_define_path, input_path, save_path, batch_size, channel, height, width):
    __import__(model_define_path.split('/')[-1].split('.')[0])
    model = torch.load(input_path)
    model.eval()
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    test_data = random.randn(batch_size, channel, height, width)
    test_data = ascontiguousarray(test_data, dtype=float64)
    print(test_data.shape)
    torch.set_printoptions(precision=7)
    pred = model(torch.Tensor(test_data))
    print(pred.size())

    net = caffe.Classifier('{}.prototxt'.format(save_path + name), '{}.caffemodel'.format(save_path + name), caffe.TEST)
    net.blobs['data'].reshape(batch_size, channel, height, width)
    net.blobs['data'].data[...] = test_data
    net.forward()
    caffe_pred = []
    for i in range(len(net.outputs)):
        caffe_pred.append(net.blobs[net.outputs[i]].data)
        print(net.blobs[net.outputs[i]].data.shape)
        # print(net.blobs[net.outputs[i]].data[:])   #1 32 1000

    sum = 0.0
    for i in range(len(pred)):
        sum += norm(caffe_pred[i] - pred[i].detach().numpy()) / norm(pred[i].detach().numpy()) * 100

    # print('output%d have %10.10f%% relative error'%(i, np.linalg.norm(caffe_pred[0] - pred.detach().numpy()) / np.linalg.norm(pred[0].detach().numpy())*100))
    # print(caffe_pred[0])
    # print(pred)
    return sum


if __name__ == '__main__':
    verify('resnet', '/home/rex/桌面/P2C_GUI/resnet.pth', '/home/rex/桌面/P2C_GUI/', 1, 3, 224, 224)
