import torch
import pytorch2caffe
import sys
import os


def convert(model_define_path, pth_path, save_path, batch_size, channel, height, width):
    try:
        rootPath0 = os.path.split(model_define_path)[0]
        # print(rootPath0)
        sys.path.append(rootPath0)
    except:
        pass
    rootPath = os.path.split(pth_path)[0]
    # print(rootPath)
    sys.path.append(rootPath)
    pytorch2caffe.RP_TRANSFERRING_FLAG = False
    pytorch2caffe.log = pytorch2caffe.TransLog()
    pytorch2caffe.layer_names = {}
    list_pth_path = pth_path.split("/")
    model_path = list_pth_path[-1]
    # print(model_path)
    name = model_path.split(".")[0]
    # moxing dingyi wenjian de daoru
    # print(model_define_path)

    try:
        __import__(model_define_path.split('/')[-1].split('.')[0])
    except:
        pass

    # print(pth_path)
    model = torch.load(pth_path)


    try:
        model.eval()
    except:
        pass
    input = torch.ones([batch_size, channel, height, width])
    print("input shape:", input.shape)
    # print(model.state_dict())
    signal = pytorch2caffe.trans_net(model, input, name)
    pytorch2caffe.save_prototxt(save_path + '{}.prototxt'.format(name))
    pytorch2caffe.save_caffemodel(save_path + "{}.caffemodel".format(name))
    pytorch2caffe.RP_TRANSFERRING_FLAG = True
    pytorch2caffe.NET_INITED = False
    return name, signal


if __name__ == '__main__':
    convert('darknet53.pth', '', 1, 3,
            224, 224)
