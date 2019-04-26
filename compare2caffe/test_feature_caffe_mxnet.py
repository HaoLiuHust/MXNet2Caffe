import mxnet as mx
from collections import namedtuple
import os
import numpy as np
import math
import cv2
import os, sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append("/home/liuhao/software/frameworks/caffe/build/install/python")
import caffe

caffe.set_mode_cpu()
net_file = '/home/liuhao/samba/96/UniubiProjects/nnie_mapper/models/feature_small/meta/mobileface_256_112x112_20190121_1M_bgr_inplace.prototxt'
net_model = '/home/liuhao/samba/96/UniubiProjects/nnie_mapper/models/feature_small/meta/mobileface_256_112x112_20190121_1M_bgr.caffemodel'

net_caffe = caffe.Net(net_file,net_model,caffe.TEST)

sym,arg_params,aux_params=mx.model.load_checkpoint("/home/liuhao/store/liuhao/releasemodel/mobileface_256_112x112_20190121_1M",0)
data_shape=[('data',[1,3,112,112])]

mod = mx.module.Module(symbol=sym,label_names=None,context=mx.gpu(0))

mod.bind(data_shape,label_shapes=None,for_training=False)
mod.set_params(arg_params,aux_params,allow_missing=True)

mean_pixels = (127.5, 127.5, 127.5)

def adjust_data(img,data_shape):
    #batch_data = np.zeros((1, 3, data_shape, data_shape))
    _mean_pixels = np.array(mean_pixels).reshape((3, 1, 1))

    data = cv2.resize(np.array(img), (data_shape, data_shape))
    data = np.transpose(data, (2, 0, 1))
    data = data.astype('float32')
    _data = (data - _mean_pixels) / 128
    return _data

def adjust_data_G(img,data_shape):
    batch_data = mx.nd.zeros((1, 3, data_shape,data_shape))
    data=cv2.resize(img,(data_shape, data_shape))
    data =mx.nd.array(data)
    data = mx.nd.transpose(data, (2,0,1))
    data = data.astype('float32')
    _data = data
    batch_data[0] = _data
    test_iter = mx.nd.array(batch_data)
    return test_iter

img_dir="/home/liuhao/store/liuhao/Evaluation/PrivateTestSet/ForID/IDTESTALL_align/00ce4d583417db936ae34499e8dcbebd/"
img_lst=os.listdir(img_dir)
img_lst.sort()
for img_name in img_lst:
    if not img_name.endswith(".bmp"):
        continue
    img=cv2.imread(os.path.join(img_dir,img_name))
    if img is None:
        continue
    img_c=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    Batch = namedtuple('Batch',['data'])
    data_shape_G = 112
    data = adjust_data_G(img_c,data_shape_G)
    mod.forward(Batch([data]))
    feature_mxnet = mod.get_outputs()[0].asnumpy()

    net_caffe.blobs['data'].reshape(1, 3, 112, 112)
    caffe_img=adjust_data(img,data_shape_G)
    net_caffe.blobs['data'].data[...] = caffe_img
    output = net_caffe.forward()
    feature_caffe = output['fc1'][0]

    print "test"
