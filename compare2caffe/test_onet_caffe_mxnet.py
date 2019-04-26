import mxnet as mx
from collections import namedtuple
import os
import numpy as np
import math
import cv2
import os, sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append("/home/liuhao/software/frameworks/nvcaffe/build/install/python")
import caffe

caffe.set_mode_cpu()
net_file = '/home/liuhao/store/liuhao/work/snapper_for_zhinuo_todo/MobilenetV2-48_0.25_5pts_eyeaug_small_size_BlurandShelterAug_Correct_5pts_lr0.010000/onet_48_260k.prototxt'
net_model = '/home/liuhao/store/liuhao/work/snapper_for_zhinuo_todo/MobilenetV2-48_0.25_5pts_eyeaug_small_size_BlurandShelterAug_Correct_5pts_lr0.010000/onet_48_260k.caffemodel'

net_caffe = caffe.Net(net_file,net_model,caffe.TEST)

sym,arg_params,aux_params=mx.model.load_checkpoint\
    ("/home/liuhao/store/liuhao/work/snapper_for_zhinuo_todo/MobilenetV2-48_0.25_5pts_eyeaug_small_size_BlurandShelterAug_Correct_5pts_lr0.010000/onet",14)
data_shape=[('data',[1,3,48,48])]


mod = mx.module.Module(symbol=sym,label_names=None,context=mx.gpu(6))

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
    _mean_pixels=mx.nd.array(mean_pixels).reshape((3,1,1))
    data=cv2.resize(img,(data_shape, data_shape))
    data =mx.nd.array(data)
    data = mx.nd.transpose(data, (2,0,1))
    data = data.astype('float32')

    _data = (data - _mean_pixels) / 128

    batch_data[0] = _data
    test_iter = mx.nd.array(batch_data)
    return test_iter

img_dir="/home/liuhao/store/liuhao/work/snapper_for_zhinuo_todo/test_img/"
img_lst=os.listdir(img_dir)
img_lst.sort()
for img_name in img_lst:
    img=cv2.imread(os.path.join(img_dir,img_name))
    if img is None:
        continue
    # img_c=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    Batch = namedtuple('Batch',['data'])
    data_shape_G = 48
    data = adjust_data_G(img,data_shape_G)
    mod.forward(Batch([data]))
    prob_race = mod.get_outputs()[0].asnumpy()

    net_caffe.blobs['data'].reshape(1, 3, data_shape_G, data_shape_G)
    caffe_img=adjust_data(img,data_shape_G)
    net_caffe.blobs['data'].data[...] = caffe_img
    output = net_caffe.forward()

    beauty = net_caffe.blobs['fc3'].data[...]
    beauty = np.squeeze(beauty)

    print("test")
