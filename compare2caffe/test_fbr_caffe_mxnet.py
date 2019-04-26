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
net_file = '/home/liuhao/CLionProjects/ucnn2/tools/tensorrt_converter/fbr1105_bgr_trt.prototxt'
net_model = '/home/liuhao/CLionProjects/ucnn2/tools/tensorrt_converter/fbr1105_bgr_trt.caffemodel'

net_caffe = caffe.Net(net_file,net_model,caffe.TEST)

sym,arg_params,aux_params=mx.model.load_checkpoint("/home/liuhao/samba/96/UniubiProjects/AIOT_demo/model_frb/1105fbr",4)
data_shape=[('data',[1,3,128,128])]

internals = sym.get_internals()
fc1 = internals['fc1_output']
fc2 = internals['softmax2_output']
fc3 = internals['fc3_output']
group = mx.symbol.Group([fc1,fc2, fc3])

mod = mx.module.Module(symbol=group,label_names=None,context=mx.gpu(6))

mod.bind(data_shape,label_shapes=None,for_training=False)
mod.set_params(arg_params,aux_params,allow_missing=True)


mean_pixels = (127.5, 127.5, 127.5)

def adjust_data(img,data_shape):
    #batch_data = np.zeros((1, 3, data_shape, data_shape))
    _mean_pixels = np.array(mean_pixels).reshape((3, 1, 1))


    data = cv2.resize(np.array(img), (data_shape, data_shape))
    data = np.transpose(data, (2, 0, 1))
    data = data.astype('float32')
    _data = (data - _mean_pixels)

    return _data

def adjust_data_G(img,data_shape):
    batch_data = mx.nd.zeros((1, 3, data_shape,data_shape))
    _mean_pixels=mx.nd.array(mean_pixels).reshape((3,1,1))
    data=cv2.resize(img,(data_shape, data_shape))
    data =mx.nd.array(data)
    data = mx.nd.transpose(data, (2,0,1))
    data = data.astype('float32')
    _data = data - _mean_pixels
    batch_data[0] = _data
    test_iter = mx.nd.array(batch_data)
    return test_iter

img_dir="/home/liuhao/store/liuhao/work/fbr_test/"
img_lst=os.listdir(img_dir)
img_lst.sort()
for img_name in img_lst:
    img=cv2.imread(os.path.join(img_dir,img_name))
    if img is None:
        continue
    img_c=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    Batch = namedtuple('Batch',['data'])
    data_shape_G = 128
    data = adjust_data_G(img_c,data_shape_G)
    mod.forward(Batch([data]))
    prob_race = mod.get_outputs()[1].asnumpy()
    prob_beauty = mod.get_outputs()[2].asnumpy()
    prob_beauty=np.squeeze(prob_beauty)

    beauty_result = min(10 * (max(-5, prob_beauty[1]) + 11), 100)

    net_caffe.blobs['data'].reshape(1, 3, 128, 128)
    caffe_img=adjust_data(img,data_shape_G)
    net_caffe.blobs['data'].data[...] = caffe_img
    output = net_caffe.forward()
    race = output['softmax2'][0]
    beauty = net_caffe.blobs['fc3'].data[...]
    beauty = np.squeeze(beauty)

    beauty_caffe = min(10 * (max(-5, beauty[1]) + 11), 100)

    print "test"
