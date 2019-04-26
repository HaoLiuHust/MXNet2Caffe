import mxnet as mx
from collections import namedtuple
import os
import numpy as np
import math
import os, sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "/home/liuhao/software/frameworks/caffe/build/install/python/"))
import caffe

net_file = '/home/liuhao/Projects/Projects/MXNet2Caffe/mobileface_13_256_20190320_lijixiao_18.prototxt'
net_model = net_file.replace(".prototxt",".caffemodel")

net_caffe = caffe.Net(net_file,net_model,caffe.TEST)



# with open(r"angle.bin",'wb') as f:
#     for layer in net_caffe.params:
#         param=net_caffe.params[layer][0].data
#         f.write(param.data)


# net_caffe.params['bn_data'][0].data[...]=net_caffe.params['bn_data'][0].data[...][::-1]
net_caffe.params['conv_1_conv2d'][0].data[...]=net_caffe.params['conv_1_conv2d'][0].data[...][:,::-1,:,:]
# with open("/home/liuhao/CLionProjects/FeatureExtractor/model/keypoint_rgb.prototxt",'w') as f:
#     f.write(str(net_caffe.to_proto()))

net_caffe.save("/home/liuhao/Projects/Projects/MXNet2Caffe/mobileface_13_256_20190320_lijixiao_18_bgr.caffemodel")
os.system("cp %s %s"%(net_file,"/home/liuhao/Projects/Projects/MXNet2Caffe/mobileface_13_256_20190320_lijixiao_18_bgr.prototxt"))

print("test")