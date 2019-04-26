import sys
import os
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "/home/liuhao/CLionProjects/ModelServing/servings/3rdparty/nvcaffe/python"))
import caffe

import numpy as np
import cv2
import mxnet as mx
import pickle

caffe.set_device(2)
caffe.set_mode_gpu()

net_file = '/home/liuhao/Projects/Projects/MXNet2Caffe/resnet100_512.prototxt'
net_model = '/home/liuhao/Projects/Projects/MXNet2Caffe/resnet100_512.caffemodel'

net = caffe.Net(net_file,net_model,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data',(2,0,1))

#transformer.set_raw_scale('data', 255.0)
#transformer.set_channel_swap('data', (2, 0, 1))
net.blobs['data'].reshape(1,3,112,112)

test_img=cv2.imread("/home/liuhao/dataset/112x112/Aaron_Eckhart/Aaron_Eckhart_1.png")

#test_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)

caffe_img = test_img.astype(np.float32)
caffe_img -= 127.5
caffe_img /= 128

net.blobs['data'].data[...] = transformer.preprocess('data', caffe_img)
blobs=net.blobs

output = net.forward()
feature = output['fc1'][0]

data_shape = [('data',[1,3,112,112])]


# sym, arg_params, aux_params = mx.model.load_checkpoint("/home/liuhao/dataset/FaceModel/model",0)
#
# # mx.viz.plot_network(sym, shape={'data': (1, 3, 112, 112), 'softmax_label': (1,)}, \
# #                     node_attrs={'shape': 'oval', 'fixedsize': 'fasl==false'}).view()
#
# # #all layer_name
# # layer_names=sym.get_internals().list_outputs()
# # layer_group=sym.get_internals()
#
# # groups=[]
# # names=[]
# # for name in layer_names:
# #         if name.endswith("output"):
# #                 groups.append(layer_group[name])
# #                 names.append(name)
# #
# # sym_group=mx.sym.Group(groups)
#
#
# ctx=mx.gpu(2)
#
# model = mx.mod.Module(
#         context=ctx,
#         symbol=sym,
# label_names=None)
#
# #dataset = lfw.load_pkl('/home/liuhao/dataset/lfw_info/90d24_90_112x112.pkl')
#
# model.bind(data_shapes=data_shape,for_training=False)
# model.set_params(arg_params,aux_params,allow_missing = True)
#
# from collections import namedtuple
# data = namedtuple("Batch",['data'])
#
# input_nd=mx.nd.array(test_img)
# input_nd=mx.nd.transpose(input_nd,(2,0,1))
#
# input_nd=mx.nd.expand_dims(input_nd,axis=0)
#
# model.forward(data([input_nd]))
#
# outputs = model.get_outputs()[0].asnumpy()
#
# print("edi dis %0.6f"%(np.sqrt(np.sum(np.square(outputs-feature)))))
#
print("test")