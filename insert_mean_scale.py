import numpy as np
import sys,os
# sys.path.append("/home/liuhao/software/frameworks/caffe/build/install/python/")
import find_caffe
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import copy
import argparse

parser = argparse.ArgumentParser(description='Convert MXNet model to Caffe model')
parser.add_argument('--in-prototxt', type=str, default='../model-r50-am-lfw/model.prototxt')
parser.add_argument('--out-prototxt', type=str, default='../model-r50-am-lfw/model.prototxt')
parser.add_argument('--mean', type=float,nargs=3, default=[0.,0.,0])
parser.add_argument('--std', type=float,nargs=3, default=[1.,1.,1.])


args = parser.parse_args()

src_model = args.in_prototxt
dst_model = args.out_prototxt

with open(src_model) as k:
    str1 = k.read()
msg1 = caffe_pb2.NetParameter()
text_format.Merge(str1, msg1)
msg2 = caffe_pb2.NetParameter()

pre_norm_layer=None
layers=[]

for i,l in enumerate(msg1.layer):
    layers.append(l)
    if l.type=="Scale" and pre_norm_layer is None:
        pre_norm_layer=copy.deepcopy(l)
        pre_norm_layer.name='pre_norm'
        pre_norm_layer.bottom.remove(l.bottom[0])
        pre_norm_layer.bottom.append(msg1.layer[0].top[0])
        pre_norm_layer.top.remove(l.top[0])
        pre_norm_layer.top.append('pre_norm')

msg1.layer[1].bottom.remove(msg1.layer[1].bottom[0])
msg1.layer[1].bottom.append(pre_norm_layer.top[0])

layers.insert(1,pre_norm_layer)

for l in layers:
    l1 = msg2.layer.add()
    l1.CopyFrom(l)

with open(dst_model, 'w') as m:
    m.write(text_format.MessageToString(msg2))

net_caffe = caffe.Net(dst_model,src_model.replace(".prototxt",".caffemodel"),caffe.TEST)
data_channel = net_caffe.params['pre_norm'][1].num
mean=-np.array(args.mean[0:data_channel])
std=1/np.array(args.std[0:data_channel])
mean=mean*std

net_caffe.params['pre_norm'][1].data.flat=np.array(mean).flat
net_caffe.params['pre_norm'][0].data.flat=np.array(std).flat

net_caffe.save(dst_model.replace(".prototxt",".caffemodel"))


print("done")