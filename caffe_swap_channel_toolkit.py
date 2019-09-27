import mxnet as mx
from collections import namedtuple
import os
import numpy as np
import math
import os, sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "/home/liuhao/software/frameworks/caffe/build/install/python/"))
import caffe
import argparse

def parser_args():
  parser = argparse.ArgumentParser(description='swap channel ')
  parser.add_argument('--cf-prototxt', type=str, default='')
  parser.add_argument('--layer-name', type=str, default='')
  parser.add_argument('--save-cf-prototxt', type=str, default='')
  return parser.parse_args()


args=parser_args()

print args

net_file = args.cf_prototxt
net_model = net_file.replace(".prototxt",".caffemodel")

try:
    net_caffe = caffe.Net(net_file,net_model,caffe.TEST)
except Exception:
    print "read model failed\n"
    exit(-1)

if not args.layer_name in net_caffe.params:
    print "layer not exist\n"
    exit(-1)

net_caffe.params[args.layer_name][0].data[...]=net_caffe.params[args.layer_name][0].data[...][:,::-1,:,:]
# with open("/home/liuhao/CLionProjects/FeatureExtractor/model/keypoint_rgb.prototxt",'w') as f:
#     f.write(str(net_caffe.to_proto()))

net_caffe.save(args.save_cf_prototxt.replace(".prototxt",".caffemodel"))
os.system("cp %s %s"%(net_file,args.save_cf_prototxt))

print("done\n")