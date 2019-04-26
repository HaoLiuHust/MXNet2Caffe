'''
Untest
'''
import numpy as np
import sys,os
sys.path.append("/home/liuhao/software/frameworks/caffe/build/install/python/")
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

src_model = '/home/liuhao/Projects/Projects/MXNet2Caffe/resnet50_512_112x112_20190320_lijixiao_10_bgr.prototxt'
dst_model = '/home/liuhao/Projects/Projects/MXNet2Caffe/resnet50_512_112x112_20190320_lijixiao_10_bgr_inplace.prototxt'

'''
 Set all BatchNorm and Scale ReLU as inplace op
'''
def inplace_conv_block(model):

    # load the prototxt file as a protobuf message
    with open(model) as k:
        str1 = k.read()
    msg1 = caffe_pb2.NetParameter()
    text_format.Merge(str1, msg1)

    replaced={}
    layers={}

    # search for bn and scale layer and remove them
    for i, l in enumerate(msg1.layer):

        for j,bottom in enumerate(msg1.layer[i].bottom):
            while True:
                if bottom in replaced and bottom!=replaced[bottom]:

                    bottom=replaced[bottom]

                else:
                    break

            msg1.layer[i].bottom[j] = bottom

        layers[msg1.layer[i].name]=l

        if l.type == "BatchNorm":
            bottom_l=layers[l.bottom[0]]

            if bottom_l.type=="ReLU" or bottom_l.type=="PReLU" or bottom_l.type=="Eltwise":
                continue

            replaced[msg1.layer[i].top[0]] = msg1.layer[i].bottom[0]

            print("inplace layer %s..." % l.name)
            msg1.layer[i].top.remove(msg1.layer[i].top[0])
            msg1.layer[i].top.append(msg1.layer[i].bottom[0])


            # if (i+1)<len(msg1.layer):
            #     msg1.layer[i+1].bottom.remove(msg1.layer[i+1].bottom[0])
            #     msg1.layer[i+1].bottom.append(msg1.layer[i].top[0])
        elif l.type == "Scale":
            replaced[msg1.layer[i].top[0]] = msg1.layer[i].bottom[0]

            print("inplace layer %s..." % l.name)
            msg1.layer[i].top.remove(msg1.layer[i].top[0])
            msg1.layer[i].top.append(msg1.layer[i].bottom[0])

            # if (i + 1) < len(msg1.layer):
            #     msg1.layer[i+1].bottom.remove(msg1.layer[i+1].bottom[0])
            #     msg1.layer[i+1].bottom.append(msg1.layer[i].top[0])
        elif l.type == "ReLU":
            replaced[msg1.layer[i].top[0]] = msg1.layer[i].bottom[0]

            print("inplace layer %s..." % l.name)
            msg1.layer[i].top.remove(msg1.layer[i].top[0])
            msg1.layer[i].top.append(msg1.layer[i].bottom[0])

            # if (i + 1) < len(msg1.layer):
            #     msg1.layer[i+1].bottom.remove(msg1.layer[i+1].bottom[0])
            #     msg1.layer[i+1].bottom.append(msg1.layer[i].top[0])



        # elif l.type == "Eltwise":
        #     if msg1.layer[i].bottom[0] in replaced:
        #         msg1.layer[i].bottom[0] = replaced[msg1.layer[i].bottom[0]]
        #     if msg1.layer[i].bottom[1] in replaced:
        #         msg1.layer[i].bottom[1] = replaced[msg1.layer[i].bottom[1]]

    return msg1

model_inplace = inplace_conv_block(src_model)

# save prototxt for inference
print "Saving inplace prototxt file..."
path = os.path.join(dst_model)
with open(path, 'w') as m:
    m.write(text_format.MessageToString(model_inplace))
