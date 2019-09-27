import argparse

import caffe_pb2
from google.protobuf import text_format

def convert_prototxt(fname, output):
    proto = caffe_pb2.NetParameter()
    with open(fname, 'r') as f:
        text_format.Merge(str(f.read()), proto)

    plugin_type = ["ELU", "PReLU"]
    count_elu = 0
    count_prelu = 0
    count_lrelu = 0
    count_slice = 0
    for layer in proto.layer:
        if layer.type == "ELU":
            count_elu += 1
            modify_elu_layer(layer, count_elu)
        elif layer.type == "PReLU":
            count_prelu += 1
            modify_prelu_layer(layer, count_prelu)
        elif layer.type == "ReLU" and layer.relu_param.negative_slope:
            count_lrelu += 1
            modify_lrelu_layer(layer, count_lrelu)
        elif layer.type == "Slice":
            count_slice += 1
            modify_slice_layer(layer, count_slice)
        else:
            pass

    with open(output + '.prototxt', 'w') as f:
        f.write(str(proto))

def convert_caffemodel(fname, output):
    proto = caffe_pb2.NetParameter()
    with open(fname, "r") as f:
        proto.ParseFromString(f.read())

    plugin_type = ["ELU", "PReLU"]    
    count_elu = 0
    count_prelu = 0
    for layer in proto.layer:
        if layer.type == "ELU":
            count_elu += 1
            modify_elu_weight(layer, count_elu)
        elif layer.type == "PReLU":
            count_prelu += 1
            modify_prelu_weight(layer, count_prelu)
        else:
            pass


    with open(output + '.caffemodel', 'wb') as f:
        f.write(proto.SerializeToString())

def modify_elu_layer(layer, count_elu):
    # layer.type = "IPlugin"
    elu_param = layer.elu_param.alpha
    layer.name = "ELU" + str(count_elu) + "_" + str(elu_param)

def modify_slice_layer(layer, count_slice):
    # layer.type = "IPlugin"
    top_num=len(layer.top)
    axis=layer.slice_param.axis
    layer.name = "Slice" + str(count_slice)+"_"+str(axis)+"_"+str(top_num)

prelu_layers = {}
def modify_prelu_layer(layer, count_prelu):
    # layer.type = "IPlugin"
    new_name = "PReLU" + str(count_prelu)
    prelu_layers[layer.name] = new_name
    layer.name = new_name
    # channel_shared = layer.prelu_param.channel_shared
    # layer.name = "PReLU" + str(count_prelu) + "_" + str(channel_shared)

def modify_lrelu_layer(layer, count_lrelu):
    negative_slope = layer.relu_param.negative_slope
    layer.name = "LReLU" + str(count_lrelu) + "_" + str(negative_slope)

# ELU layer has no parameters in weight file
# So no need to modify
def modify_elu_weight(layer, count_elu): 
    pass

def modify_prelu_weight(layer, count_prelu):
    if(layer.name in prelu_layers):
        layer.name = prelu_layers[layer.name]
    else:
        print layer.name + "    Found unknown layer in weight file"



def main():
    parser = argparse.ArgumentParser(
        description='Caffe model converter.')
    parser.add_argument('predix', help='The caffe model prefix')
    parser.add_argument('output', help='The output prefix')
    args = parser.parse_args()

    convert_prototxt(args.predix + '.prototxt', args.output)
    convert_caffemodel(args.predix + '.caffemodel', args.output)

if __name__ == '__main__':
    main()
    print "TensorRT Model is generated."
