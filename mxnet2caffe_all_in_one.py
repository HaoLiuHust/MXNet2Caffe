import sys, argparse
import find_mxnet, find_caffe
import mxnet as mx
import caffe
from json2prototxt import json2prototxt
import easydict

parser = argparse.ArgumentParser(description='Convert MXNet model to Caffe model')
parser.add_argument('--mx-model',    type=str, default='../model-r50-am-lfw/model')
parser.add_argument('--mx-epoch',    type=int, default=0)
parser.add_argument('--cf-prototxt', type=str, default='../model-r50-am-lfw/model.prototxt')
parser.add_argument('--data-shape', type=int,nargs=3, default=[3,96,96])
args = parser.parse_args()
args.cf_model=args.cf_prototxt.replace(".prototxt",".caffemodel")
print args.data_shape

json2prototxt_args=easydict.EasyDict()
json2prototxt_args.mx_json=args.mx_model+"-symbol.json"
json2prototxt_args.cf_prototxt=args.cf_prototxt
json2prototxt_args.data_shape=args.data_shape

json2prototxt(json2prototxt_args)

# ------------------------------------------
# Load
_, arg_params, aux_params = mx.model.load_checkpoint(args.mx_model, args.mx_epoch)
net = caffe.Net(args.cf_prototxt, caffe.TRAIN)

# ------------------------------------------
# Convert
all_keys = arg_params.keys() + aux_params.keys()
all_keys.sort()

print('----------------------------------\n')
print('ALL KEYS IN MXNET:')
print(all_keys)
print('%d KEYS' %len(all_keys))

print('----------------------------------\n')
print('VALID KEYS:')
for i_key,key_i in enumerate(all_keys):

  try:    
    
    if 'data' is key_i:
      pass
    elif '_weight' in key_i:
      key_caffe = key_i.replace('_weight','')
      net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat      
    elif '_bias' in key_i:
      key_caffe = key_i.replace('_bias','')
      net.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat   
    elif '_gamma' in key_i and 'relu' not in key_i:
      key_caffe = key_i.replace('_gamma','_scale')
      net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
    # TODO: support prelu
    elif '_gamma' in key_i and 'relu' in key_i:   # for prelu
      key_caffe = key_i.replace('_gamma','')
      assert (len(net.params[key_caffe]) == 1)
      net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
    elif '_beta' in key_i:
      key_caffe = key_i.replace('_beta','_scale')
      net.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat    
    elif '_moving_mean' in key_i:
      key_caffe = key_i.replace('_moving_mean','')
      net.params[key_caffe][0].data.flat = aux_params[key_i].asnumpy().flat 
      net.params[key_caffe][2].data[...] = 1 
    elif '_moving_var' in key_i:
      key_caffe = key_i.replace('_moving_var','')
      net.params[key_caffe][1].data.flat = aux_params[key_i].asnumpy().flat    
      net.params[key_caffe][2].data[...] = 1
    elif '_running_mean' in key_i:
      key_caffe = key_i.replace('_running_mean','')
      net.params[key_caffe][0].data.flat = aux_params[key_i].asnumpy().flat
      net.params[key_caffe][2].data[...] = 1
    elif '_running_var' in key_i:
      key_caffe = key_i.replace('_running_var','')
      net.params[key_caffe][1].data.flat = aux_params[key_i].asnumpy().flat
      net.params[key_caffe][2].data[...] = 1

    else:
      sys.exit("Warning!  Unknown mxnet:{}".format(key_i))
  
    print("% 3d | %s -> %s, initialized." 
           %(i_key, key_i.ljust(40), key_caffe.ljust(30)))
    
  except KeyError:
    #try gluon style
    try:

      if 'data' is key_i:
        pass
      elif '_weight' in key_i:
        key_caffe = key_i.replace('_weight', '_fwd')
        net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
      elif '_bias' in key_i:
        key_caffe = key_i.replace('_bias', '_fwd')
        net.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat
      elif '_gamma' in key_i and 'relu' not in key_i:
        key_caffe = key_i.replace('_gamma', '_fwd_scale')
        net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
      # TODO: support prelu
      elif '_gamma' in key_i and 'relu' in key_i:  # for prelu
        key_caffe = key_i.replace('_gamma', '')
        assert (len(net.params[key_caffe]) == 1)
        net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
      elif '_beta' in key_i:
        key_caffe = key_i.replace('_beta', '_fwd_scale')
        net.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat
      elif '_moving_mean' in key_i:
        key_caffe = key_i.replace('_moving_mean', '')
        net.params[key_caffe][0].data.flat = aux_params[key_i].asnumpy().flat
        net.params[key_caffe][2].data[...] = 1
      elif '_moving_var' in key_i:
        key_caffe = key_i.replace('_moving_var', '')
        net.params[key_caffe][1].data.flat = aux_params[key_i].asnumpy().flat
        net.params[key_caffe][2].data[...] = 1
      elif '_running_mean' in key_i:
        key_caffe = key_i.replace('_running_mean', '_fwd')
        net.params[key_caffe][0].data.flat = aux_params[key_i].asnumpy().flat
        net.params[key_caffe][2].data[...] = 1
      elif '_running_var' in key_i:
        key_caffe = key_i.replace('_running_var', '_fwd')
        net.params[key_caffe][1].data.flat = aux_params[key_i].asnumpy().flat
        net.params[key_caffe][2].data[...] = 1

      else:
        sys.exit("Warning!  Unknown mxnet:{}".format(key_i))

      print("% 3d | %s -> %s, initialized."
            % (i_key, key_i.ljust(40), key_caffe.ljust(30)))

    except KeyError:
      print("\nError!  key error mxnet:{}".format(key_i))
      pass
    pass
      
# ------------------------------------------
# Finish
net.save(args.cf_model)
print("\n- Finished.\n")

