import sys
import argparse
import json
from prototxt_basic import *

def parser_args():
  parser = argparse.ArgumentParser(description='Convert MXNet jason to Caffe prototxt')
  parser.add_argument('--mx-json',     type=str, default='../model-r50-am-lfw/model-symbol.modified.json')
  parser.add_argument('--cf-prototxt', type=str, default='../model-r50-am-lfw/model.prototxt')
  parser.add_argument('--data-shape', type=int,nargs=3, default=[3,96,96])

  return parser.parse_args()

def json2prototxt(args):
  with open(args.mx_json) as json_file:
    jdata = json.load(json_file)

  valid_layers=set()
  last_valid_layers=None

  with open(args.cf_prototxt, "w") as prototxt_file:
    for i_node in range(0,len(jdata['nodes'])):
      node_i    = jdata['nodes'][i_node]
      if str(node_i['op']) == 'null' and str(node_i['name']) != 'data':
        continue

      print('{}, \top:{}, name:{} -> {}'.format(i_node,node_i['op'].ljust(20),
                                          node_i['name'].ljust(30),
                                          node_i['name']).ljust(20))
      info = node_i

      info['top'] = info['name']
      info['bottom'] = []
      info['params'] = []
      for input_idx_i in node_i['inputs']:
        input_i = jdata['nodes'][input_idx_i[0]]
        if str(input_i['op']) != 'null' or (str(input_i['name']) == 'data'):
          if str(input_i['name']) in valid_layers:
            info['bottom'].append(str(input_i['name']))
          else:
            info['bottom'].append(last_valid_layers)

        if str(input_i['op']) == 'null':
          info['params'].append(str(input_i['name']))
          if not str(input_i['name']).startswith(str(node_i['name'])):
            print('           use shared weight -> %s'% str(input_i['name']))
            info['share'] = True

      if(write_node(prototxt_file, info, args.data_shape)):
        valid_layers.add(info['name'])
        last_valid_layers=info['name']

if __name__=="__main__":
  cmd_args = parser_args()
  json2prototxt(cmd_args)