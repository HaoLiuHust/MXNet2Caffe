1. symbol转prototxt: json2prototxt.py, 转出后手动把不支持的地方接上  
2. 权重转换：mxnet2caffe.py  
3. 两步合一：mxnet2caffe_all_in_one.py  
4. 调整输入顺序：caffe_swap_channel_toolkit.py  
5. inplace操作：inplace_conv_block.py, 注意转换后可能需要手动微调
