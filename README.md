1. symbol转prototxt: json2prototxt.py, 转出后手动把不支持的地方接上  
2. 权重转换：mxnet2caffe.py  
3. 如果想两步合一：mxnet2caffe_all_in_one.py，如果有不支持的层就不能2步合一了  
4. 如果需要调整输入顺序：caffe_swap_channel_toolkit.py  
5. 如果需要inplace操作：inplace_conv_block.py, 注意转换后可能需要手动微调
