1. symbol转prototxt: json2prototxt.py, 转出后手动把不支持的地方接上  
2. 权重转换：mxnet2caffe.py  
3. 如果想两步合一：mxnet2caffe_all_in_one.py，如果有不支持的层就不能2步合一了  
4. 如果需要调整输入顺序：caffe_swap_channel_toolkit.py  
5. 如果需要inplace操作：inplace_conv_block.py, 注意转换后可能需要手动微调
6. insert_mean_scale.py, 将均值和方差用Scale层的形式写入模型文件

English Guidelines:  
1. symbol file convert to prototxt: json2prototxt.py, if there are some layers skipped, modify the prototxt file mannualy
2. convert params: mxnet2caffe.py  
3. 1,2 steps in one: mxnet2caffe_all_in_one.py, however, if there are some layers skipped, we cannot use it.
4. swap input channel(bgr->rgb or vice versa): caffe_swap_channel_toolkit.py
5. inplace op: inplace_conv_block.py  
6. use Scale layer to process data mean and std: insert_mean_scale.py
