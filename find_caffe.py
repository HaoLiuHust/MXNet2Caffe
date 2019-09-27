# try:
#     import caffe
# except ImportError:
#     import os, sys
#     # sys.path.append("/home/liuhao/software/frameworks/nvcaffe/build/install/python/")
#     sys.path.append("/home/liuhao/software/frameworks/caffe/build/install/python/")
#     import caffe

import os, sys
# sys.path.append("/home/liuhao/software/frameworks/nvcaffe/build/install/python/")
sys.path.append("/home/liuhao/software/frameworks/caffe/build/install/python/")
print "start import caffe"
try:
    import caffe
except Exception:
    sys.path.append("/home/liuhao/samba/149/software/frameworks/caffe/build/install/python/")
    import caffe

print "end import caffe"
