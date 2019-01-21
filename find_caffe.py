try:
    import caffe
except ImportError:
    import os, sys
    #sys.path.append("/home/liuhao/software/frameworks/nvcaffe/build/install/python/")
    sys.path.append("/home/liuhao/software/frameworks/caffe/build/install/python/")
    import caffe
