import sys
sys.path.insert(0,"/home/liuhao/framework/sphereface/tools/caffe-sphereface/python/")

import caffe
import os
import numpy as np
import cv2
import pickle

caffe.set_device(2)
caffe.set_mode_gpu()

net_file = '/home/liuhao/framework/sphereface/train/code/sphereface_deploy.prototxt'
net_model = '/home/liuhao/framework/sphereface/train/result/MyFace/sphereface_model_finetune_iter_3000.caffemodel'

net = caffe.Net(net_file,net_model,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data',(2,0,1))

#transformer.set_raw_scale('data', 255.0)
#transformer.set_channel_swap('data', (2, 0, 1))
net.blobs['data'].reshape(1,3,112,96)

f = open("/home/liuhao/dataset/feature_mean_MyFace.txt",'r')
mean_f = f.readlines()
mean_arr = np.array(mean_f,dtype=float)
f.close()

#load database
database_path = '/home/liuhao/dataset/image112x96/'
database_picke = '/home/liuhao/dataset/faceMyFace.pkl'

face_db={}
if os.path.exists(database_picke):
    face_db = pickle.load(open(database_picke,'r'))
    load_from_file = True

img_lst=[]
if len(face_db) ==0 :
    img_lst = os.listdir(database_path)
    load_from_file = False

def extract_feature(net,test_img):
    test_img = test_img.astype(np.float32)
    test_img -= 127.5
    test_img /= 128

    net.blobs['data'].data[...] = transformer.preprocess('data', test_img)
    output = net.forward()
    feature = output['fc5'][0]

    test_flip = cv2.flip(test_img, 1)

    net.blobs['data'].data[...] = transformer.preprocess('data', test_flip)
    output = net.forward()
    feature_flip = output['fc5'][0]

    feature_con = np.concatenate((feature, feature_flip))
    return feature_con

for img_path in img_lst:
    img = cv2.imread(os.path.join(database_path,img_path))

    face_db[img_path]=extract_feature(net,img)-mean_arr

if load_from_file is not True:
    with open(database_picke,'w') as pf:
        pickle.dump(face_db,pf)

test_path = '/home/liuhao/dataset/blur_face'
test_lst = os.listdir(test_path)
test_lst.sort()
for test_img_path in test_lst:
    if test_img_path.find('face') > -1:
        print test_img_path
        test_img = cv2.imread(os.path.join(test_path, test_img_path))
        test_feature = extract_feature(net, test_img)-mean_arr

        names = []
        dis_lst = []
        for k, v in face_db.items():
            cosdistance = np.dot(v,test_feature) / (
                np.linalg.norm(v) * np.linalg.norm(test_feature) + 1e-5)
            # if cosdistance > 0.45:
            #     print k, cosdistance

            names.append(k)
            dis_lst.append(cosdistance)

        max_value = max(dis_lst)
        max_index = dis_lst.index(max_value)
        print names[max_index], max_value
        print '---------------------'

        #test_body = cv2.imread(os.path.join(test_path, test_img_path.replace('face', 'body')))
        cv2.imshow('face', test_img)
        #cv2.imshow('body', test_body)
        cv2.waitKey()




