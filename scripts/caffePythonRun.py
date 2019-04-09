# OverFeat
#
# SSD
#
# YOLO
#
# Squeezenet
# python caffePython.py ../../../caffe/models/squeezenet/deploy.prototxt ../../../caffe/models/squeezenet/squeezenet_v1.1.caffemodel data prob
#
# Squeezenet(Ristretto)
# python caffePython.py ../../../ristretto/models/SqueezeNet/quantized.prototxt ../../../ristretto/models/SqueezeNet/squeezenet_finetuned.caffemodel data prob
#
# VGG16
# python caffePython.py ../../../caffe/models/vgg16/vgg16_deploy.prototxt ../../../caffe/models/vgg16/VGG_ILSVRC_16_layers.caffemodel data prob 
#
# VGG19
# python caffePython.py ../../../caffe/models/vgg19/vgg16_deploy.prototxt ../../../caffe/models/vgg19/VGG_ILSVRC_16_layers.caffemodel data prob 
#
# GoogleNet
# python caffePython.py ../../../caffe/models/bvlc_googlenet/deploy.prototxt ../../../caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel data prob 
#
# Alexnet
# python caffePython.py ../../../caffe/models/bvlc_alexnet/deploy.prototxt ../../../caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel data prob 
#
# dcNet
# python caffePython.py ../../../caffe/models/dcNet/deploy_sqz_2.prototxt ../../../caffe/models/dcNet/sqz_rework_iter_100000.caffemodel data objectness0_soft
# 
# SSD
# python caffePython.py ../../../ssd/models/SSD/deploy.prototxt ../../../ssd/models/SSD/VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.caffemodel start end

import glob
import sys
import os
import ntpath
import xml.etree.ElementTree as ET
#sys.path.append('/home/ikenna/caffe/python/')
#sys.path.append('/home/ikenna/ssd/python/')
#sys.path.append('/home/ikenna/ristretto/python/')
#sys.path.append('/opt/caffe/python/')    
import numpy as np
import caffe
import cv2
import math
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path2deploy',
                        help="Path to deploy.prototxt")
    parser.add_argument('path2modelfile',
                        help="Path to .caffemodel")
    args = parser.parse_args()
    return args
    
def maxIndex(list):
    i = 0
    idx = 0
    max = float('-inf')
    while i < list.shape[1]:
        if list.item(0, i) > max:
            max = list.item(0, i)
            idx = i
        i = i + 1
    return idx
    
def max5Index(list):
    i = 0
    idx0 = 0
    idx1 = 0
    idx2 = 0
    idx3 = 0
    idx4 = 0
    max0 = float('-inf')
    max1 = float('-inf')
    max2 = float('-inf')
    max3 = float('-inf')
    max4 = float('-inf')
    while i < list.shape[1]:
        if list.item(0, i) > max0:
            max = list.item(0, i)
            idx0 = i
        elif list.item(0, i) > max1:
            max = list.item(0, i)
            idx = i
        
        i = i + 1
    return [idx0, idx1, idx2, idx3, idx4]

if __name__ == "__main__":

    # Begin Code ------------------------------------------------------------------------------------------------------------------------------------
    args = parse_args()
    path2deploy = args.path2deploy
    path2modelfile = args.path2modelfile   
    net = caffe.Net(path2deploy, path2modelfile, caffe.TEST)
    # End Code --------------------------------------------------------------------------------------------------------------------------------------
    
    
    fileList = glob.glob("/shared/ILSVRC2012_img_val/*.JPEG")
    netLayerHeight = net.blobs['data'].shape[2]
    netLayerWidth = net.blobs['data'].shape[3]
    with open("/shared/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt") as f:
        groundTruth = f.readlines()
    groundTruth = [x.strip() for x in groundTruth] 
    
    i = 0
    while i < len(fileList):
        # Begin Code ------------------------------------------------------------------------------------------------------------------------------------
        imgFileName = fileList[i]
        imgFileBaseName = ntpath.basename(imgFileName)
        xmlFileName = "/shared/ILSVRC2012_bbox_val_v3/val/" + os.path.splitext(imgFileBaseName)[0] + ".xml"
        xmlData = ET.parse(xmlFileName)
        xmlDataRoot = xmlData.getroot()
        xmin = int(xmlDataRoot[5][4][0].text)
        ymin = int(xmlDataRoot[5][4][1].text)
        xmax = int(xmlDataRoot[5][4][2].text)
        ymax = int(xmlDataRoot[5][4][3].text)
        gtClass = groundTruth[int(imgFileBaseName[15:23])]
        
        print(gtClass)
        
        img = cv2.imread(imgFileName);      
        img = img[ymin : ymin + (ymax - ymin), xmin : xmin + (xmax - xmin)]
        
        old_size = img.shape[:2] # old_size is in (height, width) format
        ratio = float(netLayerHeight)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = netLayerHeight - new_size[1]
        delta_h = netLayerWidth - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        color = [0, 0, 0]
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)     
        # End Code --------------------------------------------------------------------------------------------------------------------------------------

        
        # Begin Code ------------------------------------------------------------------------------------------------------------------------------------  
        net.blobs['data'].reshape(1,3,img.shape[0],img.shape[1])
        net.blobs['data'].data[...] = img.transpose(2,0,1).astype(float) - 128.0
        net_output = net.forward()
        # End Code --------------------------------------------------------------------------------------------------------------------------------------

        print type(net_output['prob'])
        quit()
        
        queryClass = maxIndex(net_output['prob'])
        print(queryClass)
        
        i = i + 1

