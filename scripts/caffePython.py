import sys
#sys.path.append('/home/ikenna/caffe/python/')
#sys.path.append('/home/ikenna/ssd/python/')
#sys.path.append('/home/ikenna/ristretto/python/')
import numpy as np
import caffe
import cv2
import math
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path2deploy',
                        help="Path to deploy.prototxt")
    parser.add_argument('path2modelfile',
                        help="Path to .caffemodel")
    parser.add_argument('beginLayer',
                    help="Layer to start from")
    parser.add_argument('endLayer',
                    help="Layer to go up to")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

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
    # python caffePython.py ../../../caffe/models/vgg16/vgg16_deploy.prototxt ../../../caffe/models/vgg16/VGG_ILSVRC_16_layers.caffemodel conv1_1 prob 
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
 
    
    args = parse_args()
    path2deploy = args.path2deploy
    path2modelfile = args.path2modelfile
    beginLayer = args.beginLayer
    endLayer = args.endLayer

    
    net = caffe.Net(path2deploy, path2modelfile, caffe.TEST)
    frame = cv2.imread('/home/ikenna/detector_test_kitti/KITTI/000000.png')
    frame = cv2.resize(frame, (net.blobs['data'].shape[2], net.blobs['data'].shape[3]), cv2.INTER_LINEAR)
    cv2.imwrite('image.png', frame);
    net.blobs['data'].reshape(1,3,frame.shape[0],frame.shape[1])
    net.blobs['data'].data[...] = frame.transpose(2,0,1).astype(float)

    start = time.time()
    net_output = net.forward(start=beginLayer, end=endLayer)
    end = time.time()
    print(end - start)  
    #quit()
    
    print net_output[endLayer].shape 
    print net_output[endLayer].ndim
    #fh = open('../build/output_golden.txt', 'w')
    fh = open('output_golden.txt', 'w')
    if(net_output[endLayer].ndim == 4):
        for i in range(0, net_output[endLayer].shape[1]):
            for j in range(0, net_output[endLayer].shape[2]):
                for k in range(0, net_output[endLayer].shape[3]):
                    fh.write(str(net_output[endLayer].item(0, i, j, k)) + ' ')
                fh.write('\n')
            if(net_output[endLayer].shape[3] != 1 and net_output[endLayer].shape[2] != 1):
                fh.write('\n')
                fh.write('\n')
                fh.write('\n')
    elif (net_output[endLayer].ndim == 2) :
        for i in range(0, net_output[endLayer].shape[1]):
            fh.write(str(net_output[endLayer].item(0, i)) + '\n')

    fh.close()      
    quit()

