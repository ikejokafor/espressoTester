import sys
sys.path.append("/home/mdl/izo5011/downloaded_repos/x86/caffe-rfcn/python/")
import numpy as np
import caffe
import cv2
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path2deploy',
        required = False,
        default = "/home/mdl/izo5011/IkennaWorkSpace/caffeModels/rfcn_resnet50/rfcn_resnet50.prototxt",                    
        help = "Path to deploy.prototxt",
        type = str 
    )
    parser.add_argument(
        '--path2modelfile',
        required = False,
        default = "/home/mdl/izo5011/IkennaWorkSpace/caffeModels/rfcn_resnet50/rfcn_resnet50.caffemodel",                    
        help = "Path to .caffemodel",
        type = str 
    )
    parser.add_argument(
        '--beginLayer',
        required = False,
        default = " ",                    
        help = "Layer to start from",
        type = str 
    )
    parser.add_argument(
        '--endLayer',
        required = False,
        default = " ",                    
        help = "Layer to go up to",
        type = str 
    )
    args = parser.parse_args()
    return args


def main(args):
    # debug
    import pdb
    pdb.set_trace()
    
    path2deploy = args.path2deploy
    path2modelfile = args.path2modelfile
    beginLayer = args.beginLayer
    endLayer = args.endLayer

    net = caffe.Net(path2deploy, path2modelfile, caffe.TEST)
    netLayerHeight = net.blobs['data'].shape[2]
    netLayerWidth = net.blobs['data'].shape[3]
    img = cv2.imread("./dog.jpg")
    old_size = img.shape[:2] # old_size is in (height, width) format
    ratio = float(netLayerHeight) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = netLayerHeight - new_size[1]
    delta_h = netLayerWidth - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)  


    net.blobs['data'].data[...] = img.transpose(2,0,1).astype(float)

    # net_output = net.forward(start=beginLayer, end=endLayer)
    net_output = net.forward()
    quit()
    
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


if __name__ == "__main__":
    args = parse_args()
    main(args)
 
