import sys
sys.path.append("/export/home/izo5011/downloaded_repos/opencv_4_5_0/build/python_loader/")
sys.path.append("/home/mdl/izo5011/downloaded_repos/x86/caffe-rfcn/python/")
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


def main(args):   
    import pdb
    pdb.set_trace()

    net = caffe.Net(
        "/home/mdl/izo5011/IkennaWorkSpace/caffeModels/mobileNetSSD/mobileNetSSD.prototxt", 
        "/home/mdl/izo5011/IkennaWorkSpace/caffeModels/mobileNetSSD/mobileNetSSD.caffemodel", 
        caffe.TEST
    )
    all_names = [n for n in net._layer_names]
    
    for layNm in all_names:
        lay = net[layNm]



if __name__ == "__main__":
    argv = parseArgs()
    main(argv)
