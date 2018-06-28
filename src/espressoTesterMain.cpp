//  OverFeat
// ----------------------------------------------
//  Convolutions:
//  Pool:
//  Max Pooling:
//  Global Pooling:
//  Average Pooling:
//  Strides:
//  Padding:
//  Norm:
//  Fully Connected + Softmax:
//
//
// SSD
// ----------------------------------------------
//  Convolutions:
//  Pool:
//  Max Pooling:
//  Global Pooling:
//  Average Pooling:
//  Strides:
//  Padding:
//  Norm:
//  Fully Connected + Softmax:
// 
//
//  YOLO
// ----------------------------------------------
//  Convolutions:
//  Pool:
//  Max Pooling:
//  Global Pooling:
//  Average Pooling:
//  Strides:
//  Padding:
//  Norm:
//  Fully Connected + Softmax:
//
//
//  SqueezeNet
// ----------------------------------------------
//  Convolutions:
//  Pool: 3x3
//  Max Pooling:
//  Global Pooling: Yes
//  Average Pooling:
//  Strides:
//  Padding:
//  Norm: No
//  Fully Connected + Softmax:
//      ./espressoTester ../../../caffe/models/squeezenet/deploy.prototxt ../../../caffe/models/squeezenet/squeezenet_v1.1.caffemodel ../scripts/image.png data prob
// 
//
//  VGG16
// ----------------------------------------------
//  Convolutions: 3x3
//  Pool: 2x2
//  Max Pooling: Yes
//  Global Pooling: No
//  Average Pooling: No
//  Strides: 1, 2
//  Padding: 0, 1
//  Norm: No
//  Fully Connected + Softmax: Yes
//      ./espressoTester ../../../caffe/models/vgg16/vgg16_deploy.prototxt ../../../caffe/models/vgg16/VGG_ILSVRC_16_layers.caffemodel ../scripts/image.png data prob 
// 
//
//  VGG19
// ----------------------------------------------
//  Convolutions: 3x3
//  Pool: 2x2
//  Max Pooling: Yes
//  Global Pooling: No
//  Average Pooling: No
//  Strides: 1, 2
//  Padding: 0, 1
//  Norm: No
//  Fully Connected + Softmax: Yes
//      ./espressoTester ../../../caffe/models/vgg19/vgg19_deploy.prototxt ../../../caffe/models/vgg19/VGG_ILSVRC_19_layers.caffemodel ../scripts/image.png data prob 
// 
//
//  GoogleNet
// ----------------------------------------------
//  Convolutions: 1x1. 3x3, 5x5, 7x7
//  Max Pool: 2x2, 3x3, 
//  Ave Pool: 7x7
//  Max Pooling: Yes
//  Global Pooling: No
//  Average Pooling: Yes
//  Strides: 1, 2
//  Padding: 0, 1, 2, 3
//  Norm: LRN
//  Fully Connected + Softmax: Yes
//      ./espressoTester ../../../caffe/models/bvlc_googlenet/deploy.prototxt ../../../caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel ../scripts/image.png data prob 
// 
//
//  AlexNet
// ----------------------------------------------
//  Convolutions: 3x3, 5x5, 11x11
//  Pool: 3x3
//  Max Pooling: Yes
//  Global Pooling: No
//  Average Pooling: No
//  Strides: 1, 2, 4
//  Padding: 0, 1, 2
//  Norm: LRN
//  Fully Connected + Softmax: Yes
//      ./espressoTester ../../../caffe/models/bvlc_alexnet/deploy.prototxt ../../../caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel ../scripts/image.png data prob 
//   
//
//  DcNet
// ----------------------------------------------
//  Convolutions:
//  Pool: 2x2
//  Max Pooling:
//  Global Pooling:
//  Average Pooling:
//  Strides:
//  Padding:
//  Norm:
//  Fully Connected + Softmax:
//      ./espressoTester ../../../caffe/models/dcNet/deploy_sqz_2.prototxt ../../../caffe/models/dcNet/sqz_rework_iter_100000.caffemodel ../../../../../detector_test_kitti/temp.png data objectness0_soft
//
//
#include "Network.hpp"
#include "caffeDataParser.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>
#include <fstream>
using namespace std;
using namespace cv;


void dataTransform(vector<espresso::layerInfo_t> &networkLayerInfo, vector<caffeDataParser::layerInfo_t> caffeDataParserLayerInfo, precision_t precision) {
    
    if(precision == FLOAT) {
        
        // Begin Code -------------------------------------------------------------------------------------------------------------------------------
        for(uint32_t i = 0; i < caffeDataParserLayerInfo.size(); i++) {
            networkLayerInfo[i].precision               = FLOAT; 
            networkLayerInfo[i].layerName               = caffeDataParserLayerInfo[i].layerName;     
            networkLayerInfo[i].topLayerNames           = caffeDataParserLayerInfo[i].topLayerNames;    
            networkLayerInfo[i].bottomLayerNames        = caffeDataParserLayerInfo[i].bottomLayerNames; 
            networkLayerInfo[i].layerType               = caffeDataParserLayerInfo[i].layerType;       
            networkLayerInfo[i].numInputRows            = caffeDataParserLayerInfo[i].numInputRows;     
            networkLayerInfo[i].numInputCols            = caffeDataParserLayerInfo[i].numInputCols;     
            networkLayerInfo[i].inputDepth              = caffeDataParserLayerInfo[i].inputDepth;       
            networkLayerInfo[i].outputDepth             = caffeDataParserLayerInfo[i].outputDepth;      
            networkLayerInfo[i].numKernelRows           = caffeDataParserLayerInfo[i].numKernelRows;    
            networkLayerInfo[i].numKernelCols           = caffeDataParserLayerInfo[i].numKernelCols;    
            networkLayerInfo[i].stride                  = caffeDataParserLayerInfo[i].stride;          
            networkLayerInfo[i].padding                 = caffeDataParserLayerInfo[i].padding;
            networkLayerInfo[i].globalPooling           = caffeDataParserLayerInfo[i].globalPooling;
            networkLayerInfo[i].group                   = caffeDataParserLayerInfo[i].group;
            networkLayerInfo[i].localSize               = caffeDataParserLayerInfo[i].localSize;
            networkLayerInfo[i].alpha                   = caffeDataParserLayerInfo[i].alpha;
            networkLayerInfo[i].beta                    = caffeDataParserLayerInfo[i].beta;
            
            if(caffeDataParserLayerInfo[i].layerType == "Convolution" || caffeDataParserLayerInfo[i].layerType == "InnerProduct") {
                networkLayerInfo[i].flFilterData = (float*)malloc(    caffeDataParserLayerInfo[i].numFilterValues
                                                                    * sizeof(float)
                                                               );
                memcpy  (   networkLayerInfo[i].flFilterData, 
                            caffeDataParserLayerInfo[i].filterData,
                            caffeDataParserLayerInfo[i].numFilterValues                                 
                            * sizeof(float)                    
                        );

                networkLayerInfo[i].flBiasData = (float*)malloc(  caffeDataParserLayerInfo[i].numBiasValues 
                                                                * sizeof(float)
                                                             );
                memcpy  (   networkLayerInfo[i].flBiasData, 
                            caffeDataParserLayerInfo[i].biasData, 
                            caffeDataParserLayerInfo[i].numBiasValues
                            * sizeof(float)
                        );
            } else {
                networkLayerInfo[i].flFilterData = NULL;
                networkLayerInfo[i].flBiasData = NULL;
            }
        }
        // End Code ---------------------------------------------------------------------------------------------------------------------------------     

    } else if(precision == FIXED) {
        
        // Begin Code -------------------------------------------------------------------------------------------------------------------------------

        for(uint32_t i = 0; i < caffeDataParserLayerInfo.size(); i++) {
            if(caffeDataParserLayerInfo[i].layerType == "LRN" || caffeDataParserLayerInfo[i].layerType == "Softmax") {
                networkLayerInfo[i].precision = FLOAT; 
            } else {
                networkLayerInfo[i].precision = FIXED; 
            }
            networkLayerInfo[i].layerName               = caffeDataParserLayerInfo[i].layerName;     
            networkLayerInfo[i].topLayerNames           = caffeDataParserLayerInfo[i].topLayerNames;    
            networkLayerInfo[i].bottomLayerNames        = caffeDataParserLayerInfo[i].bottomLayerNames; 
            networkLayerInfo[i].layerType               = caffeDataParserLayerInfo[i].layerType;       
            networkLayerInfo[i].numInputRows            = caffeDataParserLayerInfo[i].numInputRows;     
            networkLayerInfo[i].numInputCols            = caffeDataParserLayerInfo[i].numInputCols;     
            networkLayerInfo[i].inputDepth              = caffeDataParserLayerInfo[i].inputDepth;       
            networkLayerInfo[i].outputDepth             = caffeDataParserLayerInfo[i].outputDepth;  
            networkLayerInfo[i].dinFxPtLength           = ESPRO_DEF_FXPT_LEN;      
            networkLayerInfo[i].dinNumFracBits          = ESPRO_DEF_NUM_FRAC_BITS; 
            networkLayerInfo[i].whtFxPtLength           = ESPRO_DEF_FXPT_LEN;     
            networkLayerInfo[i].whtNumFracBits          = ESPRO_DEF_NUM_FRAC_BITS;
            networkLayerInfo[i].doutFxPtLength          = ESPRO_DEF_FXPT_LEN;     
            networkLayerInfo[i].doutNumFracBits         = ESPRO_DEF_NUM_FRAC_BITS;
            networkLayerInfo[i].numKernelRows           = caffeDataParserLayerInfo[i].numKernelRows;    
            networkLayerInfo[i].numKernelCols           = caffeDataParserLayerInfo[i].numKernelCols;    
            networkLayerInfo[i].stride                  = caffeDataParserLayerInfo[i].stride;          
            networkLayerInfo[i].padding                 = caffeDataParserLayerInfo[i].padding;
            networkLayerInfo[i].localSize               = caffeDataParserLayerInfo[i].localSize;
            networkLayerInfo[i].alpha                   = caffeDataParserLayerInfo[i].alpha;
            networkLayerInfo[i].beta                    = caffeDataParserLayerInfo[i].beta;  
            networkLayerInfo[i].group                   = caffeDataParserLayerInfo[i].group;   
            networkLayerInfo[i].globalPooling           = caffeDataParserLayerInfo[i].globalPooling;         
            if(caffeDataParserLayerInfo[i].layerType == "Convolution" || caffeDataParserLayerInfo[i].layerType == "InnerProduct") {
                networkLayerInfo[i].fxFilterData = (FixedPoint_t*)malloc(sizeof(FixedPoint_t) * caffeDataParserLayerInfo[i].numFilterValues); 
                for(int j = 0; j < caffeDataParserLayerInfo[i].numFilterValues; j++) {
	                networkLayerInfo[i].fxFilterData[j] = FixedPoint::create(ESPRO_DEF_NUM_FRAC_BITS, caffeDataParserLayerInfo[i].filterData[j]);
                }        
                
                networkLayerInfo[i].fxBiasData = (FixedPoint_t*)malloc(sizeof(FixedPoint_t) * caffeDataParserLayerInfo[i].numBiasValues);
                for(int j = 0; j < caffeDataParserLayerInfo[i].numBiasValues; j++) {
	                networkLayerInfo[i].fxBiasData[j] = FixedPoint::create(ESPRO_DEF_NUM_FRAC_BITS, caffeDataParserLayerInfo[i].biasData[j]);
                } 
                networkLayerInfo[i].numFilterValues = caffeDataParserLayerInfo[i].numFilterValues;
            } else {
                networkLayerInfo[i].fxFilterData = NULL;
                networkLayerInfo[i].fxBiasData = NULL;
            }
        }
        // End Code ---------------------------------------------------------------------------------------------------------------------------------     
        
    }
}


int main(int argc, char **argv) {
    
    // printModelProtocalBuffer(argv[1], argv[2]);
    // exit(0);

    
	// VGG
    // string protoTxt = "../../../caffe/models/vgg16/vgg16_deploy.prototxt";
	// string model = "../../../caffe/models/vgg16/VGG_ILSVRC_16_layers.caffemodel"; 
    // string beginLayer = "data"; 
    // string endLayer = "prob";
	
	
	// AlexNet
	string protoTxt = "../../../caffe/models/bvlc_alexnet/deploy.prototxt";
	string model = "../../../caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel"; 
	string beginLayer = "data"; 
	string endLayer = "fc6";
	
	
	// Goolge Net
	// string protoTxt = "../../../caffe/models/bvlc_googlenet/deploy.prototxt";
	// string model = "../../../caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel"; 
	// string beginLayer = "data"; 
	// string endLayer = "prob";
	
	
	
    string fixed_float = "float";    
	Mat img = imread("../scripts/image.png", IMREAD_COLOR);

   
    // Read in model
	vector<caffeDataParser::layerInfo_t> caffeDataParserLayerInfo = parseCaffeData(protoTxt, model);
    ofstream fd;
    
    if(fixed_float == "float" || fixed_float == "both") {
        vector<espresso::layerInfo_t> networkLayerInfo;
        networkLayerInfo.resize(caffeDataParserLayerInfo.size());
        dataTransform(networkLayerInfo, caffeDataParserLayerInfo, FLOAT);
        Network *network = new Network(networkLayerInfo);
        // Input Image, for dcNet remember to subtract 127 from each pixel value to compare results (float)
        Blob_t inputBlob;
        int beginLayerIdx = network->ReturnLayerIdx(beginLayer);   
        int endLayerIdx = network->ReturnLayerIdx(endLayer);
        if(endLayerIdx != -1 && beginLayerIdx != -1) {
            inputBlob.flData = new float[img.channels() * img.rows * img.cols]; 
            inputBlob.depth = img.channels();
            inputBlob.numRows = img.rows;
            inputBlob.numCols = img.cols;    
            for(int c = 0; c < img.channels(); c++) {
              for(int i=0; i < img.rows; i++) {
                  for(int j = 0; j < img.cols; j++) { 
                      int a = img.at<cv::Vec3b>(i,j)[c];
                      //index3D(img.channels(), img.rows, img.cols, inputBlob.flData, c, i, j) = (float)a - 127.0f;   // for dcNet
                      index3D(img.channels(), img.rows, img.cols, inputBlob.flData, c, i, j) = (float)a;
                  }
              }
            }  
            network->m_cnn[0]->m_inputDepth     = inputBlob.depth;
            network->m_cnn[0]->m_numInputRows   = inputBlob.numRows;
            network->m_cnn[0]->m_numInputCols   = inputBlob.numCols;
            network->m_cnn[0]->m_blob.flData    = inputBlob.flData;
            network->Forward(beginLayer, endLayer);
        
        
            fd.open("output_espresso.txt");
            if(network->m_cnn[endLayerIdx]->m_blob.depth > 1 && network->m_cnn[endLayerIdx]->m_blob.numRows > 1 && network->m_cnn[endLayerIdx]->m_blob.numCols > 1) {
                for(int i = 0; i < network->m_cnn[endLayerIdx]->m_blob.depth; i++) {
                    for(int j = 0; j < network->m_cnn[endLayerIdx]->m_blob.numRows; j++) {
                        for(int k = 0; k < network->m_cnn[endLayerIdx]->m_blob.numCols; k++) {
                            fd << index3D(  
                                            network->m_cnn[endLayerIdx]->m_blob.depth, 
                                            network->m_cnn[endLayerIdx]->m_blob.numRows, 
                                            network->m_cnn[endLayerIdx]->m_blob.numCols, 
                                            network->m_cnn[endLayerIdx]->m_blob.flData, i, j, k
                                         ) << " ";
                        }
                        fd << endl;
                    }
                    fd << endl << endl << endl;
                }
            } else {
                for(int i = 0; i < network->m_cnn[endLayerIdx]->m_blob.depth; i++) {
                    fd << network->m_cnn[endLayerIdx]->m_blob.flData[i] << endl;
                }
            }
            fd.close();
        }
        delete network;
    }    
    
    if(fixed_float == "fixed" || fixed_float == "both") {       
        vector<espresso::layerInfo_t> networkLayerInfo_fxPt;   
        networkLayerInfo_fxPt.resize(caffeDataParserLayerInfo.size());
        dataTransform(networkLayerInfo_fxPt, caffeDataParserLayerInfo, FIXED);
        Network *network_fxPt = new Network(networkLayerInfo_fxPt);
        Blob_t inputBlob_fxPt;
        int beginLayerIdx = network_fxPt->ReturnLayerIdx(beginLayer);   
        int endLayerIdx = network_fxPt->ReturnLayerIdx(endLayer);
        if(endLayerIdx != -1 && beginLayerIdx != -1) {           
            
            // Input Image, for dcNet remember to subtract 127 from each pixel value to compare results (fixedPoint)
            inputBlob_fxPt.fxData = new FixedPoint_t[img.channels() * img.rows * img.cols]; 
            inputBlob_fxPt.depth = img.channels();
            inputBlob_fxPt.numRows = img.rows;
            inputBlob_fxPt.numCols = img.cols;    
            for(int c = 0; c < img.channels(); c++) {
                for(int i=0; i < img.rows; i++) {
                    for(int j = 0; j < img.cols; j++) { 
                        int a = img.at<cv::Vec3b>(i,j)[c];
                        //float b = (float)a - 127.0f; // for dcNet
                        index3D(img.channels(), img.rows, img.cols, inputBlob_fxPt.fxData, c, i, j) = FixedPoint::create(ESPRO_DEF_NUM_FRAC_BITS, a);
                    }
                }
            } 
            network_fxPt->m_cnn[0]->m_inputDepth     = inputBlob_fxPt.depth;
            network_fxPt->m_cnn[0]->m_numInputRows   = inputBlob_fxPt.numRows;
            network_fxPt->m_cnn[0]->m_numInputCols   = inputBlob_fxPt.numCols;
            network_fxPt->m_cnn[0]->m_blob.fxData    = inputBlob_fxPt.fxData;
            network_fxPt->Forward(beginLayer, endLayer);
            
            fd.open("output_espresso_fxPt.txt");
            if(network_fxPt->m_cnn[endLayerIdx]->m_blob.depth > 1 && network_fxPt->m_cnn[endLayerIdx]->m_blob.numRows > 1 && network_fxPt->m_cnn[endLayerIdx]->m_blob.numCols > 1) {
                for(int i = 0; i < network_fxPt->m_cnn[endLayerIdx]->m_blob.depth; i++) {
                    for(int j = 0; j < network_fxPt->m_cnn[endLayerIdx]->m_blob.numRows; j++) {
                        for(int k = 0; k < network_fxPt->m_cnn[endLayerIdx]->m_blob.numCols; k++) {
                            fd << FixedPoint::toFloat(ESPRO_DEF_NUM_FRAC_BITS, index3D(  
                                                            network_fxPt->m_cnn[endLayerIdx]->m_blob.depth, 
                                                            network_fxPt->m_cnn[endLayerIdx]->m_blob.numRows, 
                                                            network_fxPt->m_cnn[endLayerIdx]->m_blob.numCols, 
                                                            network_fxPt->m_cnn[endLayerIdx]->m_blob.fxData, i, j, k)
                                                        ) << " ";
                        }
                        fd << endl;
                    }
                    fd << endl << endl << endl;
                }
            } else {
                for(int i = 0; i < network_fxPt->m_cnn[endLayerIdx]->m_blob.depth; i++) {
	                if (network_fxPt->m_cnn[endLayerIdx]->m_precision == FIXED) {
		                fd << FixedPoint::toFloat(ESPRO_DEF_NUM_FRAC_BITS, network_fxPt->m_cnn[endLayerIdx]->m_blob.fxData[i]) << endl;
	                } else {
		                fd << network_fxPt->m_cnn[endLayerIdx]->m_blob.flData[i] << endl;
	                }

                }
            }
            fd.close();
        }
        delete network_fxPt;
    }
    

    // DcNet Output
    // fd.open("obj.txt");
    // for(int i = 0; i < network->m_outputLayers[3]->m_blob.depth; i++) {
    //     for(int j = 0; j < network->m_outputLayers[3]->m_blob.numRows; j++) {
    //         for(int k = 0; k < network->m_outputLayers[3]->m_blob.numCols; k++) {
    //             fd << index3D(network->m_outputLayers[3]->m_blob.depth, network->m_outputLayers[3]->m_blob.numRows, network->m_outputLayers[3]->m_blob.numCols, network->m_outputLayers[3]->m_blob.data, i, j, k) << " ";
    //         }
    //         fd << endl;
    //     }
    //     fd << endl << endl << endl;
    // }
    // fd.close();
    // 
    // fd.open("cls.txt");
    // for(int i = 0; i < network->m_outputLayers[4]->m_blob.depth; i++) {
    //     for(int j = 0; j < network->m_outputLayers[4]->m_blob.numRows; j++) {
    //         for(int k = 0; k < network->m_outputLayers[4]->m_blob.numCols; k++) {
    //             fd << index3D(network->m_outputLayers[4]->m_blob.depth, network->m_outputLayers[4]->m_blob.numRows, network->m_outputLayers[4]->m_blob.numCols, network->m_outputLayers[4]->m_blob.data, i, j, k) << " ";
    //         }
    //         fd << endl;
    //     }
    //     fd << endl << endl << endl;
    // }
    // fd.close();
    // 
    // fd.open("loc.txt");
    // for(int i = 0; i < network->m_outputLayers[5]->m_blob.depth; i++) {
    //     for(int j = 0; j < network->m_outputLayers[5]->m_blob.numRows; j++) {
    //         for(int k = 0; k < network->m_outputLayers[5]->m_blob.numCols; k++) {
    //             fd << index3D(network->m_outputLayers[5]->m_blob.depth, network->m_outputLayers[5]->m_blob.numRows, network->m_outputLayers[5]->m_blob.numCols, network->m_outputLayers[5]->m_blob.data, i, j, k) << " ";
    //         }
    //         fd << endl;
    //     }
    //     fd << endl << endl << endl;
    // }
    // fd.close();  
    
    
    for(uint32_t i = 0; i < caffeDataParserLayerInfo.size(); i++) {
        if(caffeDataParserLayerInfo[i].layerType == "Convolution" || caffeDataParserLayerInfo[i].layerType == "InnerProduct") {
            free(caffeDataParserLayerInfo[i].filterData);
            free(caffeDataParserLayerInfo[i].biasData);
        }
    }
 
    return 0;
}
