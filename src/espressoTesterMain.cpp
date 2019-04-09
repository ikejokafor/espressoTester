// Definitions
#define THRESH			0.5f
#define HIER_THRESH		0.5f
#define NMS				0.45f


// System includes
#include <iomanip>
#include <fstream>
#include "util.hpp"
using namespace std;

// opencv includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


// darknet includes
#include "activation_layer.h"
#include "logistic_layer.h"
#include "l2norm_layer.h"
#include "activations.h"
#include "avgpool_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "connected_layer.h"
#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "cost_layer.h"
#include "crnn_layer.h"
#include "crop_layer.h"
#include "detection_layer.h"
#include "dropout_layer.h"
#include "gru_layer.h"
#include "list.h"
#include "local_layer.h"
#include "maxpool_layer.h"
#include "normalization_layer.h"
#include "option_list.h"
#include "parser.h"
#include "region_layer.h"
#include "yolo_layer.h"
#include "iseg_layer.h"
#include "reorg_layer.h"
#include "rnn_layer.h"
#include "route_layer.h"
#include "upsample_layer.h"
#include "shortcut_layer.h"
#include "softmax_layer.h"
#include "lstm_layer.h"
#include "utils.h"


// espresso inludes
#include "Network.hpp"


void darknetDataTransform(network **net, vector<espresso::layerInfo_t> &networkLayerInfo, espresso::precision_t precision, string configFileName, string weightFileName) {
	// BEGIN Code -----------------------------------------------------------------------------------------------------------------------------------
	*net = parse_network_cfg((char*)&configFileName[0]);
	network *net_ptr = *net;
	load_weights(net_ptr, (char*)&weightFileName[0]);
	networkLayerInfo.resize(net_ptr->n + 1);
	// END Code -------------------------------------------------------------------------------------------------------------------------------------
	
	
    // BEGIN Code -----------------------------------------------------------------------------------------------------------------------------------
	networkLayerInfo[0] = espresso::layerInfo_t();
	networkLayerInfo[0].precision = precision;
	networkLayerInfo[0].layerType = "Input";
	networkLayerInfo[0].layerName = "0_Data";
	networkLayerInfo[0].inputDepth = net_ptr->layers[0].c;
	networkLayerInfo[0].numInputRows = net_ptr->layers[0].h;
	networkLayerInfo[0].numInputCols = net_ptr->layers[0].w;
	// END Code -------------------------------------------------------------------------------------------------------------------------------------
	
	
    for(uint32_t i = 0, j = 1; i < net_ptr->n; i++, j++) {
	    networkLayerInfo[j] = espresso::layerInfo_t();
	    // BEGIN Code -------------------------------------------------------------------------------------------------------------------------------
	    if(net_ptr->layers[i].type == CONVOLUTIONAL) {
		    networkLayerInfo[j].layerType = "Convolution";
		    if (net_ptr->layers[i].batch_normalize) {			    
			    networkLayerInfo[j].flBeta = 0.000001f;
			    networkLayerInfo[j].darknetNormScaleBias = true;
				networkLayerInfo[j].flMeanData = (float*)malloc(net_ptr->layers[i].n * sizeof(float));
				memcpy(networkLayerInfo[j].flMeanData, net_ptr->layers[i].rolling_mean, net_ptr->layers[i].n * sizeof(float));
				networkLayerInfo[j].flVarianceData = (float*)malloc(net_ptr->layers[i].n * sizeof(float));
				memcpy(networkLayerInfo[j].flVarianceData, net_ptr->layers[i].rolling_variance, net_ptr->layers[i].n * sizeof(float));
				networkLayerInfo[j].flScaleBiasData = (float*)malloc(net_ptr->layers[i].n * sizeof(float));
				memcpy(networkLayerInfo[j].flScaleBiasData, net_ptr->layers[i].scales,  net_ptr->layers[i].n * sizeof(float));
		    }
			if(net_ptr->layers[i].activation == LEAKY) {
				networkLayerInfo[j].activation = espresso::LEAKY;
			} else if(net_ptr->layers[i].activation == LINEAR){
				networkLayerInfo[j].activation = espresso::LINEAR;
			}
		    networkLayerInfo[j].darknetAct = true;
	    } else if(net_ptr->layers[i].type == MAXPOOL) {
		    networkLayerInfo[j].layerType = "Pooling_MAX";
	    } else if(net_ptr->layers[i].type == AVGPOOL) {
		    networkLayerInfo[j].layerType = "Pooling_AVG";
	    } else if(net_ptr->layers[i].type == ROUTE) {
		    networkLayerInfo[j].layerType = "Concat";
		    for (int k = 0; k < net_ptr->layers[i].n; k++) {
				networkLayerInfo[j].bottomLayerNames.push_back(networkLayerInfo[net_ptr->layers[i].input_layers[k] + 1].layerName);
		    }
	    } else if(net_ptr->layers[i].type == SHORTCUT){
		    networkLayerInfo[j].layerType = "Residual";
		    networkLayerInfo[j].bottomLayerNames.push_back(networkLayerInfo[j - 1].layerName);
		    networkLayerInfo[j].bottomLayerNames.push_back(networkLayerInfo[net_ptr->layers[i].index + 1].layerName);
	    } else if(net_ptr->layers[i].type == YOLO) {
		    networkLayerInfo[j].layerType = "YOLO";
		    networkLayerInfo[j].darknet_n_param = net_ptr->layers[i].n;
			networkLayerInfo[j].darknet_classes_param = net_ptr->layers[i].classes;
			networkLayerInfo[j].darknet_outputs_param = net_ptr->layers[i].outputs;
			networkLayerInfo[j].precision = espresso::FLOAT;
	    } else if(net_ptr->layers[i].type == UPSAMPLE) {
		    networkLayerInfo[j].layerType = "UpSample";
	    } else {
			cout << "Skipped Darknet Layer " << i << endl;
			continue;
	    }
	    // END Code ---------------------------------------------------------------------------------------------------------------------------------


	    // BEGIN Code -------------------------------------------------------------------------------------------------------------------------------
		if(networkLayerInfo[j].layerType == "YOLO") {
			networkLayerInfo[j].precision = espresso::FLOAT;
		} else {
			networkLayerInfo[j].precision = precision;
		}
	    networkLayerInfo[j].layerName = to_string(j) + "_" + networkLayerInfo[j].layerType;
	    networkLayerInfo[j].numInputRows = net_ptr->layers[i].h;
		networkLayerInfo[j].numInputCols = net_ptr->layers[i].w;
	    networkLayerInfo[j].inputDepth = net_ptr->layers[i].c;
	    networkLayerInfo[j].outputDepth = net_ptr->layers[i].n;
	    networkLayerInfo[j].dinFxPtLength = ESPRO_DEF_FXPT_LEN;
	    networkLayerInfo[j].dinNumFracBits = ESPRO_DEF_NUM_FRAC_BITS;
		networkLayerInfo[j].whtFxPtLength = ESPRO_DEF_FXPT_LEN;
		networkLayerInfo[j].whtNumFracBits = ESPRO_DEF_NUM_FRAC_BITS;
		networkLayerInfo[j].doutFxPtLength = ESPRO_DEF_FXPT_LEN;
		networkLayerInfo[j].doutNumFracBits = ESPRO_DEF_NUM_FRAC_BITS;    
		networkLayerInfo[j].biasFxPtLength = ESPRO_DEF_FXPT_LEN;
		networkLayerInfo[j].biasNumFracBits = ESPRO_DEF_NUM_FRAC_BITS;
		networkLayerInfo[j].scaleBiasFxPtLength = ESPRO_DEF_FXPT_LEN;
		networkLayerInfo[j].scaleBiasNumFracBits = ESPRO_DEF_NUM_FRAC_BITS;
	    networkLayerInfo[j].leakyFxPtLength = ESPRO_DEF_FXPT_LEN;
	    networkLayerInfo[j].leakyNumFracBits = ESPRO_DEF_NUM_FRAC_BITS;
	    networkLayerInfo[j].numKernelRows = net_ptr->layers[i].size;
		networkLayerInfo[j].numKernelCols = net_ptr->layers[i].size;
	    networkLayerInfo[j].stride = net_ptr->layers[i].stride;
	    networkLayerInfo[j].padding = net_ptr->layers[i].pad;	    
	    networkLayerInfo[j].numFilterValues = net_ptr->layers[i].nweights;
	    networkLayerInfo[j].group = net_ptr->layers[i].groups;
	    if (networkLayerInfo[j].layerType == "Convolution" && precision == espresso::FLOAT) {
			networkLayerInfo[j].flFilterData = (float*)malloc(networkLayerInfo[j].numFilterValues * sizeof(float));
			memcpy(networkLayerInfo[j].flFilterData, net_ptr->layers[i].weights, networkLayerInfo[j].numFilterValues * sizeof(float));
			networkLayerInfo[j].flBiasData = (float*)malloc(networkLayerInfo[j].outputDepth * sizeof(float));
			memcpy(networkLayerInfo[j].flBiasData, net_ptr->layers[i].biases,  networkLayerInfo[j].outputDepth * sizeof(float));
			if (networkLayerInfo[j].darknetNormScaleBias) {
				for (int a = 0; a < networkLayerInfo[j].outputDepth; a++) {
					networkLayerInfo[j].flBiasData[a] 
						= ((-networkLayerInfo[j].flMeanData[a] * networkLayerInfo[j].flScaleBiasData[a]) / sqrt(networkLayerInfo[j].flVarianceData[a]))
							+ (networkLayerInfo[j].flBeta * networkLayerInfo[j].flScaleBiasData[a]) + networkLayerInfo[j].flBiasData[a];
					for (int b = 0; b < (networkLayerInfo[j].numFilterValues / networkLayerInfo[j].outputDepth); b++) {
						index2D(networkLayerInfo[j].outputDepth, (networkLayerInfo[j].numFilterValues / networkLayerInfo[j].outputDepth), networkLayerInfo[j].flFilterData, a, b)
							= (index2D(networkLayerInfo[j].outputDepth, (networkLayerInfo[j].numFilterValues / networkLayerInfo[j].outputDepth), networkLayerInfo[j].flFilterData, a, b) 
								* networkLayerInfo[j].flScaleBiasData[a])
								/ sqrt(networkLayerInfo[j].flVarianceData[a]);
					}
				}	    
			}
		} else if(networkLayerInfo[j].layerType == "Convolution" && precision == espresso::FIXED) {
			networkLayerInfo[j].fxFilterData = (fixedPoint_t*)malloc(networkLayerInfo[j].numFilterValues * sizeof(fixedPoint_t));
			for(int a = 0; a < networkLayerInfo[j].numFilterValues; a++){
				networkLayerInfo[j].fxFilterData[a] = fixedPoint::create(ESPRO_DEF_NUM_FRAC_BITS, net_ptr->layers[i].weights[a]);
			}
			networkLayerInfo[j].fxBiasData = (fixedPoint_t*)malloc(networkLayerInfo[j].outputDepth * sizeof(fixedPoint_t));
			for(int a = 0; a < networkLayerInfo[j].outputDepth; a++){
				networkLayerInfo[j].fxBiasData[a] = fixedPoint::create(ESPRO_DEF_NUM_FRAC_BITS, net_ptr->layers[i].biases[a]);
			}
			if (networkLayerInfo[j].darknetNormScaleBias) {
				fixedPoint_t flBeta = fixedPoint::create(ESPRO_DEF_NUM_FRAC_BITS, networkLayerInfo[j].flBeta);
				for (int a = 0; a < networkLayerInfo[j].outputDepth; a++) {
					fixedPoint_t fxScale = fixedPoint::create(ESPRO_DEF_NUM_FRAC_BITS, networkLayerInfo[j].flScaleBiasData[a]);
					fixedPoint_t fxVariance = fixedPoint::create(ESPRO_DEF_NUM_FRAC_BITS, sqrt(networkLayerInfo[j].flVarianceData[a]));
					fixedPoint_t fxMean = fixedPoint::create(ESPRO_DEF_NUM_FRAC_BITS, networkLayerInfo[j].flMeanData[a]);
					if (fxVariance == 0) {
						networkLayerInfo[j].fxBiasData[a] 
							= ((-fxMean * fxScale))
								+ (flBeta * fxScale) + networkLayerInfo[j].fxBiasData[a];
			
					} else {
						networkLayerInfo[j].fxBiasData[a] 
							= ((-fxMean * fxScale) / fxVariance)
								+ (flBeta * fxScale) + networkLayerInfo[j].fxBiasData[a];
					}
					for (int b = 0; b < (networkLayerInfo[j].numFilterValues / networkLayerInfo[j].outputDepth); b++) {
						if (fxVariance == 0 && fxScale == 0) {
							index2D(networkLayerInfo[j].outputDepth, (networkLayerInfo[j].numFilterValues / networkLayerInfo[j].outputDepth), networkLayerInfo[j].fxFilterData, a, b)
								= index2D(networkLayerInfo[j].outputDepth, (networkLayerInfo[j].numFilterValues / networkLayerInfo[j].outputDepth), networkLayerInfo[j].fxFilterData, a, b);
						} else if(fxVariance == 0 && fxScale != 0) {
							index2D(networkLayerInfo[j].outputDepth, (networkLayerInfo[j].numFilterValues / networkLayerInfo[j].outputDepth), networkLayerInfo[j].fxFilterData, a, b)
								= (index2D(networkLayerInfo[j].outputDepth, (networkLayerInfo[j].numFilterValues / networkLayerInfo[j].outputDepth), networkLayerInfo[j].fxFilterData, a, b) 
									* fxScale);
						} else if(fxVariance != 0 && fxScale == 0) {
							index2D(networkLayerInfo[j].outputDepth, (networkLayerInfo[j].numFilterValues / networkLayerInfo[j].outputDepth), networkLayerInfo[j].fxFilterData, a, b)
								= (index2D(networkLayerInfo[j].outputDepth, (networkLayerInfo[j].numFilterValues / networkLayerInfo[j].outputDepth), networkLayerInfo[j].fxFilterData, a, b) 
									/ fxVariance);
						} else {
							index2D(networkLayerInfo[j].outputDepth, (networkLayerInfo[j].numFilterValues / networkLayerInfo[j].outputDepth), networkLayerInfo[j].fxFilterData, a, b)
								= (index2D(networkLayerInfo[j].outputDepth, (networkLayerInfo[j].numFilterValues / networkLayerInfo[j].outputDepth), networkLayerInfo[j].fxFilterData, a, b) 
									* fxScale)
									/ fxVariance;
						}
					}
				}		    
			}
		}
	    // END Code ---------------------------------------------------------------------------------------------------------------------------------
    }

	
	// BEGIN Code -----------------------------------------------------------------------------------------------------------------------------------
	// Set layer output and input layers ie bottom and top layers
	for(uint32_t i = 1; i < networkLayerInfo.size(); i++) {
		if(networkLayerInfo[i].layerType != "Concat" && networkLayerInfo[i].layerType != "Residual") {
			networkLayerInfo[i].bottomLayerNames.push_back(networkLayerInfo[i - 1].layerName);
		} 
	}	
	for(uint32_t i = 1; i < networkLayerInfo.size(); i++) {
		networkLayerInfo[i].topLayerNames.push_back(networkLayerInfo[i].layerName);
	}
	// END Code -------------------------------------------------------------------------------------------------------------------------------------

	
	// Clean up
	// free(net_ptr)
}


int main(int argc, char **argv) {
	

	// BEGIN Code -----------------------------------------------------------------------------------------------------------------------------------	
	espresso::precision_t precision = espresso::FIXED;
	network *yolo_net = NULL;
	vector<espresso::layerInfo_t> networkLayerInfo;
	darknetDataTransform(&yolo_net, networkLayerInfo, precision, "/home/ikenna/IkennaWorkSpace/darknet/cfg/yolov3.cfg", "/home/ikenna/IkennaWorkSpace/darknet/cfg/yolov3.weights");
	vector<int> outputLayers;
	for(int i = 0; i < networkLayerInfo.size(); i++) {
		if (networkLayerInfo[i].layerType == "YOLO") {
			outputLayers.push_back(i);
		}
	}	
	Network *net = new Network(networkLayerInfo, outputLayers);
	// END Code -------------------------------------------------------------------------------------------------------------------------------------

	
	// BEGIN Code -----------------------------------------------------------------------------------------------------------------------------------		
	image im = load_image_color("/home/ikenna/IkennaWorkSpace/darknet/data/dog.jpg", 0, 0);
	image sized = letterbox_image(im, networkLayerInfo[0].numInputRows, networkLayerInfo[0].numInputCols);
	FILE *fd;
	// fd = fopen("input.txt", "w");
	// for (int i = 0; i < networkLayerInfo[0].inputDepth; i++) {
	// 	for (int j = 0; j < networkLayerInfo[0].numInputRows; j++) {
	// 		for (int k = 0; k < networkLayerInfo[0].numInputCols; k++) {
	// 			fprintf(fd, "%f ", index3D(networkLayerInfo[0].inputDepth, networkLayerInfo[0].numInputRows, networkLayerInfo[0].numInputCols, sized.data, i, j, k));
	// 		}
	// 		fprintf(fd, "\n");
	// 	}
	// 	fprintf(fd, "\n\n\n");
	// }
	// fclose(fd);	
	if(precision == espresso::FLOAT) {
		net->m_cnn[0]->m_blob.flData = sized.data;
	} else {
		net->m_cnn[0]->m_blob.fxData = (fixedPoint_t*)malloc((networkLayerInfo[0].numInputRows * networkLayerInfo[0].numInputCols * networkLayerInfo[0].inputDepth) * sizeof(fixedPoint_t));
		for(int i = 0; i < (networkLayerInfo[0].numInputRows * networkLayerInfo[0].numInputCols * networkLayerInfo[0].inputDepth); i++) {
			net->m_cnn[0]->m_blob.fxData[i] = fixedPoint::create(ESPRO_DEF_NUM_FRAC_BITS, sized.data[i]);
		}
	}
	// END Code -------------------------------------------------------------------------------------------------------------------------------------


	// BEGIN Code -----------------------------------------------------------------------------------------------------------------------------------		
	string beginLayerName = "0_Data";
	string endLayerName = "107_YOLO";
	int eIdx = net->ReturnLayerIdx(endLayerName);
	net->Forward(beginLayerName, endLayerName);
	// fd = fopen("output.txt", "w");
	// for (int i = 0; i < net->m_cnn[eIdx]->m_blob.depth; i++) {
	// 	for (int j = 0; j < net->m_cnn[eIdx]->m_blob.numRows; j++) {
	// 		for (int k = 0; k < net->m_cnn[eIdx]->m_blob.numCols; k++) {
	// 			if (net->m_cnn[eIdx]->m_precision == espresso::FLOAT) {
	// 				fprintf(fd, "%f ", index3D(net->m_cnn[eIdx]->m_blob.depth, net->m_cnn[eIdx]->m_blob.numRows, net->m_cnn[eIdx]->m_blob.numCols, net->m_cnn[eIdx]->m_blob.flData, i, j, k));
	// 			} else {
	// 				fprintf(fd, "%f ", fixedPoint::toFloat(net->m_cnn[eIdx]->m_doutNumFracBits, index3D(net->m_cnn[eIdx]->m_blob.depth, net->m_cnn[eIdx]->m_blob.numRows, net->m_cnn[eIdx]->m_blob.numCols, net->m_cnn[eIdx]->m_blob.fxData, i, j, k)));
	// 			}
	// 		}
	// 		fprintf(fd, "\n");
	// 	}
	// 	fprintf(fd, "\n\n\n");
	// }
	// fclose(fd);
	// END Code -------------------------------------------------------------------------------------------------------------------------------------


	// BEGIN Code -----------------------------------------------------------------------------------------------------------------------------------
	for(uint32_t i = 0; i < net->m_cnn.size(); i++){
        if(net->m_cnn[i]->m_layerType == "YOLO") {
	        layer *l = &yolo_net->layers[i - 1];
	        l->output = (float*)net->m_cnn[i]->m_blob.flData;
        }
    }
    char **names = get_labels("/home/ikenna/IkennaWorkSpace/darknet/data/coco.names");
    image **alphabet = load_alphabet();
	int nboxes = 0;
	layer l = yolo_net->layers[yolo_net->n-1];
	detection *dets = get_network_boxes(yolo_net, im.w, im.h, THRESH, HIER_THRESH, 0, 1, &nboxes);
	if(NMS) do_nms_sort(dets, nboxes, l.classes, NMS);
    draw_detections(im, dets, nboxes, 0.5f, names, alphabet, l.classes);
    save_image(im, "predictions");
	free_detections(dets, nboxes);
    free_image(im);
    free_image(sized);
	// END Code -------------------------------------------------------------------------------------------------------------------------------------
	
	
	// Clean up
	delete net;
	
	
	
    return 0;
}
