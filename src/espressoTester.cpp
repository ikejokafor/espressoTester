// Project includes
#include "espressoTester.hpp"


void setLayerConnections(std::vector<espresso::layerInfo_obj> &networkLayerInfoArr)
{
	for (int i = 0; i < networkLayerInfoArr.size(); i++) 
	{
		if (i == 0)
		{
			networkLayerInfoArr[i].bottomLayerNames.push_back(networkLayerInfoArr[i].layerName);
		}
		else if (networkLayerInfoArr[i].layerType != "Concat" && networkLayerInfoArr[i].layerType != "Residual") 
		{
			networkLayerInfoArr[i].bottomLayerNames.push_back(networkLayerInfoArr[i - 1].layerName);
		} 
	}	
	for (int i = 0; i < networkLayerInfoArr.size(); i++) 
	{
		networkLayerInfoArr[i].topLayerNames.push_back(networkLayerInfoArr[i].layerName);
	}
}


void setKrnlGrpData(espresso::layerInfo_obj &networkLayerInfo, int numKernels)
{
	int numKernelGroups = networkLayerInfo.kernel_group_arr.size();
	int krnl_idx_ofst = 0;
	int numKrnlInGrp;
	for (int i = 0; i < numKernelGroups; i++) 
	{	
		int numDepthGroups = networkLayerInfo.kernel_group_arr[i].size();
		for (int j = 0; j < numDepthGroups; j++)
		{
			kernel_group *kernel_group_i = networkLayerInfo.kernel_group_arr[i][j];
			numKrnlInGrp = kernel_group_i->get_num_kernels();
			int krnl_idx = krnl_idx_ofst;
			for (int k = 0; k < numKrnlInGrp; k++) 
			{
				kernel *kernel_i = kernel_group_i->get_kernel(k);
				int channels = kernel_i->get_length(0);
				for (int c = 0; c < channels; c++)
				{
					int height = kernel_i->get_length(1);
					for (int h = 0; h < height; h++)
					{
						int width = kernel_i->get_length(2);
						for (int w = 0; w < width; w++)
						{
							int idx = index4D(
								channels, 
								height, 
								width,
								krnl_idx,
								c,
								h,
								w
							);
							fixedPoint_t value = networkLayerInfo.fxFilterData[idx];
							kernel_i->set_weight(c, h, w, value);
						}
					}
				}
				krnl_idx++;
			}
		}
		krnl_idx_ofst += numKrnlInGrp;
	}
}


std::vector<std::vector<kernel_group*>> createKernelGroupArr(int numKernels, int height, int width, int channels, int fxPtLength, int numFracBits) 
{
	std::vector<std::vector<kernel_group*>> kernel_group_arr;
	int numKernelGroups = ceil(float(numKernels) / float(QUAD_MAX_NUM_KRNL));
	int numDepthGroups = ceil(float(channels) / float(QUAD_MAX_KRNL_DEPTH));
	kernel_group_arr.resize(numKernelGroups);
	int kernel_rem = numKernels;
	for (int i = 0; i < numKernelGroups; i++) 
	{	
		kernel_group_arr[i].resize(numDepthGroups);
		int channel_rem = channels;
		for (int j = 0; j < numDepthGroups; j++)
		{
			int numChannels = std::min(QUAD_MAX_KRNL_DEPTH, channel_rem);
			kernel_group_arr[i][j] = new kernel_group(numChannels, height, width, fxPtLength, numFracBits);
			int numKrnlInGrp = std::min(QUAD_MAX_NUM_KRNL, kernel_rem);
			for (int k = 0; k < numKrnlInGrp; k++) 
			{
				kernel_group_arr[i][j]->add_kernel(new kernel(
					numChannels,
					height, 
					width, 
					fxPtLength,
					numFracBits
				));
			}
			channel_rem -= QUAD_MAX_KRNL_DEPTH;
		}
		kernel_rem -= QUAD_MAX_NUM_KRNL;
	}
	return kernel_group_arr;
}


void createKernelGroups(espresso::layerInfo_obj &networkLayerInfo)
{
	int kernelDepth = networkLayerInfo.inputDepth;
	int numKernels = networkLayerInfo.outputDepth;
	int kernelWidth = networkLayerInfo.numKernelCols;
	int kernelHeight = networkLayerInfo.numKernelRows;
	networkLayerInfo.kernel_group_arr = createKernelGroupArr(
		numKernels, 
		kernelHeight, 
		kernelWidth, 
		kernelDepth,
		networkLayerInfo.whtFxPtLength,
		networkLayerInfo.whtNumFracBits
	);
	setKrnlGrpData(networkLayerInfo, numKernels);
}


void getFxPtWeights(espresso::layerInfo_obj &networkLayerInfo) 
{
	networkLayerInfo.fxFilterData = new fixedPoint_t[networkLayerInfo.numFilterValues];
	fixedPoint_t *fxFilterData = networkLayerInfo.fxFilterData;
	int whtFxPtLength = networkLayerInfo.whtFxPtLength;
	int whtNumFracBits = networkLayerInfo.whtNumFracBits;	
	for(int k = 0; k < networkLayerInfo.numFilterValues; k++)
	{
		fxFilterData[k] = fixedPoint::create(
			whtFxPtLength, 
			whtNumFracBits, 
			networkLayerInfo.flFilterData[k]
		);
	}
	networkLayerInfo.fxBiasData = new fixedPoint_t[networkLayerInfo.outputDepth];
	fixedPoint_t *fxBiasData = networkLayerInfo.fxBiasData;
	int biasFxPtLength = networkLayerInfo.biasFxPtLength;
	int biasNumFracBits = networkLayerInfo.biasFxPtLength;
	for(int k = 0; k < networkLayerInfo.outputDepth; k++) 
	{
		fxBiasData[k] = fixedPoint::create(
			biasFxPtLength, 
			biasNumFracBits, 
			networkLayerInfo.flBiasData[k]
		);
	}
}


void getFlPtWeights(layer *layer_i, espresso::layerInfo_obj &networkLayerInfo) 
{
	networkLayerInfo.flFilterData = new float[networkLayerInfo.numFilterValues];
	memcpy(
		networkLayerInfo.flFilterData, 
		layer_i->weights, 
		networkLayerInfo.numFilterValues * sizeof(float)
	);
	networkLayerInfo.flBiasData = new float[networkLayerInfo.outputDepth];
	memcpy(
		networkLayerInfo.flBiasData, 
		layer_i->biases, 
		networkLayerInfo.outputDepth * sizeof(float)
	);
	if(networkLayerInfo.darknetNormScaleBias) 
	{
		int outputDepth = networkLayerInfo.outputDepth;
		int numFilValPerMap = networkLayerInfo.numFilterValues / networkLayerInfo.outputDepth;
		float *flFilterData_ptr = networkLayerInfo.flFilterData;
		for (int a = 0; a < outputDepth; a++) 
		{
			float rolling_mean = layer_i->rolling_mean[a];
			float scales = layer_i->scales[a];
			float rolling_variance = layer_i->rolling_variance[a];
			float flBeta = networkLayerInfo.flBeta;
			float flBiasData_v = flFilterData_ptr[a];
			networkLayerInfo.flBiasData[a] = ((-rolling_mean * scales) / sqrtf(rolling_variance)) + (flBeta * scales) + flBiasData_v;
			for (int b = 0; b < numFilValPerMap; b++) 
			{
				int idx = index2D(numFilValPerMap, a, b);
				flFilterData_ptr[idx] = flFilterData_ptr[idx] * scales/ sqrtf(rolling_variance);
			}
		}	    
	}
}


void getWeights(espresso::precision_t precision, layer *layer_i, espresso::layerInfo_obj &networkLayerInfo, int fxPtLen, int numFracBits)
{
	getFlPtWeights(layer_i, networkLayerInfo);
	if(precision == espresso::FIXED)
	{
		getFxPtWeights(networkLayerInfo);
	}
}



void setBaseLayerInfo(int i, layer *layer_i, espresso::layerInfo_obj &networkLayerInfo, espresso::precision_t precision, int fxPtLen, int numFracBits, network *yolo_net)
{
	networkLayerInfo.numInputRows = layer_i->h;
	networkLayerInfo.numInputCols = layer_i->w;
	networkLayerInfo.inputDepth = layer_i->c;
	networkLayerInfo.outputDepth = layer_i->n;
	networkLayerInfo.numKernelRows = layer_i->size;
	networkLayerInfo.numKernelCols = layer_i->size;
	networkLayerInfo.stride = layer_i->stride;
	networkLayerInfo.padding = layer_i->pad;	    
	networkLayerInfo.numFilterValues = layer_i->nweights;
	networkLayerInfo.group = layer_i->groups;
	networkLayerInfo.dinFxPtLength = fxPtLen;
	networkLayerInfo.dinNumFracBits = numFracBits;
	networkLayerInfo.whtFxPtLength = fxPtLen;
	networkLayerInfo.whtNumFracBits = numFracBits,
	networkLayerInfo.doutFxPtLength = fxPtLen;
	networkLayerInfo.doutNumFracBits = numFracBits,	
	networkLayerInfo.biasFxPtLength = fxPtLen;
	networkLayerInfo.biasNumFracBits = numFracBits,
	networkLayerInfo.leakyFxPtLength = fxPtLen;
	networkLayerInfo.leakyNumFracBits = numFracBits,
	networkLayerInfo.precision = precision;
	networkLayerInfo.net_idx = i;
	networkLayerInfo.backend = espresso::DARKNET_BACKEND;
	networkLayerInfo.yolo_net = yolo_net;
}


void createDataLayer(network *net, espresso::layerInfo_obj& networkLayerInfo, espresso::precision_t precision)
{
	networkLayerInfo.precision = precision;
	networkLayerInfo.layerType = "Input";
	networkLayerInfo.layerName = "0_Data";
	networkLayerInfo.inputDepth = net->layers[0].c;
	networkLayerInfo.numInputRows = net->layers[0].h;
	networkLayerInfo.numInputCols = net->layers[0].w;	
}


void darknetDataTransform(
	network** net, 
	std::vector<espresso::layerInfo_obj> &networkLayerInfoArr, 
	espresso::precision_t precision,
	espresso::backend_t backend,
	char* configFileName, 
	char* weightFileName,
	int fxPtLen,
	int numFracBits
) {
	*net = parse_network_cfg(configFileName);
	network *net_ptr = *net;
	load_weights(net_ptr, weightFileName);
	int numLayers = net_ptr->n + 1;
	networkLayerInfoArr.resize(numLayers);
	int j = 0;
	createDataLayer(net_ptr, networkLayerInfoArr[0], precision);
	for(int i = 1; i < numLayers; i++) 
	{
		networkLayerInfoArr[i] = espresso::layerInfo_obj();
		switch (net_ptr->layers[j].type)
		{
			case CONVOLUTIONAL:
			{
				networkLayerInfoArr[i].layerType = "Convolution";
				networkLayerInfoArr[i].layerName = std::to_string(i) + "_" + networkLayerInfoArr[i].layerType;
				setBaseLayerInfo(j, &net_ptr->layers[j], networkLayerInfoArr[i], precision, fxPtLen, numFracBits, net_ptr);
				if (net_ptr->layers[j].batch_normalize) 
				{			    
					networkLayerInfoArr[i].flBeta = 0.000001f;
					networkLayerInfoArr[i].darknetNormScaleBias = true;
				}
				if (net_ptr->layers[j].activation == LEAKY) 
				{
					networkLayerInfoArr[i].activation = espresso::LEAKY;
				}
				else if (net_ptr->layers[j].activation == LINEAR) 
				{
					networkLayerInfoArr[i].activation = espresso::LINEAR;
				}
				networkLayerInfoArr[i].darknetAct = true;
				getWeights(precision, &net_ptr->layers[j], networkLayerInfoArr[i], fxPtLen, numFracBits);
				if (networkLayerInfoArr[i].backend == espresso::FPGA_BACKEND)
				{
					createKernelGroups(networkLayerInfoArr[i]);
				}
				break;
			}
			case ROUTE:
			{
				networkLayerInfoArr[i].layerType = "Concat";
				networkLayerInfoArr[i].layerName = std::to_string(i) + "_" + networkLayerInfoArr[i].layerType;
				setBaseLayerInfo(j, &net_ptr->layers[j], networkLayerInfoArr[i], precision, fxPtLen, numFracBits, net_ptr);
				for (int k = 0; k < net_ptr->layers[j].n; k++) 
				{
					networkLayerInfoArr[i].bottomLayerNames.push_back(networkLayerInfoArr[net_ptr->layers[j].input_layers[k]].layerName);
				}
				break;
			} 
			case SHORTCUT:
			{
				networkLayerInfoArr[i].layerType = "Residual";
				networkLayerInfoArr[i].layerName = std::to_string(i) + "_" + networkLayerInfoArr[i].layerType;
				setBaseLayerInfo(j, &net_ptr->layers[j], networkLayerInfoArr[i], precision, fxPtLen, numFracBits, net_ptr);
				networkLayerInfoArr[i].bottomLayerNames.push_back(networkLayerInfoArr[i - 1].layerName);
				networkLayerInfoArr[i].bottomLayerNames.push_back(networkLayerInfoArr[net_ptr->layers[j].index + 1].layerName);
				break;
			} 
			case YOLO:
			{
				networkLayerInfoArr[i].layerType = "YOLO";
				networkLayerInfoArr[i].layerName = std::to_string(i) + "_" + networkLayerInfoArr[i].layerType;
				setBaseLayerInfo(j, &net_ptr->layers[j], networkLayerInfoArr[i], precision, fxPtLen, numFracBits, net_ptr);
				networkLayerInfoArr[i].darknet_n_param = net_ptr->layers[j].n;
				networkLayerInfoArr[i].darknet_classes_param = net_ptr->layers[j].classes;
				networkLayerInfoArr[i].darknet_outputs_param = net_ptr->layers[j].outputs;
				break;
			} 
			case UPSAMPLE:
			{
				networkLayerInfoArr[i].layerType = "UpSample";
				networkLayerInfoArr[i].layerName = std::to_string(i) + "_" + networkLayerInfoArr[i].layerType;
				setBaseLayerInfo(j, &net_ptr->layers[j], networkLayerInfoArr[i], precision, fxPtLen, numFracBits, net_ptr);
				break;
			}
			default:
			{
				std::cout << "Skipped Darknet Layer " << i << std::endl;
				continue;
			}
		}
        j++;
	}
	setLayerConnections(networkLayerInfoArr);
}


std::vector<int> getYOLOOutputLayers(std::vector<espresso::layerInfo_obj> &networkLayerInfo) {
	std::vector<int> outputLayers;
	for (int i = 0; i < networkLayerInfo.size(); i++) {
		if (networkLayerInfo[i].layerType == "YOLO") {
			outputLayers.push_back(i);
		}
	}
	return outputLayers;
}


void post_yolo(network* yolo_net, espresso::Network* net, char* imgOut_FN, char* cocoNames_FN, image im)
{
	for (int i = 0; i < net->m_cnn.size(); i++) 
	{
		if (net->m_cnn[i]->m_layerType == "YOLO") 
		{
			layer *l = &yolo_net->layers[i - 1];
			l->output = (float*)net->m_cnn[i]->m_blob.flData;
		}
	}
	char **names = get_labels(cocoNames_FN);
	image **alphabet = load_alphabet();
	int nboxes = 0;
	layer l = yolo_net->layers[yolo_net->n - 1];
	detection *dets = get_network_boxes(yolo_net, im.w, im.h, THRESH, HIER_THRESH, 0, 1, &nboxes);
	do_nms_sort(dets, nboxes, l.classes, NMS_THRESH);
	draw_detections(im, dets, nboxes, 0.5f, names, alphabet, l.classes);
	save_image(im, "predictions");
	free_detections(dets, nboxes);
}


void cfgInputLayer(espresso::precision_t precision, espresso::layerInfo_obj& networkLayerInfo, network *yolo_net, espresso::Network* net, image im)
{
	if (precision == espresso::FLOAT) 
	{
		net->m_cnn[0]->m_blob.flData = im.data;
	}
	else 
	{
		int numValues = networkLayerInfo.numInputRows * networkLayerInfo.numInputCols * networkLayerInfo.inputDepth;
		net->m_cnn[0]->m_blob.fxData = new fixedPoint_t[numValues];
		for (int i = 0; i < numValues; i++) 
		{
			net->m_cnn[0]->m_blob.fxData[i] = fixedPoint::create(networkLayerInfo.dinFxPtLength, networkLayerInfo.dinNumFracBits, im.data[i]);
		}
	}

}


int main(int argc, char **argv) 
{
	espresso::precision_t precision = espresso::FLOAT;
	espresso::backend_t backend = espresso::DARKNET_BACKEND;
	network *yolo_net = NULL;
	std::vector<espresso::layerInfo_obj> networkLayerInfoArr;
	std::string yolov3_cfg_FN = "/home/ikenna/IkennaWorkSpace/darknet/cfg/yolov3.cfg"; 
	std::string yolov3_whts_FN = "/home/ikenna/IkennaWorkSpace/darknet/cfg/yolov3.weights";	
	darknetDataTransform(
		&yolo_net, 
		networkLayerInfoArr, 
		precision,
		backend,
		(char*)yolov3_cfg_FN.c_str(), 
		(char*)yolov3_whts_FN.c_str(),
		32,
		16
	);
	std::vector<int> outputLayers = getYOLOOutputLayers(networkLayerInfoArr);
	std::string imgFN = "/home/ikenna/IkennaWorkSpace/darknet/data/dog.jpg";
	espresso::Network net(networkLayerInfoArr, outputLayers);
	image im = load_image_color((char*)imgFN.c_str(), 0, 0);
	image sized = letterbox_image(im, networkLayerInfoArr[0].numInputRows, networkLayerInfoArr[0].numInputCols);
	cfgInputLayer(precision, networkLayerInfoArr[0], yolo_net, &net, sized);
	set_batch_network(yolo_net, 1);
	net.Forward();
	std::string imgOut_FN = "predictions";
	std::string cocoNames_FN = "/home/ikenna/IkennaWorkSpace/darknet/data/coco.names";
	post_yolo(yolo_net, &net, (char*)imgOut_FN.c_str(), (char*)cocoNames_FN.c_str(), sized);
	free_image(im);
	free_image(sized);
	
	
	// int numKernels = networkLayerInfoArr[1].kernel_group_arr[0][0]->get_num_kernels();
	// int nCols = networkLayerInfoArr[1].kernel_group_arr[0][0]->get_kernel_width();
	// int nRows = networkLayerInfoArr[1].kernel_group_arr[0][0]->get_kernel_height();
	// int nChnls = networkLayerInfoArr[1].kernel_group_arr[0][0]->get_kernel_depth();
	// float *flFilterData_ptr = networkLayerInfoArr[1].flFilterData;
	// for (int i = 0; i < numKernels; i++)
	// {
	// 	std::string fname;
	// 	if (i < 10)
	// 	{
	// 		fname = "kernels/kernel0" + std::to_string(i) + ".txt";
	// 	}
	// 	else
	// 	{
	// 		fname = "kernels/kernel" + std::to_string(i) + ".txt";
	// 	}
	// 	std::ofstream *fd = new std::ofstream(fname);
	// 	for (int d0 = 0; d0 < nChnls; d0++) 
	// 	{
	// 		for (int d1 = 0; d1 < nRows; d1++) 
	// 		{
	// 			for (int d2 = 0; d2 < nCols; d2++) 
	// 			{
	// 				int idx = index4D(
	// 					nChnls,
	// 					nRows,
	// 					nCols,
	// 					i,
	// 					d0,
	// 					d1,
	// 					d2
	// 				);
	// 				fd[0] << flFilterData_ptr[idx] << " ";
	// 			}
	// 			fd[0] << std::endl;
	// 		}
	// 		fd[0] << std::endl;
	// 		fd[0] << std::endl;
	// 	}
	// 	fd->close();
	// 	kernel *kernel_i = networkLayerInfoArr[1].kernel_group_arr[0][0]->get_kernel(i);
	// 	if (i < 10)
	// 	{
	// 		fname = "kernels_esp/kernel_esp0" + std::to_string(i) + ".txt";
	// 	}
	// 	else
	// 	{
	// 		fname = "kernels_esp/kernel_esp" + std::to_string(i) + ".txt";
	// 	}
	// 	fd = new std::ofstream(fname);
	// 	kernel_i->print(fd);
	// 	fd->close();
	// }
	// 
	// 	
	// kernel_group_config *kgc = networkLayerInfoArr[1].kernel_group_arr[0][0]->get_config();
	// kernel_group  *kg = networkLayerInfoArr[1].kernel_group_arr[0][0];
	// int nKernels = kg->get_num_kernels();
	// for (int k = 0; k < nKernels; k++) 
	// {
	// 	std::string fname;
	// 	if (k < 10)
	// 	{
	// 		fname = "kernels_cfg/kernel_cfg0" + std::to_string(k) + ".txt";
	// 	}
	// 	else
	// 	{
	// 		fname = "kernels_cfg/kernel_cfg" + std::to_string(k) + ".txt";
	// 	}
	// 	std::ofstream *fd = new std::ofstream(fname);
	// 	for (int c = 0; c < QUAD_MAX_KRNL_DEPTH; c++) 
	// 	{ 
	// 		for (int v = 0; v < NUM_KERNEL_3x3_VAL; v++) 
	// 		{
	// 			fd[0] << fixedPoint::toFloat(16, kgc->get_data(k, c, v)) << " ";
	// 		}
	// 		fd[0] << std::endl;
	// 	}
	// }
	// int length;
	// fixedPoint_t* bytes = (fixedPoint_t*)kgc->get_bytes(length);
	// for (int k = 0; k < nKernels; k++) 
	// {
	// 	std::string fname;
	// 	if (k < 10)
	// 	{
	// 		fname = "kernels_hw/kernels_hw0" + std::to_string(k) + ".txt";
	// 	}
	// 	else
	// 	{
	// 		fname = "kernels_hw/kernels_hw" + std::to_string(k) + ".txt";
	// 	}
	// 	std::ofstream *fd = new std::ofstream(fname);
	// 	for (int v = 0; v < NUM_KERNEL_3x3_VAL; v++) 
	// 	{ 
	// 		for (int c = 0; c < QUAD_MAX_KRNL_DEPTH; c++) 
	// 		{
	// 			int i0 = index3D(
	// 				NUM_KERNEL_3x3_VAL,
	// 				QUAD_MAX_KRNL_DEPTH,
	// 				k,
	// 				v,
	// 				c
	// 			);
	// 			fd[0] << fixedPoint::toFloat(16, bytes[i0]) << " ";
	// 		}
	// 		fd[0] << std::endl;
	// 	}
	// }
	
	
    return 0;
}
