// Project includes
#include "espressoTester.hpp"


void cfgInputLayer(const image& im, espresso::Network* net, const espresso::layerInfo_obj& networkLayerInfo, espresso::precision_t precision)
{
	int numValues = networkLayerInfo.numInputRows * networkLayerInfo.numInputCols * networkLayerInfo.inputDepth;
	if (precision == espresso::FLOAT) 
	{
		memcpy(net->m_cnn[0]->m_blob.flData, im.data, numValues * sizeof(float));
	}
	else 
	{
		net->m_cnn[0]->m_blob.fxData = new fixedPoint_t[numValues];
		for (int i = 0; i < numValues; i++) 
		{
			net->m_cnn[0]->m_blob.fxData[i] = fixedPoint::create(networkLayerInfo.dinFxPtLength, networkLayerInfo.dinNumFracBits, im.data[i]);
		}
	}

}


void createDataLayer(espresso::layerInfo_obj& networkLayerInfo, network* net, espresso::precision_t precision)
{
	networkLayerInfo.precision = precision;
	networkLayerInfo.layerType = espresso::INPUT;
	networkLayerInfo.layerName = "0_Data";
	networkLayerInfo.inputDepth = net->layers[0].c;
	networkLayerInfo.numInputRows = net->layers[0].h;
	networkLayerInfo.numInputCols = net->layers[0].w;	
}


std::vector<std::vector<kernel_group*>> createKernelGroupArr(int numKernels, int channels, int height, int width, int fxPtLength, int numFracBits) 
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
					numFracBits));
			}
			channel_rem -= QUAD_MAX_KRNL_DEPTH;
		}
		kernel_rem -= QUAD_MAX_NUM_KRNL;
	}
	return kernel_group_arr;
}


void createKernelGroups(espresso::layerInfo_obj& networkLayerInfo)
{
	int kernelDepth = networkLayerInfo.inputDepth;
	int numKernels = networkLayerInfo.outputDepth;
	int kernelWidth = networkLayerInfo.numKernelCols;
	int kernelHeight = networkLayerInfo.numKernelRows;
	networkLayerInfo.kernel_group_arr = createKernelGroupArr(
		numKernels,
		kernelDepth,
		kernelHeight, 
		kernelWidth, 
		networkLayerInfo.whtFxPtLength,
		networkLayerInfo.whtNumFracBits);
	setKrnlGrpData(networkLayerInfo, numKernels);
}


std::vector<espresso::layerInfo_obj> darknetDataTransform(
	network** net, 
	char* configFileName, 
	char* weightFileName,
	espresso::backend_t backend,
	espresso::precision_t precision,
	int fxPtLen,
	int numFracBits
) {
	*net = parse_network_cfg(configFileName);
	network* net_ptr = *net;
	load_weights(net_ptr, weightFileName);
	int numLayers = net_ptr->n + 1;
	std::vector<espresso::layerInfo_obj> networkLayerInfoArr;
	networkLayerInfoArr.resize(numLayers);
	int j = 0;
	createDataLayer(networkLayerInfoArr[0], net_ptr, precision);
	for (int i = 1; i < numLayers; i++) 
	{
		switch (net_ptr->layers[j].type)
		{
			case CONVOLUTIONAL:
			{
				networkLayerInfoArr[i].layerType = espresso::CONVOLUTION;
				networkLayerInfoArr[i].layerName = std::to_string(i) + "_Convolution";
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
				getWeights(networkLayerInfoArr[i], &net_ptr->layers[j], precision, fxPtLen, numFracBits);
				if (networkLayerInfoArr[i].backend == espresso::FPGA_BACKEND)
				{
					createKernelGroups(networkLayerInfoArr[i]);
				}
				break;
			}
			case ROUTE:
			{
				networkLayerInfoArr[i].layerType = espresso::CONCAT;
				networkLayerInfoArr[i].layerName = std::to_string(i) + "_Concat";
				setBaseLayerInfo(j, &net_ptr->layers[j], networkLayerInfoArr[i], precision, fxPtLen, numFracBits, net_ptr);
				for (int k = 0; k < net_ptr->layers[j].n; k++) 
				{
					networkLayerInfoArr[i].bottomLayerNames.push_back(networkLayerInfoArr[net_ptr->layers[j].input_layers[k] + 1].layerName);
				}
				break;
			} 
			case SHORTCUT:
			{
				networkLayerInfoArr[i].layerType = espresso::RESIDUAL;
				networkLayerInfoArr[i].layerName = std::to_string(i) + "_Residual";
				setBaseLayerInfo(j, &net_ptr->layers[j], networkLayerInfoArr[i], precision, fxPtLen, numFracBits, net_ptr);
				networkLayerInfoArr[i].bottomLayerNames.push_back(networkLayerInfoArr[i - 1].layerName);
				networkLayerInfoArr[i].bottomLayerNames.push_back(networkLayerInfoArr[net_ptr->layers[j].index + 1].layerName);
				break;
			} 
			case YOLO:
			{
				networkLayerInfoArr[i].layerType = espresso::YOLO;
				networkLayerInfoArr[i].layerName = std::to_string(i) + "_YOLO";
				setBaseLayerInfo(j, &net_ptr->layers[j], networkLayerInfoArr[i], precision, fxPtLen, numFracBits, net_ptr);
				networkLayerInfoArr[i].darknet_n_param = net_ptr->layers[j].n;
				networkLayerInfoArr[i].darknet_classes_param = net_ptr->layers[j].classes;
				networkLayerInfoArr[i].darknet_outputs_param = net_ptr->layers[j].outputs;
				break;
			} 
			case UPSAMPLE:
			{
				networkLayerInfoArr[i].layerType = espresso::UPSAMPLE;
				networkLayerInfoArr[i].layerName = std::to_string(i) + "_UpSample";
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
	return networkLayerInfoArr;
}


void getFlPtWeights(espresso::layerInfo_obj& networkLayerInfo, layer* layer_i) 
{
	networkLayerInfo.flFilterData = new float[networkLayerInfo.numFilterValues];
	memcpy(
		networkLayerInfo.flFilterData, 
		layer_i->weights, 
		networkLayerInfo.numFilterValues * sizeof(float));
	networkLayerInfo.flBiasData = new float[networkLayerInfo.outputDepth];
	memcpy(
		networkLayerInfo.flBiasData, 
		layer_i->biases, 
		networkLayerInfo.outputDepth * sizeof(float));
	if (networkLayerInfo.darknetNormScaleBias) 
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
				flFilterData_ptr[idx] = flFilterData_ptr[idx] * scales / sqrtf(rolling_variance);
			}
		}	    
	}
}


void getFxPtWeights(espresso::layerInfo_obj& networkLayerInfo) 
{
	networkLayerInfo.fxFilterData = new fixedPoint_t[networkLayerInfo.numFilterValues];
	fixedPoint_t *fxFilterData = networkLayerInfo.fxFilterData;
	int whtFxPtLength = networkLayerInfo.whtFxPtLength;
	int whtNumFracBits = networkLayerInfo.whtNumFracBits;	
	for (int k = 0; k < networkLayerInfo.numFilterValues; k++)
	{
		fxFilterData[k] = fixedPoint::create(
			whtFxPtLength, 
			whtNumFracBits, 
			networkLayerInfo.flFilterData[k]);
	}
	networkLayerInfo.fxBiasData = new fixedPoint_t[networkLayerInfo.outputDepth];
	fixedPoint_t *fxBiasData = networkLayerInfo.fxBiasData;
	int biasFxPtLength = networkLayerInfo.biasFxPtLength;
	int biasNumFracBits = networkLayerInfo.biasFxPtLength;
	for (int k = 0; k < networkLayerInfo.outputDepth; k++) 
	{
		fxBiasData[k] = fixedPoint::create(
			biasFxPtLength, 
			biasNumFracBits, 
			networkLayerInfo.flBiasData[k]);
	}
}


void getLayerPrec(std::vector<layerPrec_t>& layerPrecArr)
{
    layerPrecArr[0].dinNorm = true;	
	int maxtDoutIntBits = -INT_MAX;
    for(int i = 1; i < layerPrecArr.size(); i++)
    {
	    layerPrecArr[i].dinFxPtLength = layerPrecArr[i - 1].doutFxPtLength;
	    layerPrecArr[i].dinNumFracBits	= layerPrecArr[i - 1].doutNumFracBits;
	    if (layerPrecArr[i].layerType == espresso::CONVOLUTION) {
		    float maxFilMag = fabs(layerPrecArr[i].maxFilter);
		    float minFilMag = fabs(layerPrecArr[i].minFilter);
		    int numIntBits0;
		    int numIntBits1;
		    int whtNumIntBits;
		    int doutNumIntBits;
		    int dinNumIntBits = layerPrecArr[i].dinFxPtLength - layerPrecArr[i].dinNumFracBits - 1;
		    if (maxFilMag >= 1 && minFilMag >= 1) 
		    {
			    numIntBits0 = std::max(int(ceilf(log2(maxFilMag))), 1);
			    numIntBits1 = std::max(int(ceilf(log2(minFilMag))), 1);
			    whtNumIntBits = std::max(numIntBits0, numIntBits1);
			    if (whtNumIntBits > espresso::YOLO_MAX_NUM_INT_BITS) 
			    {
				    exit(1);
			    }
			    doutNumIntBits = dinNumIntBits + whtNumIntBits;
		    }
		    else if (maxFilMag < 1 && minFilMag >= 1) 
		    {
			    whtNumIntBits = std::max(int(ceilf(log2(minFilMag))), 1);
			    if (whtNumIntBits > espresso::YOLO_MAX_NUM_INT_BITS) 
			    {
				    exit(1);
			    }
			    doutNumIntBits = dinNumIntBits + whtNumIntBits;
		    }
		    else if (maxFilMag >= 1 && minFilMag < 1) 
		    {
			    whtNumIntBits = std::max(int(ceilf(log2(maxFilMag))), 1);
			    if (whtNumIntBits > espresso::YOLO_MAX_NUM_INT_BITS) 
			    {
				    exit(1);
			    }
			    doutNumIntBits = dinNumIntBits + whtNumIntBits;
		    }
		    else if (maxFilMag < 1 && minFilMag < 1) {
			    numIntBits0 = ceilf(log2(1.0f / maxFilMag));
			    numIntBits1 = ceilf(log2(1.0f / minFilMag));
			    whtNumIntBits = std::min(std::min(numIntBits0, numIntBits1), espresso::YOLO_MAX_NUM_FRAC_BITS);
			    doutNumIntBits = dinNumIntBits - whtNumIntBits;
		    }
		    if (layerPrecArr[i - 1].dinNorm) 
		    {
			    doutNumIntBits = whtNumIntBits;
		    }
		    int doutNumFracBits = std::min(espresso::YOLO_DEF_FXPT_LEN - (doutNumIntBits + 1), espresso::YOLO_MAX_NUM_FRAC_BITS);
		    if (doutNumIntBits > maxtDoutIntBits) 
		    {
			    maxtDoutIntBits = doutNumIntBits;
		    }
		    layerPrecArr[i].leakyFxPtLength = espresso::YOLO_DEF_FXPT_LEN;
		    layerPrecArr[i].leakyNumFracBits = espresso::YOLO_DEF_NUM_FRAC_BITS;
		    layerPrecArr[i].whtFxPtLength = espresso::YOLO_DEF_FXPT_LEN;
		    layerPrecArr[i].whtNumFracBits = espresso::YOLO_DEF_FXPT_LEN - (whtNumIntBits + 1);
		    layerPrecArr[i].doutFxPtLength = espresso::YOLO_DEF_FXPT_LEN;
		    layerPrecArr[i].doutNumFracBits = doutNumFracBits;	
	    }
	    else 
	    {
		    layerPrecArr[i].doutFxPtLength	= layerPrecArr[i].dinFxPtLength;
		    layerPrecArr[i].doutNumFracBits = layerPrecArr[i].dinNumFracBits;
	    }
    }
	for (int i = 0; i < layerPrecArr.size(); i++)
	{
		layerPrecArr[i].maxtDoutIntBits = maxtDoutIntBits;
	}
}


void getLayerStats(const std::vector<espresso::layerInfo_obj>& networkLayerInfoArr, std::vector<layerPrec_t>& layerPrecArr)
{
 	for(int i = 0; i < networkLayerInfoArr.size(); i++)
	{
		if (layerPrecArr[i].layerType == espresso::CONVOLUTION)
		{
			int numBias = networkLayerInfoArr[i].outputDepth;
			for (int j = 0; j < numBias; j++)
			{
				layerPrecArr[i].minBias = std::min(
				    layerPrecArr[i].minBias,
					networkLayerInfoArr[i].flBiasData[j]
				);
				layerPrecArr[i].maxBias = std::max(
					layerPrecArr[i].maxBias,
					networkLayerInfoArr[i].flBiasData[j]
				);
			}
			for (int j = 0; j < networkLayerInfoArr[i].numFilterValues; j++)
			{
				layerPrecArr[i].minFilter = std::min(
				    layerPrecArr[i].minFilter,
					networkLayerInfoArr[i].flFilterData[j]
				);
				layerPrecArr[i].maxFilter = std::max(
				    layerPrecArr[i].maxFilter,
					networkLayerInfoArr[i].flFilterData[j]
				);
				
			}
		}
	}
}


void getWeights(espresso::layerInfo_obj& networkLayerInfo, layer* layer_i, espresso::precision_t precision, int fxPtLen, int numFracBits)
{
	getFlPtWeights(networkLayerInfo, layer_i);
	if (precision == espresso::FIXED)
	{
		getFxPtWeights(networkLayerInfo);
	}
}


std::vector<int> getYOLOOutputLayers(std::vector<espresso::layerInfo_obj>& networkLayerInfo) 
{
	std::vector<int> outputLayers;
	for (int i = 0; i < networkLayerInfo.size(); i++) 
	{
		if (networkLayerInfo[i].layerType == espresso::YOLO) 
		{
			outputLayers.push_back(i);
		}
	}
	return outputLayers;
}


void post_yolo(espresso::Network* net, network* yolo_net, char* cocoNames_FN, image im, char* imgOut_FN)
{
	for (int i = 0; i < net->m_cnn.size(); i++) 
	{
		if (net->m_cnn[i]->m_layerType == espresso::YOLO) 
		{
			layer* l = &yolo_net->layers[i - 1];
			l->output = (float*)net->m_cnn[i]->m_blob.flData;
		}
	}
	char **names = get_labels(cocoNames_FN);
	image **alphabet = load_alphabet();
	int nboxes = 0;
	layer l = yolo_net->layers[yolo_net->n - 1];
	detection *dets = get_network_boxes(yolo_net, im.w, im.h, espresso::THRESH, espresso::HIER_THRESH, 0, 1, &nboxes);
	do_nms_sort(dets, nboxes, l.classes, espresso::NMS_THRESH);
	draw_detections(im, dets, nboxes, 0.5f, names, alphabet, l.classes);
	save_image(im, imgOut_FN);
	free_detections(dets, nboxes);
}


std::vector<layerPrec_t> profileYOLOWeights(const std::vector<espresso::layerInfo_obj>& networkLayerInfoArr)
{
    std::vector<layerPrec_t> layerPrecArr;
    layerPrecArr.resize(networkLayerInfoArr.size());
    for(int i = 0; i < layerPrecArr.size(); i++)
    {
	    layerPrecArr[i].dinNorm = false;
        layerPrecArr[i].minBias = FLT_MAX;
        layerPrecArr[i].maxBias = -FLT_MAX;
        layerPrecArr[i].minFilter = FLT_MAX;
	    layerPrecArr[i].maxFilter = -FLT_MAX;
	    layerPrecArr[i].dinFxPtLength = espresso::YOLO_DEF_FXPT_LEN;
	    layerPrecArr[i].dinNumFracBits = espresso::YOLO_DEF_NUM_FRAC_BITS;
	    layerPrecArr[i].whtFxPtLength = espresso::YOLO_DEF_FXPT_LEN;
	    layerPrecArr[i].whtNumFracBits = espresso::YOLO_DEF_NUM_FRAC_BITS;
	    layerPrecArr[i].doutFxPtLength = espresso::YOLO_DEF_FXPT_LEN;
	    layerPrecArr[i].doutNumFracBits = espresso::YOLO_DEF_NUM_FRAC_BITS;	
	    layerPrecArr[i].biasFxPtLength = espresso::YOLO_DEF_FXPT_LEN;
	    layerPrecArr[i].biasNumFracBits = espresso::YOLO_DEF_NUM_FRAC_BITS;
	    layerPrecArr[i].leakyFxPtLength = espresso::YOLO_DEF_FXPT_LEN;
	    layerPrecArr[i].leakyNumFracBits = espresso::YOLO_DEF_NUM_FRAC_BITS;
	    layerPrecArr[i].layerType = networkLayerInfoArr[i].layerType;
    }
	getLayerStats(networkLayerInfoArr, layerPrecArr);
    getLayerPrec(layerPrecArr);
	return layerPrecArr;
}


void setBaseLayerInfo(int i, layer* layer_i, espresso::layerInfo_obj& networkLayerInfo, espresso::precision_t precision, int fxPtLen, int numFracBits, network* yolo_net)
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


void setKrnlGrpData(espresso::layerInfo_obj& networkLayerInfo, int numKernels)
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


void setLayerConnections(std::vector<espresso::layerInfo_obj>& networkLayerInfoArr)
{
	for (int i = 0; i < networkLayerInfoArr.size(); i++) 
	{
		if (networkLayerInfoArr[i].layerType != espresso::CONCAT && networkLayerInfoArr[i].layerType != espresso::RESIDUAL) 
		{
			networkLayerInfoArr[i].bottomLayerNames.push_back(networkLayerInfoArr[i - 1].layerName);
		} 
	}	
	for (int i = 0; i < networkLayerInfoArr.size(); i++) 
	{
		networkLayerInfoArr[i].topLayerNames.push_back(networkLayerInfoArr[i].layerName);
	}
}


void setLayerPrec(std::vector<espresso::layerInfo_obj>& networkLayerInfoArr, std::vector<layerPrec_t> layerPrecArr)
{
	networkLayerInfoArr[0].dinFxPtLength = layerPrecArr[0].dinFxPtLength;
	networkLayerInfoArr[0].dinNumFracBits = layerPrecArr[0].dinNumFracBits;
	networkLayerInfoArr[0].doutFxPtLength = layerPrecArr[0].doutFxPtLength;
	networkLayerInfoArr[0].doutNumFracBits = layerPrecArr[0].doutNumFracBits;
	for (int i = 1; i < networkLayerInfoArr.size(); i++) 
	{
		if (networkLayerInfoArr[i].layerType == espresso::CONVOLUTION) 
		{
			networkLayerInfoArr[1].dinFxPtLength = (i == 1) ? networkLayerInfoArr[0].doutFxPtLength : espresso::YOLO_DEF_FXPT_LEN;
			networkLayerInfoArr[1].dinNumFracBits = (i == 1) ? networkLayerInfoArr[0].doutNumFracBits : espresso::YOLO_DEF_FXPT_LEN - (layerPrecArr[i].maxtDoutIntBits + 1);
			// assuming bias rep will never be larger than product of wht and din rep
			int resFxPtLength  = networkLayerInfoArr[i].dinFxPtLength + networkLayerInfoArr[i].whtFxPtLength;
			int resNumFracBits = networkLayerInfoArr[i].dinNumFracBits + networkLayerInfoArr[i].whtNumFracBits;				
			networkLayerInfoArr[i].biasFxPtLength = resFxPtLength;
			networkLayerInfoArr[i].biasNumFracBits = resNumFracBits;
			networkLayerInfoArr[i].doutFxPtLength = espresso::YOLO_DEF_FXPT_LEN;
			networkLayerInfoArr[i].doutNumFracBits = espresso::YOLO_DEF_FXPT_LEN - (layerPrecArr[i].maxtDoutIntBits + 1);
			fixedPoint::SetParam(
				espresso::YOLO_DEF_FXPT_LEN, 
				espresso::YOLO_DEF_NUM_FRAC_BITS, 
				layerPrecArr[i].whtFxPtLength,
				layerPrecArr[i].whtNumFracBits,
				networkLayerInfoArr[i].fxFilterData,
				networkLayerInfoArr[i].numFilterValues
			);
			fixedPoint::SetParam(
				espresso::YOLO_DEF_FXPT_LEN, 
				espresso::YOLO_DEF_NUM_FRAC_BITS,
				networkLayerInfoArr[i].biasFxPtLength, 
				networkLayerInfoArr[i].biasNumFracBits, 
				networkLayerInfoArr[i].fxBiasData, 
				networkLayerInfoArr[i].outputDepth
			);
		}
	}
}



int main(int argc, char **argv) 
{
	espresso::precision_t precision = espresso::FIXED;
	espresso::backend_t backend = espresso::DARKNET_BACKEND;
	network* yolo_net = NULL;
	std::string yolov3_cfg_FN = "/home/ikenna/IkennaWorkSpace/darknet/cfg/yolov3.cfg"; 
	std::string yolov3_whts_FN = "/home/ikenna/IkennaWorkSpace/darknet/cfg/yolov3.weights";	
	std::vector<espresso::layerInfo_obj> networkLayerInfoArr = darknetDataTransform(
		&yolo_net, 
		(char*)yolov3_cfg_FN.c_str(), 
		(char*)yolov3_whts_FN.c_str(),
		backend,
		precision,
		espresso::YOLO_DEF_FXPT_LEN,
		espresso::YOLO_DEF_NUM_FRAC_BITS
	);
	std::vector<int> outputLayers = getYOLOOutputLayers(networkLayerInfoArr);
	std::vector<layerPrec_t> layerPrecArr = profileYOLOWeights(networkLayerInfoArr);
	setLayerPrec(networkLayerInfoArr, layerPrecArr);
	std::string imgFN = "/home/ikenna/IkennaWorkSpace/darknet/data/dog.jpg";
	espresso::Network net(networkLayerInfoArr, outputLayers);
	image im = load_image_color((char*)imgFN.c_str(), 0, 0);
	image sized = letterbox_image(im, networkLayerInfoArr[0].numInputRows, networkLayerInfoArr[0].numInputCols);
	cfgInputLayer(sized, &net, networkLayerInfoArr[0], precision);
	set_batch_network(yolo_net, 1);
	net.Forward();
	std::string imgOut_FN = "predictions";
	std::string cocoNames_FN = "/home/ikenna/IkennaWorkSpace/darknet/data/coco.names";
	post_yolo(&net, yolo_net, (char*)cocoNames_FN.c_str(), sized, (char*)imgOut_FN.c_str());
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
