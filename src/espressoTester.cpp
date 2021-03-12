#include "espressoTester.hpp"
using namespace std;


#ifdef CAFFE_DATA_PARSER
espresso::layerType_t getEspLayType(string layerType)
{
    if(layerType == "Input")
    {
        return espresso::INPUT;
    }
    else if(layerType == "Convolution")
    {
        return espresso::CONVOLUTION;
    }
    else if(layerType == "Pooling_MAX")
    {
        return espresso::POOLING_MAX;
    }
    else if(layerType == "Pooling_AVE")
    {
        return espresso::POOLING_AVG;
    }
    else if(layerType == "Permute")
    {
        return espresso::PERMUTE;
    }
    else if(layerType == "Flatten")
    {
        return espresso::FLATTEN;
    }
    else if(layerType == "Eltwise")
    {
        return espresso::RESIDUAL;
    }
    else if(layerType == "DetectionOutput")
    {
        return espresso::DETECTION_OUTPUT;
    }
    else if(layerType == "PriorBox")
    {
        return espresso::PRIOR_BOX;
    }
    else if(layerType == "Reshape")
    {
        return espresso::RESHAPE;
    }
    else if(layerType == "InnerProduct")
    {
        return espresso::INNERPRODUCT;
    }
    else if(layerType == "Softmax")
    {
        return espresso::SOFTMAX;
    }
    else if(layerType == "Concat")
    {
        return espresso::CONCAT;
    }
    else if(layerType == "PSROIPooling")
    {
        return espresso::PSROIPoolingLayer;
    }
}


int findCaffeLayer(string layerName, vector<caffeDataParser::layerInfo_t> caffeLayerInfo)
{
    for(int i = 0; i < caffeLayerInfo.size(); i++)
    {
        if(caffeLayerInfo[i].layerName == layerName)
        {
            return i;
        }
    }
}


vector<espresso::layerInfo_obj*> caffeDataTransform(vector<caffeDataParser::layerInfo_t> caffeLayerInfo, espresso::backend_t backend)
{
	printf("Loading Caffe Data.....\n");
    vector<espresso::layerInfo_obj*> networkLayerInfoArr;
    espresso::layerInfo_obj* layerInfo;
    for(int i = 0; i < caffeLayerInfo.size(); i++)
    {
        if(caffeLayerInfo[i].layerType == "ReLU")
        {
            // merge with top layer
            continue;
        }
        else if(caffeLayerInfo[i].layerType == "BatchNorm")
        {
            // merge with top layer
            continue;
        }
        else if(caffeLayerInfo[i].layerType == "Scale")
        {
            // merge with top layer
            continue;
        }
        layerInfo = new espresso::layerInfo_obj();
        layerInfo->backend                 = backend;
        layerInfo->layerType               = getEspLayType(caffeLayerInfo[i].layerType);
        layerInfo->precision               = espresso::FLOAT;
        layerInfo->layerName               = caffeLayerInfo[i].layerName;
        layerInfo->topLayerNames           = caffeLayerInfo[i].topLayerNames;
        layerInfo->bottomLayerNames        = caffeLayerInfo[i].bottomLayerNames;
        layerInfo->numInputRows            = caffeLayerInfo[i].numInputRows;
        layerInfo->numInputCols            = caffeLayerInfo[i].numInputCols;
        layerInfo->inputDepth              = caffeLayerInfo[i].inputDepth;
        layerInfo->outputDepth             = caffeLayerInfo[i].outputDepth;
        layerInfo->numKernelRows           = caffeLayerInfo[i].numKernelRows;
        layerInfo->numKernelCols           = caffeLayerInfo[i].numKernelCols;
        layerInfo->stride                  = caffeLayerInfo[i].stride;
        layerInfo->padding                 = caffeLayerInfo[i].padding;
        layerInfo->globalPooling           = caffeLayerInfo[i].globalPooling;
        layerInfo->group                   = caffeLayerInfo[i].group;
        layerInfo->localSize               = caffeLayerInfo[i].localSize;
        layerInfo->alpha                   = caffeLayerInfo[i].alpha;
        layerInfo->flBeta                  = caffeLayerInfo[i].beta;
        layerInfo->numKernels              = layerInfo->outputDepth;
        layerInfo->dilation                = caffeLayerInfo[i].dilation;
        if(caffeLayerInfo[i].layerType == "Convolution" || caffeLayerInfo[i].layerType == "InnerProduct")
        {
            // layerInfo->flFilterData = (float*)malloc(    caffeLayerInfo[i].numFilterValues
            //                                                     * sizeof(float)
            //                                                 );
            // memcpy  (   layerInfo->flFilterData,
            //             caffeLayerInfo[i].filterData,
            //             caffeLayerInfo[i].numFilterValues
            //             * sizeof(float)
            //         );
            // layerInfo->flBiasData = (float*)malloc(  caffeLayerInfo[i].numBiasValues
            //                                                 * sizeof(float)
            //                                              );
            // memcpy  (   layerInfo->flBiasData,
            //             caffeLayerInfo[i].biasData,
            //             caffeLayerInfo[i].numBiasValues
            //             * sizeof(float)
            //         );
        }
        else
        {
            layerInfo->flFilterData = NULL;
            layerInfo->flBiasData = NULL;
        }
        networkLayerInfoArr.push_back(layerInfo);
    }
    for(int i = 1; i < networkLayerInfoArr.size(); i++)
    {
        networkLayerInfoArr[i]->topLayerNames.clear();
        networkLayerInfoArr[i]->topLayerNames.push_back(networkLayerInfoArr[i]->layerName);
    }
    return networkLayerInfoArr;
}
#endif

void cfgInputLayer(const image& im, espresso::CNN_Network* net, const espresso::layerInfo_obj* networkLayerInfo, espresso::precision_t precision)
{
    int numValues = networkLayerInfo->numInputRows * networkLayerInfo->numInputCols * networkLayerInfo->inputDepth;
    net->m_cnn[0]->m_blob.flData = new float[numValues];
    net->m_cnn[0]->m_blob.fxData = new fixedPoint_t[numValues];
    if (precision == espresso::FLOAT)
    {
        net->m_cnn[0]->m_blob.flData = new float[numValues * sizeof(float)];
        memcpy(net->m_cnn[0]->m_blob.flData, im.data, numValues * sizeof(float));
        net->m_cnn[0]->m_precision = precision;
    }
    else
    {
        for (int i = 0; i < numValues; i++)
        {
            net->m_cnn[0]->m_blob.fxData[i] = fixedPoint::create(networkLayerInfo->dinFxPtLength, networkLayerInfo->dinNumFracBits, im.data[i]);
        }
        net->m_cnn[0]->m_precision = precision;
    }
}


void createDataLayer(espresso::layerInfo_obj* networkLayerInfo, network* net, espresso::precision_t precision)
{
    networkLayerInfo->precision = precision;
    networkLayerInfo->layerType = espresso::INPUT;
    networkLayerInfo->layerName = "0_Data";
    networkLayerInfo->inputDepth = net->layers[0].c;
    networkLayerInfo->numInputRows = net->layers[0].h;
    networkLayerInfo->numInputCols = net->layers[0].w;
    networkLayerInfo->yolo_net = net;
}


vector<espresso::layerInfo_obj*> darknetDataTransform(
    network** net,
    char* configFileName,
    char* weightFileName,
    espresso::backend_t backend,
    espresso::precision_t precision,
    int fxPtLen,
    int numFracBits
) {
	printf("Loading Darknet Data.....\n");
    *net = parse_network_cfg(configFileName);
    network* net_ptr = *net;
    load_weights(net_ptr, weightFileName);
    int numLayers = net_ptr->n + 1;
    vector<espresso::layerInfo_obj*> networkLayerInfoArr;
    networkLayerInfoArr.resize(numLayers);
    int j = 0;
    networkLayerInfoArr[0] = new espresso::layerInfo_obj();
    createDataLayer(networkLayerInfoArr[0], net_ptr, precision);
    for (int i = 1; i < numLayers; i++)
    {
        networkLayerInfoArr[i] = new espresso::layerInfo_obj();
        switch (net_ptr->layers[j].type)
        {
            case CONVOLUTIONAL:
            {
                networkLayerInfoArr[i]->layerType = espresso::CONVOLUTION;
                networkLayerInfoArr[i]->layerName = to_string(i) + "_Convolution";
                setBaseLayerInfo(j, &net_ptr->layers[j], networkLayerInfoArr[i], precision, fxPtLen, numFracBits, backend, net_ptr);
                if (net_ptr->layers[j].batch_normalize)
                {
                    networkLayerInfoArr[i]->flBeta = 0.000001f;
                    networkLayerInfoArr[i]->darknetNormScaleBias = true;
                }
                if (net_ptr->layers[j].activation == LEAKY)
                {
                    networkLayerInfoArr[i]->activation = espresso::LEAKY;
                }
                else if (net_ptr->layers[j].activation == LINEAR)
                {
                    networkLayerInfoArr[i]->activation = espresso::LINEAR;
                }
                networkLayerInfoArr[i]->darknetAct = true;
                getWeights(networkLayerInfoArr[i], &net_ptr->layers[j], precision, fxPtLen, numFracBits);
                break;
            }
            case ROUTE:
            {
                networkLayerInfoArr[i]->layerType = espresso::CONCAT;
                networkLayerInfoArr[i]->layerName = to_string(i) + "_Concat";
                setBaseLayerInfo(j, &net_ptr->layers[j], networkLayerInfoArr[i], precision, fxPtLen, numFracBits, backend, net_ptr);
                for (int k = 0; k < net_ptr->layers[j].n; k++)
                {
                    networkLayerInfoArr[i]->bottomLayerNames.push_back(networkLayerInfoArr[net_ptr->layers[j].input_layers[k] + 1]->layerName);
                }
                break;
            }
            case SHORTCUT:
            {
                networkLayerInfoArr[i]->layerType = espresso::RESIDUAL;
                networkLayerInfoArr[i]->layerName = to_string(i) + "_Residual";
                setBaseLayerInfo(j, &net_ptr->layers[j], networkLayerInfoArr[i], precision, fxPtLen, numFracBits, backend, net_ptr);
                networkLayerInfoArr[i]->bottomLayerNames.push_back(networkLayerInfoArr[i - 1]->layerName);
                networkLayerInfoArr[i]->bottomLayerNames.push_back(networkLayerInfoArr[net_ptr->layers[j].index + 1]->layerName);
                break;
            }
            case YOLO:
            {
                networkLayerInfoArr[i]->layerType = espresso::YOLO;
                networkLayerInfoArr[i]->layerName = to_string(i) + "_YOLO";
                setBaseLayerInfo(j, &net_ptr->layers[j], networkLayerInfoArr[i], precision, fxPtLen, numFracBits, backend, net_ptr);
                networkLayerInfoArr[i]->darknet_n_param = net_ptr->layers[j].n;
                networkLayerInfoArr[i]->darknet_classes_param = net_ptr->layers[j].classes;
                networkLayerInfoArr[i]->darknet_outputs_param = net_ptr->layers[j].outputs;
                break;
            }
            case UPSAMPLE:
            {
                networkLayerInfoArr[i]->layerType = espresso::UPSAMPLE;
                networkLayerInfoArr[i]->layerName = to_string(i) + "_UpSample";
                setBaseLayerInfo(j, &net_ptr->layers[j], networkLayerInfoArr[i], precision, fxPtLen, numFracBits, backend, net_ptr);
                break;
            }
            default:
            {
                cout << "Skipped Darknet Layer " << i << endl;
                continue;
            }
        }
        j++;
    }
    set_batch_network(net_ptr, 1);
    setLayerConnections(networkLayerInfoArr);
    return networkLayerInfoArr;
}


void getFlPtWeights(espresso::layerInfo_obj* networkLayerInfo, layer* layer_i)
{
    networkLayerInfo->flFilterData = new float[networkLayerInfo->numFilterValues];
    memcpy(
        networkLayerInfo->flFilterData,
        layer_i->weights,
        networkLayerInfo->numFilterValues * sizeof(float));
    networkLayerInfo->flBiasData = new float[networkLayerInfo->outputDepth];
    memcpy(
        networkLayerInfo->flBiasData,
        layer_i->biases,
        networkLayerInfo->outputDepth * sizeof(float));
    if(networkLayerInfo->darknetNormScaleBias)
    {
        int outputDepth = networkLayerInfo->outputDepth;
        int numFilValPerMap = networkLayerInfo->numFilterValues / networkLayerInfo->outputDepth;
        float *flFilterData_ptr = networkLayerInfo->flFilterData;
        for (int a = 0; a < outputDepth; a++)
        {
            float rolling_mean = layer_i->rolling_mean[a];
            float scales = layer_i->scales[a];
            float rolling_variance = layer_i->rolling_variance[a];
            float flBeta = networkLayerInfo->flBeta;
            float flBiasData_v = flFilterData_ptr[a];
            networkLayerInfo->flBiasData[a] = ((-rolling_mean * scales) / sqrtf(rolling_variance)) + (flBeta * scales) + flBiasData_v;
            for (int b = 0; b < numFilValPerMap; b++)
            {
                int idx = index2D(numFilValPerMap, a, b);
                flFilterData_ptr[idx] = flFilterData_ptr[idx] * scales / sqrtf(rolling_variance);
            }
        }
    }
}


void getFxPtWeights(espresso::layerInfo_obj* networkLayerInfo)
{
    networkLayerInfo->fxFilterData = new fixedPoint_t[networkLayerInfo->numFilterValues];
    fixedPoint_t *fxFilterData = networkLayerInfo->fxFilterData;
    int whtFxPtLength = networkLayerInfo->whtFxPtLength;
    int whtNumFracBits = networkLayerInfo->whtNumFracBits;
    for (int k = 0; k < networkLayerInfo->numFilterValues; k++)
    {
        fxFilterData[k] = fixedPoint::create(
            whtFxPtLength,
            whtNumFracBits,
            networkLayerInfo->flFilterData[k]);
    }
    networkLayerInfo->fxBiasData = new fixedPoint_t[networkLayerInfo->outputDepth];
    fixedPoint_t *fxBiasData = networkLayerInfo->fxBiasData;
    int biasFxPtLength = networkLayerInfo->biasFxPtLength;
    int biasNumFracBits = networkLayerInfo->biasFxPtLength;
    for (int k = 0; k < networkLayerInfo->outputDepth; k++)
    {
        fxBiasData[k] = fixedPoint::create(
            biasFxPtLength,
            biasNumFracBits,
            networkLayerInfo->flBiasData[k]);
    }
}


void getLayerPrec(vector<layerPrec_t>& layerPrecArr)
{
    layerPrecArr[0].dinNorm = true;
    int maxtDoutIntBits = -INT_MAX;
    for(int i = 1; i < layerPrecArr.size(); i++)
    {
        layerPrecArr[i].dinFxPtLength = layerPrecArr[i - 1].doutFxPtLength;
        layerPrecArr[i].dinNumFracBits    = layerPrecArr[i - 1].doutNumFracBits;
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
                numIntBits0 = max(int(ceilf(log2(maxFilMag))), 1);
                numIntBits1 = max(int(ceilf(log2(minFilMag))), 1);
                whtNumIntBits = max(numIntBits0, numIntBits1);
                if (whtNumIntBits > espresso::YOLO_MAX_NUM_INT_BITS)
                {
                    exit(1);
                }
                doutNumIntBits = dinNumIntBits + whtNumIntBits;
            }
            else if (maxFilMag < 1 && minFilMag >= 1)
            {
                whtNumIntBits = max(int(ceilf(log2(minFilMag))), 1);
                if (whtNumIntBits > espresso::YOLO_MAX_NUM_INT_BITS)
                {
                    exit(1);
                }
                doutNumIntBits = dinNumIntBits + whtNumIntBits;
            }
            else if (maxFilMag >= 1 && minFilMag < 1)
            {
                whtNumIntBits = max(int(ceilf(log2(maxFilMag))), 1);
                if (whtNumIntBits > espresso::YOLO_MAX_NUM_INT_BITS)
                {
                    exit(1);
                }
                doutNumIntBits = dinNumIntBits + whtNumIntBits;
            }
            else if (maxFilMag < 1 && minFilMag < 1) {
                numIntBits0 = ceilf(log2(1.0f / maxFilMag));
                numIntBits1 = ceilf(log2(1.0f / minFilMag));
                whtNumIntBits = min(min(numIntBits0, numIntBits1), espresso::YOLO_MAX_NUM_FRAC_BITS);
                doutNumIntBits = dinNumIntBits - whtNumIntBits;
            }
            if (layerPrecArr[i - 1].dinNorm)
            {
                doutNumIntBits = whtNumIntBits;
            }
            int doutNumFracBits = min(espresso::YOLO_DEF_FXPT_LEN - (doutNumIntBits + 1), espresso::YOLO_MAX_NUM_FRAC_BITS);
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
            layerPrecArr[i].doutFxPtLength    = layerPrecArr[i].dinFxPtLength;
            layerPrecArr[i].doutNumFracBits = layerPrecArr[i].dinNumFracBits;
        }
    }
    for (int i = 0; i < layerPrecArr.size(); i++)
    {
        layerPrecArr[i].maxtDoutIntBits = maxtDoutIntBits;
    }
}


void getLayerStats(const vector<espresso::layerInfo_obj*>& networkLayerInfoArr, vector<layerPrec_t>& layerPrecArr)
{
     for(int i = 0; i < networkLayerInfoArr.size(); i++)
    {
        if (layerPrecArr[i].layerType == espresso::CONVOLUTION)
        {
            int numBias = networkLayerInfoArr[i]->outputDepth;
            for (int j = 0; j < numBias; j++)
            {
                layerPrecArr[i].minBias = min(
                    layerPrecArr[i].minBias,
                    networkLayerInfoArr[i]->flBiasData[j]
                );
                layerPrecArr[i].maxBias = max(
                    layerPrecArr[i].maxBias,
                    networkLayerInfoArr[i]->flBiasData[j]
                );
            }
            for (int j = 0; j < networkLayerInfoArr[i]->numFilterValues; j++)
            {
                layerPrecArr[i].minFilter = min(
                    layerPrecArr[i].minFilter,
                    networkLayerInfoArr[i]->flFilterData[j]
                );
                layerPrecArr[i].maxFilter = max(
                    layerPrecArr[i].maxFilter,
                    networkLayerInfoArr[i]->flFilterData[j]
                );

            }
        }
    }
}


void getWeights(espresso::layerInfo_obj* networkLayerInfo, layer* layer_i, espresso::precision_t precision, int fxPtLen, int numFracBits)
{
    getFlPtWeights(networkLayerInfo, layer_i);
}


vector<int> getYOLOOutputLayers(vector<espresso::layerInfo_obj*>& networkLayerInfo)
{
    vector<int> outputLayers;
    for (int i = 0; i < networkLayerInfo.size(); i++)
    {
        if (networkLayerInfo[i]->layerType == espresso::YOLO)
        {
            outputLayers.push_back(i);
        }
    }
    return outputLayers;
}


void post_yolo(espresso::CNN_Network* net, network* yolo_net, char* cocoNames_FN, image im, char* imgOut_FN)
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


vector<layerPrec_t> profileYOLOWeights(const vector<espresso::layerInfo_obj*>& networkLayerInfoArr)
{
    vector<layerPrec_t> layerPrecArr;
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
        layerPrecArr[i].layerType = networkLayerInfoArr[i]->layerType;
    }
    getLayerStats(networkLayerInfoArr, layerPrecArr);
    getLayerPrec(layerPrecArr);
    return layerPrecArr;
}


void setBaseLayerInfo(int i, layer* layer_i, espresso::layerInfo_obj* networkLayerInfo, espresso::precision_t precision, int fxPtLen, int numFracBits, espresso::backend_t backend, network* yolo_net)
{
    networkLayerInfo->numInputRows = layer_i->h;
    networkLayerInfo->numInputCols = layer_i->w;
    networkLayerInfo->inputDepth = layer_i->c;
    networkLayerInfo->outputDepth = layer_i->n;
    networkLayerInfo->numKernelRows = layer_i->size;
    networkLayerInfo->numKernelCols = layer_i->size;
    networkLayerInfo->numKernels = layer_i->n;
    networkLayerInfo->kernelDepth = layer_i->c;
    networkLayerInfo->stride = layer_i->stride;
    networkLayerInfo->padding = layer_i->pad;
    networkLayerInfo->numFilterValues = layer_i->nweights;
    networkLayerInfo->group = layer_i->groups;
    networkLayerInfo->dinFxPtLength = fxPtLen;
    networkLayerInfo->dinNumFracBits = numFracBits;
    networkLayerInfo->whtFxPtLength = fxPtLen;
    networkLayerInfo->whtNumFracBits = numFracBits,
    networkLayerInfo->doutFxPtLength = fxPtLen;
    networkLayerInfo->doutNumFracBits = numFracBits,
    networkLayerInfo->biasFxPtLength = fxPtLen;
    networkLayerInfo->biasNumFracBits = numFracBits,
    networkLayerInfo->leakyFxPtLength = fxPtLen;
    networkLayerInfo->leakyNumFracBits = numFracBits,
    networkLayerInfo->precision = precision;
    networkLayerInfo->net_idx = i;
    networkLayerInfo->backend = backend;
    networkLayerInfo->yolo_net = yolo_net;
}


void setLayerConnections(vector<espresso::layerInfo_obj*>& networkLayerInfoArr)
{
    networkLayerInfoArr[0]->bottomLayerNames.push_back(networkLayerInfoArr[0]->layerName);
    for (int i = 1; i < networkLayerInfoArr.size(); i++)
    {
        if (networkLayerInfoArr[i]->layerType != espresso::CONCAT && networkLayerInfoArr[i]->layerType != espresso::RESIDUAL)
        {
            networkLayerInfoArr[i]->bottomLayerNames.push_back(networkLayerInfoArr[i - 1]->layerName);
        }
    }
    for (int i = 1; i < networkLayerInfoArr.size(); i++)
    {
        networkLayerInfoArr[i]->topLayerNames.push_back(networkLayerInfoArr[i]->layerName);
    }
}


int main(int argc, char **argv)
{
    string WSpath = string(getenv("WORKSPACE_PATH"));

    SYSC_FPGA_hndl* m_sysc_fpga_hndl = new SYSC_FPGA_hndl();
    // if(m_sysc_fpga_hndl->software_init(NULL) == -1)
    // {
    //     cout << "Software Init Failed" << endl;
    //     // exit(1);
    // }

    // YOLOv3
    espresso::precision_t precision = espresso::FLOAT;
    espresso::backend_t backend = espresso::FPGA_BACKEND;
    network* yolo_net = NULL;
    string yolov3_cfg_FN = WSpath + "/darknet/cfg/yolov3.cfg";
    string yolov3_whts_FN = WSpath + "/darknet/cfg/yolov3.weights";
    string yolov3_mrgd_fm_FN = WSpath + "/darknet/cfg/yolov3_merged_fmt.txt";
    vector<espresso::layerInfo_obj*> networkLayerInfoArr = darknetDataTransform(
        &yolo_net,
        (char*)yolov3_cfg_FN.c_str(),
        (char*)yolov3_whts_FN.c_str(),
        backend,
        precision,
        espresso::YOLO_DEF_FXPT_LEN,
        espresso::YOLO_DEF_NUM_FRAC_BITS
    );
    vector<int> outputLayers = getYOLOOutputLayers(networkLayerInfoArr);
    // vector<layerPrec_t> layerPrecArr = profileYOLOWeights(networkLayerInfoArr);
    string imgFN = WSpath + "/darknet/data/dog.jpg";
    espresso::CNN_Network net(networkLayerInfoArr, outputLayers);
    image im = load_image_color((char*)imgFN.c_str(), 0, 0);
    image sized = letterbox_image(im, networkLayerInfoArr[0]->numInputRows, networkLayerInfoArr[0]->numInputCols);
    cfgInputLayer(sized, &net, networkLayerInfoArr[0], espresso::FLOAT);
    // net.cfgFPGALayers(yolov3_mrgd_fm_FN);
	net.cfgFPGALayers();
    // net.printMemBWStats();
    net.setHardware(m_sysc_fpga_hndl);
    if(argc == 2)
    {
        net.Forward(argv[1]);
    }
    else if(argc == 3)
    {
        net.Forward(argv[1], argv[2]);
    }
    else
    {
        net.Forward();
    }
	net.printAccelPerfAnalyStats();
    string imgOut_FN = "predictions";
    string cocoNames_FN = WSpath + "/darknet/data/coco.names";
    post_yolo(&net, yolo_net, (char*)cocoNames_FN.c_str(), sized, (char*)imgOut_FN.c_str());
    free_image(im);
    free_image(sized);


    // MobileNetSSD
    // string protoTxt = WSpath + "/caffeModels/mobileNetSSD/mobileNetSSD.prototxt";
    // string model = WSpath + "/caffeModels/mobileNetSSD/mobileNetSSD.caffemodel";
    // string mergdFMT = WSpath + "/caffeModels/mobileNetSSD/mobileNetSSD_merged.txt";
    // vector<caffeDataParser::layerInfo_t> caffeLayerInfo = parseCaffeData(protoTxt, model);
    // vector<int> outputLayers;
    // // vector<int> outputLayers = getSSDOutputLayers();
    // vector<espresso::layerInfo_obj*> networkLayerInfoArr = caffeDataTransform(caffeLayerInfo, espresso::FPGA_BACKEND);
    // if(networkLayerInfoArr[0]->layerType != espresso::INPUT)
    // {
    //     espresso::layerInfo_obj* layerInfo = new espresso::layerInfo_obj();
    //     networkLayerInfoArr[0]->bottomLayerNames[0] = "Data";
    //     layerInfo->layerName = "Data";
    //     layerInfo->layerType = espresso::INPUT;
    //     layerInfo->inputDepth = 3;
    //     layerInfo->numInputRows = 300;
    //     layerInfo->numInputCols = 300;
    //     vector<espresso::layerInfo_obj*>::iterator it = networkLayerInfoArr.begin();
    //     networkLayerInfoArr.insert(it, layerInfo);
    // }
    // espresso::CNN_Network net(networkLayerInfoArr, outputLayers);
    // net.cfgFPGALayers(mergdFMT);
    // net.printMemBWStats();
    // net.setHardware(m_sysc_fpga_hndl);
    // net.Forward();


    // // RFCN-Resnet101
    // string protoTxt = WSpath + "/caffeModels/rfcn_resnet101/rfcn_resnet101.prototxt";
    // string model = WSpath + "/caffeModels/rfcn_resnet101/rfcn_resnet101.caffemodel";
    // string mergdFMT = WSpath + "/caffeModels/rfcn_resnet101/rfcn_resnet101_merged.txt";
    // vector<caffeDataParser::layerInfo_t> caffeLayerInfo = parseCaffeData(protoTxt, model);
    // vector<int> outputLayers;
    // // vector<int> outputLayers = getRFCN_Resnet101OutputLayers();
    // vector<espresso::layerInfo_obj*> networkLayerInfoArr = caffeDataTransform(caffeLayerInfo, espresso::FPGA_BACKEND);
    // if(networkLayerInfoArr[0]->layerType != espresso::INPUT)
    // {
    //     espresso::layerInfo_obj* layerInfo = new espresso::layerInfo_obj();
    //     networkLayerInfoArr[0]->bottomLayerNames[0] = "Data";
    //     layerInfo->layerName = "Data";
    //     layerInfo->layerType = espresso::INPUT;
    //     layerInfo->inputDepth = 3;
    //     layerInfo->numInputRows = 224;
    //     layerInfo->numInputCols = 224;
    //     vector<espresso::layerInfo_obj*>::iterator it = networkLayerInfoArr.begin();
    //     networkLayerInfoArr.insert(it, layerInfo);
    // }
    // espresso::CNN_Network net(networkLayerInfoArr, outputLayers);
    // // net.cfgFPGALayers(mergdFMT);
	// net.cfgFPGALayers();
    // net.setHardware(m_sysc_fpga_hndl);
    // if(argc == 2)
    // {
    //     net.Forward(argv[1]);
    // }
    // else if(argc == 3)
    // {
    //     net.Forward(argv[1], argv[2]);
    // }
    // else
    // {
    //     net.Forward();
    // }
	// net.printAccelPerfAnalyStats();
	
	
	// // RFCN-Resnet50
    // string protoTxt = WSpath + "/caffeModels/rfcn_resnet50/rfcn_resnet50.prototxt";
    // string model = WSpath + "/caffeModels/rfcn_resnet50/rfcn_resnet50.caffemodel";
    // string mergdFMT = WSpath + "/caffeModels/rfcn_resnet50/rfcn_resnet50_merged.txt";
    // vector<caffeDataParser::layerInfo_t> caffeLayerInfo = parseCaffeData(protoTxt, model);
    // vector<int> outputLayers;
    // // vector<int> outputLayers = getRFCN_Resnet101OutputLayers();
    // vector<espresso::layerInfo_obj*> networkLayerInfoArr = caffeDataTransform(caffeLayerInfo, espresso::FPGA_BACKEND);
    // if(networkLayerInfoArr[0]->layerType != espresso::INPUT)
    // {
    //     espresso::layerInfo_obj* layerInfo = new espresso::layerInfo_obj();
    //     networkLayerInfoArr[0]->bottomLayerNames[0] = "Data";
    //     layerInfo->layerName = "Data";
    //     layerInfo->layerType = espresso::INPUT;
    //     layerInfo->inputDepth = 3;
    //     layerInfo->numInputRows = 224;
    //     layerInfo->numInputCols = 224;
    //     vector<espresso::layerInfo_obj*>::iterator it = networkLayerInfoArr.begin();
    //     networkLayerInfoArr.insert(it, layerInfo);
    // }
    // espresso::CNN_Network net(networkLayerInfoArr, outputLayers);
    // // net.cfgFPGALayers(mergdFMT);
	// net.cfgFPGALayers();
    // net.setHardware(m_sysc_fpga_hndl);
    // if(argc == 2)
    // {
    //     net.Forward(argv[1]);
    // }
    // else if(argc == 3)
    // {
    //     net.Forward(argv[1], argv[2]);
    // }
    // else
    // {
    //     net.Forward();
    // }
	// net.printAccelPerfAnalyStats();
    
    
    // // Resnet50
    // string protoTxt = WSpath + "/caffeModels/resnet50/resnet50.prototxt";
    // string model = WSpath + "/caffeModels/resnet50/resnet50.caffemodel";
    // vector<caffeDataParser::layerInfo_t> caffeLayerInfo = parseCaffeData(protoTxt, model);
    // vector<int> outputLayers;
    // vector<espresso::layerInfo_obj*> networkLayerInfoArr = caffeDataTransform(caffeLayerInfo, espresso::FPGA_BACKEND);
    // if(networkLayerInfoArr[0]->layerType != espresso::INPUT)
    // {
    //     espresso::layerInfo_obj* layerInfo = new espresso::layerInfo_obj();
    //     networkLayerInfoArr[0]->bottomLayerNames[0] = "Data";
    //     layerInfo->layerName = "Data";
    //     layerInfo->layerType = espresso::INPUT;
    //     layerInfo->inputDepth = 3;
    //     layerInfo->numInputRows = 224;
    //     layerInfo->numInputCols = 224;
    //     vector<espresso::layerInfo_obj*>::iterator it = networkLayerInfoArr.begin();
    //     networkLayerInfoArr.insert(it, layerInfo);
    // }
    // espresso::CNN_Network net(networkLayerInfoArr, outputLayers);
	// net.cfgFPGALayers();
    // net.setHardware(m_sysc_fpga_hndl);
    // if(argc == 2)
    // {
    //     net.Forward(argv[1]);
    // }
    // else if(argc == 3)
    // {
    //     net.Forward(argv[1], argv[2]);
    // }
    // else
    // {
    //     net.Forward();
    // }


    return 0;
}
