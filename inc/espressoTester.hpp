// System includes
#include <iomanip>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>



// Project includes
#include "Network.hpp"


class layerPrec_obj
{
	public:
		bool dinNorm;
		float minBias;
		float maxBias;
		float minFilter;
		float maxFilter;
		int dinFxPtLength;
		int dinNumFracBits; 
		int doutFxPtLength;
		int doutNumFracBits;
		int leakyFxPtLength;
		int leakyNumFracBits;
	    int whtFxPtLength;
	    int whtNumFracBits;
		int maxtDoutIntBits;
		espresso::layerType_t layerType;
};


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void cfgInputLayer(const image& im, espresso::Network* net, const espresso::layerInfo_obj& networkLayerInfo, espresso::precision_t precision);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void createDataLayer(espresso::layerInfo_obj& networkLayerInfo, network* net, espresso::precision_t precision);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
std::vector<std::vector<kernel_group*>> createKernelGroupArr(int numKernels, int channels, int height, int width, int fxPtLength, int numFracBits);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void createKernelGroups(espresso::layerInfo_obj& networkLayerInfo);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
std::vector<espresso::layerInfo_obj> darknetDataTransform(
	network** net, 
	char* configFileName, 
	char* weightFileName,
	espresso::backend_t backend,
	espresso::precision_t precision,
	int fxPtLen = espresso::ESPRO_DEF_FXPT_LEN,
	int numFracBits = espresso::ESPRO_DEF_NUM_FRAC_BITS
);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void getFlPtWeights(espresso::layerInfo_obj& networkLayerInfo, layer* layer_i);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void getFxPtWeights(espresso::layerInfo_obj& networkLayerInfo);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void getLayerPrec(std::vector<layerPrec_obj> layerPrecArr);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void getLayerStats(const std::vector<espresso::layerInfo_obj>& networkLayerInfoArr, std::vector<layerPrec_obj> layerPrecArr);



// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void getWeights(espresso::layerInfo_obj& networkLayerInfo, layer* layer_i, espresso::precision_t precision, int fxPtLen, int numFracBits);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
std::vector<int> getYOLOOutputLayers(std::vector<espresso::layerInfo_obj> &networkLayerInfo);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void post_yolo(espresso::Network* net, network* yolo_net, char* cocoNames_FN, image im, char* imgOut_FN);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
std::vector<layerPrec_obj> profileYOLOWeights(const std::vector<espresso::layerInfo_obj>& networkLayerInfoArr);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------	
void setBaseLayerInfo(int i, layer* layer_i, espresso::layerInfo_obj& networkLayerInfo, espresso::precision_t precision, int fxPtLen, int numFracBits, network* yolo_net);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void setKrnlGrpData(espresso::layerInfo_obj& networkLayerInfo, int numKernels);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void setLayerConnections(std::vector<espresso::layerInfo_obj>& networkLayerInfoArr);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void setLayerPrec(std::vector<espresso::layerInfo_obj>& networkLayerInfoArr,);
