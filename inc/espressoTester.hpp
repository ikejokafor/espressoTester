// Definitions
#define THRESH						0.5f
#define HIER_THRESH					0.5f
#define NMS_THRESH					0.45f
#define YOLO_DEF_FXPT_LEN			16
#define YOLO_DEF_NUM_FRAC_BITS		14
#define YOLO_MAX_NUM_INT_BITS		(YOLO_DEF_FXPT_LEN - 1)
#define YOLO_MAX_NUM_FRAC_BITS      (YOLO_DEF_FXPT_LEN - 2)


// System includes
#include <iomanip>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>



// Project includes
#include "Network.hpp"


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void setLayerConnections(std::vector<espresso::layerInfo_obj> &networkLayerInfoArr);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void setKrnlGrpData(espresso::layerInfo_obj &networkLayerInfo, int numKernels);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
std::vector<std::vector<kernel_group*>> createKernelGroupArr(int numKernels, int height, int width, int channels);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void getFxPtWeights(espresso::layerInfo_obj &networkLayerInfo);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void getFlPtWeights(layer *layer_i, espresso::layerInfo_obj &networkLayerInfo);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void getWeights(espresso::precision_t precision, layer *layer_i, espresso::layerInfo_obj &networkLayerInfo, int fxPtLen, int numFracBits);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------	
void setBaseLayerInfo(int i, layer *layer_i, espresso::layerInfo_obj &networkLayerInfo, espresso::precision_t precision, int fxPtLen, int numFracBits, network yolo_net);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void createDataLayer(network *net, std::vector<espresso::layerInfo_obj> &networkLayerInfoArr, espresso::precision_t precision);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void createKernelGroups(espresso::layerInfo_obj &networkLayerInfo);


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
void post_yolo(network *yolo_net, espresso::Network *net);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void createDataLayer(network *net, espresso::layerInfo_obj& networkLayerInfo, espresso::precision_t precision);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void cfgInputLayer(espresso::precision_t precision, espresso::layerInfo_obj& networkLayerInfo, network *yolo_net, espresso::Network* net, image im);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void darknetDataTransform(
	network **net, 
	std::vector<espresso::layerInfo_obj> &networkLayerInfoArr, 
	espresso::precision_t precision,
	espresso::backend_t backend,
	std::string configFileName, 
	std::string weightFileName,
	int fxPtLen = espresso::ESPRO_DEF_FXPT_LEN,
	int numFracBits = espresso::ESPRO_DEF_NUM_FRAC_BITS
);
