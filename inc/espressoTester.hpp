// System includes
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cstdlib>

// Project includes
#include "util.hpp"
#include "CNN_Network.hpp"
#ifdef CAFFE_DATA_PARSER
#include "caffeDataParser.hpp"
#endif

typedef struct
{
	bool dinNorm;
	float minBias;
	float maxBias;
	float minFilter;
	float maxFilter;
	int dinFxPtLength;
	int dinNumFracBits;
	int doutFxPtLength;
	int doutNumFracBits;
	int biasFxPtLength;
	int biasNumFracBits;
	int leakyFxPtLength;
	int leakyNumFracBits;
	int whtFxPtLength;
	int whtNumFracBits;
	int maxtDoutIntBits;
	espresso::layerType_t layerType;
} layerPrec_t;


#ifdef CAFFE_DATA_PARSER
// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
espresso::layerType_t getEspLayType(std::string layerType);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
std::vector<espresso::layerInfo_obj*> caffeDataTransform(std::vector<caffeDataParser::layerInfo_t> caffeDataParserLayerInfo, espresso::backend_t backend);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
int findCaffeLayer(std::string layerName, std::vector<caffeDataParser::layerInfo_t> caffeLayerInfo);
#endif

// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void cfgInputLayer(const image& im, espresso::CNN_Network* net, const espresso::layerInfo_obj* networkLayerInfo, espresso::precision_t precision);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void createDataLayer(espresso::layerInfo_obj* networkLayerInfo, network* net, espresso::precision_t precision);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure                   LOYOLO_DEF_FXPT_LEN,
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
std::vector<espresso::layerInfo_obj*> darknetDataTransform(
	network** net,
	char* configFileName,
	char* weightFileName,
	espresso::backend_t backend,
	espresso::precision_t precision,
	int fxPtLen,
	int numFracBits
);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void getFlPtWeights(espresso::layerInfo_obj* networkLayerInfo, layer* layer_i);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void getLayerPrec(std::vector<layerPrec_t>& layerPrecArr);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void getLayerStats(const std::vector<espresso::layerInfo_obj*>& networkLayerInfoArr, std::vector<layerPrec_t>& layerPrecArr);



// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void getWeights(espresso::layerInfo_obj* networkLayerInfo, layer* layer_i, espresso::precision_t precision, int fxPtLen, int numFracBits);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
std::vector<int> getYOLOOutputLayers(std::vector<espresso::layerInfo_obj*>& networkLayerInfo);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void post_yolo(espresso::CNN_Network* net, network* yolo_net, char* cocoNames_FN, image im, char* imgOut_FN);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
std::vector<layerPrec_t> profileYOLOWeights(const std::vector<espresso::layerInfo_obj*>& networkLayerInfoArr);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void setBaseLayerInfo(int i, layer* layer_i, espresso::layerInfo_obj* networkLayerInfo, espresso::precision_t precision, int fxPtLen, int numFracBits, espresso::backend_t backend, network* yolo_net);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void setLayerConnections(std::vector<espresso::layerInfo_obj*>& networkLayerInfoArr);
