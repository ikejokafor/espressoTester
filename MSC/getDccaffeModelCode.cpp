using namespace caffe;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;


	NetParameter param;
	LayerParameter *lparam;
	int fd = open("/home/ikenna/caffe-master/models/dcNet/deploy_sqz_2.prototxt", O_RDONLY);
	google::protobuf::io::FileInputStream* input = new google::protobuf::io::FileInputStream(fd);
	bool success = google::protobuf::TextFormat::Parse(input, &param);
    close(fd);
	if(!success) {
		exit(1);
	}
    
    NetParameter weightparam;
    fd = open("/home/ikenna/detector_test_kitti/detector_kitti/sqz_rework_iter_100000.caffemodel", O_RDONLY);
    ZeroCopyInputStream* raw_input = new FileInputStream(fd);
    CodedInputStream* coded_input = new CodedInputStream(raw_input);
    coded_input->SetTotalBytesLimit(INT_MAX, 536870912);
    success = weightparam.ParseFromCodedStream(coded_input);
    delete coded_input;
    delete raw_input;
    if(!success) {
        exit(1);
    }
    
    
    
	for (int nlayers = 0; nlayers < param.layer_size(); nlayers++) { 
        lparam = param.::caffe::NetParameter::mutable_layer(nlayers);
		if (lparam->has_convolution_param()) {
            for (int j = 0; j < weightparam.layer_size(); j++) { 
                if(lparam->name() == weightparam.layer(j).name()){
                    lparam->add_blobs();
                    lparam->add_blobs();
                    
                    BlobShape *shape1 = new BlobShape(); 
                    shape1->add_dim(weightparam.layer(j).blobs(0).shape().dim(0));
                    shape1->add_dim(weightparam.layer(j).blobs(0).shape().dim(1));
                    shape1->add_dim(weightparam.layer(j).blobs(0).shape().dim(2));
                    shape1->add_dim(weightparam.layer(j).blobs(0).shape().dim(3));
                    lparam->mutable_blobs(0)->set_allocated_shape(shape1);
                    
                    BlobShape *shape2 = new BlobShape(); 
                    shape2->add_dim(weightparam.layer(j).blobs(0).shape().dim(0));
                    lparam->mutable_blobs(1)->set_allocated_shape(shape2);
                    
                    for(int k = 0; k < weightparam.layer(j).blobs(0).data_size(); k++) {
                        lparam->mutable_blobs(0)->add_data(weightparam.layer(j).blobs(0).data(k));
                    }
                    for(int k = 0; k < weightparam.layer(j).blobs(1).data_size(); k++) {
                        lparam->mutable_blobs(1)->add_data(weightparam.layer(j).blobs(1).data(k));
                    }
                }
            }
		}
	}
    
    
    
    fstream output("dcNet_deploy_sq_2.caffemodel", ios::out | ios::trunc | ios::binary);
    param.SerializeToOstream(&output);
    exit(0);
    