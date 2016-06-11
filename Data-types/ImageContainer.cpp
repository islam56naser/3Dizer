#include "ImageContainer.h"

namespace reconstruct3D{

	ImageContainer::ImageContainer(Mat img){
		//blur(img, this->image, Size(7, 7));
		img.copyTo(this->image);
		Size imgSize = image.size();
		this->segImage = Mat::zeros(imgSize, CV_8U);
	}

}