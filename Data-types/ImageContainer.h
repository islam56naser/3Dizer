#pragma once
#include "opencv2\imgproc.hpp"
#include <iostream>

namespace reconstruct3D{
	using namespace std;
	using namespace cv;
	class ImageContainer
	{
	public:
		ImageContainer(Mat img);
		Mat image;
		Mat grayImage;
		Mat segImage;
		Mat mask ;
		Mat descriptor;
		Mat rectifiedImg;
		vector<KeyPoint>imgKeys;
		vector<Vec6f> triList;
	};
}
