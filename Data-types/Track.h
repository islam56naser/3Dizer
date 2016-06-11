#pragma once
#include "opencv2\core.hpp"
#include <iostream>

namespace reconstruct3D{
	using namespace std;
	using namespace cv;
	class Track
	{
	public:
		Track(size_t idx1);
		vector<Point2f>pointList;
		size_t firstImgIdx;
		Point3f scenePoint;
		int colorCode;
		void print();

	};





}

