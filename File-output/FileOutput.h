#pragma once
#include "opencv2\core.hpp"
#include "Track.h"
#include <fstream>

namespace reconstruct3D{
	using namespace std;
	using namespace cv;

	class FileOutput
	{
	public:
		static void outputPCD(const String& fileName, vector<Point3f>& cloud);
		static void outputPCD(const String& fileName, vector<Point3f>& cloud, vector<int>& colorCode);
		static void outputPCD(const String& fileName, vector<Track>& pointTracks);
	};


}


