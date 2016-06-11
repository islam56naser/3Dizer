#pragma once
#include "opencv2\core.hpp"
#include "Track.h"

namespace reconstruct3D{
	using namespace std;
	using namespace cv;
	class Filter
	{
	public:
		static void filterOutliers(vector<Point3f>& cloud, vector<Point3f>& filteredCloud, double distThreshold, double depthThreshold);
		static void filterOutliers(vector<Track>& pointTracks, vector<Track>& filteredTracks, vector<Point3f>& filteredCloud,
			double distThreshold, double depthThreshold);


		static void filterKeyPoints(vector<KeyPoint>& keys, vector<KeyPoint>& filteredKeys,
			Size imgSize, size_t xBins = 10, size_t yBins = 10);
		static void filterKeyPoints(vector<vector<KeyPoint>>& keys, vector<vector<KeyPoint>>& filteredKeys,
			Size imgSize, size_t xBins = 10, size_t yBins = 10);
		static void histFilter(vector<Point3f>& cloud, vector<Point3f>& filteredCloud,size_t xBins, size_t yBins,size_t zBins);
	private:
		static void distFromCentroid(vector<Point3f>& cloud, Point3f centroid, vector<double>& distances);
		static void depthFromCentroid(vector<Point3f>& cloud, Point3f centroid, vector<double>& depths);
		static bool isBigger(Point3f pt, Point3f refPt);
		static bool isSmaller(Point3f pt, Point3f refPt);

	};



}


