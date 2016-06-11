#pragma once
#include "Track.h"
#include "cvsba.h"
#include "Utility.h"

namespace reconstruct3D{
	class BundAdj
	{
	public:
		static void updateBundAdjParams(vector<Mat>& cams, vector<Mat>& distortions, vector<Mat>& rotations, vector<Mat>& translations, 
			Mat& rot, Mat& trans, Mat& cam, Mat& distor);
		static void initBundAdjParams(vector<Mat>& cams, vector<Mat>& distortions, vector<Mat>& rotations,
			vector<Mat>& translations, Mat& rot1, Mat& rot2, Mat& trans1, Mat& trans2, Mat& cam, Mat& distor);
		static void runBundAdj(vector<Point3f>& cloud, vector<reconstruct3D::Track>& pointTracks, vector<vector<Point2f>>& imgPoints, vector<vector<int>>& vis
			, vector<Mat>& rotations, vector<Mat>& translations, vector<Mat>& cams, vector<Mat>& distortions, vector<Mat>& projections,
			cvsba::Sba::Params param,bool update = true, int nConstFrames = 0);
		static void generateBundAdjData(vector<Track>& pointTracks, vector<vector<Point2f>>& imgPoints,
			vector<vector<int>>& vis, size_t nViews, size_t cloudSize);

	private:
		static void updateAfterBA(vector<Point3f>& cloud, vector<Mat>& rotations, vector<Mat>& translations,
			vector<reconstruct3D::Track>& pointTracks, vector<Mat>& projections,cvsba::Sba::Params params);
		
	};
}


