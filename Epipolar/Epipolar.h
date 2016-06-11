#pragma once
#include "ImageContainer.h"
#include <opencv2/calib3d/calib3d.hpp>
#include "Utility.h"

namespace reconstruct3D{
	class Epipolar
	{
	public:
		static bool chooseValidPair(vector<ImageContainer>& imgCont, vector<vector<DMatch>>& goodMatchesM,
			Mat& cam, Mat& distor, Mat& ess, double inlierPercent = 0.95);
		static void getEssMatrix(Mat& rot1, Mat& rot2, Mat& trans1, Mat& trans2, Mat& approxEss);
		static void getEssMatrix(Mat& relRot, Mat& relTrans, Mat& approxEss);
		static void correctKeyPointMatches(vector<KeyPoint>& keys1, vector<KeyPoint>& keys2, vector<DMatch>& matches, Mat& fund);
		static void correctKeyPointMatches(vector<ImageContainer>& imgCont, vector<vector<DMatch>>& matches, vector<Mat>& fundM);

		static void fundFromEss(Mat& essential, Mat& fundamental, Mat& cam);
		static void fundFromEss(vector<Mat>& essM, vector<Mat>& fundM, Mat& cam);

		static void calcEssMatrix(vector<vector<DMatch>>& goodMatchesM, vector<ImageContainer>& imgCont,
			Mat& cam, Mat& distor, vector<Mat>& essM, double inlierPercent = 0.95);
		static void calcEssMatrix(vector<DMatch>& goodMatches, vector<KeyPoint>& keys1, vector<KeyPoint>& keys2,
			Mat& cam, Mat& distor, Mat& ess, double inlierPercent = 0.95);
		static double calcEpipError(Mat& fund, vector<Point2f>& pts1, vector<Point2f>& pts2);
	private:
		static bool isValidPair(vector<DMatch>& matches, vector<KeyPoint>& key1, vector<KeyPoint>& key2, Mat& cam, Mat& distor, Mat& ess, Mat& inliersMask, double inlierPercent);


	};
}
