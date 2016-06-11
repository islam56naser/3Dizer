#include "Epipolar.h"

namespace reconstruct3D{
	bool Epipolar::chooseValidPair(vector<ImageContainer>& imgCont, vector<vector<DMatch>>& goodMatchesM, Mat& cam, Mat& distor
		, Mat& ess, double inlierPercent){
		size_t count = 0;
		size_t nImages = imgCont.size();
		Mat inliersMask;
		while (true)
		{
			if (isValidPair(goodMatchesM[0], imgCont[0].imgKeys, imgCont[1].imgKeys, cam, distor, ess, inliersMask, inlierPercent)) break;
			if (count > nImages - 3){
				return false;
			}
			imgCont.erase(imgCont.begin(), imgCont.begin() + 1);
			goodMatchesM.erase(goodMatchesM.begin(), goodMatchesM.begin() + 1);
			count++;
		}
		vector<DMatch>unrefinMatches = goodMatchesM[0];
		goodMatchesM[0].clear();
		for (size_t i = 0; i < unrefinMatches.size(); i++)
		{
			if ((int)inliersMask.at<uchar>(i, 0) == 1)
			{
				goodMatchesM[0].push_back(unrefinMatches[i]);
			}
		}
		return true;
	}

	double Epipolar::calcEpipError(Mat& fund, vector<Point2f>& pts1, vector<Point2f>& pts2){
		double err = 0;
		size_t nPoints = pts1.size();
		for (size_t i = 0; i < nPoints; i++)
		{
			Mat pt1 = (Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
			Mat pt2 = (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);
			Mat errMat = pt2.t()* fund * pt1;
			err += errMat.at<double>(0, 0);
		}
		return err / nPoints;
	}


	void Epipolar::getEssMatrix(Mat& rot1, Mat& rot2, Mat& trans1, Mat& trans2, Mat& approxEss){
		Mat rot1Mat, rot2Mat;
		if (rot1.size() == Size(3, 3))
		{
			rot1Mat = rot1;
		}
		else
		{
			Rodrigues(rot1, rot1Mat);
		}
		if (rot2.size() == Size(3, 3))
		{
			rot2Mat = rot2;
		}
		else
		{
			Rodrigues(rot2, rot2Mat);
		}
		Mat relRot = rot1Mat.inv() * rot2Mat;
		Mat relTrans = trans2 - trans1;
		getEssMatrix(relRot, relTrans, approxEss);
	}
	void Epipolar::getEssMatrix(Mat& relRot, Mat& relTrans, Mat& approxEss){
		double t1 = relTrans.at<double>(0, 0);
		double t2 = relTrans.at<double>(1, 0);
		double t3 = relTrans.at<double>(2, 0);
		Mat tranCross = (Mat_<double>(3, 3) << 0.0, -t3, t2,
			t3, 0.0, -t1,
			-t2, t1, 0.0);
		approxEss = tranCross * relRot;

	}

	void Epipolar::calcEssMatrix(vector<vector<DMatch>>& goodMatchesM, vector<ImageContainer>& imgCont, Mat& cam, Mat& distor,
		vector<Mat>& essM, double inlierPercent){
		size_t nImages = imgCont.size();
		essM.resize(nImages - 1);
		for (size_t i = 1; i < nImages; i++)
		{
			calcEssMatrix(goodMatchesM[i - 1], imgCont[i - 1].imgKeys, imgCont[i].imgKeys, cam, distor, essM[i - 1], inlierPercent);

		}

	}

	void Epipolar::calcEssMatrix(vector<DMatch>& goodMatches, vector<KeyPoint>& keys1, vector<KeyPoint>& keys2, Mat& cam, Mat& distor, Mat& ess, double inlierPercent){
		int inliers = 0;
		vector<Point2f>pts1, pts2;
		Mat rot, trans, inlierMask;
		size_t n = goodMatches.size();
		for (size_t j = 0; j < n; j++)
		{
			pts1.push_back(keys1[goodMatches[j].trainIdx].pt);
			pts2.push_back(keys2[goodMatches[j].queryIdx].pt);

		}
		undistortPoints(pts1, pts1, cam, distor);
		undistortPoints(pts2, pts2, cam, distor);
		ess = findEssentialMat(pts1, pts2, 1.0, Point(0, 0), RANSAC, 0.999, 1.25, inlierMask);
		inliers = recoverPose(ess, pts1, pts2, rot, trans, 1.0, Point(0, 0), inlierMask);
		if ((double)inliers / n < inlierPercent){
			ess.release();
		}
	}


	void Epipolar::fundFromEss(Mat& essential, Mat& fundamental, Mat& cam){
		assert(!essential.empty());
		fundamental = cam.t().inv()*essential * cam.inv();

	}
	void Epipolar::fundFromEss(vector<Mat>& essM, vector<Mat>& fundM, Mat& cam){
		size_t n = essM.size();
		fundM.resize(n);
		for (size_t i = 0; i < n; i++)
		{
			fundFromEss(essM[i], fundM[i], cam);
		}

	}
	void Epipolar::correctKeyPointMatches(vector<ImageContainer>& imgCont, vector<vector<DMatch>>& matches, vector<Mat>& fundM){
		for (size_t i = 1; i < imgCont.size(); i++)
		{
			correctKeyPointMatches(imgCont[i - 1].imgKeys, imgCont[i].imgKeys, matches[i - 1], fundM[i - 1]);

		}
	}


	void Epipolar::correctKeyPointMatches(vector<KeyPoint>& keys1, vector<KeyPoint>& keys2, vector<DMatch>& matches, Mat& fund){

		assert(!fund.empty());
		for (size_t i = 0; i < matches.size(); i++)
		{

			vector<Point2f> pt1, pt2;
			int idx1, idx2;
			idx1 = matches[i].trainIdx;
			idx2 = matches[i].queryIdx;
			pt1.push_back(keys1[idx1].pt);
			pt2.push_back(keys2[idx2].pt);
			correctMatches(fund, pt1, pt2, pt1, pt2);
			keys1[idx1].pt = pt1[0];
			keys2[idx2].pt = pt2[0];

		}


	}

	bool Epipolar::isValidPair(vector<DMatch>& matches, vector<KeyPoint>& key1, vector<KeyPoint>& key2, Mat& cam, Mat& distor, Mat& ess, Mat& inliersMask, double inlierPercent){
		vector<Point2f>pts1, pts2;
		inliersMask.deallocate();
		Mat rot, trans;
		size_t n = matches.size();
		Utility:: getPointMatches(key1, key2, matches, pts1, pts2);
		undistortPoints(pts1, pts1, cam, distor);
		undistortPoints(pts2, pts2, cam, distor);
		ess = findEssentialMat(pts1, pts2, 1.0, Point(0, 0), RANSAC, 0.999, 1.25, inliersMask);
		int inliers = recoverPose(ess, pts1, pts2, rot, trans, 1.0, Point(0, 0), inliersMask);
		return ((double)inliers / n) > inlierPercent;
	}



}