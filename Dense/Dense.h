#pragma once
#include "opencv2\xfeatures2d.hpp"
#include "Track.h"
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "ImageContainer.h"
#include "opencv2\ximgproc\sparse_match_interpolator.hpp"
#include "Utility.h"
namespace reconstruct3D{
	using namespace cv;
	using namespace std;

	class Dense
	{
	public:
		static void getNewKeyPoints(vector<ImageContainer>& imgCont, vector<vector<KeyPoint>>& newKeys, bool drawPoints = false,
			float size = 4,float angle = -1.0f, float response = 0, int octave = 0, int classID = -1);
		static void extendPointTracks(vector<Track>& oldTracks, vector<Track>& newTracks);
		static void extendCloud(vector<Point3f>& oldCloud, vector<Point3f>& newCloud);
		static void extendImgKeyPoints(vector<ImageContainer>& imgCont, vector<vector<KeyPoint>>& newKeys);
		static void generateFlow(ImageContainer& imgCont1, ImageContainer& imgCont2, vector<DMatch>& matches, Mat& flow);
		static void generateFlow(vector<ImageContainer>& imgCont, vector<vector<DMatch>>& matches, vector<Mat>& flow);
		static void rectifyImgPair(vector<Point2f>& pts1, vector<Point2f>& pts2, Mat& img1, Mat& img2, 
			Mat& fund, Mat& cam, Mat& dist, Mat& img1Rect, Mat& img2Rect);
		static void rectifyAllImgs(vector<reconstruct3D::ImageContainer>& imgConts, vector<vector<DMatch>>& matches, vector<Mat>& fundamentals,Mat& cams, Mat& dist);

	private:
		static void pointsToKeyPoints(vector<Point2f>& pts, vector<KeyPoint>& keyPoints,float size,
			float angle,float response,int octave ,int classID );
		static void getNewPoints(ImageContainer& imgCont, vector<Point2f>& newPoints, bool drawPoints = false);
		static void getCentroids(vector<Vec6f>& triangleList, vector<Point2f>& centroids);
		static void draw_point(Mat& img, Point2f fp, Scalar color);
		static void draw_delaunay(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color);
	};
}


