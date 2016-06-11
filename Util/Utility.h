#pragma once
#include "Track.h"
#include "ImageContainer.h"
#include <opencv2/calib3d/calib3d.hpp>
#include"opencv2\imgproc.hpp"
#include "Feature.h"
#include "Filter.h"
#include "Dense.h"
#include <assert.h>

namespace reconstruct3D{
	class Utility
	{
	public:
		
	
		static void generatePointTracks(vector<ImageContainer>& imgCont, vector<vector<DMatch>>& goodMatchesM, vector<Track>& pointTracks);
		static void generatePointTracks(vector<Mat>& images, vector<vector<KeyPoint>>& keyPoints, vector<vector<DMatch>>& goodMatchesM, vector<Track>& pointTracks);
		static void generatePointTracks(vector<ImageContainer>& imgCont, vector<Mat>& flows, vector<Track>& denseTracks);
		static void generatePointTracksNoMask(vector<ImageContainer>& imgCont, vector<Mat>& flows, vector<Track>& denseTracks);
	
		static void getPointMatches(ImageContainer& imgCont1, ImageContainer& imgCont2, vector<DMatch>& matches, vector<Point2f>& pts1, vector<Point2f>& pts2);
		static void getPointMatches(vector<KeyPoint>& keys1, vector<KeyPoint>& keys2, vector<DMatch>& matches, vector<Point2f>& pts1, vector<Point2f>& pts2);

		static double triangulate(vector<Point2f>& pts1, vector<Point2f>& pts2, vector<Point3f>& cloud,
			Mat& P1, Mat& P2, Mat& cam, Mat& distor);
		static void triangulateAndUpdateTracks(vector<Track>& pointTracks, vector<Point2f>& pts1, vector<Point2f>& pts2, vector<Point3f>& cloud,
			Mat& P1, Mat& P2,Mat& cam,Mat& distor, bool correctedM = false, bool showStats = true);
		static void generateMatchVects(vector<Track>& pointTracks, size_t firstSeenIdx,
			vector<Point2f>& pts1, vector<Point2f>& pts2, size_t existCloudSize = 0);
		static void get3d2dCorres(vector<Track>& pointTracks, vector<Point3f>& cloudPoints, vector<Point2f>& viewPoints, size_t viewIdx, size_t cloudSize);
		
		static void getExistMatchVects(vector<Track>& pointTracks, vector<Point3f>& cloudPoints, vector<Point2f>& pts1,
			vector<Point2f>& pts2, size_t viewIdx, size_t cloudSize);

		static void composeProjectionMatrix(Mat& R, Mat& T, Mat& P);
		static void convertFromHomogeneous(Mat& XYZW, Mat& XYZ);
		static void maskViewOutliers(vector<Track>& pointTracks, size_t viewPointsSize, size_t viewIdx, Mat& inlierIndices, size_t cloudSize);



	private:

	
		static bool notValid(Track myTrack);
	};



}


