#include "Dense.h"


namespace reconstruct3D{
	
	void Dense::pointsToKeyPoints(vector<Point2f>& pts, vector<KeyPoint>& keyPoints,float size,float angle, float response, int octave, int classID){
		for (size_t i = 0; i < pts.size(); i++)
		{
			keyPoints.push_back(KeyPoint(pts[i], size,angle,response,octave,classID));
		}
	
	}
	void Dense::extendPointTracks(vector<Track>& oldTracks, vector<Track>& newTracks){
		for (size_t i = 0; i < newTracks.size(); i++)
		{
			oldTracks.push_back(newTracks[i]);

		}
	}

	void Dense::generateFlow(ImageContainer& imgCont1, ImageContainer& imgCont2, vector<DMatch>& matches, Mat& flow){
		Ptr<ximgproc::EdgeAwareInterpolator> interpol = ximgproc::createEdgeAwareInterpolator();
		vector<Point2f> pts1, pts2;
		vector<KeyPoint> keys1, keys2;
		keys1 = imgCont1.imgKeys;
		keys2 = imgCont2.imgKeys;
		assert(keys1.size() > 0 && keys2.size() > 0);
		for (size_t i = 0; i < matches.size(); i++)
		{
			size_t idx1 = matches[i].trainIdx;
			size_t idx2 = matches[i].queryIdx;
			pts1.push_back(keys1[idx1].pt);
			pts2.push_back(keys2[idx2].pt);

		}
		int k = 0;
		if (pts1.size() > 180)
		{
			k = 128;
		}
		else
		{
			k = (int) pts1.size() / 2;
		}

		interpol->setK(k);
		interpol->interpolate(imgCont1.grayImage, pts1, imgCont2.grayImage, pts2, flow);
	}

	void Dense:: rectifyImgPair(vector<Point2f>& pts1, vector<Point2f>& pts2, Mat& img1, Mat& img2, 
		Mat& fund, Mat& cam, Mat& dist, Mat& img1Rect, Mat& img2Rect){
		Size imgSize = img1.size();
		Mat H1, H2, firstImgMap1, firstImgMap2, secImgMap1, secImgMap2, R1, R2;
		stereoRectifyUncalibrated(pts1, pts2, fund, imgSize, H1, H2);
		R1 = cam.inv() * H1 * cam;
		R2 = cam.inv() * H2 * cam;
		initUndistortRectifyMap(cam, dist, R1, cam, imgSize, CV_32FC1, firstImgMap1, firstImgMap2);
		initUndistortRectifyMap(cam, dist, R2, cam, imgSize, CV_32FC1, secImgMap1, secImgMap2);
		remap(img1, img1Rect, firstImgMap1, firstImgMap2, INTER_LINEAR);
		remap(img2, img2Rect, secImgMap1, secImgMap2, INTER_LINEAR);

	}

	void Dense:: rectifyAllImgs(vector<reconstruct3D::ImageContainer>& imgConts, vector<vector<DMatch>>& matches, vector<Mat>& fundamentals, Mat& cam, Mat& dist){
		size_t nImages = imgConts.size();
		for (size_t i = 1; i < nImages; i++)
		{
			vector<Point2f>ptsV1, ptsV2;
			reconstruct3D::Utility::getPointMatches(imgConts[i - 1], imgConts[i], matches[i - 1], ptsV1, ptsV2);
			reconstruct3D::Utility::getPointMatches(imgConts[i - 1], imgConts[i], matches[i - 1], ptsV1, ptsV2);
			rectifyImgPair(ptsV1, ptsV2, imgConts[i - 1].grayImage, imgConts[i].grayImage, fundamentals[i - 1], cam, dist,
				imgConts[i - 1].rectifiedImg, imgConts[i].rectifiedImg);
			imshow("img2", imgConts[i - 1].rectifiedImg);
			imshow("img3", imgConts[i].rectifiedImg);
			waitKey(0);
		}

	}



	void Dense::generateFlow(vector<ImageContainer>& imgCont, vector<vector<DMatch>>& matches, vector<Mat>& flow){
		size_t nImages = imgCont.size();
		flow.resize(nImages - 1);
		for (size_t i = 1; i < nImages; i++)
		{
			generateFlow(imgCont[i - 1], imgCont[i], matches[i - 1], flow[i - 1]);
		}
	}
	void Dense::extendImgKeyPoints(vector<ImageContainer>& imgCont, vector<vector<KeyPoint>>& newKeys){
		size_t nImages = imgCont.size();

		for (size_t i = 0; i < nImages; i++)
		{
			for (size_t j = 0; j < newKeys[i].size(); j++)
			{
				imgCont[i].imgKeys.push_back(newKeys[i][j]);

			}

		}

	}

	void Dense::extendCloud(vector<Point3f>& oldCloud, vector<Point3f>& newCloud){
		for (size_t i = 0; i < newCloud.size(); i++)
		{
			oldCloud.push_back(newCloud[i]);

		}


	}

	void Dense:: draw_delaunay(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color)
	{

		vector<Vec6f> triangleList;
		subdiv.getTriangleList(triangleList);
		vector<Point> pt(3);
		Size size = img.size();
		Rect rect(0, 0, size.width, size.height);

		for (size_t i = 0; i < triangleList.size(); i++)
		{
			Vec6f t = triangleList[i];
			pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
			pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
			pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

			// Draw rectangles completely inside the image.
			if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
			{
				line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
				line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
				line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
			}
		}
	}




	void Dense :: draw_point(Mat& img, Point2f fp, Scalar color)
	{
		circle(img, fp, 2, color, CV_FILLED, CV_AA, 0);
	}



	void Dense:: getCentroids(vector<Vec6f>& triangleList, vector<Point2f>& centroids){
		for (size_t i = 0; i < triangleList.size(); i++)
		{
			Vec6f t = triangleList[i];
			Point2f pt1 = Point2f(t[0], t[1]);
			Point2f pt2 = Point2f(t[2], t[3]);
			Point2f pt3 = Point2f(t[4], t[5]);
			//---add centroid if the triangle area > threshold
			Mat cross = (Mat_<float>(3, 3) << 1.f, 1.f, 1.f, pt1.x, pt2.x, pt3.x, pt1.y, pt2.y, pt3.y);
			if (0.5 * determinant(cross) > 12)
			{
				centroids.push_back((pt1 + pt2 + pt3) / 3);

			}
		}


	}

	void Dense::getNewKeyPoints(vector<ImageContainer>& imgCont, vector<vector<KeyPoint>>& newKeys, bool drawPoints,
		float size, float angle, float response, int octave, int classID){
		size_t nImages = imgCont.size();
		vector<vector<Point2f>> newPoints(nImages);
		for (size_t i = 0; i < nImages; i++)
		{
			getNewPoints(imgCont[i], newPoints[i], drawPoints);
			pointsToKeyPoints(newPoints[i], newKeys[i],size,angle,response,octave,classID);

		}

	}

	void Dense :: getNewPoints(ImageContainer& imgCont, vector<Point2f>& newPoints, bool drawPoints){

		Subdiv2D sub(Rect(0, 0, imgCont.image.cols, imgCont.image.rows));
		for (size_t i = 0; i < imgCont.imgKeys.size(); i++)
		{
			sub.insert(imgCont.imgKeys[i].pt);
		}
		sub.getTriangleList(imgCont.triList);
		getCentroids(imgCont.triList, newPoints);

		if (drawPoints)
		{
			Mat imgCopy = imgCont.image.clone();
			for (size_t i = 0; i < newPoints.size(); i++)
			{
				draw_point(imgCopy, newPoints[i], Scalar(50, 70, 110));
			}

			draw_delaunay(imgCopy, sub, Scalar(100, 150, 200));
			imshow("Delaunay", imgCopy);
			waitKey(0);
		}

	}
}