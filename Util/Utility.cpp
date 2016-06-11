#include "Utility.h"

namespace reconstruct3D{

	void Utility:: generatePointTracks(vector<Mat>& images,vector<vector<KeyPoint>>& keyPoints, vector<vector<DMatch>>& goodMatchesM, vector<Track>& pointTracks){
		size_t nImages = images.size();
		assert(nImages > 1 && keyPoints.size() == nImages && goodMatchesM.size() == nImages - 1);
		bool isRGB = true;
		for (size_t i = 0; i < nImages; i++)
		{
			if (images[i].type() != CV_8UC3)
			{
				isRGB = false;
				break;
			}

		}

		pointTracks.clear();
		vector<vector<Point2i>>keyIndicesM(nImages);

		for (size_t i = 0; i < nImages; i++)
		{
			keyIndicesM[i].resize(keyPoints[i].size());
			for (size_t j = 0; j < keyPoints[i].size(); j++)
			{

				keyIndicesM[i][j] = Point(-1, -1);

			}

		}

		for (size_t i = 0; i <nImages - 1; i++)
		{
			for (size_t j = 0; j < goodMatchesM[i].size(); j++)
			{
				size_t idx1, idx2;
				idx1 = goodMatchesM[i][j].trainIdx;
				idx2 = goodMatchesM[i][j].queryIdx;
				if (keyIndicesM[i][idx1].y == -1)
				{
					keyIndicesM[i][idx1].y = idx2;
					keyIndicesM[i + 1][idx2].x = idx1;
				}

			}

		}

		for (size_t i = 0; i < nImages; i++)
		{
			for (size_t j = 0; j < keyIndicesM[i].size(); j++)
			{
				if ((keyIndicesM[i][j].x = -1) && (keyIndicesM[i][j].y != -1))
				{
					Track newTrack(i);
					Point pos = keyPoints[i][j].pt;
					if (isRGB)
					{
						Vec3b rgb = images[i].at<Vec3b>(pos);
						newTrack.colorCode =  ((int)rgb[2]) << 16 | ((int)rgb[1]) << 8 | ((int)rgb[0]);
					}
					newTrack.pointList.push_back(keyPoints[i][j].pt);
					size_t L = i + 1;
					short nextIdx = keyIndicesM[i][j].y;
					short prevIdx = -1;
					while ((nextIdx != -1) && L < nImages)
					{
						newTrack.pointList.push_back(keyPoints[L][nextIdx].pt);
						prevIdx = nextIdx;
						nextIdx = keyIndicesM[L][prevIdx].y;
						L++;

					}
					 pointTracks.push_back(newTrack);
				}
			}
		}


	}

	void Utility::generatePointTracks(vector<ImageContainer>& imgCont, vector<vector<DMatch>>& goodMatchesM, vector<Track>& pointTracks){
		size_t nImages = imgCont.size();
		assert(nImages > 1 && goodMatchesM.size() == nImages - 1);
		vector<vector<KeyPoint>> keyPoints;
		vector<Mat> images;
		for (size_t i = 0; i < nImages; i++)
		{
			keyPoints.push_back(imgCont[i].imgKeys);
			images.push_back(imgCont[i].image);
		}
		generatePointTracks(images, keyPoints, goodMatchesM, pointTracks);

	}

	 




	double Utility::triangulate(vector<Point2f>& pts1, vector<Point2f>& pts2, vector<Point3f>& cloud,
		Mat& P1, Mat& P2, Mat& cam, Mat& distor){
		size_t ptsSize = pts1.size();

		assert(pts2.size() == ptsSize && ptsSize != 0 && !P1.empty() && !P2.empty() && !cam.empty() && !distor.empty());

		Mat XYZW(4, ptsSize, CV_32F), XYZ;
		vector<Point2f> normPts1, normPts2;
		undistortPoints(pts1, normPts1, cam, distor);
		undistortPoints(pts2, normPts2, cam, distor);
		triangulatePoints(P1, P2, normPts1, normPts2, XYZW);
		convertFromHomogeneous(XYZW, XYZ);
		size_t unrefCloud = XYZ.cols;
		for (size_t i = 0; i < unrefCloud; i++)
		{
			Point3f scenePoint = Point3f(XYZ.at<float>(0, i), XYZ.at<float>(1, i), XYZ.at<float>(2, i));
			cloud.push_back(scenePoint);
		}
		Mat r1 = P1(Rect(0, 0, 3, 3));
		Mat t1 = P1(Rect(3, 0, 1, 3));
		Mat rVec1;
		Rodrigues(r1, rVec1);
		vector<Point2f> projPoints;
		cv::projectPoints(cloud, rVec1, t1, cam, distor, projPoints);
		double reproj = 0;
		for (size_t i = 0; i < ptsSize; i++)
		{
			reproj += norm(projPoints[i] - pts1[i]);
		}

		return reproj / unrefCloud;
	}


	void Utility :: triangulateAndUpdateTracks(vector<Track>& pointTracks, vector<Point2f>& pts1, vector<Point2f>& pts2, 
		vector<Point3f>& cloud, Mat& P1, Mat& P2, Mat& cam, Mat& distor,bool correctedM, bool showStats){
		size_t ptsSize = pts1.size();
		size_t nPrev = cloud.size();
		assert(pointTracks.size() >= ptsSize);
		vector<Point3f> unrefCloud;
		double reproj = triangulate(pts1, pts2, unrefCloud, P1, P2, cam, distor);

		if (reproj < 4)
		{
			for (size_t i = 0; i < ptsSize; i++)
			{
				Point3f scenePoint = unrefCloud[i];
				if (scenePoint.z > 0)
				{
					cloud.push_back(scenePoint);
					pointTracks[i + nPrev].scenePoint = scenePoint;
					if (correctedM)
					{
						pointTracks[i + nPrev].pointList[0] = pts1[i];
						pointTracks[i + nPrev].pointList[1] = pts2[i];
					}

				}
			}
		}
	

		size_t nNew = cloud.size();
		unsigned int negZ = ptsSize - (nNew - nPrev);
		

		if (showStats)
		{
			cout << "The number of points in this view : " << nNew - nPrev << endl;
			cout << "No of -Z points in this pair : " << negZ << endl;
			cout << "Avg reproj error : " << reproj << endl;
		}

		vector<Track>::iterator bgn = pointTracks.begin() + nPrev;
		vector<Track>::iterator it = remove_if(bgn, bgn + ptsSize, notValid);
		pointTracks.erase(it, it + negZ);
	}


	void Utility::generateMatchVects(vector<Track>& pointTracks, size_t firstSeenIdx,
		vector<Point2f>& pts1, vector<Point2f>& pts2,size_t existCloudSize){

		assert(existCloudSize <= pointTracks.size());
		for (size_t j = existCloudSize; j < pointTracks.size(); j++)
		{
			if (pointTracks[j].firstImgIdx == firstSeenIdx)
			{
				pts1.push_back(pointTracks[j].pointList[0]);
				pts2.push_back(pointTracks[j].pointList[1]);

			}
		}
		
	}

	void Utility::generatePointTracksNoMask(vector<ImageContainer>& imgCont, vector<Mat>& flows, vector<Track>& denseTracks){
		cout << "NOT MASKED " << endl;
		size_t nImages = imgCont.size();
		Size imgsSize = imgCont[0].image.size();
		bool isRGB = true;
		vector<Mat>imgMasks;
	
		for (size_t i = 0; i < nImages; i++)
		{
			Mat mask = Mat::zeros(imgsSize, CV_32S);
			imgMasks.push_back(mask);
			if (imgCont[i].image.type() != CV_8UC3)
			{
				isRGB = false;
				break;
			}
		}

		vector<Scalar> meus(nImages - 1), stds(nImages - 1);
		for (size_t i = 0; i < nImages - 1; i++)
		{
			meanStdDev(flows[i], meus[i], stds[i]);

		}

		size_t imgWidth = imgsSize.width;
		size_t imgHeight = imgsSize.height;
		Rect imgWindow = Rect(0, 0,imgWidth,imgHeight);
		for (size_t n = 0; n < nImages; n++)
		{

			for (size_t i = 0; i < imgHeight; i++)
			{
				for (size_t j = 0; j < imgWidth; j++)
				{

					if (imgMasks[n].at<int>(i, j) == 0)
					{
						Track newTrack(n);
						Point2f pt1 = Point2f((float)j, (float)i);
						if (isRGB)
						{
							Vec3b bgr = imgCont[n].image.at<Vec3b>(i, j);
							newTrack.colorCode = ((int)bgr[2]) << 16 | ((int)bgr[1]) << 8 | ((int)bgr[0]);
						}	
						newTrack.pointList.push_back(pt1);
						for (size_t k = n; k < nImages - 1; k++)
						{
							Vec2f motion = flows[k].at<Vec2f>(i, j);
							bool accurateMot = (motion[0] < meus[k][0] + 1.2 * stds[k][0] && motion[0] > meus[k][0] - 1.2 * stds[k][0] &&
								motion[1] < meus[k][1] + 1.2 * stds[k][1] && motion[1] > meus[k][1] - 1.2 * stds[k][1]); //0.5 to 2
							pt1 += Point2f(motion[0], motion[1]);
							size_t rowIdx = (size_t)round(pt1.y);
							size_t colIdx = (size_t)round(pt1.x);
							if (!imgWindow.contains(Point(colIdx,rowIdx))|| !accurateMot) break;
							newTrack.pointList.push_back(pt1);
							imgMasks[k + 1].at<int>(rowIdx, colIdx) = 1;
						}

						if (newTrack.pointList.size() > 1) denseTracks.push_back(newTrack);
					}
				}
			}

		}

	}

	void Utility::generatePointTracks(vector<ImageContainer>& imgCont, vector<Mat>& flows, vector<Track>& denseTracks){
		size_t nImages = imgCont.size() ;
		Size imgsSize = imgCont[0].grayImage.size();
		assert(nImages > 1 && flows.size() == (nImages - 1));
		bool isMasked = true;
		for (size_t i = 0; i < nImages; i++)
		{
			if (imgCont[i].mask.empty())
			{
				isMasked = false;
				break;
			}
		}
		

		if (isMasked)
		{
			cout << " MASKED " << endl;
			bool isRGB = true;
			vector<Mat>imgMasks;
		
			for (size_t i = 0; i < nImages; i++)
			{
				Mat mask = Mat::zeros(imgsSize, CV_16S);
				imgMasks.push_back(mask);
				if (imgCont[i].image.type() != CV_8UC3)
				{
					isRGB = false;
					break;
				}
			}
			vector<Scalar> meus(nImages - 1), stds(nImages - 1);
			for (size_t i = 0; i < nImages - 1; i++)
			{
				meanStdDev(flows[i], meus[i], stds[i]);

			}
			size_t imgWidth = imgsSize.width;
			size_t imgHeight = imgsSize.height;
			Rect imgWindow = Rect(0, 0, imgWidth, imgHeight);
			for (size_t n = 0; n < nImages; n++)
			{
				for (size_t i = 0; i < imgHeight; i++)
				{
					for (size_t j = 0; j < imgWidth; j++)
					{
						if (imgMasks[n].at<int>(i, j) == 0 && (int)imgCont[n].mask.at<uchar>(i, j) == 255)
						{
							Track newTrack(n);
							//--Get color code from first image
							Point2f pt1 = Point2f((float)j, (float)i);
							if (isRGB)
							{
								Vec3b rgb = imgCont[n].image.at<Vec3b>(i, j);
								newTrack.colorCode = ((int)rgb[2]) << 16 | ((int)rgb[1]) << 8 | ((int)rgb[0]);
							}
						
							newTrack.pointList.push_back(pt1);
							for (size_t k = n; k < min(nImages - 1, n + 3); k++)
							{
								Vec2f motion = flows[k].at<Vec2f>(i, j);
								bool notAccurate = (abs(motion[0]) > abs(meus[i][0]) + 2 * stds[i][0] || abs(motion[1]) > abs(meus[i][1]) + 2 * stds[i][1]);
								pt1 += Point2f(motion[0], motion[1]);
								size_t rowIdx = (size_t)round(pt1.y);
								size_t colIdx = (size_t)round(pt1.x);
								if (!imgWindow.contains(Point(colIdx,rowIdx)) || notAccurate) break;
								newTrack.pointList.push_back(pt1);
								imgMasks[k + 1].at<int>(rowIdx, colIdx) = 1;
							}

							if (newTrack.pointList.size() > 1) denseTracks.push_back(newTrack);
						}
					}

				}

			}
		}
		else
		{
			generatePointTracksNoMask(imgCont, flows, denseTracks);
		}
	}


	void Utility:: maskViewOutliers(vector<Track>& pointTracks, size_t viewPointsSize, size_t viewIdx, Mat& inlierIndices, size_t cloudSize){
		assert(cloudSize <= pointTracks.size() && viewPointsSize <= cloudSize && !inlierIndices.empty());

		vector<bool> mask(viewPointsSize, false);
		size_t maskIdx = 0;
		size_t inlierSize = inlierIndices.rows;
		for (size_t i = 0; i < inlierSize; i++)
		{
			mask[inlierIndices.at<int>(i, 0)] = true;


		}
		for (size_t i = 0; i < cloudSize; i++)
		{
			size_t nPoints = pointTracks[i].pointList.size();
			size_t firstSeenIdx = pointTracks[i].firstImgIdx;

			if ((nPoints + firstSeenIdx) > viewIdx)
			{
				if (!mask[maskIdx])
				{		
					pointTracks[i].pointList[viewIdx - firstSeenIdx] = Point2f(-1, -1);
				}
			
				maskIdx++;

			}

		}
	}




	void Utility::get3d2dCorres(vector<Track>& pointTracks, vector<Point3f>& cloudPoints, vector<Point2f>& viewPoints, size_t viewIdx, size_t cloudSize){
		assert(cloudSize <= pointTracks.size());
		for (size_t j = 0; j < cloudSize; j++)
		{
			size_t nPoints = pointTracks[j].pointList.size();
			size_t firstSeenIdx = pointTracks[j].firstImgIdx;
			if ((nPoints + firstSeenIdx) > viewIdx)
			{

				viewPoints.push_back(pointTracks[j].pointList[viewIdx - firstSeenIdx]);
				cloudPoints.push_back(pointTracks[j].scenePoint);
			}

		}

	}

	void Utility::getExistMatchVects(vector<Track>& pointTracks, vector<Point3f>& cloudPoints, vector<Point2f>& pts1,
		vector<Point2f>& pts2, size_t viewIdx, size_t cloudSize){
		for (size_t j = 0; j < cloudSize; j++)
		{
			size_t nPoints = pointTracks[j].pointList.size();
			size_t firstSeenIdx = pointTracks[j].firstImgIdx;
			if ((nPoints + firstSeenIdx) > viewIdx + 1)
			{

				pts1.push_back(pointTracks[j].pointList[viewIdx - firstSeenIdx]);
				pts2.push_back(pointTracks[j].pointList[viewIdx - firstSeenIdx + 1]);
				cloudPoints.push_back(pointTracks[j].scenePoint);
			}

		}


	}



	void Utility :: composeProjectionMatrix(Mat& R, Mat& T, Mat& P){
		Mat rot;
		if (P.size() == Size(0,0))
		{
			P = Mat::zeros(3, 4, CV_64F);
		}

		assert((R.size() == Size(3, 3) || R.size() == Size(3, 1) || R.size() == Size(1, 3)) &&
			(T.size() == Size(1, 3) || T.size() == Size(3, 1)) && P.size() == Size(4, 3));

		if (R.size() != Size(3,3))
		{
			Rodrigues(R, rot);
		}
		else 
		{
			rot = R;
		}

		Mat ROI;
		ROI = P(Rect(0, 0, 3, 3));
		rot.copyTo(ROI);
		ROI = P(Rect(3, 0, 1, 3));
		T.copyTo(ROI);
	}

	void Utility :: convertFromHomogeneous(Mat& XYZW, Mat& XYZ){
		Mat W, W_vect, XYZScaled;
		XYZScaled = XYZW.rowRange(0, 3);
		W = XYZW.rowRange(3, 4);
		W_vect = Mat::ones(3, 1, CV_32F)* W;
		XYZ = XYZScaled.mul(1 / W_vect);

	}

	bool Utility :: notValid(Track myTrack){
		return (myTrack.scenePoint == Point3f(-1, -1, -1));
	}


	void Utility :: getPointMatches(ImageContainer& imgCont1, ImageContainer& imgCont2, vector<DMatch>& matches, vector<Point2f>& pts1, vector<Point2f>& pts2){

		getPointMatches(imgCont1.imgKeys, imgCont2.imgKeys, matches, pts1, pts2);


	}
	void Utility :: getPointMatches(vector<KeyPoint>& keys1, vector<KeyPoint>& keys2, vector<DMatch>& matches, vector<Point2f>& pts1, vector<Point2f>& pts2){

		size_t n = matches.size();
		for (size_t i = 0; i < n; i++)
		{
			pts1.push_back(keys1[matches[i].trainIdx].pt);
			pts2.push_back(keys2[matches[i].queryIdx].pt);

		}

	}




}


