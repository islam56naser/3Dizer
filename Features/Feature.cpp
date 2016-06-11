#include "Feature.h"

namespace reconstruct3D{
	void Feature::sfmDetectAndCompFeatures(vector<ImageContainer>& imgCont,const String& detectorType, const String& descripType){
		sfmDetectFeatures(imgCont, detectorType);
		sfmComputeFeatures(imgCont, descripType);

	}

	void Feature::sfmDetectFeatures(vector<ImageContainer>& imgCont, const String& detectorType){
		Ptr<FeatureDetector>detector;
		assert(detectorType == "SURF" || detectorType == "SIFT" || detectorType == "FAST");
		size_t nImages = imgCont.size();
		if (detectorType == "SURF")
		{
			detector = SURF::create(400);
		}
		else if (detectorType == "SIFT")
		{
			detector = SIFT::create();
		}
		else if (detectorType == "FAST")
		{
			detector = FastFeatureDetector::create();

		}

		for (size_t i = 0; i < nImages; i++)
		{
			detector->detect(imgCont[i].grayImage, imgCont[i].imgKeys);

		}



	}








	void Feature::sfmComputeFeatures(vector<ImageContainer>& imgCont, const String& descripType){
		assert(descripType == "SURF" || descripType == "SIFT" || descripType == "FREAK");

		size_t nImages = imgCont.size();
		if (descripType == "SURF")
		{
			Ptr<SURF> extractor = SURF::create();
			for (size_t i = 0; i < nImages; i++)
			{
				extractor->compute(imgCont[i].grayImage, imgCont[i].imgKeys,imgCont[i].descriptor);

			}

		}
		else if (descripType == "SIFT")
		{
			Ptr<SIFT> extractor = SIFT::create();
			for (size_t i = 0; i < nImages; i++)
			{
				extractor->compute(imgCont[i].grayImage, imgCont[i].imgKeys, imgCont[i].descriptor);

			}

		}
		else if (descripType == "FREAK")
		{
			Ptr<FREAK> extractor = FREAK::create();
			for (size_t i = 0; i < nImages; i++)
			{
				extractor->compute(imgCont[i].grayImage, imgCont[i].imgKeys, imgCont[i].descriptor);

			}
			
		}
		
	}
	void Feature::sfmComputeFeatures(vector<Mat>& images, vector<vector<KeyPoint>>& keys, vector<Mat>& descriptors, const String& descripType){
		assert(descripType == "SURF" || descripType == "SIFT" || descripType == "FREAK");

		if (descripType == "SURF"){
			Ptr<SURF> extractor = SURF::create();
			extractor->compute(images,keys, descriptors);


		}
		else if (descripType == "SIFT")
		{
			Ptr<SIFT> extractor = SIFT::create();
			extractor->compute(images, keys, descriptors);

		}
		else if (descripType == "FREAK")
		{
			Ptr<FREAK> extractor = FREAK::create();
			extractor->compute(images, keys, descriptors);

		}

	}

	void Feature::sfmBFMatcher(vector<Mat>& images, vector<vector<KeyPoint>>& keys, vector<Mat>& descriptors, vector<vector<DMatch>>& goodMatchesM
		, double distance, bool crossCheck, bool binary, bool showMatches){
		size_t nImages = images.size();
		assert(nImages > 1);
		if (!binary)
		{
			BFMatcher matcher(NORM_L2, crossCheck);
			goodMatchesM.resize(nImages - 1);
			
			for (size_t i = 1; i < nImages; i++)
			{
				vector<DMatch> matches;
				matcher.match(descriptors[i], descriptors[i - 1], matches);
				for (size_t j = 0; j < matches.size(); j++)
				{

					Point2f to = keys[i][matches[j].queryIdx].pt;
					Point2f from = keys[i - 1][matches[j].trainIdx].pt;
					if (norm(to-from) < distance)
					{
						goodMatchesM[i - 1].push_back(matches[j]);

					}

				}


				if (showMatches)
				{
					Mat img_matches;
					drawMatches(images[i], keys[i], images[i - 1], keys[i - 1], goodMatchesM[i - 1], img_matches);
					imshow("Matches", img_matches);
					waitKey(0);
				}
			}


		}
		else
		{
			

			BFMatcher matcher(NORM_HAMMING2, crossCheck);
			goodMatchesM.resize(nImages - 1);
			for (size_t i = 1; i < nImages; i++)
			{
				vector<DMatch> matches;
				matcher.match(descriptors[i], descriptors[i-1], matches);
				for (size_t j = 0; j < matches.size(); j++)
				{

					Point2f to = keys[i][matches[j].queryIdx].pt;
					Point2f from = keys[i - 1][matches[j].trainIdx].pt;
					if (norm(to-from))
					{
						goodMatchesM[i - 1].push_back(matches[j]);

					}

				}


				if (showMatches)
				{
					Mat img_matches;
					drawMatches(images[i], keys[i], images[i - 1], keys[i - 1], goodMatchesM[i - 1], img_matches);
					imshow("Matches", img_matches);
					waitKey(0);
				}
			}


		}



	}



	void Feature::sfmFlannMatcher(vector<ImageContainer>& imgCont, vector<vector<DMatch>>& goodMatchesM,double distance, size_t nMatches, bool showMatches){
		size_t nImages = imgCont.size();
		assert(nImages > 1);
		FlannBasedMatcher matcher;
		for (size_t m = 1; m < nImages; m++)
		{
			vector<vector<DMatch>> matches;
			vector<DMatch>goodMatches;
			matcher.knnMatch(imgCont[m].descriptor, imgCont[m - 1].descriptor, matches, nMatches);
			for (size_t i = 0; i < matches.size(); i++){
				for (size_t j = 0; j < nMatches; j++){

					Point2f to = imgCont[m].imgKeys[matches[i][j].queryIdx].pt;
					Point2f from = imgCont[m-1].imgKeys[matches[i][j].trainIdx].pt;
					if (norm(to-from) < distance){
						goodMatches.push_back(matches[i][j]);
						j = nMatches;
					}

				}

			}
			goodMatchesM.push_back(goodMatches);
			if (showMatches)
			{
				Mat img_matches;
				drawMatches(imgCont[m].grayImage, imgCont[m].imgKeys, imgCont[m - 1].grayImage, imgCont[m - 1].imgKeys, goodMatches, img_matches);
				imshow("Matches", img_matches);
				waitKey(0);
			}

		}
	}





	void Feature::sfmBFMatcher(vector<ImageContainer>& imgCont, vector<vector<DMatch>>& goodMatchesM, double ratio, double distance, bool showMatches){
		BFMatcher matcher;
		size_t nImages = imgCont.size();
		assert(nImages > 1);
		for (size_t m = 1; m < nImages; m++)
		{
			vector<vector<DMatch>> matches;
			vector<DMatch>goodMatches;
			matcher.knnMatch(imgCont[m].descriptor, imgCont[m - 1].descriptor, matches, 2);
			for (size_t i = 0; i < matches.size(); i++)
			{
				Point2f to = imgCont[m].imgKeys[matches[i][1].queryIdx].pt;
				Point2f from = imgCont[m - 1].imgKeys[matches[i][0].trainIdx].pt;
				if (matches[i][0].distance < ratio * matches[i][1].distance && norm(to - from) < distance)
				{
					goodMatches.push_back(matches[i][0]);
				}

			}
			goodMatchesM.push_back(goodMatches);
			if (showMatches)
			{
				Mat img_matches;
				drawMatches(imgCont[m].grayImage, imgCont[m].imgKeys, imgCont[m - 1].grayImage, imgCont[m - 1].imgKeys, goodMatches, img_matches);
				imshow("Matches", img_matches);
				waitKey(0);
			}
		}

	}



	void Feature::sfmBFMatcher(vector<ImageContainer>& imgCont, vector<vector<DMatch>>& goodMatchesM
		, double distance, bool crossCheck, bool binary, bool showMatches){
		size_t nImages = imgCont.size();
		assert(nImages > 1);		
		if (!binary)
		{
			BFMatcher matcher(NORM_L2, crossCheck);
			goodMatchesM.resize(nImages - 1);
			for (size_t i = 1; i < nImages; i++)
			{
				vector<DMatch> matches;
				matcher.match(imgCont[i].descriptor, imgCont[i - 1].descriptor, matches);
				for (size_t j = 0; j < matches.size(); j++)
				{

					Point2f to = imgCont[i].imgKeys[matches[j].queryIdx].pt;
					Point2f from = imgCont[i - 1].imgKeys[matches[j].trainIdx].pt;
					if (norm(to - from) < distance)
					{
						goodMatchesM[i - 1].push_back(matches[j]);

					}

				}


				if (showMatches)
				{
					Mat img_matches;
					drawMatches(imgCont[i].grayImage, imgCont[i].imgKeys, imgCont[i - 1].grayImage, imgCont[i - 1].imgKeys, goodMatchesM[i - 1], img_matches);
					imshow("Matches", img_matches);
					waitKey(0);
				}
			}


		}
		else
		{
			BFMatcher matcher(NORM_HAMMING2, crossCheck);
			goodMatchesM.resize(nImages - 1);
			for (size_t i = 1; i < nImages; i++)
			{
				vector<DMatch> matches;
				matcher.match(imgCont[i].descriptor, imgCont[i - 1].descriptor, matches);
				for (size_t j = 0; j < matches.size(); j++)
				{

					Point2f to = imgCont[i].imgKeys[matches[j].queryIdx].pt;
					Point2f from = imgCont[i - 1].imgKeys[matches[j].trainIdx].pt;
					if (norm(to - from) < distance)
					{
						goodMatchesM[i - 1].push_back(matches[j]);

					}

				}


				if (showMatches)
				{
					Mat img_matches;
					drawMatches(imgCont[i].grayImage, imgCont[i].imgKeys, imgCont[i - 1].grayImage, imgCont[i - 1].imgKeys, goodMatchesM[i - 1], img_matches);
					imshow("Matches", img_matches);
					waitKey(0);
				}
			}


		}

	}

}


