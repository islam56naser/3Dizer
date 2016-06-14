#pragma once
#include "opencv2\xfeatures2d.hpp"
#include "opencv2\features2d.hpp"
#include "opencv2\highgui.hpp"
#include "ImageContainer.h"
#include <iostream>


namespace reconstruct3D{
	using namespace cv;
	using namespace std;
	using namespace xfeatures2d;
	class Feature
	{
	public:
		static void sfmDetectAndCompFeatures(vector<ImageContainer>& imgCont,const String& detectorType, const String& descripType);
		static void sfmDetectFeatures(vector<ImageContainer>& imgConts, const String& detectorType);
		static void sfmComputeFeatures(vector<ImageContainer>& imgCont, const String& descripType);

		static void sfmFlannMatcher(vector<ImageContainer>& imgCont, vector<vector<DMatch>>& goodMatchesM,
			double distance, size_t nMatches, bool showMatches);


		static void sfmBFMatcher(vector<ImageContainer>& imgCont, vector<vector<DMatch>>& goodMatchesM, double ratio, double distance, bool showMatches);

		static void sfmBFMatcher(vector<ImageContainer>& imgCont, vector<vector<DMatch>>& goodMatchesM,double distance, bool binary, bool showMatches);

	




	private:
	
	};


}


