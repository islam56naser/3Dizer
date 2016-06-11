#include "BundAdj.h"


namespace reconstruct3D{
	void BundAdj:: generateBundAdjData(vector<Track>& pointTracks, vector<vector<Point2f>>& imgPoints,
		vector<vector<int>>& vis, size_t nViews, size_t cloudSize){
		imgPoints.clear();
		vis.clear();
		imgPoints.resize(nViews);
		vis.resize(nViews);

		for (size_t i = 0; i < cloudSize; i++)
		{			
			size_t firstSeenIdx = pointTracks[i].firstImgIdx;
			size_t nPoints = pointTracks[i].pointList.size();

			for (size_t j = 0; j < firstSeenIdx; j++)
			{
				imgPoints[j].push_back(Point2f(-1, -1));
				vis[j].push_back(0);

			}
			
			for (size_t j = firstSeenIdx; j < (min(firstSeenIdx + nPoints, nViews)); j++)
			{
				Point2f pt = pointTracks[i].pointList[j - firstSeenIdx];
				imgPoints[j].push_back(pt);
				if (pt == Point2f(-1, -1))
				{
					vis[j].push_back(0);

				}
				else
				{
					vis[j].push_back(1);

				}

			}
			
			


			for (size_t j = (firstSeenIdx + min(nPoints,nViews)); j < nViews; j++)
			{
				imgPoints[j].push_back(Point(-1, -1));
				vis[j].push_back(0);

			}
		}
	


	}






	void BundAdj:: updateBundAdjParams(vector<Mat>& cams, vector<Mat>& distortions, vector<Mat>& rotations, vector<Mat>& translations,
		Mat& rot, Mat& trans, Mat& cam, Mat& distor){
		assert((rot.size() == Size(3, 3) || rot.size() == Size(3, 1) || rot.size() == Size(1, 3)) && 
			(trans.size() == Size(1, 3) || trans.size() == Size(3, 1)) && (cam.size() == Size(3,3)) && 
			(distor.size() == Size(4, 1) || distor.size() == Size(1, 4) || distor.size() == Size(5, 1) || distor.size() == Size(1, 5)));
	
		distortions.push_back(distor);
		cams.push_back(cam);
		translations.push_back(trans);
		rotations.push_back(rot);

	}

	void BundAdj:: initBundAdjParams(vector<Mat>& cams, vector<Mat>& distortions, vector<Mat>& rotations,
		vector<Mat>& translations, Mat& rot1, Mat& rot2, Mat& trans1, Mat& trans2, Mat& cam, Mat& distor){

		assert((rot1.size() == Size(3, 3) || rot1.size() == Size(3, 1) || rot1.size() == Size(1, 3)) &&
			(rot2.size() == Size(3, 3) || rot2.size() == Size(3, 1) || rot2.size() == Size(1, 3))&&
			(trans1.size() == Size(1, 3) || trans1.size() == Size(3, 1)) && 
			(trans2.size() == Size(1, 3) || trans2.size() == Size(3, 1)) &&
			(cam.size() == Size(3, 3)) && 
			(distor.size() == Size(4, 1) || distor.size() == Size(1, 4) || distor.size() == Size(5, 1) || distor.size() == Size(1, 5)));
		rotations.push_back(rot1);
		rotations.push_back(rot2);
		cams.push_back(cam);
		cams.push_back(cam);
		distortions.push_back(distor);
		distortions.push_back(distor);
		translations.push_back(trans1);
		translations.push_back(trans2);


	}

	void BundAdj:: updateAfterBA(vector<Point3f>& cloud, vector<Mat>& rotations, vector<Mat>& translations,
		vector<reconstruct3D::Track>& pointTracks, vector<Mat>& projections, cvsba::Sba::Params params){
		size_t cloudSize = cloud.size();

		if (params.type == cvsba::Sba::STRUCTURE)
		{
			for (size_t j = 0; j < cloudSize; j++)
			{
				pointTracks[j].scenePoint = cloud[j];
			}
		}
		else if (params.type == cvsba::Sba::MOTION)
		{
			for (size_t j = 0; j < rotations.size(); j++)
			{
				Utility::composeProjectionMatrix(rotations[j], translations[j], projections[j]);
			}
		}
		else
		{
			for (size_t j = 0; j < rotations.size(); j++)
			{
				Utility::composeProjectionMatrix(rotations[j], translations[j], projections[j]);

			}
			for (size_t j = 0; j < cloudSize; j++)
			{
				pointTracks[j].scenePoint = cloud[j];
			}
		}
	

	
	}


	void BundAdj:: runBundAdj(vector<Point3f>& cloud, vector<reconstruct3D::Track>& pointTracks, vector<vector<Point2f>>& imgPoints, vector<vector<int>>& vis
		, vector<Mat>& rotations, vector<Mat>& translations, vector<Mat>& cams, vector<Mat>& distortions, vector<Mat>& projections,
		cvsba::Sba::Params param,  bool update, int nConstFrames){

		size_t nViews = rotations.size();
		assert(translations.size() == nViews && cams.size() == nViews && distortions.size() == nViews && projections.size() == nViews && nConstFrames < nViews);
		cvsba::Sba bundAdjust;
		bundAdjust.setParams(param);
		cout << "Starting bundle adjustment" << endl;
		size_t cloudSize = cloud.size();
		generateBundAdjData(pointTracks, imgPoints, vis, nViews,cloudSize);
		cout << "generated BA data" << endl;
		bundAdjust.run(cloud, imgPoints, vis, cams, rotations, translations, distortions, nConstFrames);
		if (update)
		{
			updateAfterBA(cloud, rotations, translations, pointTracks, projections, param);
		}
		cout << "Initial Error : " << bundAdjust.getInitialReprjError() << endl;
		cout << "Final Error : " << bundAdjust.getFinalReprjError() << endl;

	}

}