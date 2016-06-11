#include "Inclusions.h"



using namespace cv;
using namespace std;


bool takePictures(size_t nImages, vector<reconstruct3D::ImageContainer>& imgCont){
	VideoCapture cap(0);

	Size imgSize = Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));

	if (!cap.isOpened()){
		cout << "Couldn't open the camera" << endl;
		return false;
	}

	short imgCounter = 0;
	Mat buffer;
	while (true){
		cap.read(buffer);
		if (imgCounter > (nImages - 1)){
			destroyWindow("Video");
			break;
		}
		char c = waitKey(25);
		if (c == 's'){
			reconstruct3D::ImageContainer cont(buffer);
			imwrite("Inputs/img" + to_string(imgCounter) + ".jpg", buffer);
			imgCont.push_back(cont);
			imgCounter++;
		}
		putText(buffer, "Press S to capture an image", Point(200, 400), CV_FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(200));
		putText(buffer, to_string(imgCounter) + "/" + to_string(nImages), Point(200, 460), CV_FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(200));
		imshow("Video", buffer);
	}
	return true;
}

bool readImages(size_t nImages, vector<reconstruct3D::ImageContainer>& imgConts){
	for (size_t i = 0; i < nImages; i++)
	{
		Mat buffer = imread("Inputs/img" + to_string(i) + ".jpg");
		if (buffer.empty()){
			cout << "Couldn't read images" << endl;
			return false;
		} 
		reconstruct3D::ImageContainer cont(buffer);
		imgConts.push_back(cont);
	}
	return true;
}


//void getBinMask(const Mat& gcMask, Mat& binMask){
//	int erosion_size = 5;
//	Mat element = getStructuringElement(cv::MORPH_ELLIPSE, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
//	Mat bin = gcMask & 1;
//	binMask = bin.mul(Mat::ones(bin.size(), CV_8U), 255);
//	dilate(binMask, binMask, element);
//	erode(binMask, binMask, element);
//
//}

//reconstruct3D::GCApplication gcapp;

//static void on_mouse(int event, int x, int y, int flags, void* param)
//{
//	gcapp.mouseClick(event, x, y, flags, param);
//
//}


void featureProcess(vector<reconstruct3D::ImageContainer>& imgConts, vector<vector<DMatch>>& matches){
	//--Feature detection and matching for each pair of images
	//----Step 1 : feature detection and description
	Size imgSize = imgConts[0].image.size();
	reconstruct3D::Feature::sfmDetectFeatures(imgConts, "SURF");
	reconstruct3D::Feature::sfmComputeFeatures(imgConts, "SURF");
	//----Step 2 : Feature matching with BF
	//--Very sensitive to this step 
	reconstruct3D::Feature::sfmBFMatcher(imgConts, matches, 0.8, imgSize.width / 8, true);

}

bool loadCamParams(Mat& cam, Mat& distortion){
	FileStorage Params("Inputs/out_camera_data.xml", FileStorage::READ);
	if (!Params.isOpened()){
		cout << "Couldn't open camera parameters file" << endl;
		return false;
	}
	Params["camera_matrix"] >> cam;
	Params["distortion_coefficients"] >> distortion;
	Params.release();
	cout << "Camera parameters loaded" << endl;
	return true;
}

void preprocessImgs(vector<reconstruct3D::ImageContainer>& imgConts){
	size_t nImages = imgConts.size();
	//string mode;
	//bool validInp = false,
	//	interactive = false;
	//while (! validInp){
	//	cout << "Enter the mode (person) or (scene) : ";
	//	getline(cin, mode);
	//	if (mode == "person"){
	//		validInp = true;
	//		interactive = true;
	//	}
	//	else if (mode == "scene"){
	//		validInp = true;
	//		interactive = false;
	//	}

	//}

	//if (!isOk){
	//	return -1;
	//}

	//if (interactive){
	//	const string winName = "image";
	//	namedWindow(winName, WINDOW_AUTOSIZE);
	//	setMouseCallback(winName, on_mouse, 0);
	//	for (size_t i = 0; i < nImages; i++)
	//	{
	//		gcapp.setImageAndWinName(imgCont[i].image, winName);
	//		gcapp.showImage();
	//		Mat mask, binMask;
	//		while (true)
	//		{
	//			char c = waitKey(0);
	//			switch (c)
	//			{
	//			case 27:
	//				cout << "Exiting ..." << endl;
	//				mask = gcapp.getMask();
	//				goto exit_main;
	//			case 'r':
	//				cout << endl;
	//				gcapp.reset();
	//				gcapp.showImage();
	//				break;
	//			case 'n':
	//				int iterCount = gcapp.getIterCount();
	//				cout << "<" << iterCount << "... ";
	//				int newIterCount = gcapp.nextIter();
	//				if (newIterCount > iterCount)
	//				{
	//					gcapp.showImage();
	//					cout << iterCount << ">" << endl;
	//				}
	//				else
	//					cout << "rect must be determined>" << endl;
	//				break;
	//			}
	//		}

	//	exit_main:
	//		getBinMask(mask, binMask);
	//		binMask.copyTo(imgCont[i].mask);
	//	}
	//}
	//--Convert all images to gray-scale
	for (size_t i = 0; i < nImages; i++)
	{
		cvtColor(imgConts[i].image, imgConts[i].grayImage, CV_BGR2GRAY);
		//if (interactive)
		//{
		//	imgCont[i].grayImage.copyTo(imgCont[i].segImage, imgCont[i].mask);

		//}
	}
}



int main(){

	//--Take M images from a video stream and do some preprocessing 
	size_t nImages = 10;
	vector<reconstruct3D::ImageContainer> imgCont;
	bool isOk = readImages(nImages, imgCont);
	if (!isOk)		return -1;
	preprocessImgs(imgCont);

	//--Load camera parameters
	clock_t start = clock();
	Mat k, dist;
	if (!loadCamParams(k, dist))	return -1;

	//--Feature detection & matching
	vector<vector<DMatch>> goodMatchesM;
	featureProcess(imgCont, goodMatchesM);
	printf("Time taken to match features across M images : %.2fs\n", (double)(clock() - start) / CLOCKS_PER_SEC);


	//First Method (incremental SFM)------------------------------------
	//--choose the initial pair and calculate Ess 
	Mat ess1, fund1;
	bool pairChosen = reconstruct3D::Epipolar::chooseValidPair(imgCont, goodMatchesM, k, dist, ess1, 0.98);
	
	if (!pairChosen){
		cout << "Couldn't find any pair to initialize reconstruction" << endl;
		return -1;
	}
	
	nImages = imgCont.size();
	cout << "First pair chosen " << endl;
	reconstruct3D::Epipolar::fundFromEss(ess1, fund1, k);

	//--Generate Point tracks
	vector<reconstruct3D::Track>tracks;
	reconstruct3D::Utility::generatePointTracks(imgCont, goodMatchesM, tracks);

	//--For the first pair of images do the following
	//----Step 1 : Calculate 2 vectors of matching points
	vector<Point2f> pts1, pts2;
	reconstruct3D::Utility::generateMatchVects(tracks, 0, pts1, pts2);
	correctMatches(fund1, pts1, pts2, pts1, pts2);


	//----Step 3 : calculate R ,T
	Mat rot2, trans2;
	double focal = k.at<double>(0, 0);
	Point2d principalPoint(k.at<double>(0, 2), k.at<double>(1, 2));
	int inliers = recoverPose(ess1, pts1, pts2, rot2, trans2, focal, principalPoint);
	cout << "The number of points for the first pair : " << pts1.size() << endl;
	cout << "The number of inliers for first pair : " << inliers << endl;


	//----Step 4 : calculate two projection matrices
	Mat proj1(3, 4, CV_64F), proj2(3, 4, CV_64F), rot1, trans1;
	rot1 = Mat::eye(3, 3, CV_64F);
	trans1 = Mat::zeros(3, 1, CV_64F);
	reconstruct3D::Utility::composeProjectionMatrix(rot1, trans1, proj1);
	reconstruct3D::Utility::composeProjectionMatrix(rot2, trans2, proj2);

	//----Step 5 : triangulate 2D points to get initial scene and store 3D point in its track
	vector<Point3f>scene;
	reconstruct3D::Utility::triangulateAndUpdateTracks(tracks, pts1, pts2, scene, proj1, proj2, k, dist, true);
	size_t newCloudSize = scene.size();
	size_t prevCloudSize = newCloudSize;
	
	//----Step 6 : output partial structure
	reconstruct3D::FileOutput::outputPCD("Outputs/scene1.pcd", tracks);
	vector<Mat> projM;
	projM.push_back(proj1);
	projM.push_back(proj2);
	


	//--Step 7 : BA init
	Mat rotVec1, rotVec2;
	Rodrigues(rot1, rotVec1);
	Rodrigues(rot2, rotVec2);
	vector<Mat> rotM, transM, camsM, distM;
	reconstruct3D::BundAdj:: initBundAdjParams(camsM, distM, rotM, transM, rot1, rot2, trans1, trans2, k, dist); 
	
	cvsba::Sba::Params sbaParams;
	sbaParams.fixedDistortion = 5;
	sbaParams.fixedIntrinsics = 5;
	sbaParams.iterations = 200;
	sbaParams.verbose = true;
	sbaParams.type = cvsba::Sba::STRUCTURE;

	vector<vector<Point2f>>imagePoints;
	vector <vector<int>>visibility;

	//--For each additional view i
	for (size_t i = 2; i < nImages; i++)
	{
		cout << "Adding view no : " << i + 1 << endl;
		//----Step 1 : Find view's existing 3D points in the cloud
		vector<Point2f>viewExist2DPoints;
		vector<Point3f>viewExist3DPoints;
		reconstruct3D::Utility::get3d2dCorres(tracks, viewExist3DPoints, viewExist2DPoints, i, newCloudSize);
		cout << "view's existing points in cloud are determined" << endl;

		//----Step 2 : Calculate the projection matrix for the view and Epipolar geometry with the previous view
		Mat rotVectV, transV, projV(3, 4, CV_64F), rotV, inliersV;		
		solvePnPRansac(viewExist3DPoints, viewExist2DPoints, k, dist, rotVectV, transV, false, 100, 3, 0.999, inliersV, SOLVEPNP_ITERATIVE); //1.5 to 3
		reconstruct3D::Utility::maskViewOutliers(tracks, viewExist2DPoints.size(), i, inliersV, newCloudSize);
		size_t inlierSize =  inliersV.size().height;
		cout << "The ratio of inliers to total points in this view : " << (float)inlierSize / viewExist2DPoints.size() << endl;
		Rodrigues(rotVectV, rotV);
		reconstruct3D::Utility::composeProjectionMatrix(rotV, transV, projV);
		projM.push_back(projV);
		cout << "view's projection matrix calculated" << endl;

		//----Step 4 : triangulate new points 
		vector<Point2f> ptsV1, ptsV2;
		reconstruct3D::Utility::generateMatchVects(tracks, i - 1, ptsV1, ptsV2, newCloudSize);

		if (ptsV1.size() > 0)
		{
			reconstruct3D::Utility::triangulateAndUpdateTracks(tracks, ptsV1, ptsV2, scene, projM[i - 1], projM[i], k, dist);
			newCloudSize = scene.size();

			cout << "New 3D points added" << endl;
	

			//----Step 5 : output partial structure 
			String fileName = "Outputs/scene";
			fileName += to_string(i);
			fileName += ".pcd";
			reconstruct3D::FileOutput::outputPCD(fileName, tracks);
			prevCloudSize = newCloudSize;

		}
		//-- BA here (Structure only works well)
		reconstruct3D::BundAdj::updateBundAdjParams(camsM, distM, rotM, transM, rotV, transV, k, dist);
		reconstruct3D::BundAdj::runBundAdj(scene, tracks, imagePoints, visibility, rotM, transM, camsM, distM, projM, sbaParams, true);
	
	}

	cout << "total number of 3D points :  " << newCloudSize << endl;
	sbaParams.iterations = 600;
	sbaParams.type = cvsba::Sba::MOTIONSTRUCTURE;
	reconstruct3D::BundAdj::runBundAdj(scene, tracks, imagePoints, visibility, rotM, transM, camsM, distM, projM, sbaParams, true, 1);
	reconstruct3D::FileOutput::outputPCD("Outputs/Complete scene.pcd", tracks);



	//--Calculate fundamentals
	vector<Mat> fundM;
	for (size_t i = 1; i < nImages; i++)
	{
		Mat essV, fundV;
		reconstruct3D::Epipolar::getEssMatrix(rotM[i - 1], rotM[i], transM[i - 1], transM[i], essV);
		reconstruct3D::Epipolar::fundFromEss(essV, fundV, k);
		fundM.push_back(fundV);
	}

	//--DOESN'T WORK well
	vector<Mat>flows;
	reconstruct3D::Dense::generateFlow(imgCont, goodMatchesM, flows);
	vector<reconstruct3D::Track>densTracks;
	vector<Point3f>densCloud;
	reconstruct3D::Utility::generatePointTracks(imgCont, flows, densTracks);
	
	for (size_t i = 1; i < nImages; i++)
	{
		vector<Point2f>dens1, dens2;
		reconstruct3D::Utility::generateMatchVects(densTracks, i - 1, dens1, dens2, densCloud.size());
		if (dens1.size() > 0)
		{
			correctMatches(fundM[i - 1], dens1, dens2, dens1, dens2);			//increases reprojection error sometimes
			reconstruct3D::Utility::triangulateAndUpdateTracks(densTracks, dens1, dens2, densCloud, projM[i - 1], projM[i], k, dist,false);
			cout << "Avg Epip error : " << reconstruct3D::Epipolar::calcEpipError(fundM[i - 1], dens1, dens2) << endl;

		}
		else{
			cout << "No points to triangulate in loop : " << i << endl;
		}

	}
	
	reconstruct3D::FileOutput::outputPCD("Outputs/Complete dense.pcd", densTracks);
	//

	////--filter outliers
	//reconstruct3D::Filter::filterOutliers(densTracks, densTracks, densCloud, 4, 2.5);
	//reconstruct3D::FileOutput::outputPCD("filtered1.pcd", densTracks);

	//reconstruct3D::Filtering::histFilter(denseScene, denseScene, 10, 10, 10);
	//reconstruct3D::FileOutput::outputPCD("filtered2.pcd", denseScene);


	////-- Load input file into a PointCloud<T> with an appropriate type
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	//pcl::PCLPointCloud2 cloudBlob;
	//pcl::io::loadPCDFile("Complete dense.pcd", cloudBlob);
	//pcl::fromPCLPointCloud2(cloudBlob, *cloud);



	////--Filtering using radius_outlier_removal
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
	//pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> radOutRem;
	//radOutRem.setInputCloud(cloud);
	//radOutRem.setRadiusSearch(0.04);			//--Tweak for optimizing radius outlier removal 
	//radOutRem.setMinNeighborsInRadius(12);		//--Tweak for optimizing radius outlier removal 
	//radOutRem.filter(*cloud_filtered);

	////--Output filtered cloud
	//pcl::io::savePCDFile("filtWithPCL.pcd", *cloud_filtered);



	////-- Normal estimation 
	//pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> n;
	//pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	//pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
	//tree->setInputCloud(cloud_filtered);
	//n.setInputCloud(cloud_filtered);
	//n.setSearchMethod(tree);
	//n.setKSearch(50);							//--Tweak for optimizing normal estimation of cloud 
	//n.compute(*normals);

	////-- Concatenate the XYZ and normal fields
	//pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	//pcl::concatenateFields(*cloud_filtered, *normals, *cloud_with_normals);

	////-- Create search tree
	//pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
	//tree2->setInputCloud(cloud_with_normals);

	////-- Initialize objects
	//pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;
	//pcl::PolygonMesh triangles;

	////-- Set the maximum distance between connected points (maximum edge length)
	//gp3.setSearchRadius(0.08);					//change it for dense to 0.0XX   

	////-- Set typical values for the parameters
	//gp3.setMu(2.5);								//typical is 2.5 to 3
	//gp3.setMaximumNearestNeighbors(50);			//tweak along with search radius for optimal surface meshing 
	//gp3.setMaximumSurfaceAngle(M_PI / 4);		// 45 degrees
	//gp3.setMinimumAngle(M_PI / 18);				// 10 degrees
	//gp3.setMaximumAngle(2 * M_PI / 3);			// 120 degrees
	//gp3.setNormalConsistency(false);

	////-- Get result
	//gp3.setInputCloud(cloud_with_normals);
	//gp3.setSearchMethod(tree2);
	//gp3.reconstruct(triangles);


	////--Save the mesh
	//pcl::io::saveVTKFile("mesh.vtk", triangles);



	printf("Runtime: %.2fs\n", (double)(clock() - start) / CLOCKS_PER_SEC);
	return 0;
}