#include "Filter.h"



namespace reconstruct3D{
	void Filter::distFromCentroid(vector<Point3f>& cloud, Point3f centroid, vector<double>& distances){
		for (size_t i = 0; i < cloud.size(); i++)
		{
			distances.push_back(norm(cloud[i] - centroid));

		}


	}

	void Filter::filterKeyPoints(vector<vector<KeyPoint>>& keys, vector<vector<KeyPoint>>& filteredKeys,
		Size imgSize, size_t xBins, size_t yBins){
		filteredKeys.resize(keys.size());
		for (size_t i = 0; i < keys.size(); i++)
		{
			filterKeyPoints(keys[i], filteredKeys[i], imgSize, xBins, yBins);
		}
	}

	bool Filter:: isBigger(Point3f pt, Point3f refPt){
		return (pt.x >= refPt.x && pt.y >= refPt.y && pt.z >= refPt.z);
	}

	bool Filter:: isSmaller(Point3f pt, Point3f refPt){
		return (pt.x <= refPt.x && pt.y <= refPt.y && pt.z <= refPt.z);
	}




	void Filter::histFilter(vector<Point3f>& cloud,vector<Point3f>& filteredCloud, size_t xBins, size_t yBins, size_t zBins){
		size_t cloudSize = cloud.size();
		vector<vector<Point3f>> filteredHist;
		//--Determine extreme points
		Point3f minPt = Point3f(1000, 1000, 1000);
		Point3f maxPt = Point3f(0, 0, 0);
		for (size_t i = 0; i < cloudSize ; i++)
		{
			Point3f pt = cloud[i];
			if (pt.x < minPt.x)
			{
				minPt.x = pt.x;
			}
			else if (pt.x > maxPt.x)
			{
				maxPt.x = pt.x;
			}
			if (pt.y < minPt.y)
			{
				minPt.y = pt.y;
			}
			else if (pt.y > maxPt.y)
			{
				maxPt.y = pt.y;
			}
			if (pt.z < minPt.z)
			{
				minPt.z = pt.z;
			}
			else if (pt.z > maxPt.z)
			{
				maxPt.z = pt.z;
			}
		}
		float binLenX = (maxPt.x - minPt.x) / xBins;
		float binLenY = (maxPt.y - minPt.y) / yBins;
		float binLenZ = (maxPt.z - minPt.z) / zBins;
		vector<vector<vector<vector<Point3f>>>> pointHist(xBins);
		for (size_t i = 0; i < xBins; i++)
		{
			pointHist[i].resize(yBins);
			for (size_t j = 0; j < yBins; j++)
			{
				pointHist[i][j].resize(zBins);

			}

		}

		for (size_t n = 0; n < cloudSize; n++)
		{
			//--Loop over all cubes 
			bool found = false;
			for (size_t i = 0; i < xBins; i++)
			{
				for (size_t j = 0; j < yBins; j++)
				{
					for (size_t k = 0; k < zBins; k++)
					{
						Point3f binMinPt = Point3f(i*binLenX, j*binLenY, k*binLenZ) + minPt;
						Point3f binMaxPt = binMinPt + Point3f(binLenX, binLenY, binLenZ);
						if (isBigger(cloud[n],binMinPt) && isSmaller(cloud[n],binMaxPt))
						{
							pointHist[i][j][k].push_back(cloud[n]);
							found = true;
							break;
						}

					}
					if (found) break;
				}
				if (found) break;
			}
		}

		size_t maxSize = 0;
		for (size_t i = 0; i < xBins; i++)
		{
			for (size_t j = 0; j < yBins; j++)
			{
				for (size_t k = 0; k < zBins; k++)
				{
					size_t currentSize = pointHist[i][j][k].size();
					if (currentSize > maxSize)
					{
						maxSize = currentSize;
					}

				}

			}

		}
		vector<short>sizes;
		for (size_t i = 0; i < xBins; i++)
		{
			for (size_t j = 0; j < yBins; j++)
			{
				for (size_t k = 0; k < zBins; k++)
				{
					size_t currentSize = pointHist[i][j][k].size();
					if (currentSize > 0.15 * maxSize)
					{
						sizes.push_back(currentSize);
					}

				}

			}

		}
		Scalar mu, sigma;
		meanStdDev(sizes, mu, sigma);
		size_t threshold = (size_t)ceil( mu[0] - 2 * sigma[0]);
		for (size_t i = 0; i < xBins; i++)
		{
			for (size_t j = 0; j < yBins ; j++)
			{
				for (size_t k = 0; k < zBins; k++)
				{
					size_t pointSize = pointHist[i][j][k].size();
					if (pointSize > threshold)
					{
						filteredHist.push_back(pointHist[i][j][k]);
					}
				}

			}

		}
		filteredCloud.clear();
		for (size_t i = 0; i < filteredHist.size(); i++)
		{
			for (size_t j = 0; j < filteredHist[i].size(); j++)
			{
				filteredCloud.push_back(filteredHist[i][j]);
			}
		}

	}



	void Filter::filterKeyPoints(vector<KeyPoint>& keys, vector<KeyPoint>& filteredKeys,
		Size imgSize, size_t xBins, size_t yBins){

		float imgWidth = (float)imgSize.width;
		float imgHeight = (float)imgSize.height;
		Size2f binSize = Size2f(imgWidth / xBins, imgHeight / yBins);
		vector<vector<vector<KeyPoint>>> keyHist(xBins);
		for (size_t i = 0; i < xBins; i++)
		{
			keyHist[i].resize(yBins);
		}

		for (size_t i = 0; i < keys.size(); i++)
		{
			Point2f pt = keys[i].pt;

			//--check if the point lies in up left quarter of image
			if (Rect2f(0, 0, imgWidth / 2, imgHeight / 2).contains(pt))
			{
				size_t xEnd = (size_t)ceil(xBins / 2);
				size_t yEnd = (size_t)ceil(yBins / 2);
				bool found = false;
				for (size_t w = 0; w < xEnd; w++)
				{

					for (size_t h = 0; h < yEnd; h++)
					{
						if (Rect2f(w*binSize.width, h*binSize.height, binSize.width, binSize.height).contains(pt))
						{
							keyHist[w][h].push_back(keys[i]);
							found = true;
							break;
						}

					}
					if (found) break;
				}


			}
			//--check if the point lies in up right quarter of image
			else if (Rect2f(imgWidth / 2, 0, imgWidth / 2, imgHeight / 2).contains(pt))
			{
				size_t xStart = (size_t)floor(xBins / 2);
				size_t yEnd = (size_t)ceil(yBins / 2);
				bool found = false;
				for (size_t w = xStart; w < xBins; w++)
				{

					for (size_t h = 0; h < yEnd; h++)
					{
						if (Rect2f(w*binSize.width, h*binSize.height, binSize.width, binSize.height).contains(pt))
						{
							keyHist[w][h].push_back(keys[i]);
							found = true;
							break;
						}

					}
					if (found) break;
				}
			}
			//--check if the point lies in down left quarter of image
			else if (Rect2f(0, imgHeight / 2, imgWidth / 2, imgHeight / 2).contains(pt))
			{
				size_t xEnd = (size_t)ceil(xBins / 2);
				size_t yStart = (size_t)floor(yBins / 2);
				bool found = false;
				for (size_t w = 0; w < xEnd; w++)
				{

					for (size_t h = yStart; h < yBins; h++)
					{
						if (Rect2f(w*binSize.width, h*binSize.height, binSize.width, binSize.height).contains(pt))
						{
							keyHist[w][h].push_back(keys[i]);
							found = true;
							break;
						}

					}
					if (found) break;
				}

			}
			else
			{
				size_t xStart = (size_t)floor(xBins / 2);
				size_t yStart = (size_t)floor(yBins / 2);
				bool found = false;
				for (size_t w = xStart; w < xBins; w++)
				{

					for (size_t h = yStart; h < yBins; h++)
					{
						if (Rect2f(w*binSize.width, h*binSize.height, binSize.width, binSize.height).contains(pt))
						{
							keyHist[w][h].push_back(keys[i]);
							found = true;
							break;
						}

					}
					if (found) break;
				}
			}
		}
		size_t maxSize = 0;
		vector<short> sizes;
		for (size_t i = 0; i < xBins; i++)
		{
			for (size_t j = 0; j < yBins; j++)
			{
				size_t currentSize = keyHist[i][j].size();
				if (currentSize > 0)
				{
					sizes.push_back(currentSize);
					if (currentSize  > maxSize)
					{
						maxSize = currentSize;
					}

				}
				
		
			}
		}
		Scalar mu, sigma;
		meanStdDev(sizes, mu, sigma);
		double threshold = mu[0] - sigma[0];
		for (size_t i = 0; i < xBins; i++)
		{
			for (size_t j = 0; j < yBins; j++)
			{
				size_t keySize = keyHist[i][j].size();
				if (keySize > threshold){
					for (size_t k = 0; k <keyHist[i][j].size(); k++)
					{
						filteredKeys.push_back(keyHist[i][j][k]);

					}

				}

			}

		}
	}

	void Filter::depthFromCentroid(vector<Point3f>& cloud, Point3f centroid, vector<double>& depths){
		for (size_t i = 0; i < cloud.size(); i++)
		{
			depths.push_back(abs(cloud[i].z - centroid.z));

		}
	}


	void Filter::filterOutliers(vector<Point3f>& cloud, vector<Point3f>& filteredCloud, double distThreshold, double depthThreshold){
		
		int oldSize = cloud.size();
		int newSize = 0;
		vector<Point3f> lastFilteredCloud;
		filteredCloud = cloud;
		while (newSize < oldSize)
		{
			lastFilteredCloud.clear();
			lastFilteredCloud = filteredCloud;
			oldSize = lastFilteredCloud.size();
			filteredCloud.clear();
			Scalar Centroid = (sum(lastFilteredCloud) / oldSize);
			Point3d centroid = Point3d(Centroid.val[0], Centroid.val[1], Centroid.val[2]);
			vector<double>distances;
			vector<double>depths;
			distFromCentroid(lastFilteredCloud, centroid, distances);
			depthFromCentroid(lastFilteredCloud, centroid, depths);
			Scalar mu, sigma, mu2, sigma2;
			meanStdDev(distances, mu, sigma);
			meanStdDev(depths, mu2, sigma2);

			for (size_t i = 0; i < lastFilteredCloud.size(); i++)
			{
				if (distances[i] < mu.val[0] + distThreshold * sigma.val[0] && depths[i] < mu2.val[0] + depthThreshold * sigma2.val[0]){
					filteredCloud.push_back(lastFilteredCloud[i]);
				}

			}

			newSize = filteredCloud.size();
		}
	}


	 void Filter:: filterOutliers(vector<Track>& pointTracks, vector<Track>& filteredTracks, vector<Point3f>& filteredCloud,
		double distThreshold, double depthThreshold){
		vector<Point3f> cloud;
		for (size_t i = 0; i < pointTracks.size(); i++)
		{
			cloud.push_back(pointTracks[i].scenePoint);
		
		}

		int oldSize = cloud.size();
		int newSize = 0;
		vector<Point3f> lastFilteredCloud;
		vector<Track> lastFiltTracks;
		filteredCloud = cloud;
		filteredTracks = pointTracks;
		while (newSize < oldSize)
		{
			lastFilteredCloud.clear();
			lastFiltTracks.clear();
			lastFilteredCloud = filteredCloud;
			lastFiltTracks = filteredTracks;
			oldSize = lastFilteredCloud.size();
			filteredCloud.clear();
			filteredTracks.clear();
			Scalar Centroid = (sum(lastFilteredCloud) / oldSize);
			Point3d centroid = Point3d(Centroid.val[0], Centroid.val[1], Centroid.val[2]);
			vector<double>distances;
			vector<double>depths;
			distFromCentroid(lastFilteredCloud, centroid, distances);
			depthFromCentroid(lastFilteredCloud, centroid, depths);
			Scalar mu, sigma, mu2, sigma2;
			meanStdDev(distances, mu, sigma);
			meanStdDev(depths, mu2, sigma2);

			for (size_t i = 0; i < lastFilteredCloud.size(); i++)
			{
				if (distances[i] < mu.val[0] + distThreshold * sigma.val[0] && depths[i] < mu2.val[0] + depthThreshold * sigma2.val[0]){
					filteredCloud.push_back(lastFilteredCloud[i]);
					filteredTracks.push_back(lastFiltTracks[i]);
				}

			}

			newSize = filteredCloud.size();
		}



	}


}

