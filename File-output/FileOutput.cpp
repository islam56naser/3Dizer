#include "FileOutput.h"

namespace reconstruct3D{
	void FileOutput:: outputPCD(const String& fileName, vector<Point3f>& cloud){
		ofstream output(fileName);
		size_t N = cloud.size();
		output << "# .PCD v.7 - Point Cloud Data file format" << endl;
		output << "VERSION .7" << endl;
		output << "FIELDS x y z" << endl;
		output << "SIZE 4 4 4" << endl;
		output << "TYPE F F F" << endl;
		output << "COUNT 1 1 1" << endl;
		output << "WIDTH " << N << endl;
		output << "HEIGHT 1" << endl;
		output << "VIEWPOINT 0 0 0 1 0 0 0" << endl;
		output << "POINTS " << N << endl;
		output << "DATA ascii " << endl;

		for (size_t i = 0; i < N; i++)
		{
			output << cloud[i].x << " " << cloud[i].y << " " << cloud[i].z << endl;

		}
		output.close();
	}
	 void FileOutput:: outputPCD(const String& fileName, vector<Point3f>& cloud, vector<int>& colorCode){
		 ofstream output(fileName);
		 size_t N = cloud.size();
		 if (colorCode.size() != N)
		 {
			 cout << "Size mismatch between points and color values" << endl;
			 return;
		 }
		 output << "# .PCD v.7 - Point Cloud Data file format" << endl;
		 output << "VERSION .7" << endl;
		 output << "FIELDS x y z rgb" << endl;
		 output << "SIZE 4 4 4 4" << endl;
		 output << "TYPE F F F F" << endl;
		 output << "COUNT 1 1 1 1" << endl;
		 output << "WIDTH " << N << endl;
		 output << "HEIGHT 1" << endl;
		 output <<"VIEWPOINT 0 0 0 1 0 0 0" << endl;
		 output << "POINTS " << N << endl;
		 output << "DATA ascii " << endl;

		 for (size_t i = 0; i < N; i++)
		 {
			 output << cloud[i].x << " " << cloud[i].y << " " << cloud[i].z <<" "<< colorCode[i] << endl;

		 }
		 output.close();


	}

	 void FileOutput::outputPCD(const String& fileName, vector<Track>& pointTracks){
		 vector<Point3f> cloud;
		 vector<int> colorCodes;
		 for (size_t i = 0; i < pointTracks.size(); i++)
		 {
			 Point3f scenePoint = pointTracks[i].scenePoint;
			 if (scenePoint != Point3f(-1,-1,-1))
			 {
				 cloud.push_back(scenePoint);
				 colorCodes.push_back(pointTracks[i].colorCode);
			 }
			
		 }
		 outputPCD(fileName, cloud, colorCodes);

	 }


}
