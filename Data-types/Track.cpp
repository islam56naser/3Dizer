#include "Track.h"

namespace reconstruct3D{

	Track::Track(size_t idx1)
	{
		this->firstImgIdx = idx1;
		this->scenePoint = Point3f(-1, -1, -1);
		this->colorCode = 0;

	}

	void Track::print(){
		cout << "First seen in image : " << this->firstImgIdx << ",";
		cout << "Scene Point : " <<this->scenePoint << endl;
		cout << this->pointList << endl;

	}






}



