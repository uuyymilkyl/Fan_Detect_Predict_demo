#include "Dnn_Deploy.h"


DnnDeploy::DnnDeploy()
{

}

DnnDeploy::~DnnDeploy()
{
}

Net DnnDeploy::DnnReader(std::string _cfgpath, std::string _wgtpath)
{
	dnn::Net Darknet;
	Darknet = readNetFromDarknet(_cfgpath, _wgtpath);
	Darknet.setPreferableBackend(DNN_BACKEND_OPENCV);
	Darknet.setPreferableTarget(DNN_TARGET_CPU);

	return Darknet;
}
cv::Point calculateCenter(const int imgWidth, const int imgHeight, const cv::Mat& roi) 
{
	cv::Mat blueMask;
	cv::inRange(roi, cv::Scalar(100, 0, 0), cv::Scalar(255, 50, 50), blueMask); // 可根据实际情况调整阈值

	std::vector<cv::Point> bluePoints;
	cv::findNonZero(blueMask, bluePoints);

	cv::Point center(0, 0);
	for (const auto& point : bluePoints) {
		center += point;
	}

	if (!bluePoints.empty()) {
		center.x /= bluePoints.size();
		center.y /= bluePoints.size();
	}

	center.x += imgWidth / 2;
	center.y += imgHeight / 2;

	return center;
}

void DnnDeploy::DnnDeployMain(Net& _net, Mat& _image)
{
	out_Boxes.clear();
	out_Idx.clear();
	out_Scores.clear();
	classIds.clear();
	m_vdResultSrtuct.clear();
	float confThreshold = 0.85;
	float nmsThreshold = 0.6;
	if (DnnPreprocess(_net, _image, inputWidth, inputHeight, 3) != 0) {
		DnnPostProcess(_image, confThreshold, nmsThreshold);
	}

}

int DnnDeploy::DnnPreprocess(dnn::Net& _net, cv::Mat& _image, int _W, int _H, int _C)
{
	int imgW = _W;
	int imgH = _H;
	int imgChannels = _C;
	m_net = _net;

	Mat image = _image;
	if (image.empty() == true)
	{
		std::cout << " Darknet Input is Empty !" << std::endl;
		return 0;
	}
	dnn::blobFromImage(image, m_blob, 1 / 255.0, Size(imgW, imgH), Scalar(0, 0, 0), true, false);
	m_net.setInput(m_blob);
	return 1;
}

std::vector<Mat> DnnDeploy::DnnOutputs()
{
	std::vector<String> vstNames;
	std::vector<Mat>    vmatOutputs;
	if (vstNames.empty())
	{
		std::vector<int> vnOutLayers = m_net.getUnconnectedOutLayers();
		std::vector<String> vstLayersNames = m_net.getLayerNames();

		vstNames.resize(vnOutLayers.size());
		for (unsigned i = 0; i < vnOutLayers.size(); i++)
		{
			vstNames[i] = vstLayersNames[vnOutLayers[i] - 1];
		}

		m_net.forward(vmatOutputs, vstNames);

	}
	return vmatOutputs;
}

void DnnDeploy::DnnPostProcess(cv::Mat& _image, float& confThred, float& nmsThred)
{
	vector<int> vnClassIdx;
	vector<float> vnConfidence;
	vector<Rect>  vrBoxes;

#ifdef DEBUGMODE
	Mat show_image = _image;
	m_show_image = show_image;

#endif

	vector<Mat> vmOutputs = DnnOutputs();

	for (unsigned i = 0; i < vmOutputs.size(); ++i)
	{
		float* data = (float*)vmOutputs[i].data;
		for (int j = 0; j < vmOutputs[i].rows; ++j, data += vmOutputs[i].cols)
		{
			Mat scores = vmOutputs[i].row(j).colRange(5, vmOutputs[i].cols);
			Point classIdPoint;
			double confidence;

			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThred)
			{
				int centerX = (int)(data[0] * _image.cols);
				int centerY = (int)(data[1] * _image.rows);
				int width = (int)(data[2] * _image.cols);
				int height = (int)(data[3] * _image.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				
				vnClassIdx.push_back(classIdPoint.x);
				vnConfidence.push_back((float)confidence);
				vrBoxes.push_back(Rect(left, top, width, height));
			}
		}
	}
	DnnResult result;     ///< 声明结构体变量
	cv::Point2f center;
	vector<int> vnIndeces;
	NMSBoxes(vrBoxes, vnConfidence, confThred, nmsThred, vnIndeces);

	for (unsigned q = 0; q < vnIndeces.size(); ++q)
	{
		int idx = vnIndeces[q];
		Rect box = vrBoxes[idx];
		float score = vnConfidence[idx];
		int classId = vnClassIdx[idx];

		/* 原本的结果 */
		out_Idx.push_back(classId);
		out_Boxes.push_back(box);
		out_Scores.push_back(score);
		/* 用结构体放置结果 */
		result.classId = classId;
		result.box = box;
		result.score = score;

		int centerx = box.x + box.width / 2;
		int centery = box.y + box.height / 2;
		center.x = centerx;  //< 这里我直接拿预测框中心去进预测 实际上需要用传统视觉重新定位靶面center
		center.y = centery;
		result.center = center;
		m_vdResultSrtuct.push_back(result);  ///< 将一帧图像的所有结果放回 vector<DnnResult> (是类的公有变量）
	}

	if (out_Boxes.size() >= 1)
	{
		Mat image = _image.clone();
		Mat show_image = _image.clone();
		for (int p = 0; p < out_Boxes.size(); p++)
		{
			if (out_Boxes[p].x < 0)
				out_Boxes[p].x = 0;
			if (out_Boxes[p].y < 0)
				out_Boxes[p].y = 0;
			if ((out_Boxes[p].x + out_Boxes[p].width) > image.rows)
				out_Boxes[p].x = image.rows - 1 - out_Boxes[p].width;
			if ((out_Boxes[p].y + out_Boxes[p].height) > image.cols)
				out_Boxes[p].y = image.cols - 1 - out_Boxes[p].height;
			Mat cut_Area = image(out_Boxes[p]).clone();
			m_vmDetectAreaList.push_back(cut_Area);
			
			rectangle(show_image, out_Boxes[p], Scalar(0, 0, 255), 2, LINE_8, 0);
			Point draw;
			draw.x = out_Boxes[p].x;
			draw.y = out_Boxes[p].y;
			putText(show_image, "TYPE:[" + to_string(out_Idx[p]) + "]", draw, FONT_HERSHEY_COMPLEX_SMALL, 1.5, Scalar(0, 0, 255), 2);
			
		}
		m_show_image = show_image;
	}

}

