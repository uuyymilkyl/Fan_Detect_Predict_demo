#ifndef _DNN_DEPLOY_H
#define _DNN_DEPLOY_H

#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include<string>

using namespace cv;
using namespace dnn;
using namespace std;

#define DEBUGMODE 0 

/// < ���ÿ��
#define inputWidth 640
#define inputHeight 480

constexpr auto configs = "./models/yolov4-tiny.cfg";
constexpr auto weights = "./models/yolov4-tiny_last.weights";


struct DnnResult 
{

	Rect box;
	Point2f center;
	float score ;
	int classId ;

};

class DnnDeploy
{
public:
	DnnDeploy();
	~DnnDeploy();

public:

	void DnnDeployMain(Net& _net, Mat& _image);

	/**
	* 
	* @brief  ��ȡDnn�����ļ�
	* @param  _cfgpath config·��
	* @param  _wgtpath weight·��
	* @return Net      ����
	* @author �����
	* @date   2023.9.10
	*/
	static dnn::Net DnnReader(std::string _cfgpath, std::string _wgtpath);

    /**
	* 
	* @brief  ��ȡDnn�����ļ�
	* @param  _cfgpath config·��
	* @param  _wgtpath weight·��
	* @return Net      ����
	* @author �����
	* @date   2023.9.10
	*/
	int DnnPreprocess(dnn::Net& _net, cv::Mat& _image, int _W, int _H, int _C);

	/**
	* 
	* @brief  ��ȡDnn������
	* @return vector<Mat> vmatOutputs��������������
	* @author �����
	* @date   2023.9.10
	*/
	std::vector<Mat> DnnOutputs();

	/**
	* 
	* @brief  Dnn����������
	* @param  _image ԭͼ
	* @param  confThred ���Ŷ�
	* @return nmsThered �Ǽ���ֵ����
	* @author �����
	* @date   2023.9.10
	*/
	void DnnPostProcess(cv::Mat& _image, float& confThred, float& nmsThred);



public:
	/*�������*/
	dnn::Net m_net;
	cv::Mat m_blob;

	/*���ӻ�ͼ��*/
	cv::Mat m_show_image;

	/*�������*/
	vector<int> out_Idx;          // Ŀǰд��demo��û�����ֺ�����������Ҳû���õ�  �� 0�� 1�� ��
	vector<float> out_Scores;     // �������
	vector<Rect> out_Boxes;       // �����
	vector<int> classIds;         // �����

	vector<Mat> m_vmDetectAreaList;   // ����������û�õ����

	vector<DnnResult> m_vdResultSrtuct; ///< Ϊ�˺�������򵥵Ľ���ṹ�� 

};



#endif