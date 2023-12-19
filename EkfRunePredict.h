#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;


#define MAXROUNDCOUNT 1024
class EKFPredict
{
public:
	EKFPredict();
	~EKFPredict();

public:
	/**
	 * @brief  ��չ������Ԥ��������
	 *         ��1���ۼ�7���������Ƕ�Ӧ��ʱ���ǩ  ������[1]typeһ�� [2]framecount������ [3]��һ����ʹ˵����ؾ���С��80
	 *         ��2�����������˲�Ԥ����һ����
	 * @param  _timestamp  ʱ���
	 * @param  _framecount ֡����
	 * @return _thisPoint  ��֡�����������ؼ����
	 * 
	 * @author �����
	 * @date   2023.9.18
	 */
	Point2f EKFPredict_Main(double& _timestamp ,int &_framecount, Point2f &_thisPoint, int &_thisType);

	/**
	 * @brief  �ۼƹ۲�
	 *         ��1��framecountҪ����   �������
	 *         ��2����һ����ʹ˵���벻�ܴ���12  ������� 
	 *         ��3����������������Vector��4�� ������ǰŲһλ������ĸ��� 
	 * @param  _timestamp  ʱ���
	 * @param  _framecount ֡����
	 * @return _thisPoint  ��֡�����������ؼ����
	 * 
	 * @author �����
	 * @date   2023.9.19
	 */
	int EKFPredict_Accumulate(int& _framecount, Point2f& _thisPoint,int& _thisType, double& _timestamp);

	/**
	 * @brief  ����EKF�˲�������

	 * @param  
	 * 
	 * @author �����
	 * @date   2023.9.19
	 */
	void EKFPredict_EkfSetting();


	/**
	 * @brief  ���õ�һ���۲�ֵ
	 *
	 * @param 
	 *
	 * @author �����
	 * @date   2023.9.19
	 */
	void EKFPredict_EkfSetFirstState(Point2f& _firstPoint, float & _vx,float& _vy);

	Point2f EKFPredict_Predict(vector<double> &_timestampofpts , vector<Point2f> &_ptsofobervation);



public:
	int m_lastCount;
	int m_currentCount;

	int m_lastType;
	int m_currentType;

	Point2f m_lastPoint;
	Point2f m_currentPoint;

	vector<Point2f> m_ObservationPoints;
	vector<double> m_TimestampofPoints;  

	// �����������˲���
	KalmanFilter m_ekf;
	Mat m_F;
	Mat m_H;

	// m_AccmulateNum ���ۼ������ۼƷ���������Point 
	int m_AccmulateNum;

	// m_RenewJudgementֻ�����0/4��������һ��Ԥ���count ÿ���л�Ŀ����m_RenewJudgement = 0 �������ж���θ��¿������˲����ĳ�ʼstate ;
	int m_RenewJudgement;

	

	// ��һ������������ʱ�����һ��statepose���б�[0] [1]֮���vx vy�ٽ��и���
	float m_vx01, m_vy01;

	int m_fps;
};

static float distanceOfPoints(const Point2f& p1, const Point2f& p2) {
	float dx = p2.x - p1.x;
	float dy = p2.y - p1.y;
	return sqrt(dx * dx + dy * dy);
}

