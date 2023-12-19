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
	 * @brief  拓展卡尔曼预测主流程
	 *         【1】累计7个点与他们对应的时间标签  条件：[1]type一致 [2]framecount数连续 [3]上一个点和此点像素距离小于80
	 *         【2】进卡尔曼滤波预测下一个点
	 * @param  _timestamp  时间戳
	 * @param  _framecount 帧计数
	 * @return _thisPoint  此帧检测的能量机关激活点
	 * 
	 * @author 黄敏瑜
	 * @date   2023.9.18
	 */
	Point2f EKFPredict_Main(double& _timestamp ,int &_framecount, Point2f &_thisPoint, int &_thisType);

	/**
	 * @brief  累计观测
	 *         【1】framecount要连续   否则清空
	 *         【2】上一个点和此点距离不能大于12  否则清空 
	 *         【3】符合条件且已有Vector满4后 后三点前挪一位放入第四个点 
	 * @param  _timestamp  时间戳
	 * @param  _framecount 帧计数
	 * @return _thisPoint  此帧检测的能量机关激活点
	 * 
	 * @author 黄敏瑜
	 * @date   2023.9.19
	 */
	int EKFPredict_Accumulate(int& _framecount, Point2f& _thisPoint,int& _thisType, double& _timestamp);

	/**
	 * @brief  设置EKF滤波器参数

	 * @param  
	 * 
	 * @author 黄敏瑜
	 * @date   2023.9.19
	 */
	void EKFPredict_EkfSetting();


	/**
	 * @brief  设置第一个观测值
	 *
	 * @param 
	 *
	 * @author 黄敏瑜
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

	// 创建卡尔曼滤波器
	KalmanFilter m_ekf;
	Mat m_F;
	Mat m_H;

	// m_AccmulateNum 是累计数，累计符合条件的Point 
	int m_AccmulateNum;

	// m_RenewJudgement只会等于0/4，开启下一次预测的count 每次切换目标则m_RenewJudgement = 0 ，用于判断如何更新卡尔曼滤波器的初始state ;
	int m_RenewJudgement;

	

	// 下一个点若是连续时，则第一个statepose用列表[0] [1]之间的vx vy再进行更替
	float m_vx01, m_vy01;

	int m_fps;
};

static float distanceOfPoints(const Point2f& p1, const Point2f& p2) {
	float dx = p2.x - p1.x;
	float dy = p2.y - p1.y;
	return sqrt(dx * dx + dy * dy);
}

