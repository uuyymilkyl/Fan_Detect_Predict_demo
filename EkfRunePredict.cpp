#include "EkfRunePredict.h"


/*  注释 带_xxx 表示函数输入的变量   带m_xxx 表示类的公有变量，请注意理解*/
EKFPredict::EKFPredict()
{
}

EKFPredict::~EKFPredict()
{
}

Point2f EKFPredict::EKFPredict_Main(double& _timestamp, int& _framecount, Point2f& _thisPoint, int&_thisType)
{
    /* 如果累计了7个连续的点，则将点集和时间戳都放进卡尔曼滤波器  */
    int AccmulateNum = EKFPredict_Accumulate(_framecount, _thisPoint,_thisType,_timestamp);
    Point2f PreDictPoint;
    if (AccmulateNum == 7)
    {
        float vx = 0;
        float vy = 0;
        vector < Point2f > PredicePoints = m_ObservationPoints;
        vector <double > PredictTimes = m_TimestampofPoints;
        
        EKFPredict_EkfSetting(); ///< 首次满7，重新设置滤波器
        EKFPredict_EkfSetFirstState(m_ObservationPoints[0], vx, vy);
        PreDictPoint = EKFPredict_Predict(PredictTimes,PredicePoints);
        return PreDictPoint;
    }
    else if (AccmulateNum > 7) {
        vector < Point2f > PredicePoints = m_ObservationPoints;
        vector <double > PredictTimes = m_TimestampofPoints;

        //EKFPredict_EkfSetting();
        EKFPredict_EkfSetFirstState(m_ObservationPoints[0], m_vx01, m_vy01);
        PreDictPoint = EKFPredict_Predict(PredictTimes, PredicePoints);
        return PreDictPoint;
    }
   
    //不符合 则返回原点
    return _thisPoint;
    
}

int EKFPredict::EKFPredict_Accumulate(int& _framecount, Point2f& _thisPoint, int& _thisType, double& _timestamp)
{

    /* 第一次计数 初始化前后计数和前后点 */
    if (_framecount == 0)
    {
        m_lastType = m_currentType = _thisType;
        m_lastCount = m_currentCount = _framecount;
        m_lastPoint = m_currentPoint = _thisPoint;
        m_AccmulateNum = 0;
        return m_AccmulateNum;

    }
    else /* 第二次开始 用前后点先后赋值再进行作差 */
    {
        m_lastCount = m_currentCount;
        m_currentCount = _framecount;

        m_lastPoint = m_currentPoint;
        m_currentPoint = _thisPoint;
        /* 这里的三种情况可以用一条或判断语句全部写完 但是为了方便看，我这里分开写*/


        /* 情况一 ：前后类型不一致 返回0*/
        if (_thisType != m_lastType)
        {
            m_AccmulateNum = 0;
            m_ObservationPoints.clear();
            m_TimestampofPoints.clear();

            return m_AccmulateNum;
        }
        /* 情况二：前后帧数差 > 1 返回0*/
        /* 轮的次数是1024 （0~1023） 因此会出现前1023 后 0 的情况*/
        if ((m_currentCount - m_lastCount > 2) && (m_currentCount - m_lastCount != -(MAXROUNDCOUNT -1)))
        {
            m_AccmulateNum = 0;
            m_ObservationPoints.clear();
            m_TimestampofPoints.clear();

            return m_AccmulateNum;
        }
        /* 情况三：前后中心点距离 > 80 返回0*/
        if (distanceOfPoints(m_currentPoint, m_lastPoint) > 80) 
        {
            m_ObservationPoints.clear();
            m_TimestampofPoints.clear();
            m_AccmulateNum = 0;
            return m_AccmulateNum;
        }

        m_AccmulateNum = m_AccmulateNum + 1;
        if (m_AccmulateNum <= 5) 
        {
            if (m_AccmulateNum == 1) {
                m_ObservationPoints.clear();
                m_TimestampofPoints.clear();
            }
            m_ObservationPoints.push_back(_thisPoint);
            m_TimestampofPoints.push_back(_timestamp);
        }
        else {
            m_ObservationPoints.erase(m_ObservationPoints.begin());
            m_ObservationPoints.push_back(_thisPoint);
            m_TimestampofPoints.erase(m_TimestampofPoints.begin());
            m_TimestampofPoints.push_back(_timestamp);

        }
        return m_AccmulateNum;
    }
}

void EKFPredict::EKFPredict_EkfSetting()
{
    KalmanFilter InitKF(4, 2, 0);


    // 定义状态转移矩阵F
    m_F = (Mat_<float>(4, 4) <<
        1  , 0  , 1  , 0  ,
        0  , 1  , 0  , 1  ,
        0  , 0  ,1.6  , 0  ,
        0  , 0  , 0.8  , 1.6);

    // 定义测量矩阵H
    m_H = Mat::eye(2, 4, CV_32F);

    // 设置过程噪声协方差矩阵和测量噪声协方差矩阵
    setIdentity(InitKF.processNoiseCov, Scalar::all(1e-1));
    setIdentity(InitKF.measurementNoiseCov, Scalar::all(1e-2));

    
    setIdentity(InitKF.transitionMatrix);
    setIdentity(InitKF.measurementMatrix);



    m_ekf = InitKF;
}

void EKFPredict::EKFPredict_EkfSetFirstState(Point2f& _firstPoint, float& _vx, float& _vy)
{

        // 定义初始状态向量
        Mat state(4, 1, CV_32F); // [x, y, vx, vy]
        state.at<float>(0) = _firstPoint.x;
        state.at<float>(1) = _firstPoint.y;
        state.at<float>(2) = _vx;
        state.at<float>(3) = _vy;

        m_ekf.statePost = state;

}

Point2f EKFPredict::EKFPredict_Predict(vector<double>& _timestampofpts, vector<Point2f>& _ptsofobervation)
{
    
    Mat measurement = Mat::zeros(2, 1, CV_32F);  // 设置测量值 [x, y]
    //cout << " -------------error start 1 ---------------" << endl;
    double dt1 = _timestampofpts[1] - _timestampofpts[0]; 
    
    // 更新状态转移矩阵
    //m_ekf.transitionMatrix.at<float>(0, 2) = dt1*5;
    //m_ekf.transitionMatrix.at<float>(1, 3) = dt1 * (1.0 / 3.0)*5;
    //cout << " -------------error start 2 ---------------" << endl;
    // 使用前7帧的数据更新滤波器的状态
    for (int i = 0; i < 5; i++)
    {
        if (i > 1)
        {
            double nextTimestamp = _timestampofpts[i] + (_timestampofpts[i] - _timestampofpts[i-1]);
            float dt = nextTimestamp - _timestampofpts[i];
            m_F.at<float>(0, 2) = dt;
            m_F.at<float>(1, 3) = dt;
        }
        measurement.at<float>(0) = _ptsofobervation[i].x;
        measurement.at<float>(1) = _ptsofobervation[i].y;

        // 预测步骤
        Mat prediction = m_F * m_ekf.statePost;
        m_ekf.statePre = prediction;
        m_ekf.errorCovPre = m_F * m_ekf.errorCovPost * m_F.t() + m_ekf.processNoiseCov;

        // 更新步骤
        Mat y = measurement - m_H * m_ekf.statePre;
        Mat S = m_H * m_ekf.errorCovPre * m_H.t() + m_ekf.measurementNoiseCov;
        Mat K = m_ekf.errorCovPre * m_H.t() * S.inv();
        m_ekf.statePost = m_ekf.statePre + K *y;
        m_ekf.errorCovPost = (Mat::eye(4, 4, CV_32F) - K * m_H) * m_ekf.errorCovPre;
        //cout << " -------------error end ---------------" << endl;
    }

    Mat prediction = m_F * m_ekf.statePost;

    // 提取预测结果
    Point2f nextPoint(prediction.at<float>(0), prediction.at<float>(1));

    m_vx01 = (_ptsofobervation[1].x- _ptsofobervation[0].x)/ (m_fps*(_timestampofpts[1] - _timestampofpts[0]));
    //m_vx01 = prediction.at<float>(2);
    m_vy01 = (_ptsofobervation[1].y- _ptsofobervation[0].y)/ (m_fps*(_timestampofpts[1] - _timestampofpts[0]));
    //m_vy01 = prediction.at<float>(3);
    //cout << "下一帧位置：" << nextPoint << endl;

    return nextPoint;
}
