#include "EkfRunePredict.h"


/*  ע�� ��_xxx ��ʾ��������ı���   ��m_xxx ��ʾ��Ĺ��б�������ע�����*/
EKFPredict::EKFPredict()
{
}

EKFPredict::~EKFPredict()
{
}

Point2f EKFPredict::EKFPredict_Main(double& _timestamp, int& _framecount, Point2f& _thisPoint, int&_thisType)
{
    /* ����ۼ���7�������ĵ㣬�򽫵㼯��ʱ������Ž��������˲���  */
    int AccmulateNum = EKFPredict_Accumulate(_framecount, _thisPoint,_thisType,_timestamp);
    Point2f PreDictPoint;
    if (AccmulateNum == 7)
    {
        float vx = 0;
        float vy = 0;
        vector < Point2f > PredicePoints = m_ObservationPoints;
        vector <double > PredictTimes = m_TimestampofPoints;
        
        EKFPredict_EkfSetting(); ///< �״���7�����������˲���
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
   
    //������ �򷵻�ԭ��
    return _thisPoint;
    
}

int EKFPredict::EKFPredict_Accumulate(int& _framecount, Point2f& _thisPoint, int& _thisType, double& _timestamp)
{

    /* ��һ�μ��� ��ʼ��ǰ�������ǰ��� */
    if (_framecount == 0)
    {
        m_lastType = m_currentType = _thisType;
        m_lastCount = m_currentCount = _framecount;
        m_lastPoint = m_currentPoint = _thisPoint;
        m_AccmulateNum = 0;
        return m_AccmulateNum;

    }
    else /* �ڶ��ο�ʼ ��ǰ����Ⱥ�ֵ�ٽ������� */
    {
        m_lastCount = m_currentCount;
        m_currentCount = _framecount;

        m_lastPoint = m_currentPoint;
        m_currentPoint = _thisPoint;
        /* ������������������һ�����ж����ȫ��д�� ����Ϊ�˷��㿴��������ֿ�д*/


        /* ���һ ��ǰ�����Ͳ�һ�� ����0*/
        if (_thisType != m_lastType)
        {
            m_AccmulateNum = 0;
            m_ObservationPoints.clear();
            m_TimestampofPoints.clear();

            return m_AccmulateNum;
        }
        /* �������ǰ��֡���� > 1 ����0*/
        /* �ֵĴ�����1024 ��0~1023�� ��˻����ǰ1023 �� 0 �����*/
        if ((m_currentCount - m_lastCount > 2) && (m_currentCount - m_lastCount != -(MAXROUNDCOUNT -1)))
        {
            m_AccmulateNum = 0;
            m_ObservationPoints.clear();
            m_TimestampofPoints.clear();

            return m_AccmulateNum;
        }
        /* �������ǰ�����ĵ���� > 80 ����0*/
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


    // ����״̬ת�ƾ���F
    m_F = (Mat_<float>(4, 4) <<
        1  , 0  , 1  , 0  ,
        0  , 1  , 0  , 1  ,
        0  , 0  ,1.6  , 0  ,
        0  , 0  , 0.8  , 1.6);

    // �����������H
    m_H = Mat::eye(2, 4, CV_32F);

    // ���ù�������Э�������Ͳ�������Э�������
    setIdentity(InitKF.processNoiseCov, Scalar::all(1e-1));
    setIdentity(InitKF.measurementNoiseCov, Scalar::all(1e-2));

    
    setIdentity(InitKF.transitionMatrix);
    setIdentity(InitKF.measurementMatrix);



    m_ekf = InitKF;
}

void EKFPredict::EKFPredict_EkfSetFirstState(Point2f& _firstPoint, float& _vx, float& _vy)
{

        // �����ʼ״̬����
        Mat state(4, 1, CV_32F); // [x, y, vx, vy]
        state.at<float>(0) = _firstPoint.x;
        state.at<float>(1) = _firstPoint.y;
        state.at<float>(2) = _vx;
        state.at<float>(3) = _vy;

        m_ekf.statePost = state;

}

Point2f EKFPredict::EKFPredict_Predict(vector<double>& _timestampofpts, vector<Point2f>& _ptsofobervation)
{
    
    Mat measurement = Mat::zeros(2, 1, CV_32F);  // ���ò���ֵ [x, y]
    //cout << " -------------error start 1 ---------------" << endl;
    double dt1 = _timestampofpts[1] - _timestampofpts[0]; 
    
    // ����״̬ת�ƾ���
    //m_ekf.transitionMatrix.at<float>(0, 2) = dt1*5;
    //m_ekf.transitionMatrix.at<float>(1, 3) = dt1 * (1.0 / 3.0)*5;
    //cout << " -------------error start 2 ---------------" << endl;
    // ʹ��ǰ7֡�����ݸ����˲�����״̬
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

        // Ԥ�ⲽ��
        Mat prediction = m_F * m_ekf.statePost;
        m_ekf.statePre = prediction;
        m_ekf.errorCovPre = m_F * m_ekf.errorCovPost * m_F.t() + m_ekf.processNoiseCov;

        // ���²���
        Mat y = measurement - m_H * m_ekf.statePre;
        Mat S = m_H * m_ekf.errorCovPre * m_H.t() + m_ekf.measurementNoiseCov;
        Mat K = m_ekf.errorCovPre * m_H.t() * S.inv();
        m_ekf.statePost = m_ekf.statePre + K *y;
        m_ekf.errorCovPost = (Mat::eye(4, 4, CV_32F) - K * m_H) * m_ekf.errorCovPre;
        //cout << " -------------error end ---------------" << endl;
    }

    Mat prediction = m_F * m_ekf.statePost;

    // ��ȡԤ����
    Point2f nextPoint(prediction.at<float>(0), prediction.at<float>(1));

    m_vx01 = (_ptsofobervation[1].x- _ptsofobervation[0].x)/ (m_fps*(_timestampofpts[1] - _timestampofpts[0]));
    //m_vx01 = prediction.at<float>(2);
    m_vy01 = (_ptsofobervation[1].y- _ptsofobervation[0].y)/ (m_fps*(_timestampofpts[1] - _timestampofpts[0]));
    //m_vy01 = prediction.at<float>(3);
    //cout << "��һ֡λ�ã�" << nextPoint << endl;

    return nextPoint;
}
