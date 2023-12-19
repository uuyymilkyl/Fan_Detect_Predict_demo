#include<opencv2/opencv.hpp>
#include "Dnn_Deploy.h"
#include "EkfRunePredict.h"

#define DEPLOYTIME 0   ///< �����趨��ӡ����ʱ��


int main()
{
    // ��������� ���ú�ֻ��Ҫ�����д���
    DnnDeploy Deploy;
    EKFPredict Predict;
    
    // Ԥ����
    dnn::Net DarkNet = DnnDeploy::DnnReader(configs, weights);    ///< ��ȡ���ã���������
    Predict.EKFPredict_EkfSetting();                              ///< ���ؿ������˲���


    // ����Ƶ�ļ�
    cv::VideoCapture cap("./videos/11.mp4");
    if (!cap.isOpened())
    {
        //std::cout << "�޷�����Ƶ�ļ���" << std::endl;
        return -1;
    }

    // ��ȡ��Ƶ�Ļ�����Ϣ
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);     ///<����Ƶ��˵ fps���ȶ��ģ�ʵ����FPS��λ����Ҫ���е�������˲���fps��Ϊ�ٶȿ�����׼������ʹ��delayʱ��
    
    double timestamp = 0.0;

    // ������Ƶ���������������Ƶ�ļ�
    cv::VideoWriter outputVideo("output_video.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 30, cv::Size(frameWidth, frameHeight));
   
    int count = -1;  ///< ʱ���ǩcount

    Point2f TarCenter;  //< ������ʶ���õ��İ������ĵ� 
    int TarType;        //< �����ǩ 0�� 1��
    Point2f TarPredict; //< ����Ԥ���õ��İ������ĵ�

    // ѭ����ȡ��Ƶ֡
    while (true)
    {
        cv::Mat showframe;
        cv::Mat frame;
        cap >> frame;

        // ����Ƿ񵽴���Ƶĩβ
        if (frame.empty())
        {
            break;
        }

        /* --------------------------  ������ -----------------------------*/

        double time_start = double(cv::getTickCount());

        Deploy.DnnDeployMain(DarkNet, frame);              ///< �������� ��ʱ�н�����������

        showframe = Deploy.m_show_image;                   ///< ȡ������->���ӻ�ͼ��

        double time_end = (double(cv::getTickCount()) - time_start) * 1000 / cv::getTickFrequency();

        //std::cout << "  Deployment tooks  " << time_end << "  ms." << std::endl;


        /* --------------------------  Ԥ�ⲿ��  ----------------------------*/

        // ��ȡ��Ƶ֡������֡ʱ���ǩ�� ��1024����һ��
        if (count < MAXROUNDCOUNT)
            count += 1;
        else
            count = 0;

        // ��ȡʱ��������ڼ�����չ�������˲��ļ��ʱ������ǰ�ٶ�
        timestamp += 1.0 / fps;
        Predict.m_fps = fps;

        if (Deploy.m_vdResultSrtuct.size() == 0)
            continue;
        //cout << "  this erroe   1  ------------" << endl;
        
        // ��ȡ�������ĵ㡾����Ҫ�ġ� 
        TarCenter = Deploy.m_vdResultSrtuct[0].center;       ///< ��������ʾ����Ƶֻ��һ��Ŀ�꣬��ʵ�ʴ�����ж��Ŀ�꣬�Ұ����಻��������ʶ��Ŀǰ�뵽�����ú������ص��жϣ��������϶�����Ǵ�������棬���ٵ����Ѽ���
        TarType = Deploy.m_vdResultSrtuct[0].classId;        ///< Ҫȷ��Ŀ����ͬһ�����Ͳ��ܽ���Ԥ��


        cv::circle(showframe, cv::Point(TarCenter.x, TarCenter.y), 5, Scalar(222, 255, 255), 5);

        // ���������˲�,���Ԥ���
        TarPredict = Predict.EKFPredict_Main(timestamp,count,TarCenter,TarType);

        if (TarCenter.x != 0) 
        {
            if (TarPredict.x < 14)
                TarPredict.x = 14;
            if (TarPredict.y < 14)
                TarPredict.y = 14;
            if ((TarPredict.x > frame.rows-15))
                TarPredict.x = frame.rows-14;
            if ((TarPredict.y > frame.cols-15))
                TarPredict.y = frame.cols-14;
            cv::circle(showframe, cv::Point(TarPredict.x, TarPredict.y), 3, Scalar(0, 0, 255), 5);
            //cout << " " << endl;
        }

        // ��ͼ����д�����ӳ�ʱ��
        //double delay = cap.get(cv::CAP_PROP_POS_MSEC);
        //std::string text = "TotalDelay: " + std::to_string(delay) + "ms";
        //cv::putText(showframe, text, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX,3, cv::Scalar(0, 0, 255), 3);

        // ��֡д�������Ƶ�ļ�
        outputVideo.write(showframe);

        // ��ʾ��ǰ֡
        //cv::imshow("Frame", showframe);

        // ����ESC���˳�ѭ��
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // �ͷ���Դ
    cap.release();
    outputVideo.release();

    // �رմ���
   cv::destroyAllWindows();

    return 0;

}