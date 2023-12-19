#include<opencv2/opencv.hpp>
#include "Dnn_Deploy.h"
#include "EkfRunePredict.h"

#define DEPLOYTIME 0   ///< 用于设定打印推理时间


int main()
{
    // 类的先声明 调用后只需要在类中传参
    DnnDeploy Deploy;
    EKFPredict Predict;
    
    // 预加载
    dnn::Net DarkNet = DnnDeploy::DnnReader(configs, weights);    ///< 读取配置，加载网络
    Predict.EKFPredict_EkfSetting();                              ///< 加载卡尔曼滤波器


    // 打开视频文件
    cv::VideoCapture cap("./videos/11.mp4");
    if (!cap.isOpened())
    {
        //std::cout << "无法打开视频文件！" << std::endl;
        return -1;
    }

    // 获取视频的基本信息
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);     ///<对视频来说 fps是稳定的，实际上FPS的位置需要进行调整，因此不用fps作为速度考量标准，而是使用delay时间
    
    double timestamp = 0.0;

    // 创建视频编码器并打开输出视频文件
    cv::VideoWriter outputVideo("output_video.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 30, cv::Size(frameWidth, frameHeight));
   
    int count = -1;  ///< 时间标签count

    Point2f TarCenter;  //< 神经网络识别后得到的靶面中心点 
    int TarType;        //< 靶面标签 0蓝 1红
    Point2f TarPredict; //< 经过预测后得到的靶面中心点

    // 循环读取视频帧
    while (true)
    {
        cv::Mat showframe;
        cv::Mat frame;
        cap >> frame;

        // 检查是否到达视频末尾
        if (frame.empty())
        {
            break;
        }

        /* --------------------------  推理部分 -----------------------------*/

        double time_start = double(cv::getTickCount());

        Deploy.DnnDeployMain(DarkNet, frame);              ///< 进入推理 此时有结果在类变量中

        showframe = Deploy.m_show_image;                   ///< 取出变量->可视化图像

        double time_end = (double(cv::getTickCount()) - time_start) * 1000 / cv::getTickFrequency();

        //std::cout << "  Deployment tooks  " << time_end << "  ms." << std::endl;


        /* --------------------------  预测部分  ----------------------------*/

        // 获取视频帧计数（帧时间标签） 满1024重新一轮
        if (count < MAXROUNDCOUNT)
            count += 1;
        else
            count = 0;

        // 获取时间戳，用于计算拓展卡尔曼滤波的间隔时间来求当前速度
        timestamp += 1.0 / fps;
        Predict.m_fps = fps;

        if (Deploy.m_vdResultSrtuct.size() == 0)
            continue;
        //cout << "  this erroe   1  ------------" << endl;
        
        // 获取靶面中心点【尤其要改】 
        TarCenter = Deploy.m_vdResultSrtuct[0].center;       ///< 这里由于示例视频只有一个目标，而实际大符会有多个目标，且靶面差距不大，容易误识别，目前想到的是用红蓝像素点判断，点数量较多的则是待激活靶面，较少的是已激活
        TarType = Deploy.m_vdResultSrtuct[0].classId;        ///< 要确认目标是同一个类型才能进入预测


        cv::circle(showframe, cv::Point(TarCenter.x, TarCenter.y), 5, Scalar(222, 255, 255), 5);

        // 进卡尔曼滤波,获得预测点
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

        // 在图像上写入总延迟时间
        //double delay = cap.get(cv::CAP_PROP_POS_MSEC);
        //std::string text = "TotalDelay: " + std::to_string(delay) + "ms";
        //cv::putText(showframe, text, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX,3, cv::Scalar(0, 0, 255), 3);

        // 将帧写入输出视频文件
        outputVideo.write(showframe);

        // 显示当前帧
        //cv::imshow("Frame", showframe);

        // 按下ESC键退出循环
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // 释放资源
    cap.release();
    outputVideo.release();

    // 关闭窗口
   cv::destroyAllWindows();

    return 0;

}