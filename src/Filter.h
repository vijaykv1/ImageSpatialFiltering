#ifndef FILTER_H
#define FILTER_H

#include <opencv2/core/core.hpp>

//creation of a filter class for the sole purpose of filtering the image ... 
class Filter
{
public:
    Filter();

    ~Filter();

    void convolve_cv(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);
    void convolve_3x3(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);
    void convolve_generic(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);
    void getAbsOfSobel(const cv::Mat &input_1, const cv::Mat &input_2, cv::Mat &output);
    void scaleSobelImage(const cv::Mat &input, cv::Mat &output);
    void printMatrix (const cv::Mat &input);
    
    cv::Mat calcBinomial(uchar size);
    cv::Mat calcSobelX(uchar size);
    cv::Mat calcSobelY(uchar size);

    cv::Mat getBinomial(uchar size);
    cv::Mat getSobelX(uchar size);
    cv::Mat getSobelY(uchar size);

    cv::Mat get1x3();
    cv::Mat get3x1();

    cv::Mat get1x5(); 
    cv::Mat get5x1(); 

private:
    cv::Mat Binomial3, Binomial5;
    cv::Mat Sobel3_X, Sobel3_Y;
    cv::Mat Sobel5_X, Sobel5_Y;
    cv::Mat Exp13,Exp31;
    cv::Mat Exp51,Exp15; 
    
    int calcBinomialCoefficient(int n, int k);
};

#endif /* FILTER_H */
