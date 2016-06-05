#include <iostream>
#include <math.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Filter.h"

////////////////////////////////////////////////////////////////////////////////////
// constructor. Initialize the kernels
////////////////////////////////////////////////////////////////////////////////////
Filter::Filter()
{
    // Initialize Binomial kernels
    // 3x3
    char kernelB3[3 * 3] =  {1, 2, 1,
                             2, 4, 2,
                             1, 2, 1};

    Binomial3 = cv::Mat(3, 3, CV_8S, kernelB3).clone(); // creating a clone of the said array ... 

    // 5x5
    char kernelB5[5 * 5] =  {1,  4,  6,  4, 1,
                             4, 16, 24, 16, 4,
                             6, 24, 36, 24, 6,
                             4, 16, 24, 16, 4,
                             1,  4,  6,  4, 1};

    Binomial5 = cv::Mat(5, 5, CV_8S, kernelB5).clone(); // creating a clone of the above generated array ... so we can manipulate this for our operations 

    // initialize Sobel kernels
    // 3x3 in X direction
    char kernelS1[3 * 3] =  {-1, 0, 1,
                             -2, 0, 2,
                             -1, 0, 1};

    Sobel3_X = cv::Mat(3, 3, CV_8S, kernelS1).clone(); 

    // 3x3 in Y direction
    char kernelS2[3 * 3] =  {-1, -2, -1,
                              0,  0,  0,
                              1,  2,  1};

    Sobel3_Y = cv::Mat(3, 3, CV_8S, kernelS2).clone();

    // 5x5 in X direction
    char kernelS3[5 * 5] =  { -2, -1, 0, 1,  2,
                              -8, -4, 0, 4,  8,
                             -12, -6, 0, 6, 12,
                              -8, -4, 0, 4,  8,
                              -2, -1, 0, 1,  2};

    Sobel5_X = cv::Mat(5, 5, CV_8S, kernelS3).clone();

    // 5x5 in Y direction
    char kernelS4[5 * 5] =  {-2, -8, -12, -8, -2,
                             -1, -4,  -6, -4, -1,
                              0,  0,   0,  0,  0,
                              1,  4,   6,  4,  1,
                              2,  8,  12,  8,  2};

    Sobel5_Y = cv::Mat(5, 5, CV_8S, kernelS4).clone();

    char kernelA1[1 * 3] = { 1,
                             2,
                             4 };

    Exp13 = cv::Mat(1,3,CV_8S,kernelA1).clone(); 

    char kernelA2 [1 * 5] = {1,
                             3,
                             5,
                             7,
                             9 }; 

    Exp15 = cv::Mat(1,5,CV_8S,kernelA2).clone(); 

    char kernelA3 [3 * 1] = {1,2,3}; 

    Exp31 = cv::Mat(3,1,CV_8S,kernelA3).clone(); 

    char kernelA4 [5 * 1] = {1,3,5,7,9}; 

    Exp51 = cv::Mat(5,1,CV_8S,kernelA4).clone();
}

Filter::~Filter(){}

////////////////////////////////////////////////////////////////////////////////////
// convolve the image with the kernel using the OpenCV function - only for reference
////////////////////////////////////////////////////////////////////////////////////
void Filter::convolve_cv(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
{
    int normFactor = 0;
    
    int rows = kernel.rows;
    int cols = kernel.cols;
    
    if (kernel.isContinuous())
    {
        cols = rows * cols;
        rows = 1;
    }
    
    // calculate the normalisation factor from the filter kernel
    for (int r = 0; r < rows; ++r)
    {
        const char *pKernel = kernel.ptr<char>(r);
        
        for (int c = 0; c < cols; ++c)
        {
            normFactor += abs(*pKernel); // add the absolute value of the pKernel values ... 
            ++pKernel;
        }
    }
    
    if (input.empty() || kernel.empty())
    {  
        std::cout << "One ore more inputs are empty!" << std::endl;
        return;
    }

    cv::Mat floatInput, floatKernel;
    input.convertTo(floatInput, CV_32F);
    kernel.convertTo(floatKernel, CV_32F);

    cv::filter2D(floatInput, output, floatInput.depth(), floatKernel, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);

    output /= (normFactor);
}

///////////////////////////////////////////////////////////////////////////////
// convolve the image with the square-sized 3x3 kernel using pointer access -- > FIXME 
///////////////////////////////////////////////////////////////////////////////
void Filter::convolve_3x3(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
{
    if (input.empty() || kernel.empty())
    {
        std::cout << "One ore more inputs are empty!" << std::endl;
        return;
    }

    // for the image matrix 
    int rows = input.rows;
    int cols = input.cols;

    // for the kernel matrix
    int kern_rows = kernel.rows; 
    int kern_cols = kernel.cols; 

    // declare a normalisation variable ... 
    int normFactor = 0 ;

    output.release(); // why do we need this ? Not clear ... clarify in the next lab please ... 
    
    // create a float image initialized withe zeros
    output = cv::Mat::zeros(rows, cols, CV_32F);


    //Normalisation
    for (int  r = 0 ; r < kern_rows ; ++r )
    {
        for (int c = 0 ; c < kern_cols ; ++c )
        {
            normFactor += abs((int)kernel.at<char>(r,c));
        }
    }

    //Convolution
    for (int r = 1; r < (rows -1) ; ++r)
    {
        for(int c = 1 ; c < (cols-1) ; ++c )
        {
            for (int i = 0 ; i<kern_rows  ; ++i )
            {
                const char *pKernel = kernel.ptr<char>(i);
                for (int j = 0 ; j<kern_cols ; ++j)
                {
                    output.at<float>(r,c) += (*pKernel * input.at<float>(r-i,c-j))/normFactor;
                    ++pKernel;
                } 
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
// Matrix printer -- will print any matrix necessary
//////////////////////////////////////////////////////////////////////////////
void Filter::printMatrix (const cv::Mat &input)
{
    int rows = input.rows;
    int cols = input.cols; 

    std::cout << " ++++ Matrix used: +++++ " << std::endl;
    for (int i = 0 ; i < rows ; ++i)
    {
        const char *pInput = input.ptr<char>(i);
        for (int j = 0 ; j < cols ; ++j )
        {
            std::cout << (int)*pInput << " "; 
            ++pInput;            
        }
        std::cout << std::endl; 
    }  
    std::cout << " ++++ Kernel used: +++++ " << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
// convolve the image with any filter kernel using pointer access --> FIXME
///////////////////////////////////////////////////////////////////////////////
void Filter::convolve_generic(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
{   
    if (input.empty() || kernel.empty())
    {
        std::cout << "One ore more inputs are empty!" << std::endl;
        return;
    }
    
    //for the image
    int rows = input.rows;
    int cols = input.cols;
    
    // for the kernel matrix
    int kern_rows = kernel.rows;
    int kern_cols = kernel.cols;
    
    //Normalisation Constant
    int normFactor = 0 ;
    
    output.release();
    // create a float image initialized withe zeros
    output = cv::Mat::zeros(rows, cols, CV_32F);
    
    //Normalisation
    for ( int  r = 0 ; r < kern_rows ; ++r )
    {
        for ( int c = 0 ; c < kern_cols ; ++c )
        {
            normFactor += abs((int)kernel.at<char>(r,c));
        }
    }
    
    //Convolution
    for ( int r = floor(kern_rows/2); r < (rows -(floor(kern_rows/2))) ; ++r)
    {
        for( int c = floor(kern_cols/2) ; c < (cols-floor(kern_cols/2)) ; ++c )
        {
            for ( int i = 0 ; i<kern_rows  ; ++i )
            {
                const char *pKernel = kernel.ptr<char>(i);
                

                for ( int j = 0 ; j<kern_cols ; ++j)
                {
                    output.at<float>(r,c) += (*pKernel * input.at<float>(r-i,c-j))/normFactor;
                    ++pKernel;
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// calculate the abs() of the x-Sobel and y-Sobel image
///////////////////////////////////////////////////////////////////////////////
void Filter::getAbsOfSobel(const cv::Mat &input_1, const cv::Mat &input_2, cv::Mat &output)
{
    if (input_1.empty() || input_2.empty())
    {
        std::cout << "One ore more inputs are empty!" << std::endl;
        return;
    }

    int rows = input_1.rows;
    int cols = input_1.cols;

    output.release();
    // create a float image initialized withe zeros
    output = cv::Mat::zeros(rows, cols, CV_32F);

    for (int r = 0 ; r < rows ; ++r)
    {
        const char *pRows_1 = input_1.ptr<char>(r);
        const char *pRows_2 = input_2.ptr<char>(r);

        for (int c = 0 ; c < cols ; ++c )
        {
            output.at<float>(r,c) = sqrt (pow(input_1.at<float>(r,c),2)+ pow(input_2.at<float>(r,c),2));  
            ++pRows_1;
            ++pRows_2;
        }
    }    
}

///////////////////////////////////////////////////////////////////////////////
// return the Binomial kernel
///////////////////////////////////////////////////////////////////////////////
cv::Mat Filter::getBinomial(uchar size)
{
    if (size == 3)
        return Binomial3;
    else if (size == 5)
        return Binomial5;
    else
        return cv::Mat();
}

///////////////////////////////////////////////////////////////////////////////
// return the Sobel kernel in X direction
///////////////////////////////////////////////////////////////////////////////
cv::Mat Filter::getSobelX(uchar size)
{
    if (size == 3)
        return Sobel3_X;
    else if (size == 5)
        return Sobel5_X;
    else
        return cv::Mat();
}

///////////////////////////////////////////////////////////////////////////////
// return the Sobel kernel in Y direction
///////////////////////////////////////////////////////////////////////////////
cv::Mat Filter::getSobelY(uchar size)
{
    if (size == 3)
        return Sobel3_Y;
    else if (size == 5)
        return Sobel5_Y;
    else
        return cv::Mat();
}

///////////////////////////////////////////////////////////////////////////////
// return the kernel for 3*1 and 1*3
///////////////////////////////////////////////////////////////////////////////
cv::Mat Filter::get1x3()
{
    return Exp13; 
}

cv::Mat Filter::get3x1()
{
    return Exp31; 
}

///////////////////////////////////////////////////////////////////////////////
// return the kernel for 5*1 and 1*5
///////////////////////////////////////////////////////////////////////////////
cv::Mat Filter::get5x1()
{
    return Exp51; 
}

cv::Mat Filter::get1x5()
{
    return Exp15; 
}

///////////////////////////////////////////////////////////////////////////////
// convert an image to CV_8U; 0 corresponds to 127; brightest pixel to 255
///////////////////////////////////////////////////////////////////////////////
void Filter::scaleSobelImage(const cv::Mat &input, cv::Mat &output)
{
    // find max value
    double min, max;
    cv::minMaxLoc(input, &min, &max);

    // scale the image
    input.convertTo(output, CV_32F, (0.5f / max), 0.5f);
}