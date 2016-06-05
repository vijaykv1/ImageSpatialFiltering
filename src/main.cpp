#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Threshold.h"
#include "Histogram.h"
#include "PointOperations.h"
#include "Filter.h"
#include "Timer.h"
#include "imshow_multiple.h"

int main(int argc, char *argv[])
{
    //read image
    cv::Mat img = cv::imread(INPUTIMAGE);

    //convert to grayscale
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, CV_BGR2GRAY);

    // convert to a float image
    cv::Mat imgGray_float;
    imgGray.convertTo(imgGray_float, CV_32F, 1.0/255.0);
    
    //declare output variables
    cv::Mat imgSmoothed3x3, imgSmoothed5x5;
    
    cv::Mat img_x_Sobel3x3, img_y_Sobel3x3, img_x_Sobel5x5, img_y_Sobel5x5;
    
    cv::Mat imgAbsSobel3x3, imgAbsSobel5x5;

    //extra filters
    cv::Mat img_exp_1x3,img_exp_3x1;

    //extra filters 
    cv::Mat img_exp_1x5,img_exp_5x1;

    //create class instances
    Threshold *threshold = new Threshold();
    Histogram *histogram = new Histogram();
    PointOperations *pointOperations = new PointOperations();
    Filter *filter = new Filter();

    //start the timer for this convolution process 

    ///////////////////////////////////////////////////////////////////////////////
    // Binomial filter with pre-defined kernels
    //////////////////////////////////////////////////////////////////////////////
    auto startBinomial3x3 = std::chrono::high_resolution_clock::now(); 
    filter->convolve_3x3(imgGray_float, imgSmoothed3x3, filter->getBinomial(3));
    auto timeTakenBinomial3x3 = (startBinomial3x3 - std::chrono::high_resolution_clock::now()); 

    auto startBinomial5x5 = std::chrono::high_resolution_clock::now(); 
    filter->convolve_generic(imgGray_float, imgSmoothed5x5, filter->getBinomial(5));
    auto timeTakenBinomial5x5 = (startBinomial5x5 - std::chrono::high_resolution_clock::now()); 

    imshow_multiple("Gaussian Filter", 3, &imgGray_float, &imgSmoothed3x3, &imgSmoothed5x5);
    
    //////////////////////////////////////////////////////////////////////////////
    // x-Sobel filter with pre-defined kernels
    //////////////////////////////////////////////////////////////////////////////
    auto startSobelx_3x3 = std::chrono::high_resolution_clock::now(); 
    filter->convolve_3x3(imgGray_float, img_x_Sobel3x3, filter->getSobelX(3));
    auto timeTakenSobelx_3x3 = (startSobelx_3x3 - std::chrono::high_resolution_clock::now()); 

    auto startSobelx_5x5 = std::chrono::high_resolution_clock::now(); 
    filter->convolve_generic(imgGray_float, img_x_Sobel5x5, filter->getSobelX(5));
    auto timeTakenSobelx_5x5 = (startSobelx_5x5 - std::chrono::high_resolution_clock::now()); 

    imshow_multiple("Sobel Filter X", 3, &imgGray_float, &img_x_Sobel3x3, &img_x_Sobel5x5);

    // new image value scale -> so we can see more
    filter->scaleSobelImage(img_x_Sobel3x3, img_x_Sobel3x3);
    filter->scaleSobelImage(img_x_Sobel5x5, img_x_Sobel5x5);
    imshow_multiple("Sobel Filter X", 3, &imgGray_float, &img_x_Sobel3x3, &img_x_Sobel5x5);

    
    //////////////////////////////////////////////////////////////////////////////
    // y - Sobel filter with pre-defined kernels
    //////////////////////////////////////////////////////////////////////////////
    auto startSobely_3x3 = std::chrono::high_resolution_clock::now(); 
    filter->convolve_3x3(imgGray_float, img_y_Sobel3x3, filter->getSobelY(3));
    auto timeTakenSobely_3x3 = (startSobely_3x3 - std::chrono::high_resolution_clock::now()); 

    auto startSobely_5x5 = std::chrono::high_resolution_clock::now(); 
    filter->convolve_generic(imgGray_float, img_y_Sobel5x5, filter->getSobelY(5));
    auto timeTakenSobely_5x5 = (startSobely_5x5 - std::chrono::high_resolution_clock::now()); 

    // new image value scale -> so we can see more
    filter->scaleSobelImage(img_y_Sobel3x3, img_y_Sobel3x3);
    filter->scaleSobelImage(img_y_Sobel5x5, img_y_Sobel5x5);
    imshow_multiple("Sobel Filter Y", 3, &imgGray_float, &img_y_Sobel3x3, &img_y_Sobel5x5);

    
    //////////////////////////////////////////////////////////////////////////////
    // calculate the abs() of the Sobel images
    //////////////////////////////////////////////////////////////////////////////
    filter->getAbsOfSobel(img_x_Sobel5x5, img_y_Sobel5x5, imgAbsSobel3x3);
    filter->getAbsOfSobel(img_x_Sobel5x5, img_y_Sobel5x5, imgAbsSobel5x5);
    imshow_multiple("Abs() Sobel", 2, &imgAbsSobel3x3, &imgAbsSobel5x5);

    
    /////////////////////////////////////////////////////////////////////////////
    // Other sets of Kernel designs
    /////////////////////////////////////////////////////////////////////////////

    //1 x 3 and 3 x 1 filter types
    filter->convolve_generic(imgGray_float,img_exp_1x3,filter->get1x3()); 
    //filter->scaleSobelImage(img_exp_1x3,img_exp_1x3);
    filter->convolve_generic(imgGray_float,img_exp_3x1,filter->get3x1());  
    //filter->scaleSobelImage(img_exp_3x1,img_exp_3x1);
    imshow_multiple("Any Filter Type 1x3 and 3x1",2,&img_exp_1x3,&img_exp_3x1); 

    //1 x 5 and 5 x 1 filter types
    filter->convolve_generic(imgGray_float,img_exp_1x5,filter->get1x5());
    filter->convolve_generic(imgGray_float,img_exp_5x1,filter->get5x1());  
    imshow_multiple("Any Filter Type 1x5 and 5x1",2,&img_exp_1x5,&img_exp_5x1); 

    //////////////////////////////////////////////////////////////////////////////
    //doing calculations for the different sets of Kernel
    //////////////////////////////////////////////////////////////////////////////

    //Binomial Filter calculations
    auto diff_binomial_3x3_5x5 = timeTakenSobelx_5x5 - timeTakenBinomial3x3; 

    // Sobel X Filter calculations 
    auto diff_Sobel_x_3x3_5x5 = timeTakenSobelx_5x5 - timeTakenSobelx_3x3; 

    // Sobel Y Filter calculations
    auto diff_Sobel_y_3x3_5x5 = timeTakenSobely_5x5 - timeTakenSobely_3x3; 

    //Display the differences on terminal
    std::cout << "########################################### " << std::endl; 
    std::cout << "difference between Binomial 3x3 and 5x5 :" << std::abs(std::chrono::duration_cast<std::chrono::nanoseconds> (diff_binomial_3x3_5x5).count()) << std::endl ;
    std::cout << "difference between Sobel Y 3x3 and 5x5 : " << std::abs(std::chrono::duration_cast<std::chrono::nanoseconds> (diff_Sobel_y_3x3_5x5).count()) << std::endl ;
    std::cout << "difference between Sobel X 3x3 and 5x5 : " << std::abs(std::chrono::duration_cast<std::chrono::nanoseconds> (diff_Sobel_x_3x3_5x5).count()) << std::endl ; 
    std::cout << "########################################### " << std::endl;

    // end processing /////////////////////////////////////////////////////////////

    //wait for key pressed
    cv::waitKey();

    return 0;
}
