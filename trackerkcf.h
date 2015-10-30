#ifndef TRACKERKCF_H
#define TRACKERKCF_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <list>
#include <cmath>

#include "colornames.h"
#include "lbp.hpp"
#include "fhog.h"

using namespace cv;
using namespace std;

enum MODE {GRAY, LBP, CN, CN2, RGB, ICF, FHOG, LBP_DIFF};

class KCFParams
{
public:
    KCFParams();
    double sigma;                 //!<  gaussian kernel bandwidth
    double lambda;                //!<  regularization
    double interp_factor;         //!<  linear interpolation factor for adaptation
    double output_sigma_factor;   //!<  spatial bandwidth (proportional to target)
    double pca_learning_rate;     //!<  compression learning rate
    bool resize;                  //!<  activate the resize feature to improve the processing speed
    bool split_coeff;             //!<  split the training coefficients into two matrices
    bool wrap_kernel;             //!<  wrap around the kernel values
    bool compress_feature;        //!<  activate the pca method to compress the features
    int max_patch_size;           //!<  threshold for the ROI size
    int compressed_size;          //!<  feature size after compression
    int binSize;
    double wPadding;
    double hPadding;
    double PSRthreshold;
    double SCCMthreshold;
    MODE descriptor;              //!<  descriptor type
};

class TrackerKCF
{
public:
    TrackerKCF(const KCFParams &parameters);
    bool init(const cv::Rect_<double>& boundingBox );
    bool update(const cv::Mat &image, cv::Rect_<double>& boundingBox, double& PSR, double &SCCM, const Mat &diffMap = Mat());
    void predict(const cv::Mat &img, cv::Rect_<double>& boundingBox, double &PSR, double &SCCM, const Mat &diffMap = Mat());
    bool train(const cv::Mat &img, const Mat &diffMap = Mat());

    void createHanningWindow(cv::OutputArray _dst, const cv::Size winSize, const int type);
    void fft2(const cv::Mat &src, cv::Mat & dest);
    void fft2(const cv::Mat &src, std::vector<cv::Mat> & dest);
    void ifft2(const cv::Mat& src, cv::Mat & dest);
    bool getSubWindow(const cv::Mat &img, const cv::Rect _roi, cv::Mat& patch, const Mat &diffMap = Mat(), double angle = 0);
    void extractCN(cv::Mat& _patch, cv::Mat& cnFeatures);
    void compress(const cv::Mat& _proj_mtx, const cv::Mat& src, cv::Mat& dest);
    void denseGaussKernel(const double sigma, const cv::Mat _x, const cv::Mat _y, cv::Mat & _k);
    void pixelWiseMult(const std::vector<cv::Mat> &src1, const std::vector<cv::Mat> &src2,
                       std::vector<cv::Mat>& dest, const int flags, const bool conjB=false);
    void sumChannels(std::vector<cv::Mat> &src, cv::Mat & dest);
    void shiftRows(cv::Mat& mat);
    void shiftRows(cv::Mat& mat, int n);
    void shiftCols(cv::Mat& mat, int n);
    void calcResponse(const cv::Mat &_alphaf, const cv::Mat &_k, cv::Mat & _response);
    void calcResponse(const cv::Mat &_alphaf, const cv::Mat &_alphaf_den, const cv::Mat &_k, cv::Mat & _response);
    double calcPSR(const cv::Mat & _response, int _col, int _row, int _mask_size);
    void updateProjectionMatrix(const cv::Mat src, cv::Mat & old_cov, cv::Mat &  _proj_mtx,
                                double pca_rate, int compressed_sz);
    cv::Rect_<double> getRoi() {
        if (resizeImage)
            return cv::Rect_<double>(roi.x * 2, roi.y * 2, roi.width * 2, roi.height * 2);
        return roi;
    }

private:

    KCFParams params;

    double output_sigma;
    cv::Rect_<double> roi;
    cv::Rect_<double> bb;
    cv::Mat hann; 	//hann window filter

    cv::Mat y,yf; 	// training response and its FFT
    cv::Mat x,xf; 	// observation and its FFT
    cv::Mat k,kf;	// dense gaussian kernel and its FFT
    cv::Mat kf_lambda; // kf+lambda
    cv::Mat new_alphaf, alphaf;	// training coefficients
    cv::Mat new_alphaf_den, alphaf_den; // for splitted training coefficients
    cv::Mat z, new_z; // model
    cv::Mat response; // detection result
    cv::Mat old_response;
    cv::Mat old_cov_mtx, proj_mtx; // for feature compression

    bool isInit;
    bool resizeImage; // resize the image whenever needed and the patch size is large

//    const float gray_range[2] = { 0, 255 };
//    const float* ranges = { gray_range };
    Mat hist;

    int frame;
};

#endif // TRACKERKCF_H
