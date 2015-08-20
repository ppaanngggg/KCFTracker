#include "trackerkcf.h"

double CalculateAvg(const std::list<double> &list)
{
    double avg = 0;
    std::list<double>::const_iterator it;
    for(it = list.begin(); it != list.end(); it++) avg += *it;
    avg /= list.size();
//    cout<<avg<<endl;
    return avg;
}

void rotate(cv::Mat& src, double angle, cv::Mat& dst)
{
    cv::Point2f pt(src.cols/2., src.rows/2.);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);

    cv::warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
}

void computeChannels(InputArray image, vector<Mat>& channels)
{
    Mat_<float> grad;
    Mat_<float> angles;
    Mat luv, gray, src;

    if(image.getMat().channels() > 1)
    {
      src = Mat(image.getMat().rows, image.getMat().cols, CV_32FC3);
      image.getMat().convertTo(src, CV_32FC3, 1./255);

      cvtColor(src, gray, CV_RGB2GRAY);
      cvtColor(src, luv, CV_RGB2Luv);
    }
    else
    {
      src = Mat(image.getMat().rows, image.getMat().cols, CV_32FC1);
      image.getMat().convertTo(src, CV_32FC1, 1./255);
      src.copyTo(gray);
    }

    Mat_<float> row_der, col_der;
    Sobel(gray, row_der, CV_32F, 0, 1);
    Sobel(gray, col_der, CV_32F, 1, 0);

    cartToPolar(col_der, row_der, grad, angles, true);
    //magnitude(row_der, col_der, grad);

    Mat_<Vec6f> hist = Mat_<Vec6f>::zeros(grad.rows, grad.cols);
    //const float to_deg = 180 / 3.1415926f;
    for (int row = 0; row < grad.rows; ++row) {
        for (int col = 0; col < grad.cols; ++col) {
            //float angle = atan2(row_der(row, col), col_der(row, col)) * to_deg;
            float angle = angles(row, col);
            if (angle > 180)
                angle -= 180;
            if (angle < 0)
                angle += 180;
            int ind = (int)(angle / 30);

            // If angle == 180, prevent index overflow
            if (ind == 6)
                ind = 5;

            hist(row, col)[ind] = grad(row, col) * 255;
        }
    }

    channels.clear();

    if(image.getMat().channels() > 1)
    {
      Mat luv_channels[3];
      split(luv, luv_channels);
      for( int i = 0; i < 3; ++i )
          channels.push_back(luv_channels[i]);
    }

    channels.push_back(grad);

    vector<Mat> hist_channels;
    split(hist, hist_channels);

    for( size_t i = 0; i < hist_channels.size(); ++i )
        channels.push_back(hist_channels[i]);
}

TrackerKCF::TrackerKCF(const KCFParams &parameters) :
    params(parameters)
{
    isInit = false;
    resizeImage = false;
}

bool TrackerKCF::init(const cv::Rect_<double> &boundingBox)
{
    frame = 0;
    bb = roi = boundingBox;

    //calclulate output sigma
    output_sigma=std::sqrt(roi.width*roi.height)*params.output_sigma_factor;
    if (params.descriptor == FHOG)
        output_sigma /= params.binSize;
    output_sigma=-0.5/(output_sigma*output_sigma);

    //resize the ROI whenever needed
    if(params.resize && roi.width*roi.height>params.max_patch_size){
        resizeImage=true;
        roi.x/=2.0;
        roi.y/=2.0;
        roi.width/=2.0;
        roi.height/=2.0;
    }

    // add padding to the roi
    roi.x-=roi.width * params.wPadding / 2.;
    roi.y-=roi.height * params.hPadding / 2.;
    roi.width*=(1 + params.wPadding);
    roi.height*=(1 + params.hPadding);

    Size target_sz(roi.width, roi.height);
    if (params.descriptor == FHOG) {
        target_sz.width /= params.binSize;
        target_sz.height /= params.binSize;
    }
//    exit(0)
//     initialize the hann window filter
    createHanningWindow(hann, target_sz, CV_64F);
//    cv::imshow("hann", hann);
//    cv::waitKey();
    if(params.descriptor==CN || params.descriptor==ICF){
        cv::Mat layers[] = {hann, hann, hann, hann, hann, hann, hann, hann, hann, hann};
        cv::merge(layers, 10, hann);
    } else if(params.descriptor==CN2 || params.descriptor==RGB){
        cv::Mat layers[] = {hann, hann, hann};
        cv::merge(layers, 3, hann);
    } else if(params.descriptor==FHOG){
        cv::Mat layers[32];
        for (int i=0; i < 32; i++)
            layers[i] = hann;
        cv::merge(layers, 32, hann);
    } else if(params.descriptor==LBP_DIFF){
        cv::Mat layers[] = {hann, hann};
        cv::merge(layers, 2, hann);
    }

    // create gaussian response
    y = cv::Mat::zeros(target_sz.height,target_sz.width,CV_64F);
    for(int i=0;i<target_sz.height;i++){
        for(int j=0;j<target_sz.width;j++){
            y.at<double>(i,j) = (i-target_sz.height/2+1) * (i-target_sz.height/2+1) +
                    (j-target_sz.width/2+1) * (j-target_sz.width/2+1);
        }
    }

    y*=(double)output_sigma;
    cv::exp(y,y);
//    cv::imshow("y",y);
//    cv::waitKey();

    // perform fourier transfor to the gaussian response
    fft2(y,yf);

    return true;
}

bool TrackerKCF::update(const cv::Mat &image, cv::Rect_<double> &boundingBox, double& PSR, double &SCCM, const Mat &diffMap)
{
    cv::Mat img=image.clone();
    cv::Mat diffMap_tmp;
    if (!diffMap.empty())
        diffMap_tmp = diffMap.clone();
    // resize the image whenever needed
    if (resizeImage) {
        cv::resize(img, img, cv::Size(img.cols/2,img.rows/2));
        if (!diffMap.empty())
            cv::resize(diffMap_tmp, diffMap_tmp, cv::Size(img.cols/2, img.rows/2));
    }
    //do detect
    if (frame > 0){
        predict(img, boundingBox ,PSR, SCCM, diffMap_tmp);
        bb = boundingBox;
        if (PSR < params.PSRthreshold)
            return false;
    }

    bool ret = train(img, diffMap_tmp);
    frame++;

    return ret;
}

void TrackerKCF::predict(const Mat &img, cv::Rect_<double> &boundingBox, double &PSR, double &SCCM, const Mat &diffMap)
{
    PSR = -1;
   // const double scaleList[] = {0.95, 1, 1.05};
    const double scaleList[] = {1};
    double PSRList [sizeof(scaleList)/sizeof(double)];
    double maxResList [sizeof(scaleList)/sizeof(double)];
    Mat responseList [sizeof(scaleList)/sizeof(double)];
    Point maxLocList [sizeof(scaleList)/sizeof(double)];

    double minVal, maxVal;	// min-max response
    cv::Point minLoc,maxLoc;	// min-max location
    cv::Mat zc;

    cv::Point2d center(roi.x+roi.width/2., roi.y+roi.height/2.);
    cv::Rect_<double> tmpRoi;
    for (unsigned int i=0; i<sizeof(scaleList)/sizeof(double); i++){
        tmpRoi.x = center.x - roi.width/2. * scaleList[i];
        tmpRoi.y = center.y - roi.height/2. * scaleList[i];
        tmpRoi.width = roi.width * scaleList[i];
        tmpRoi.height = roi.height * scaleList[i];
        // extract and pre-process the patch
        if(!getSubWindow(img, tmpRoi, x, diffMap))
            return;
        //compute the gaussian kernel
        if(params.compress_feature){
            compress(proj_mtx,x,x);
            compress(proj_mtx,z,zc);
            denseGaussKernel(params.sigma,x,zc,k);
        }else
            denseGaussKernel(params.sigma,x,z,k);

        // calculate filter response
        if(params.split_coeff)
            calcResponse(alphaf,alphaf_den,k,responseList[i]);
        else
            calcResponse(alphaf,k,responseList[i]);

        // extract the maximum response
        cv::minMaxLoc( responseList[i], &minVal, &maxVal, &minLoc, &maxLoc );

        maxResList[i] = maxVal;
        PSRList[i] = calcPSR( responseList[i], maxLoc.x, maxLoc.y, 1);
        maxLocList[i] = maxLoc;
    }
    // find the max PSR scale
    double maxValue = PSRList[0];
    int maxIndex = 0;
    for(unsigned int i=1;i<sizeof(scaleList)/sizeof(double);i++){
        if (PSRList[i] > maxValue) {
            maxValue = PSRList[i];
            maxIndex = i;
        }
    }

    response  = responseList[maxIndex];

    //update thr roi
    double shift_x = (maxLocList[maxIndex].x-response.cols/2+1)*scaleList[maxIndex];
    double shift_y = (maxLocList[maxIndex].y-response.rows/2+1)*scaleList[maxIndex];

    shiftCols(response, shift_x);
    shiftRows(response, shift_y);
    if (!old_response.empty()) {
        SCCM = cv::norm(old_response, response, cv::NORM_RELATIVE | cv::NORM_L2);
//        cout<<"diff :"<<diff<<endl;
//        cout<<"SCCM : "<<SCCM<<endl;
    } else {
        SCCM = 0;
    }
    old_response = response;

    if (params.descriptor==FHOG) {
        shift_x*=params.binSize;
        shift_y*=params.binSize;
    }

    roi.x += shift_x;
    roi.y += shift_y;
    roi.width *= scaleList[maxIndex];
    roi.height *= scaleList[maxIndex];

    // update the bounding box
    boundingBox.x = roi.x + roi.width / (1+params.wPadding) * params.wPadding / 2.;
    boundingBox.y = roi.y + roi.height / (1+params.hPadding) * params.hPadding / 2.;
    boundingBox.width = roi.width / (1+params.wPadding);
    boundingBox.height = roi.height / (1+params.hPadding);
    if (resizeImage) {
        boundingBox.x *= 2;
        boundingBox.y *= 2;
        boundingBox.width *= 2;
        boundingBox.height *= 2;
    }

    PSR = PSRList[maxIndex];
}

bool TrackerKCF::train(const Mat &img, const Mat &diffMap){
    // extract the patch for learning purpose
    cv::Mat new_x;
    if(!getSubWindow(img,roi, new_x, diffMap))return false;
    x= new_x;
//    x = 0.5 * new_x;
//    if(!getSubWindow(img,roi, new_x, -15)) return false;
//    x += 0.25 * new_x;
//    if(!getSubWindow(img,roi, new_x, 15)) return false;
//    x += 0.25 * new_x;

    //update the training data
    new_z=x.clone();
    if(frame==0)
      z=x.clone();
    else
      z=(1.0-params.interp_factor)*z+params.interp_factor*new_z;

    if(params.compress_feature){
      // feature compression
      updateProjectionMatrix(z,old_cov_mtx,proj_mtx,params.pca_learning_rate,params.compressed_size);
      compress(proj_mtx,x,x);
    }

    // Kernel Regularized Least-Squares, calculate alphas
    denseGaussKernel(params.sigma,x,x,k);

    fft2(k,kf);
    kf_lambda=kf+params.lambda;

    /* TODO: optimize this element-wise division
     * new_alphaf=yf./kf
     * z=(a+bi)/(c+di)[(ac+bd)+i(bc-ad)]/(c^2+d^2)
     */
//    new_alphaf=cv::Mat_<cv::Vec2d >(yf.rows, yf.cols);
    std::complex<double> temp;

    if(params.split_coeff){
      cv::mulSpectrums(yf,kf,new_alphaf,0);
      cv::mulSpectrums(kf,kf_lambda,new_alphaf_den,0);
    }else{
      for(int i=0;i<yf.rows;i++){
        for(int j=0;j<yf.cols;j++){
          temp=std::complex<double>(yf.at<cv::Vec2d>(i,j)[0],yf.at<cv::Vec2d>(i,j)[1])/
                  (std::complex<double>(kf_lambda.at<cv::Vec2d>(i,j)[0],kf_lambda.at<cv::Vec2d>(i,j)[1])
                  /*+std::complex<double>(0.0000000001,0.0000000001)*/);
          new_alphaf.at<cv::Vec2d >(i,j)[0]=temp.real();
          new_alphaf.at<cv::Vec2d >(i,j)[1]=temp.imag();
        }
      }
    }

    // update the RLS model
    if(frame==0){
      alphaf=new_alphaf.clone();
      if(params.split_coeff)alphaf_den=new_alphaf_den.clone();
    }else{
      alphaf=(1.0-params.interp_factor)*alphaf+params.interp_factor*new_alphaf;
      if(params.split_coeff)alphaf_den=(1.0-params.interp_factor)*alphaf_den+params.interp_factor*new_alphaf_den;
    }
    return true;
}

void TrackerKCF::createHanningWindow(cv::OutputArray _dst, const cv::Size winSize, const int type)
{
    CV_Assert( type == CV_32FC1 || type == CV_64FC1 );

    _dst.create(winSize, type);
    cv::Mat dst = _dst.getMat();

    int rows = dst.rows, cols = dst.cols;

    cv::AutoBuffer<double> _wc(cols);
    double * const wc = (double *)_wc;

    double coeff0 = 2.0 * CV_PI / (double)(cols - 1), coeff1 = 2.0f * CV_PI / (double)(rows - 1);
    for(int j = 0; j < cols; j++)
        wc[j] = 0.5 * (1.0 - std::cos(coeff0 * j));

    if(dst.depth() == CV_32F){
        for(int i = 0; i < rows; i++){
            float* dstData = dst.ptr<float>(i);
            double wr = 0.5 * (1.0 - cos(coeff1 * i));
            for(int j = 0; j < cols; j++)
                dstData[j] = (float)(wr * wc[j]);
        }
    }else{
        for(int i = 0; i < rows; i++){
            double* dstData = dst.ptr<double>(i);
            double wr = 0.5 * (1.0 - cos(coeff1 * i));
            for(int j = 0; j < cols; j++)
                dstData[j] = wr * wc[j];
        }
    }

    // perform batch sqrt for SSE performance gains
    //cv::sqrt(dst, dst); //matlab do not use the square rooted version
}

void TrackerKCF::fft2(const cv::Mat &src, cv::Mat &dest)
{
    std::vector<cv::Mat> layers(src.channels());
    std::vector<cv::Mat> outputs(src.channels());

    cv::split(src, layers);

    for(int i=0;i<src.channels();i++){
        cv::dft(layers[i],outputs[i],cv::DFT_COMPLEX_OUTPUT);
    }

    merge(outputs,dest);
}

void TrackerKCF::fft2(const cv::Mat &src, std::vector<cv::Mat> &dest)
{
    std::vector<cv::Mat> layers(src.channels());
    dest.clear();
    dest.resize(src.channels());

    cv::split(src, layers);

    for(int i=0;i<src.channels();i++){
        cv::dft(layers[i],dest[i],cv::DFT_COMPLEX_OUTPUT);
    }
}

void TrackerKCF::ifft2(const cv::Mat &src, cv::Mat &dest)
{
    cv::idft(src,dest,cv::DFT_SCALE+cv::DFT_REAL_OUTPUT);
}

bool TrackerKCF::getSubWindow(const cv::Mat &img, const cv::Rect _roi, cv::Mat &patch, const cv::Mat &diffMap, double angle)
{
    cv::Rect region = _roi;
    cv::Mat_<double> diff_patch;

    // return false if roi is outside the image
    if (
            (_roi.x+_roi.width<0)
            ||(_roi.y+_roi.height<0)
            ||(_roi.x>=img.cols)
            ||(_roi.y>=img.rows)
        )
        return false;

    // extract patch inside the image
    if(_roi.x<0) {region.x=0;region.width+=_roi.x;}
    if(_roi.y<0) {region.y=0;region.height+=_roi.y;}
    if(_roi.x+_roi.width>img.cols) region.width=img.cols-_roi.x;
    if(_roi.y+_roi.height>img.rows) region.height=img.rows-_roi.y;
    if(region.width>img.cols) region.width=img.cols;
    if(region.height>img.rows) region.height=img.rows;

    patch=img(region).clone();
    if (!diffMap.empty())
        diffMap(region).copyTo(diff_patch);

    // add some padding to compensate when the patch is outside image border
    int addTop,addBottom, addLeft, addRight;
    addTop = region.y-_roi.y;
    addBottom = (_roi.height+_roi.y>img.rows?_roi.height+_roi.y-img.rows:0);
    addLeft = region.x-_roi.x;
    addRight = (_roi.width+_roi.x>img.cols?_roi.width+_roi.x-img.cols:0);

    cv::copyMakeBorder(patch,patch,addTop,addBottom,addLeft,addRight,cv::BORDER_REPLICATE);
    if (patch.cols == 0 || patch.rows == 0)
        return false;
    if (!diffMap.empty()) {
        cv::copyMakeBorder(diff_patch, diff_patch, addTop, addBottom, addLeft, addRight,
                           cv::BORDER_CONSTANT, cv::Scalar(0));
        resize(diff_patch, diff_patch, Size(hann.cols, hann.rows));
    }
    if (patch.rows==0 || patch.cols==0)
        return false;


    if (params.descriptor == FHOG)
        resize(patch, patch, Size(hann.cols * params.binSize, hann.rows * params.binSize));
    else
        resize(patch, patch, Size(hann.cols, hann.rows));

//    cout<<patch.rows<<" "<<patch.cols<<endl;
    if(angle != 0)
        rotate(patch, angle, patch);
//    cout<<patch.rows<<" "<<patch.cols<<endl;
//    cv::imshow("patch", patch);
//    cv::waitKey();

//    //hist backproject
//    Mat backproj;
//    calcBackProject( &patch, 1, 0, hist, backproj, &ranges, 1, true );
//    cv::imshow("backproj", backproj*100);
//    cv::waitKey();


    cv::Mat lbp_patch;
    std::vector<cv::Mat> tmp_patch;
    vector<Mat> ch;
    // extract the desired descriptors
    switch(params.descriptor){
    case GRAY:
        if(img.channels()>1)
            cvtColor(patch,patch, CV_BGR2GRAY);
        patch.convertTo(patch,CV_64F);
        patch=patch/255.0-0.5; // normalize to range -0.5 .. 0.5
        break;
    case LBP:
        if(img.channels()>1)
            cvtColor(patch,patch, CV_BGR2GRAY);
        lbp::OLBP(patch,lbp_patch);
        cv::copyMakeBorder(lbp_patch,lbp_patch,1,1,1,1,cv::BORDER_REPLICATE);
        patch = lbp_patch;
        patch.convertTo(patch,CV_64F);
        patch=patch/255.0-0.5;
        break;
    case LBP_DIFF:
        if(img.channels()>1)
            cvtColor(patch,patch, CV_BGR2GRAY);
        lbp::OLBP(patch,lbp_patch);
        cv::copyMakeBorder(lbp_patch,lbp_patch,1,1,1,1,cv::BORDER_REPLICATE);
        patch = lbp_patch;
        patch.convertTo(patch,CV_64F);
        patch=patch/255.0-0.5;
        tmp_patch.push_back(patch);
        tmp_patch.push_back(diff_patch / 255);
        LOGD("sum of diff_patch :%lf", cv::sum(tmp_patch[1])[0]);
        merge(tmp_patch,patch);
        break;
    case CN:
        // CV_Assert(img.channels() == 3);
        if (img.channels() == 1)
            cvtColor(patch, patch, CV_GRAY2BGR);
        extractCN(patch,patch);
        break;
    case CN2:
        CV_Assert(img.channels() == 3);
        cv::split(patch, tmp_patch);
        for (int i = 0; i < 3; i++){
            lbp::OLBP(tmp_patch[i], lbp_patch);
            cv::copyMakeBorder(lbp_patch,lbp_patch,1,1,1,1,cv::BORDER_REPLICATE);
            tmp_patch[i] = lbp_patch;
            tmp_patch[i].convertTo(tmp_patch[i],CV_64F);
            tmp_patch[i] = tmp_patch[i]/255.0-0.5;
        }
        cv::merge(tmp_patch, patch);
        break;
    case RGB:
        CV_Assert(img.channels() == 3);
        patch.convertTo(patch,CV_64F);
        patch=patch/255.0-0.5; // normalize to range -0.5 .. 0.5
        break;
    case ICF:
        computeChannels(patch, ch);
//        cout<<ch.size()<<endl;
        for(uint i = 0; i < ch.size(); i++){
            double minValue, maxValue;
            minMaxIdx(ch[i], &minValue, &maxValue);
            ch[i] = (ch[i] - minValue) / (maxValue - minValue);
//            cv::imshow("ch",ch[i]);
//            cv::waitKey();
            ch[i]-=0.5;
        }
        cv::merge(ch, patch);
        patch.convertTo(patch,CV_64FC(10));
        break;
    case FHOG:
        patch = fhog::fhog(patch, params.binSize);
        if (!diffMap.empty()){
            vector<Mat_<double> > patch_list;
            split(patch, patch_list);
            patch_list.pop_back();
            patch_list.push_back(diff_patch / 125);
            merge(patch_list,patch);
        }
//        patch.convertTo(patch,CV_64FC(32));
        break;
    }

    patch=patch.mul(hann); // hann window filter
    return true;
}

void TrackerKCF::extractCN(cv::Mat &_patch, cv::Mat &cnFeatures)
{
    cv::Vec3b & pixel = _patch.at<cv::Vec3b>(0,0);
    unsigned index;

    cv::Mat temp = cv::Mat::zeros(_patch.rows,_patch.cols,CV_64FC(10));

    for(int i=0;i<_patch.rows;i++){
        for(int j=0;j<_patch.cols;j++){
            pixel=_patch.at<cv::Vec3b>(i,j);
            index=(unsigned)(std::floor(pixel[2]/8)+32*std::floor(pixel[1]/8)+32*32*std::floor(pixel[0]/8));
            //copy the values
            for(int _k=0;_k<10;_k++){
                temp.at<cv::Vec<double,10> >(i,j)[_k]=ColorNames[index][_k];
            }
        }
    }
    cnFeatures=temp.clone();
}

void TrackerKCF::compress(const cv::Mat &_proj_mtx, const cv::Mat &src, cv::Mat &dest)
{
    cv::Mat data=src.reshape(1,src.rows*src.cols);
    cv::Mat compressed=data*_proj_mtx;
    dest=compressed.reshape(_proj_mtx.cols,src.rows).clone();
}

void TrackerKCF::denseGaussKernel(const double sigma, const cv::Mat _x, const cv::Mat _y, cv::Mat &_k)
{
    std::vector<cv::Mat> _xf,_yf,xyf_v;
    cv::Mat xy,xyf;
    double normX, normY;

    fft2(_x,_xf);
    fft2(_y,_yf);

    normX=cv::norm(_x);
    normX*=normX;
    normY=cv::norm(_y);
    normY*=normY;

    pixelWiseMult(_xf,_yf,xyf_v,0,true);
    sumChannels(xyf_v,xyf);
    ifft2(xyf,xyf);

    if(params.wrap_kernel){
      shiftRows(xyf, _x.rows/2);
      shiftCols(xyf, _x.cols/2);
    }

    //(xx + yy - 2 * xy) / numel(x)
    xy=(normX+normY-2*xyf)/(_x.rows*_x.cols*_x.channels());

    // TODO: check wether we really need thresholding or not
    //threshold(xy,xy,0.0,0.0,THRESH_TOZERO);//max(0, (xx + yy - 2 * xy) / numel(x))
    for(int i=0;i<xy.rows;i++){
      for(int j=0;j<xy.cols;j++){
        if(xy.at<double>(i,j)<0.0)xy.at<double>(i,j)=0.0;
      }
    }

    double sig=-1.0/(sigma*sigma);
    xy=sig*xy;
    cv::exp(xy,_k);
}

void TrackerKCF::pixelWiseMult(const std::vector<cv::Mat> &src1, const std::vector<cv::Mat> &src2,
                               std::vector<cv::Mat> &dest, const int flags, const bool conjB)
{
    dest.clear();
    dest.resize(src1.size());

    for(unsigned i=0;i<src1.size();i++){
        cv::mulSpectrums(src1[i], src2[i], dest[i],flags,conjB);
    }
}

void TrackerKCF::sumChannels(std::vector<cv::Mat>& src, cv::Mat &dest)
{
    dest=src[0].clone();
    for(unsigned i=1;i<src.size();i++){
        dest+=src[i];
    }
}

void TrackerKCF::shiftRows(cv::Mat &mat)
{
    cv::Mat temp;
    cv::Mat m;
    int _k = (mat.rows-1);
    mat.row(_k).copyTo(temp);
    for(; _k > 0 ; _k-- ) {
        m = mat.row(_k);
        mat.row(_k-1).copyTo(m);
    }
    m = mat.row(0);
    temp.copyTo(m);
}

void TrackerKCF::shiftRows(cv::Mat &mat, int n)
{
    if( n < 0 ) {
        n = -n;
        flip(mat,mat,0);
        for(int _k=0; _k < n;_k++) {
            shiftRows(mat);
        }
        flip(mat,mat,0);
    }else{
        for(int _k=0; _k < n;_k++) {
            shiftRows(mat);
        }
    }
}

void TrackerKCF::shiftCols(cv::Mat& mat, int n)
{
    if(n < 0){
      n = -n;
      cv::flip(mat,mat,1);
      cv::transpose(mat,mat);
      shiftRows(mat,n);
      cv::transpose(mat,mat);
      flip(mat,mat,1);
    }else{
      cv::transpose(mat,mat);
      shiftRows(mat,n);
      cv::transpose(mat,mat);
    }
}

void TrackerKCF::calcResponse(const cv::Mat& _alphaf, const cv::Mat& _k, cv::Mat &_response)
{
    cv::Mat _kf;
    fft2(_k,_kf);
    cv::Mat spec;
    cv::mulSpectrums(_alphaf,_kf,spec,0,false);
    ifft2(spec,_response);
}

void TrackerKCF::calcResponse(const cv::Mat &_alphaf, const cv::Mat &_alphaf_den, const cv::Mat &_k, cv::Mat &_response)
{
    cv::Mat _kf;
    fft2(_k,_kf);
    cv::Mat spec;
    cv::Mat spec2=cv::Mat_<cv::Vec2d >(_k.rows, _k.cols);
    std::complex<double> temp;

    cv::mulSpectrums(_alphaf,_kf,spec,0,false);

    for(int i=0;i<_k.rows;i++){
      for(int j=0;j<_k.cols;j++){
        temp=std::complex<double>(spec.at<cv::Vec2d>(i,j)[0],spec.at<cv::Vec2d>(i,j)[1])/
                (std::complex<double>(_alphaf_den.at<cv::Vec2d>(i,j)[0],_alphaf_den.at<cv::Vec2d>(i,j)[1])
                /*+std::complex<double>(0.0000000001,0.0000000001)*/);
        spec2.at<cv::Vec2d >(i,j)[0]=temp.real();
        spec2.at<cv::Vec2d >(i,j)[1]=temp.imag();
      }
    }

    ifft2(spec2,_response);
}

void TrackerKCF::updateProjectionMatrix(const cv::Mat src, cv::Mat &old_cov, cv::Mat &_proj_mtx,
                                        double pca_rate, int compressed_sz)
{
    // compute average
    std::vector<cv::Mat> layers(src.channels());
    std::vector<cv::Scalar> average(src.channels());
    cv::split(src,layers);

    for (int i=0;i<src.channels();i++){
      average[i]=cv::mean(layers[i]);
      layers[i]-=average[i];
    }

    // calc covariance matrix
    cv::Mat data,new_cov;
    cv::merge(layers,data);
    data=data.reshape(1,src.rows*src.cols);

    new_cov=1.0/(double)(src.rows*src.cols-1)*(data.t()*data);
    if(old_cov.rows==0)old_cov=new_cov.clone();

    // calc PCA
    cv::Mat w, u, vt;
    cv::SVD::compute((1.0-pca_rate)*old_cov+pca_rate*new_cov, w, u, vt);

    // extract the projection matrix
    _proj_mtx=u(cv::Rect(0,0,compressed_sz,src.channels())).clone();
    cv::Mat proj_vars=cv::Mat::eye(compressed_sz,compressed_sz,_proj_mtx.type());
    for(int i=0;i<compressed_sz;i++){
      proj_vars.at<double>(i,i)=w.at<double>(i);
    }

    // update the covariance matrix
    old_cov=(1.0-pca_rate)*old_cov+pca_rate*_proj_mtx*proj_vars*_proj_mtx.t();
}

double TrackerKCF::calcPSR(const cv::Mat & _response, int _col, int _row, int _mask_size)
{
    cv::Scalar mean, stddev;

    cv::Mat mask = cv::Mat::ones(_response.rows, _response.cols, CV_8U);
    int left = (_col - _mask_size) < 0 ? 0 : ( _col - _mask_size );
    int right = (_col + _mask_size) >= _response.cols ? (_response.cols - 1) : (_col + _mask_size);
    int up = (_row - _mask_size) < 0 ? 0 : ( _row - _mask_size );
    int down = (_row + _mask_size) >= _response.rows ? (_response.rows - 1) : (_row + _mask_size);

    cv::Mat tmpMask(mask, cv::Range(up, down+1), cv::Range(left, right+1));
    tmpMask=cv::Scalar_<uchar>(0);

    cv::meanStdDev(_response, mean, stddev, mask);
    return (_response.at<double>(_row,_col)-mean.val[0])/stddev.val[0];
}

/*-----------------------------------------------*/

KCFParams::KCFParams()
{
    sigma=0.2;
    lambda=0.01;
    interp_factor=0.0075;
    output_sigma_factor=1.0/16.0;
    resize=true;
    max_patch_size=70*70;
    descriptor=FHOG;
    split_coeff=true;
    wrap_kernel=false;
    wPadding = 1;
    hPadding = 1;

    //feature compression
    compress_feature=false;
    compressed_size=2;
    pca_learning_rate=0.15;

    binSize = 4;
    PSRthreshold = 5.0;
    SCCMthreshold = 1.0;
}
