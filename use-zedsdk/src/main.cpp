///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2017, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////


/***************************************************************************************************
 ** This sample demonstrates how to grab images and depth/disparity map with the ZED SDK          **
 ** Both images and depth/disparity map are displayed with OpenCV                                 **
 ** Most of the functions of the ZED SDK are linked with a key press event (using OpenCV)         **
 ***************************************************************************************************/

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <sl/Core.hpp>
#include <sl/defines.hpp>

typedef struct mouseOCVStruct {
    sl::Mat depth;
    cv::Size _resize;
} mouseOCV;

mouseOCV mouseStruct;

static void onMouseCallback(int32_t event, int32_t x, int32_t y, int32_t flag, void * param);
cv::Mat slMat2cvMat(sl::Mat& input);


int main(int argc, char **argv) {

    // Create a ZED camera object
    sl::Camera zed;

    // Set configuration parameters
    sl::InitParameters init_params;
    init_params.camera_resolution = sl::RESOLUTION_VGA;
    init_params.depth_mode = sl::DEPTH_MODE_MEDIUM;
    init_params.coordinate_units = sl::UNIT_METER;

    // Open the camera
    sl::ERROR_CODE err = zed.open(init_params);
    if (err != sl::SUCCESS)
        return 1;

    // Set runtime parameters after opening the camera
    sl::RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = sl::SENSING_MODE_STANDARD; // Use STANDARD sensing mode

    // Create sl and cv Mat to get ZED left image and depth image
    // Best way of sharing sl::Mat and cv::Mat :
    // Create a sl::Mat and then construct a cv::Mat using the ptr to sl::Mat data.
    sl::Resolution image_size = zed.getResolution();
    sl::Mat image_zed(image_size, sl::MAT_TYPE_8U_C4); // Create a sl::Mat to handle Left image
    cv::Mat image_ocv = slMat2cvMat(image_zed);
    sl::Mat depth_image_zed(image_size, sl::MAT_TYPE_8U_C4);
    cv::Mat depth_image_ocv = slMat2cvMat(depth_image_zed);

    // Create OpenCV images to display (lower resolution to fit the screen)
    cv::Size displaySize(672, 376);
    cv::Mat image_ocv_display(displaySize, CV_8UC4);
    cv::Mat depth_image_ocv_display(displaySize, CV_8UC4);

    // Mouse callback initialization
    mouseStruct.depth.alloc(image_size, sl::MAT_TYPE_32F_C1);
    mouseStruct._resize = displaySize;

    // Give a name to OpenCV Windows
    cv::namedWindow("Depth", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Depth", onMouseCallback, (void*) &mouseStruct);

    // Jetson only. Execute the calling thread on 2nd core
    sl::Camera::sticktoCPUCore(2);

    // Image Processing
    sl::Mat depth_image_raw_sl(image_size, sl::MAT_TYPE_32F_C4);
    cv::Mat depth_image_raw_ocv = slMat2cvMat(depth_image_raw_sl);
    cv::Mat X, Y, Z;
    cv::Mat Y_re, Z_re;
    cv::Mat mask;
    std::vector<cv::Mat> XYZ;
    std::vector<cv::Mat> image_channels;
    cv::Mat result_image;

    // Loop until 'q' is pressed
    char key = ' ';
    while (key != 'q') {

        // Grab and display image and depth
        if (zed.grab(runtime_parameters) == sl::SUCCESS) {

        zed.retrieveImage(image_zed, sl::VIEW_LEFT); // Retrieve the left image
        zed.retrieveImage(depth_image_zed, sl::VIEW_DEPTH); //Retrieve the depth view (image)
        zed.retrieveMeasure(depth_image_raw_sl, sl::MEASURE_XYZ); // Retrieve the XYZ measure
        zed.retrieveMeasure(mouseStruct.depth, sl::MEASURE_DEPTH); // Retrieve the depth measure (32bits)

        // split
        cv::split(depth_image_raw_ocv, XYZ);
        X = XYZ[0];
        Y = XYZ[1];
        Z = XYZ[2];

        // create mask
        //cv::inRange(Y, cv::Scalar(0.39), cv::Scalar(1.0), Y_re);
        //cv::inRange(Z, cv::Scalar(1.5), cv::Scalar(10.0), Z_re);
        //cv::bitwise_and(Y_re, Z_re, mask);

        //image_ocv.setTo(cv::Scalar(200.0, 0.0, 100.0), Y > 0.390);

        image_ocv.setTo(cv::Scalar(255.0, 0.0, 0.0), Z > 0.7);
        image_ocv.setTo(cv::Scalar(0.0, 255.0, 0.0), Z > 1.0);
        image_ocv.setTo(cv::Scalar(0.0, 0.0, 255.0), Z > 1.5);
        image_ocv.setTo(cv::Scalar(50.0, 200.0, 50.0), Z > 2.0);

        // Resize and display with OpenCV
        cv::resize(image_ocv, image_ocv_display, displaySize);
        cv::imshow("Image", image_ocv_display);
        //image_ocv.copyTo(result_image, Y_re);
        cv::resize(depth_image_ocv, depth_image_ocv_display, displaySize);
        cv::imshow("Depth", depth_image_ocv_display);
        //cv::imshow("result", Y_re);
        key = cv::waitKey(10);
        }
    }

    zed.close();
    return 0;
}

static void onMouseCallback(int32_t event, int32_t x, int32_t y, int32_t flag, void * param) {
    if (event == CV_EVENT_LBUTTONDOWN) {
        mouseOCVStruct* data = (mouseOCVStruct*) param;
        int y_int = (y * data->depth.getHeight() / data->_resize.height);
        int x_int = (x * data->depth.getWidth() / data->_resize.width);

        sl::float1 dist;
        data->depth.getValue(x_int, y_int, &dist);

        std::cout << std::endl;
        if (isValidMeasure(dist))
            std::cout << "Depth at (" << x_int << "," << y_int << ") : " << dist << "m";
        else {
            std::string depth_status;
            if (dist == TOO_FAR) depth_status = ("Depth is too far.");
            else if (dist == TOO_CLOSE) depth_status = ("Depth is too close.");
            else depth_status = ("Depth not available");
            std::cout << depth_status;
        }
        std::cout << std::endl;
    }
}


cv::Mat slMat2cvMat(sl::Mat& input)
{

	//convert MAT_TYPE to CV_TYPE
	int cv_type = -1;
	switch (input.getDataType())
	{
	case sl::MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
	case sl::MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
	case sl::MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
	case sl::MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
	case sl::MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
	case sl::MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
	case sl::MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
	case sl::MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
	default: break;
	}

	// cv::Mat data requires a uchar* pointer. Therefore, we get the uchar1 pointer from sl::Mat (getPtr<T>())
	//cv::Mat and sl::Mat will share the same memory pointer
	return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM_CPU));
}
