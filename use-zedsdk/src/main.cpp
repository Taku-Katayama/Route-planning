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
#include <numeric>
#include <vector>
#include <chrono>
#include <fstream> // file I/O
#include <windows.h> // IPC


typedef struct mouseOCVStruct {
    sl::Mat depth;
    cv::Size _resize;
} mouseOCV;

mouseOCV mouseStruct;

static void onMouseCallback(int32_t event, int32_t x, int32_t y, int32_t flag, void * param);
cv::Mat slMat2cvMat(sl::Mat& input);
inline double pid_controller_near(int ref, double delta);
inline double pid_controller_far(int ref, double delta);

double Kp_n = 0.3;
double Ki_n = 1.5;
double Kd_n = 0.015;
double Kp_f = 0.42;
double Ki_f = 2.1;
double kd_f = 0.029;
double P = 0;
double I = 0;
double D = 0;
double error[2] = { 0 };
double cx = 338.334;
double integral = 0;

int main(int argc, char **argv) 
{

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
    cv::Mat X, Y, Z, X_disp, Y_disp;
    cv::Mat Y_re, Z_re;
    cv::Mat mask;
    cv::Mat result_bw_image(displaySize, CV_8UC1);
    cv::Mat result_z_clustering(displaySize, CV_8UC3);
    cv::Mat result_float_image(displaySize, CV_32S);
    std::vector<cv::Mat> XYZ;
    std::vector<cv::Mat> image_channels;
    cv::Mat result_image;
    cv::Mat result_path(displaySize, CV_8UC1);
    cv::Mat result_path_3d(displaySize, CV_8UC1);
    double agv_width = 0.350;
    int save_2d = 0, save_3d = 0;
    int save_x = 0;

    std::chrono::system_clock::time_point start, end;
    double time_nano = 0, time_sec = 0, time_output = 0;;

    std::string output_file_name = "path.txt";
    std::string output_U_file_name = "U.txt";
    std::ofstream output_file, output_U_file;
    output_file.open(output_file_name, std::ios::out);
    output_U_file.open(output_U_file_name, std::ios::out);
    output_file << "path" << std::endl;
    output_U_file << "U data" << std::endl;
    output_U_file << "delta[s],U,ref point[pix]" << std::endl;

    // IPC
    HANDLE hPipe;
    char szBuff[32];
    DWORD dwNumberOfBytesRead;
    DWORD dwNumberOfBytesWritten;

    hPipe = CreateFile("\\\\.\\pipe\\pwm",
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL);

    std::cout << "Connecting server ..." << std::endl;

    if (hPipe == INVALID_HANDLE_VALUE)
    {
        std::cout << "Connect client : False" << std::endl;
        return 1;
    }

    std::cout << "Connect client : Success" << std::endl;

    // Loop until 'q' is pressed
    char key = ' ';
    while (key != 'q') 
    {

        start = std::chrono::system_clock::now();

        result_bw_image.setTo(0);
        result_z_clustering.setTo(0);
        result_path.setTo(0);

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

        /////////////////////////
        // detect path process //
        /////////////////////////

        std::vector<int> ref_point;
        std::vector<double> ref_width;
        std::vector<float> ref_depth;
        std::vector<float> ref_depth_3d;
        std::vector<float> error_x_3d;
        std::vector<int> save_y, save_y_3d, save_ref_2d, save_ref_3d;
        double cx_l = 338.334;
        double X_camera_center = 0.06;
        int output_ref_point_2d = 0;
        double output_ref_width_3d = 0;
        int ref_z_max, ref_z_min, est_x_index, ref_3d_min, ref_3d_max;

        result_bw_image.setTo(200, Y > 0.390);
        result_bw_image.setTo(0, Y > 500);

        for (int y = 0; y < Z.rows; ++y)
        {
            ref_point.clear();
            ref_width.clear();
            for(int x = 0; x < Z.cols; ++x)
            {
                if (result_bw_image.at<uchar>(y, x) == 200)
                {
                    int count = 1;
                    int start_index = x;
                    while (result_bw_image.at<uchar>(y, x + count) == 200 || (x + count) == Z.cols)
                    {
                        count++;
                    }
                    int end_index = start_index + count;

                    /////////////////////////////////
                    // use image axis only process //
                    /////////////////////////////////

                    output_ref_point_2d = (start_index + end_index) / 2;
                    ref_point.push_back(output_ref_point_2d);
                    std::vector<double> error;
                    for (int i = 0; i < ref_point.size(); ++i)
                    {
                        error.push_back(abs((double)ref_point[i] - cx_l));
                    }
                    std::vector<double>::iterator error_min = std::min_element(error.begin(), error.end());
                    int minIndex = std::distance(error.begin(), error_min);
                    output_ref_point_2d = ref_point[minIndex];

                    //////////////////////////
                    // use 3d point process //
                    //////////////////////////

                    double path_width = X.at<float>(y, end_index) - X.at<float>(y, start_index);
                    if (path_width >= agv_width) 
                    {
                    double center_width = path_width / 2;
                    output_ref_width_3d = X.at<float>(y, start_index) + center_width;
                    ref_width.push_back(output_ref_width_3d);
                    std::vector<double> error_3d;
                    for (int i = 0; i < ref_width.size(); ++i)
                    {
                        error_3d.push_back(abs(ref_width[i] - X_camera_center));
                    }
                    std::vector<double>::iterator error_3d_min = std::min_element(error_3d.begin(), error_3d.end());
                    int error_min_index = std::distance(error_3d.begin(), error_3d_min);
                    output_ref_width_3d = ref_width[error_min_index];

                    ref_depth.push_back(Z.at<float>(y, output_ref_point_2d));
                    save_y.push_back(y);
                    save_ref_2d.push_back(output_ref_point_2d);
                    std::vector<float>::iterator ref_z_min_index = std::min_element(ref_depth.begin(), ref_depth.end());
                    std::vector<float>::iterator ref_z_max_index = std::max_element(ref_depth.begin(), ref_depth.end());
                    ref_z_min = std::distance(ref_depth.begin(), ref_z_min_index);
                    ref_z_max = std::distance(ref_depth.begin(), ref_z_max_index);
                    
                    //result_path.at<uchar>(y, output_ref_point_2d) = (char)255;
                    //cv::circle(image_ocv, cv::Point(output_ref_point_2d, y), 3, cv::Scalar(0.0, 0.0, 255.0), -1);
                    //std::cout << "ref= " << output_ref_width_3d << " y= " << y << std::endl;
                    //std::cout << "y= " << y << " start= " << start_index << " end= " << end_index << " width= " << path_width << std::endl;
                    }
                    x += count;
                }
            }
        }

        if (!save_y.empty() || !save_ref_2d.empty())
        {
            result_path.at<uchar>(save_y[ref_z_max], save_ref_2d[ref_z_max]) = (char)255;
            result_path.at<uchar>(save_y[ref_z_min], save_ref_2d[ref_z_min]) = (char)255;
            cv::circle(image_ocv, cv::Point(save_ref_2d[ref_z_max], save_y[ref_z_max]), 3, cv::Scalar(0.0, 0.0, 255.0), -1);
            cv::circle(image_ocv, cv::Point(save_ref_2d[ref_z_min], save_y[ref_z_min]), 3, cv::Scalar(255.0, 0.0, 0.0), -1);
            //cv::circle(image_ocv, cv::Point(save_ref_3d[ref_z_max], save_y[ref_z_max]), 3, cv::Scalar(0.0, 255.0, 0.0), -1);
            //cv::circle(image_ocv, cv::Point(save_ref_3d[ref_z_min], save_y[ref_z_min]), 3, cv::Scalar(255.0, 0.0, 255.0), -1);

            //std::cout << "pre= " << save_2d << " now= " << save_ref_2d[ref_z_max] << std::endl;

            ////////////////////
            // pid controller //
            ////////////////////

            end = std::chrono::system_clock::now();
            time_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            time_sec = time_nano * pow(10, -9);
            time_output += time_sec;

            if (save_2d - save_ref_2d[ref_z_min] <= 50)
            {
                double U = pid_controller_near(save_ref_2d[ref_z_min], time_sec);
                output_U_file << time_output << "," << U << "," << save_ref_2d[ref_z_min] << std::endl;
                sprintf(szBuff, "%lf", U);
                WriteFile(hPipe, szBuff, strlen(szBuff), &dwNumberOfBytesWritten, NULL);
                std::cout << "near" << std::endl;
            }
            else
            {
                double U = pid_controller_far(save_ref_2d[ref_z_max], time_sec);
                output_U_file << time_output << "," << U << "," << save_ref_2d[ref_z_max] << std::endl;
                sprintf(szBuff, "%lf", U);
                WriteFile(hPipe, szBuff, strlen(szBuff), &dwNumberOfBytesWritten, NULL);
                std::cout << "far" << std::endl;
            }

            save_2d = save_ref_2d[ref_z_min];
            save_3d = save_ref_3d[ref_3d_max];
        }

        // Resize and display with OpenCV
        cv::resize(image_ocv, image_ocv_display, displaySize);
        cv::imshow("Image", image_ocv_display);
        cv::resize(depth_image_ocv, depth_image_ocv_display, displaySize);
        cv::imshow("Depth", depth_image_ocv_display);
        cv::imshow("bw result", result_bw_image);
        cv::imshow("result path", result_path);
        //cv::imshow("X", X);

        
        //std::string save_image_name = "Data/2d_result/image/image_" + std::to_string(time) + ".png";
        //std::string save_depth_name = "Data/2d_result/depth/depth_" + std::to_string(time) + ".png";
        //std::string save_y_name = "Data/2d_result/y/y_" + std::to_string(time) + ".png";
        //std::string save_path_name = "Data/2d_result/path/path_" + std::to_string(time) + ".png";

        //cv::imwrite(save_image_name, image_ocv);
        //cv::imwrite(save_depth_name, depth_image_ocv);
        //cv::imwrite(save_y_name, result_bw_image);
        //cv::imwrite(save_path_name, result_path);

        key = cv::waitKey(10);
        }
    }

    //CloseHandle(hPipe);
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

inline double pid_controller_near(int ref, double delta)
{
    error[0] = error[1];
    error[1] = cx - ref;
    integral += (error[0] + error[1]) * delta / 2.0;
    P = Kp_n * error[1];
    I = Ki_n * integral;
    D = Kd_n * (error[1] - error[0]) / delta;

    return P + I + D;
}

inline double pid_controller_far(int ref, double delta)
{
    error[0] = error[1];
    error[1] = cx - ref;
    integral += (error[0] + error[1]) * delta / 2.0;
    P = Kp_f * error[1];
    I = Ki_f * integral;
    D = kd_f * (error[1] - error[0]) / delta;

    return P + I + D;
}
