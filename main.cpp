/**
 * カメラ映像から経路を検出する
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
//#include <runCtrl.h>
//#include <vutils.h>

using namespace cv;
using namespace std;

// To Color Depth image
void to_Color(Mat InputDepthArray, Mat &OutputDepthmapArray){
	double min, max;
	minMaxLoc(InputDepthArray, &min, &max);
	InputDepthArray.convertTo(InputDepthArray, CV_8UC1, 255 / (max - min), -255 * min / (max - min));
	equalizeHist(InputDepthArray, InputDepthArray);
	Mat channel[3];
	channel[0] = Mat(InputDepthArray.size(), CV_8UC1);
	channel[1] = Mat(InputDepthArray.size(), CV_8UC1, 255);
	channel[2] = Mat(InputDepthArray.size(), CV_8UC1, 255);
	Mat hsv;
	int d;
	for (int y = 0; y < InputDepthArray.rows; y++){
		for (int x = 0; x < InputDepthArray.cols; x++){
			d = InputDepthArray.ptr<uchar>(y)[x];
			channel[0].ptr<uchar>(y)[x] = (255 - d) / 2;
		}
		merge(channel, 3, hsv);
		cvtColor(hsv, OutputDepthmapArray, CV_HSV2BGR);
	}
}

char key = ' ';

int main(int argc, const char* argv[])
{
	ofstream reference("reference.csv");
	ofstream duty_left("duty_left.csv");
	ofstream duty_right("duty_right.csv");
	ofstream encorder_left("encorder_left.csv");
	ofstream encorder_right("encorder_right.csv");

	// Setting camera intrinsic parameter and distortion coefficient
	// Left camera intrinsic parameter
	const double fku_l = 353.600559219653;
	const double fkv_l = 352.562464480179;
	const double cx_l = 320.306982522657;
	const double cy_l = 191.383465238258;

	// Right camera intrinsic parameter
	const double fku_r = 355.659530311593;
	const double fkv_r = 354.734600040007;
	const double cx_r = 335.004584045585;
	const double cy_r = 180.558275004874;

	// Create intrinsic parameter matrix
	Mat cameraParameter_l = (Mat_<double>(3, 3) << fku_l, 0., cx_l, 0., fkv_l, cy_l, 0., 0., 1.);
	Mat cameraParameter_r = (Mat_<double>(3, 3) << fku_r, 0., cx_r, 0., fkv_r, cy_r, 0., 0., 1.);

	// Left camera distortion coefficient
	const double k1_l = -0.173747838157089;
	const double k2_l = 0.0272481881774572;
	const double p1_l = 0.0;
	const double p2_l = 0.0;

	// Right camera distortion coefficient
	const double k1_r = -0.176327723277872;
	const double k2_r = 0.0286008197857787;
	const double p1_r = 0.0;
	const double p2_r = 0.0;

	// Create distortion coefficient matrix
	Mat distCoeffs_l = (Mat_<double>(1, 4) << k1_l, k2_l, p1_l, p2_l);
	Mat distCoeffs_r = (Mat_<double>(1, 4) << k1_r, k2_r, p1_r, p2_r);

	// Setting camera resolution
	VideoCapture cap(0);
	if (!cap.isOpened()) return -1;
	Size cap_size(1344, 376);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, cap_size.width);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, cap_size.height);

	// Setting recording
	double rec_fps = 30.0;
	Size rec_size(cap_size.width / 2, cap_size.height);
	VideoWriter rec("rec.avi", VideoWriter::fourcc('M', 'P', '4', '2'), rec_fps, rec_size, true);
	if (!rec.isOpened()) return -1;

	// Setting Semi-Global block matching parameter
	int minDisparity = 16 * 0;
	int numDisparities = 16 * 4;
	int blockSize = 3;
	int P1 = 0;
	int P2 = 0;
	int disp12MaxDiff = 0;
	int preFilterCap = 0;
	int uniquenessRatio = 0;
	int speckleWindowSize = 0;
	int speckleRange = 1;

	Ptr<StereoSGBM> sgbm = StereoSGBM::create(
		minDisparity,
		numDisparities,
		blockSize,
		P1,
		P2,
		disp12MaxDiff,
		preFilterCap,
		uniquenessRatio,
		speckleWindowSize,
		speckleRange,
		StereoSGBM::MODE_SGBM_3WAY);

	// Setting block matching parameter
	Ptr<StereoBM> bm = StereoBM::create(0, 15);

	// Setting variable
	double baseline = 120.0;	// [mm]
	double width_robot = 35.0;	// [cm]
	double error = 0;
	double pre_error = 0;
	double Kp = 0.1;
	double Ki = 0.05;
	double Kd = 0.05;
	double P = 0;
	double I = 0;
	double D = 0;
	double U = 0;
	double integral = 0;
	double r = 0;
	double D_r = 0;
	double D_l = 0;

	// Setting robot
	// 1 : turn on , 0 : turn off
	//int robot_switch = 0;
	//static RunCtrl run;
	//run.connect("COM6");
	//const int motor_r = 0;
	//const int motor_l = 1;
	//if (robot_switch == 0){
	//	run.setWheelVel(motor_r, 0);
	//	run.setWheelVel(motor_l, 0);
	//}
	//else{
	//	run.setWheelVel(motor_r, 5);
	//	run.setWheelVel(motor_l, 5);
	//}

	// Start time measuremen
	auto startTime = chrono::system_clock::now();
	double processingTime = 0;
	double previousTime = 0;


	// main roop
	while (key != 'q') {

		// 1.Get frame
		Mat frame;
		cap >> frame;

		// 2.Split left images and right image from stereo image
		Mat frame_l = frame(Rect(0, 0, frame.cols / 2, frame.rows));
		Mat frame_r = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));
		Size frameSize(frame_l.cols, frame_l.rows);

		// 3.Correct distortion
		Mat undistorted_l, undistorted_r;
		Mat mapx_l, mapy_l, mapx_r, mapy_r;
		initUndistortRectifyMap(cameraParameter_l, distCoeffs_l, Mat(), cameraParameter_l, frameSize, CV_32FC1, mapx_l, mapy_l);
		initUndistortRectifyMap(cameraParameter_r, distCoeffs_r, Mat(), cameraParameter_r, frameSize, CV_32FC1, mapx_r, mapy_r);
		remap(frame_l, undistorted_l, mapx_l, mapy_l, INTER_LINEAR);
		remap(frame_r, undistorted_r, mapx_r, mapy_r, INTER_LINEAR);

		// 4.Change grayscale
		Mat gray_l, gray_r;
		cvtColor(undistorted_l, gray_l, CV_BGR2GRAY);
		cvtColor(undistorted_r, gray_r, CV_BGR2GRAY);

		// 5.Compute disparity
		Mat disparity;
		//sgbm->compute(gray_l, gray_r, disparity);
		sgbm->compute(gray_l, gray_r, disparity);

		// CV_16S -> CV_64F
		Mat disparity_64f;
		disparity.convertTo(disparity_64f, CV_64F);

		// disparity / 16
		//disparity_64f = disparity_64f / 16;

		// 6.Compute depth
		Mat depth = fku_l * baseline / disparity_64f;

		// Find route process

		// 7.Cut the lower half of the depth image
		Mat depth_clone = depth.clone();
		Mat cut = depth_clone(Rect(64, depth_clone.rows - (depth_clone.rows / 2), depth_clone.cols - 64, depth_clone.rows / 2));

		// 8.If cut elements value < 30, the elements change to 0
		for (int y = 0; y < cut.rows; y++) {
			for (int x = 0; x < depth_clone.cols; x++) {
				if (cut.ptr<double>(y)[x] <= 0) cut.ptr<double>(y)[x] = double(0);
			}
		}

		// 9.Compute element average
		//vector<double> ave;
		double ave[608] = { 0 };
		double ave_total = 0;
		double total = 0;
		int element_num = 0;

		for (int x = 0; x < cut.cols; x++) {
			for (int y = 0; y < cut.rows; y++) {
				total += cut.ptr<double>(y)[x];	// total of element values
				element_num++;					// total number of element
			}
			double ave_comp = total / (double)element_num;	// average of rows
			ave[x] = ave_comp;
			ave_total += ave_comp;							// total of average value
			total = 0;										// reset
			element_num = 0;								// reset
		}

		double ave_ave = ave_total / cut.cols;			// average of average value

		// 10.Search for places where the largest width exists
		// If ave array elements > ave_ave, the elements change to 0
		// 0 is judged as a passable area
		for (int ave_num = 0; ave_num < cut.cols; ave_num++) {
			if (ave[ave_num] > ave_ave) ave[ave_num] = 0;
		}

		double J = 0;
		int zero_count = 0;
		int zero_count_max = 0;
		int zero_count_num = 0;

		for (int ave_num = 1; ave_num < cut.cols; ave_num++) {
			// compute the difference from adjacent pixels
			J = ave[ave_num] - ave[ave_num - 1];
			if (J == 0) {
				zero_count++;
				// extract the maximum value of zero_count
				if (zero_count_max < zero_count) {
					zero_count_max = zero_count;	// maximum value
					zero_count_num = ave_num;		// maximum value pixel's end index
				}

			}
			else {
				zero_count = 0;						//reset
			}
		}

		int start = zero_count_num - zero_count_max;	// maximum value pixel's start index

		// 11.compute 3D width
		double x_s = start * cut.ptr<double>(cut.rows / 2)[start] / fku_l;
		double x_e = zero_count_num * cut.ptr<double>(cut.rows / 2)[zero_count_num] / fku_l;
		double width_x = abs(x_e - x_s);

		// 12.compute reference
		if (width_x > width_robot) {
			r = ((zero_count_num + start) / 2);
			Point run_reference(r, undistorted_l.rows * 3 / 4);
			circle(undistorted_l, run_reference, 15, Scalar(0, 0, 200), 5, CV_AA);

			auto sampling = chrono::system_clock::now();
			double sampling_time = chrono::duration_cast<std::chrono::milliseconds>(sampling - startTime).count();

			// PID controller
			pre_error = error;
			error = cx_l - r;
			integral += ((error + pre_error) * sampling_time * pow(10, -3)) / 2;

			P = Kp * error;
			I = Ki * integral;
			D = Kd * ((pre_error - error) / sampling_time * pow(10, -3));

			U = P + I + D;

			// Convert U to PWM
			D_l = 180 - U;
			D_r = 180 + U;

			//	if (robot_switch == 0){
			//		cout << "error sum : " << error << endl
			//			<< "error now : " << error - pre_error << endl
			//			<<"Left Moter Output : " << D_l << endl
			//			<< "Right Motor Output : " << D_r << endl;
			//	}
			//	else{
			//		run.setMotorPwm(motor_r, (uchar)D_r);
			//		run.setMotorPwm(motor_l, (uchar)D_l);
			//	}
			//}
			//else{	// If r not detected, lower the robot speed
			//	if (robot_switch == 0){
			//		cout << "error sum : " << error << endl
			//			<< "error now : " << error - pre_error << endl
			//			<< "Left Moter Output : " << D_l << endl
			//			<< "Right Motor Output : " << D_r << endl;
			//	}
			//	else{
			//		uchar D_r_red = (uchar)D_r - 20;
			//		uchar D_l_red = (uchar)D_l - 20;
			//		run.setMotorPwm(motor_r, D_r_red);
			//		run.setMotorPwm(motor_l, D_l_red);
			//	}
	     }

			// recording left frame
			rec << undistorted_l;

			// compute process time and elapsed time
			auto checkTime = chrono::system_clock::now();
			double elapsedTime = chrono::duration_cast<std::chrono::milliseconds>(checkTime - startTime).count();
			processingTime = elapsedTime - previousTime;
			previousTime = elapsedTime;
			ostringstream elapsed, processing;
			elapsed << elapsedTime;
			processing << processingTime;
			string elapsedTimeStr = "elapsed time : " + elapsed.str() + "msec";
			string processingTimeStr = "processing time : " + processing.str() + "msec";
			cout << elapsedTimeStr << " " << processingTimeStr << endl;

			int enc_l = 0;
			int enc_r = 0;
			//run.getEncoderVel(motor_l, &enc_l);
			//run.getEncoderVel(motor_r, &enc_r);

			reference << elapsedTime << "," << r << endl;
			duty_left << elapsedTime << "," << D_l << endl;
			duty_right << elapsedTime << "," << D_r << endl;
			encorder_left << elapsedTime << "," << enc_l << endl;
			encorder_right << elapsedTime << "," << enc_r << endl;

			// preview
			//Mat depth_map;
			//to_Color(depth,depth_map);
			imshow("left", undistorted_l);
			imshow("right", undistorted_r);
			imshow("depth", depth);

			// Loop break when the enter key is pressed
			key = waitKey(15);
		}

		// robot stop
		//run.setMotorPwm(motor_r, 0);
		//run.setMotorPwm(motor_l, 0);

		return 0;
	}