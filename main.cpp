/**
 * デプスイメージから走行出力を決定する
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <stdio.h>
#include <windows.h>
#include <string>

// デプスイメージをカラーイメージに変換
void to_Color(cv::Mat InputDepthArray, cv::Mat &OutputDepthmapArray)
{
	double min, max;
	cv::minMaxLoc(InputDepthArray, &min, &max);
	InputDepthArray.convertTo(InputDepthArray, CV_8UC1, 255 / (max - min), -255 * min / (max - min));
	cv::equalizeHist(InputDepthArray, InputDepthArray);
	cv::Mat channel[3];
	channel[0] = cv::Mat(InputDepthArray.size(), CV_8UC1);
	channel[1] = cv::Mat(InputDepthArray.size(), CV_8UC1, 255);
	channel[2] = cv::Mat(InputDepthArray.size(), CV_8UC1, 255);
	cv::Mat hsv;
	int d;
	for (int y = 0; y < InputDepthArray.rows; y++)
	{
		for (int x = 0; x < InputDepthArray.cols; x++)
		{
			d = InputDepthArray.ptr<uchar>(y)[x];
			channel[0].ptr<uchar>(y)[x] = (255 - d) / 2;
		}
		cv::merge(channel, 3, hsv);
		cv::cvtColor(hsv, OutputDepthmapArray, CV_HSV2BGR);
	}
}

char key = ' ';

int main(int argc, const char* argv[])
{

	// 名前つきパイプを作成
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

	if (hPipe == INVALID_HANDLE_VALUE)
	{
		return 1;
	}

	// データ保存ファイル
	std::ofstream reference("reference_7.csv");
	std::ofstream duty_left("duty_left_7.csv");
	std::ofstream duty_right("duty_right_7.csv");
	std::ofstream encorder_left("encorder_left.csv");
	std::ofstream encorder_right("encorder_right.csv");
	std::ofstream data_width("width_0520_7.txt");
	std::ofstream data_ave("ave_7.csv");

	// 内部パラメータと歪みパラメータの設定
	// 左カメラの内部パラメータ
	const double fku_l = 353.600559219653;
	const double fkv_l = 352.562464480179;
	const double cx_l = 320.306982522657;
	const double cy_l = 191.383465238258;

	// 右カメラの内部パラメータ
	const double fku_r = 355.659530311593;
	const double fkv_r = 354.734600040007;
	const double cx_r = 335.004584045585;
	const double cy_r = 180.558275004874;

	// 内部パラメータ行列
	cv::Mat cameraParameter_l = (cv::Mat_<double>(3, 3) << fku_l, 0., cx_l, 0., fkv_l, cy_l, 0., 0., 1.);
	cv::Mat cameraParameter_r = (cv::Mat_<double>(3, 3) << fku_r, 0., cx_r, 0., fkv_r, cy_r, 0., 0., 1.);

	// 左カメラの歪みパラメータ
	const double k1_l = -0.173747838157089;
	const double k2_l = 0.0272481881774572;
	const double p1_l = 0.0;
	const double p2_l = 0.0;

	// 右カメラの歪みパラメータ
	const double k1_r = -0.176327723277872;
	const double k2_r = 0.0286008197857787;
	const double p1_r = 0.0;
	const double p2_r = 0.0;

	// 歪みパラメータ行列
	cv::Mat distCoeffs_l = (cv::Mat_<double>(1, 4) << k1_l, k2_l, p1_l, p2_l);
	cv::Mat distCoeffs_r = (cv::Mat_<double>(1, 4) << k1_r, k2_r, p1_r, p2_r);

	// カメラ設定
	cv::VideoCapture cap(0);
	cv::Size cap_size(1344, 376);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, cap_size.width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, cap_size.height);
	if (!cap.isOpened()) 
	{ 
		return -1;
	}

	// 録画設定
	double rec_fps = 30.0;
	cv::Size rec_size(cap_size.width / 2, cap_size.height);
	cv::VideoWriter rec("rec_l_7.avi", cv::VideoWriter::fourcc('M', 'P', '4', '2'), rec_fps, rec_size, true);
	cv::VideoWriter rec1("rec_r_7.avi", cv::VideoWriter::fourcc('M', 'P', '4', '2'), rec_fps, rec_size, true);
	if (!rec.isOpened() || !rec1.isOpened()) 
	{
		return -1;
	}

	// Semi-Global block matching のパラメータ設定
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

	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
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
		cv::StereoSGBM::MODE_SGBM_3WAY);

	// block matching のパラメータ設定
	cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(0, 15);

	// 変数
	double baseline = 120.0;	// [mm]
	double width_robot = 350.0;	// [mm]
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

	// 時間測定
	auto startTime = std::chrono::system_clock::now();
	double processingTime = 0;
	double previousTime = 0;

	int frame_count = 0;

	while (key != 'q') 
	{
		// フレーム取得
		cv::Mat frame;
		cap >> frame;

		// ステレオイメージの分割
		cv::Mat frame_l = frame(cv::Rect(0, 0, frame.cols / 2, frame.rows));
		cv::Mat frame_r = frame(cv::Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));
		cv::Size frameSize(frame_l.cols, frame_l.rows);

		// 歪み補正
		cv::Mat undistorted_l, undistorted_r;
		cv::Mat mapx_l, mapy_l, mapx_r, mapy_r;
		initUndistortRectifyMap(cameraParameter_l, distCoeffs_l, cv::Mat(), cameraParameter_l, frameSize, CV_32FC1, mapx_l, mapy_l);
		initUndistortRectifyMap(cameraParameter_r, distCoeffs_r, cv::Mat(), cameraParameter_r, frameSize, CV_32FC1, mapx_r, mapy_r);
		remap(frame_l, undistorted_l, mapx_l, mapy_l, cv::INTER_LINEAR);
		remap(frame_r, undistorted_r, mapx_r, mapy_r, cv::INTER_LINEAR);

		// グレースケールに変換
		cv::Mat gray_l, gray_r;
		cvtColor(undistorted_l, gray_l, CV_BGR2GRAY);
		cvtColor(undistorted_r, gray_r, CV_BGR2GRAY);

		// 視差計算
		cv::Mat disparity;
		sgbm->compute(gray_l, gray_r, disparity);

		// CV_16S を CV_64F に変換(計算精度の向上のため)
		cv::Mat disparity_64f;
		disparity.convertTo(disparity_64f, CV_64F);

		// 視差を実数値に戻す(OpenCVの関数は視差値が16倍されて返ってくるため)
		disparity_64f = disparity_64f / 16;

		// デプスイメージの計算
		cv::Mat depth = fku_l * baseline / disparity_64f;

		// 走行出力の検出

		// デプスイメージの下半分を抜き出す
		cv::Mat depth_clone = depth.clone();
		cv::Mat cut = depth_clone(cv::Rect(64, depth_clone.rows - (depth_clone.rows / 2), depth_clone.cols - 64, depth_clone.rows / 2));

		// 700[mm]より近い物体があるか?
		cut.setTo(700.0, cut < 700.0);
/*		for (int y = 0; y < cut.rows; y++)
		{
			for (int x = 0; x < depth_clone.cols; x++)
			{
				if (cut.ptr<double>(y)[x] < 700.0)
				{ 
					cut.ptr<double>(y)[x] = double(700.0);
				}
			}
		}
*/
		// 列要素の平均値を計算
		std::vector<double> ave;
		//double ave[608] = { 0 };
		double ave_total = 0;
		double total = 0;

		for (int x = 0; x < cut.cols; ++x) 
		{
			for (int y = 0; y < cut.rows; ++y)
			{
				total += cut.ptr<double>(y)[x];	// 列要素の合計
			}
			double ave_compute = total / (double)cut.rows;	// 列要素の平均値
			ave.push_back(ave_compute);	// 結果配列に格納
			total = 0;		// 値のリセット
		}

		// 結果配列の値が2000[mm]以上なら, "0"に変換
		for (int ave_num = 0; ave_num < cut.cols; ++ave_num)
		{
			if (ave[ave_num] > 2000)
			{
				ave[ave_num] = 0;
			}
			data_ave << ave[ave_num] << ",";	// データの保存
		}
		data_ave << "\n";

		double J = 0;
		int zero_count = 0;
		std::vector<double> path_width;
		std::vector<int> reference_point;

		for (int ave_num = 0; ave_num < ave.size() - 1; ++ave_num) 
		{	// 結果配列の値が"0"か?
			if (ave[ave_num] == 0)
			{	// "0"の要素がいくつ連続しているか?
				J = ave[ave_num + 1] - ave[ave_num];
				if (J == 0)
				{
					zero_count++;	// "0"が連続している数
				}
			}
			else
			{	// 通路幅の空間幅の計算
				double X_left = ave[ave_num - zero_count - 1] * ((ave_num - zero_count - 1) - cx_l) / fku_l;	// 経路幅の左端の空間座標[mm]
				double X_right = ave[ave_num + 1] * ((ave_num + 1) - cx_l) / fku_l;	// 経路幅の右端の空間座標[mm]
				path_width.push_back(abs(X_right - X_left));	// 経路幅の長さを格納
				int reference_axis = (2 * ave_num - zero_count) / 2;	// 経路中心座標
				reference_point.push_back(reference_axis);	// 経路中心座標を格納
				zero_count = 0; // 値のリセット
			}
		}

		// 経路幅の計算結果が複数あるか?(移動出力が複数得られているか?)
		if(path_width.size() >= 2)
		{
			std::vector<double>::iterator max_path_width = std::max_element(path_width.begin(), path_width.end());	// 計算結果配列の最大値を取得
			size_t maxIndex = std::distance(path_width.begin(), max_path_width);	// 最大値が格納されているインデックスを取得
			path_width[0] = path_width[maxIndex];	// path_width[0]に最大値を格納
			reference_point[0] = reference_point[maxIndex];	//reference_point[0]に幅が最大の経路の中心座標を格納
		}

		// 移動出力の計算

		// 経路の幅は, AGVの幅より広いか?
		if (path_width[0] > width_robot)
		{
			//std::cout << "width: " << width_x << std::endl;
			data_width << frame_count << "," << path_width[0] << "\n";
			cv::Point run_reference(reference_point[0], undistorted_l.rows * 3 / 4);
			cv::circle(undistorted_l, run_reference, 15, cv::Scalar(0, 0, 200), 5, CV_AA);

			auto sampling = std::chrono::system_clock::now();
			double sampling_time = std::chrono::duration_cast<std::chrono::milliseconds>(sampling - startTime).count();

			// PIDコントローラ
			pre_error = error;
			error = cx_l - r;
			integral += ((error + pre_error) * sampling_time * pow(10, -3)) / 2;

			P = Kp * error;
			I = Ki * integral;
			D = Kd * ((pre_error - error) / sampling_time * pow(10, -3));

			U = P + I + D;

			// PWMに変換
			D_l = 80 - U;
			D_r = 80 + U;

			//std::cout << "D_l:" << D_l << ", D_r: " << D_r << "\n";

			sprintf(szBuff, "%lf,%lf", D_l, D_r);

			//std::cout << "szBuff: " << szBuff << "\n";

			// 走行用プログラムにPWM信号を送信
			WriteFile(hPipe, szBuff, strlen(szBuff), &dwNumberOfBytesWritten, NULL);
	     }

			// 録画
		rec << undistorted_l;
		rec1 << undistorted_r;

		// 経過時間と処理時間の計測, 表示
		auto checkTime = std::chrono::system_clock::now();
		double elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(checkTime - startTime).count();
		processingTime = elapsedTime - previousTime;
		previousTime = elapsedTime;
		std::ostringstream elapsed, processing;
		elapsed << elapsedTime;
		processing << processingTime;
		std::string elapsedTimeStr = "elapsed time : " + elapsed.str() + "[msec]";
		std::string processingTimeStr = "processing time : " + processing.str() + "[msec]";
		//cout << elapsedTimeStr << " " << processingTimeStr << endl;

		reference << elapsedTime << "," << r << "\n";
		duty_left << elapsedTime << "," << D_l << "\n";
		duty_right << elapsedTime << "," << D_r << "\n";

		// 表示
		//Mat depth_map;
		//to_Color(depth,depth_map);
		cv::imshow("left", undistorted_l);
		cv::imshow("right", undistorted_r);
		cv::imshow("depth", depth);
		cv::imshow("cut", cut);

		// 停止トリガー 'q'で停止
		key = cv::waitKey(15);

		// 1000フレーム経過で終了
		frame_count++;
		if (frame_count == 1000) 
		{
			break;
		}
	}
	// パイプの切断
	CloseHandle(hPipe);
	return 0;
}