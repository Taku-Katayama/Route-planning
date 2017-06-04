#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

void to_Color(cv::Mat InputDepthArray, cv::Mat &OutputDepthmapArray);

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

cv::Mat disp(cv::Mat input)
{
	cv::Mat output;
	double min, max;
	cv::minMaxLoc(input, &min, &max);
	input.convertTo(output, CV_8UC1, 255 / (max - min), -255 * min / (max - min));
	return output;
}

int main(int argc, const char* argv[])
{

	char key = ' ';

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

	// 動画読み込み
	cv::VideoCapture cap_l("data/rec_l_1000mm.avi");
	cv::VideoCapture cap_r("data/rec_r_1000mm.avi");
	if (!cap_l.isOpened()) 
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

	// 変数
	double baseline = 120.0;	// [mm]

	// 時間測定
	auto startTime = std::chrono::system_clock::now();
	double processingTime = 0;
	double previousTime = 0;

	// Mat配列
	cv::Mat undis_l, undis_r;
	cv::Mat gray_l, gray_r;
	cv::Mat disparity, disparity_64f;
	cv::Mat depth;
	cv::Mat X(376, 672, CV_64F);
	cv::Mat Y(376, 672, CV_64F);
	cv::Mat view;

	while (key != 'q') 
	{

		// 動画からフレーム取得
		cap_l >> undis_l;
		cap_r >> undis_r;

		// グレースケールに変換
		cvtColor(undis_l, gray_l, CV_BGR2GRAY);
		cvtColor(undis_r, gray_r, CV_BGR2GRAY);

		// 視差計算
		sgbm->compute(gray_l, gray_r, disparity);

		// CV_16S を CV_64F に変換(計算精度の向上のため)
		disparity.convertTo(disparity_64f, CV_64F);

		// 視差を実数値に戻す(OpenCVの関数は視差値が16倍されて返ってくるため)
		disparity_64f = disparity_64f / 16;

		// 計算
		depth = fku_l * baseline / disparity_64f;
		depth.setTo(0.0, depth < 300.0);
		for (int y = 0; y < depth.rows; ++y)
		{
			for (int x = 0; x < depth.cols; ++x)
			{
				X.at<double>(y, x) = (x - cx_l) * depth.at<double>(y, x) / fku_l;
				Y.at<double>(y, x) = (y - cy_l) * depth.at<double>(y, x) / fkv_l;
			}
		}

		undis_l.setTo(cv::Scalar(200, 100, 0), depth>1100);

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
		std::cout << elapsedTimeStr << " " << processingTimeStr << std::endl;

		view = disp(depth);

		// 表示
		cv::imshow("left", undis_l);
		cv::imshow("depth", view);

		// 停止トリガー 'q'で停止
		key = cv::waitKey(15);
	}
	return 0;
}