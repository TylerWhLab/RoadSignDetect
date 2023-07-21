#include "./Common.h"

int main()
{
	std::string fileDir = "D:\\Pub\\KCCImageNet\\images\\road_signs\\";

	std::vector<std::string> filelist;
	std::vector<std::string> target_list;

	for (const auto& entry : fs::directory_iterator(fileDir)) {
		if (entry.is_regular_file() && entry.path().extension() == ".jpg" || entry.is_regular_file() && entry.path().extension() == ".png") {

			if (entry.path().filename().string().find("street_") != std::string::npos) {
				filelist.push_back(entry.path().string());
			}

			else if (entry.path().filename().string().find("target_") != std::string::npos) {
				target_list.push_back(entry.path().string());
			}

		}
	}
	// End for file search

	int i = 0;

	// origin
	for (const std::string& filePath : filelist)
	{
		Mat search_img = cv::imread(filePath, cv::ImreadModes::IMREAD_ANYCOLOR);
		Mat draw_color = search_img.clone();

		// target
		for (const std::string& filePath_target : target_list)
		{
			std::string target_name = "";
			std::vector<std::string> vt = { "attention" , "deadend" };
			for (const std::string& t : vt) {
				if (filePath_target.find(t) != std::string::npos) {
					target_name = t;
				}
			}

			Mat pattern_img = cv::imread(filePath_target, cv::ImreadModes::IMREAD_ANYCOLOR);
			Mat search_gray_img, pattern_gray_img;
			cvtColor(search_img, search_gray_img, ColorConversionCodes::COLOR_BGR2GRAY);
			cvtColor(pattern_img, pattern_gray_img, ColorConversionCodes::COLOR_BGR2GRAY);

			// 동일한 pixel match
			cv::Mat result;
			cv::matchTemplate(search_gray_img, pattern_gray_img, result, cv::TM_CCOEFF_NORMED);

			// 매칭 결과를 정규화하여 최대값 위치 찾기
			double minVal, maxVal;
			cv::Point minLoc, maxLoc;
			cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

			// threshold_matching 기준으로 contours에 담기
			double threshold_matching = 0.95;
			Mat result_areas;
			threshold(result, result_areas, threshold_matching, 255, ThresholdTypes::THRESH_BINARY);
			result_areas.convertTo(result_areas, CV_8UC1);
			std::vector<std::vector<cv::Point>> contours;
			std::vector<cv::Vec4i> hierarchy;
			cv::findContours(result_areas, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

			RNG rng(12345);

			for (size_t i = 0; i < contours.size(); i++)
			{
				std::cout << std::format("\n===== {} 표지판 발견 ({}) =====\n", target_name, filePath);
				cv::Rect rt = cv::boundingRect(contours[i]);
				cv::drawMarker(draw_color, rt.tl(), Scalar(255, 255, 255), MarkerTypes::MARKER_CROSS);
				cv::rectangle(draw_color, Rect(rt.x, rt.y, pattern_gray_img.cols, pattern_gray_img.rows), cv::Scalar(0, 255, 0), 1);
				//std::cout << "\n";

				double length = cv::arcLength(contours[i], false);
				double area = cv::contourArea(contours[i]);
				Point2f center(0, 0);
				float radius = 0;
				cv::minEnclosingCircle(contours[i], center, radius);

				string desc = "";
				desc += std::format("id [{}]\n", i);
				desc += std::format("len {:.2f}\n", length);
				desc += std::format("area {:.2f}\n", area);
				desc += std::format("min radius {:.2f}\n", radius);
				desc += std::format("x {:.2f}\n", center.x);
				desc += std::format("y {:.2f}\n", center.y);

				// 특징 출력
				cout << desc << "\n";

				// 박스 위에 index 출력
				// putText(draw_color, std::format("id[{}]", i), Point(rt.x, rt.y), 1, 1, Scalar(0, 0, 0), 1, 8);

				// target 이름 출력
				putText(draw_color
					, target_name
					, Point(rt.x, rt.y)
					, 1                 // font face
					, 2                 // font size
					, Scalar(0, 0, 255) // font color
					, 1
					, 8);

			}
			// End for draw

		}
		// End for target

		// 결과 이미지 출력
		const char* draw_window = "draw image";
		namedWindow(draw_window);
		imshow(draw_window, draw_color);
		waitKey(1);

		// 결과 이미지 파일로 저장
		cv::imwrite(std::format("{}\\result\\result_{}.bmp", fileDir, i), draw_color);
		i++;

	}
	// End for filelist

	return 0;
}
