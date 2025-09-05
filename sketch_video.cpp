#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

cv::Mat apply_sketch_effect(const cv::Mat& frame) {
    cv::Mat gray, inverted, blurred, inverted_blurred, sketch;
    
    // Convert to grayscale
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    
    // Invert and blur
    cv::bitwise_not(gray, inverted);
    cv::GaussianBlur(inverted, blurred, cv::Size(21, 21), 0);
    
    // Blend with original grayscale
    cv::bitwise_not(blurred, inverted_blurred);
    cv::divide(gray, inverted_blurred, sketch, 256.0);
    
    cv::Mat result;
    cv::cvtColor(sketch, result, cv::COLOR_GRAY2BGR);
    return result;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv << " <input_video> <output_video>" << std::endl;
        return -1;
    }
    
    std::string input_path = argv[15];
    std::string output_path = argv[16];
    
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video " << input_path << std::endl;
        return -1;
    }
    
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m','p','4','v'), 
                          fps, cv::Size(width, height));
    
    if (!writer.isOpened()) {
        std::cerr << "Error: Cannot create output video " << output_path << std::endl;
        return -1;
    }
    
    cv::Mat frame;
    std::cout << "Processing video..." << std::endl;
    
    while (cap.read(frame)) {
        cv::Mat sketched = apply_sketch_effect(frame);
        writer.write(sketched);
    }
    
    cap.release();
    writer.release();
    
    std::cout << "Video processing complete: " << output_path << std::endl;
    return 0;
}
