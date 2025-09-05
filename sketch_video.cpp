#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <vector>
#include <iostream>
#include <chrono>
#include <future>

cv::Mat apply_sketch_effect(const cv::Mat& frame) {
    cv::Mat gray, inverted, blurred, sketch;
    
    // Convert to grayscale
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    
    // Invert and blur
    cv::bitwise_not(gray, inverted);
    cv::GaussianBlur(inverted, blurred, cv::Size(21, 21), 0);
    
    cv::bitwise_not(blurred, inverted);
    
    // Divide operation for sketch effect
    cv::divide(gray, inverted, sketch, 256.0);
    
    cv::Mat result;
    cv::cvtColor(sketch, result, cv::COLOR_GRAY2BGR);
    return result;
}

void process_video(const std::string& input_path, const std::string& output_path, int num_threads) {
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Cannot open video: " + input_path);
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m','p','4','v'), 
                          fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        throw std::runtime_error("Cannot create output video: " + output_path);
    }

    std::cout << "Processing " << total_frames << " frames at " << fps << " FPS using " << num_threads << " threads..." << std::endl;

    // Enable OpenCV optimizations
    cv::setUseOptimized(true);
    cv::setNumThreads(num_threads);

    // Process in batches for better performance
    const int batch_size = num_threads * 2;
    std::vector<cv::Mat> frame_batch;
    std::vector<std::future<cv::Mat>> futures;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    int processed_frames = 0;

    while (true) {
        // Read batch of frames
        frame_batch.clear();
        for (int i = 0; i < batch_size; i++) {
            cv::Mat frame;
            if (!cap.read(frame)) break;
            frame_batch.push_back(frame);
        }
        
        if (frame_batch.empty()) break;

        // Process batch in parallel using std::async
        futures.clear();
        for (const auto& frame : frame_batch) {
            futures.push_back(std::async(std::launch::async, 
                [frame]() { return apply_sketch_effect(frame); }));
        }

        // Collect results and write
        for (auto& future : futures) {
            cv::Mat result = future.get();
            writer.write(result);
            processed_frames++;
        }

        // Progress update
        if (processed_frames % (batch_size * 10) == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            double progress = static_cast<double>(processed_frames) / total_frames * 100.0;
            double fps_current = (elapsed > 0) ? static_cast<double>(processed_frames) / elapsed : 0.0;
            std::cout << "Progress: " << progress << "% (" << processed_frames 
                     << "/" << total_frames << ") - " << fps_current << " FPS" << std::endl;
        }
    }

    cap.release();
    writer.release();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    double avg_fps = (total_time > 0) ? static_cast<double>(processed_frames) / total_time : 0.0;
    
    std::cout << "Processing complete!" << std::endl;
    std::cout << "Total frames: " << processed_frames << std::endl;
    std::cout << "Total time: " << total_time << " seconds" << std::endl;
    std::cout << "Average FPS: " << avg_fps << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_video> <output_video> [threads] [use_gpu]" << std::endl;
        return -1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    int threads = (argc > 3) ? std::atoi(argv[3]) : std::thread::hardware_concurrency();
    
    try {
        std::cout << "Starting optimized video processing..." << std::endl;
        std::cout << "Input: " << input_path << std::endl;
        std::cout << "Output: " << output_path << std::endl;
        std::cout << "Threads: " << threads << std::endl;

        process_video(input_path, output_path, threads);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

