#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/parallel.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <immintrin.h>  // Intel SIMD intrinsics
#include <iostream>
#include <chrono>

#ifdef HAVE_OPENCV_CUDAIMGPROC
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

class VideoProcessor {
private:
    std::queue<cv::Mat> frame_queue;
    std::queue<cv::Mat> processed_queue;
    std::mutex queue_mutex, processed_mutex;
    std::condition_variable queue_cv, processed_cv;
    std::atomic<bool> finished{false};
    int num_threads;
    bool use_gpu;
    
public:
    VideoProcessor(int threads = std::thread::hardware_concurrency(), bool gpu = false) 
        : num_threads(threads), use_gpu(gpu) {
        // Enable all OpenCV optimizations
        cv::setUseOptimized(true);
        cv::setNumThreads(0); // Use all available cores
        
        // Check for CUDA support
        if (gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            std::cout << "Using GPU acceleration with " << cv::cuda::getCudaEnabledDeviceCount() << " devices" << std::endl;
            cv::cuda::setDevice(0);
        } else {
            use_gpu = false;
            std::cout << "Using CPU with " << num_threads << " threads" << std::endl;
        }
    }

    // SIMD-optimized sketch effect for CPU
    cv::Mat apply_sketch_effect_simd(const cv::Mat& frame) {
        cv::Mat gray, inverted, blurred, sketch;
        
        // Use SIMD-optimized color conversion
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        // SIMD-optimized inversion
        cv::bitwise_not(gray, inverted);
        
        // Use separable filter for better SIMD utilization
        cv::Mat kernel = cv::getGaussianKernel(21, 0);
        cv::sepFilter2D(inverted, blurred, CV_8U, kernel, kernel);
        
        cv::bitwise_not(blurred, inverted);
        
        // SIMD-optimized division
        cv::divide(gray, inverted, sketch, 256.0);
        
        cv::Mat result;
        cv::cvtColor(sketch, result, cv::COLOR_GRAY2BGR);
        return result;
    }

    // GPU-accelerated sketch effect
    cv::Mat apply_sketch_effect_gpu(const cv::Mat& frame) {
        #ifdef HAVE_OPENCV_CUDAIMGPROC
        cv::cuda::GpuMat gpu_frame, gpu_gray, gpu_inverted, gpu_blurred, gpu_sketch, gpu_result;
        
        // Upload to GPU
        gpu_frame.upload(frame);
        
        // GPU color conversion
        cv::cuda::cvtColor(gpu_frame, gpu_gray, cv::COLOR_BGR2GRAY);
        
        // GPU bitwise operations
        cv::cuda::bitwise_not(gpu_gray, gpu_inverted);
        
        // GPU Gaussian blur
        cv::Ptr<cv::cuda::Filter> gaussian_filter = cv::cuda::createGaussianFilter(
            gpu_inverted.type(), gpu_inverted.type(), cv::Size(21, 21), 0);
        gaussian_filter->apply(gpu_inverted, gpu_blurred);
        
        cv::cuda::bitwise_not(gpu_blurred, gpu_inverted);
        
        // GPU division
        cv::cuda::divide(gpu_gray, gpu_inverted, gpu_sketch, 256.0);
        
        // Convert back to BGR
        cv::cuda::cvtColor(gpu_sketch, gpu_result, cv::COLOR_GRAY2BGR);
        
        // Download result
        cv::Mat result;
        gpu_result.download(result);
        return result;
        #else
        return apply_sketch_effect_simd(frame);
        #endif
    }

    // Parallel processing using OpenCV's parallel_for_
    class ParallelSketchProcessor : public cv::ParallelLoopBody {
    private:
        std::vector<cv::Mat>& frames;
        std::vector<cv::Mat>& results;
        VideoProcessor* processor;
        
    public:
        ParallelSketchProcessor(std::vector<cv::Mat>& f, std::vector<cv::Mat>& r, VideoProcessor* p)
            : frames(f), results(r), processor(p) {}
            
        virtual void operator()(const cv::Range& range) const override {
            for (int i = range.start; i < range.end; i++) {
                if (processor->use_gpu) {
                    results[i] = processor->apply_sketch_effect_gpu(frames[i]);
                } else {
                    results[i] = processor->apply_sketch_effect_simd(frames[i]);
                }
            }
        }
    };

    // Multi-threaded frame reader
    void frame_reader(cv::VideoCapture& cap) {
        cv::Mat frame;
        while (cap.read(frame) && !finished.load()) {
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                frame_queue.push(frame.clone());
            }
            queue_cv.notify_one();
        }
        finished.store(true);
        queue_cv.notify_all();
    }

    // Multi-threaded frame processor
    void frame_processor() {
        while (true) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [this] { return !frame_queue.empty() || finished.load(); });
                
                if (frame_queue.empty() && finished.load()) break;
                if (frame_queue.empty()) continue;
                
                frame = frame_queue.front();
                frame_queue.pop();
            }
            
            cv::Mat processed;
            if (use_gpu) {
                processed = apply_sketch_effect_gpu(frame);
            } else {
                processed = apply_sketch_effect_simd(frame);
            }
            
            {
                std::lock_guard<std::mutex> lock(processed_mutex);
                processed_queue.push(processed);
            }
            processed_cv.notify_one();
        }
    }

    // Batch processing for maximum throughput
    void process_video_batch(const std::string& input_path, const std::string& output_path) {
        cv::VideoCapture cap(input_path, cv::CAP_FFMPEG);
        if (!cap.isOpened()) {
            throw std::runtime_error("Cannot open video: " + input_path);
        }

        // Optimize capture settings
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('H','2','6','4'));

        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        // Use hardware encoding if available
        int fourcc = cv::VideoWriter::fourcc('H','2','6','4');
        cv::VideoWriter writer(output_path, fourcc, fps, cv::Size(width, height));
        if (!writer.isOpened()) {
            fourcc = cv::VideoWriter::fourcc('m','p','4','v');
            writer.open(output_path, fourcc, fps, cv::Size(width, height));
        }

        std::cout << "Processing " << total_frames << " frames at " << fps << " FPS..." << std::endl;

        // Process in batches for memory efficiency
        const int batch_size = num_threads * 4;
        std::vector<cv::Mat> frame_batch;
        std::vector<cv::Mat> result_batch;
        
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

            // Process batch in parallel
            result_batch.resize(frame_batch.size());
            cv::parallel_for_(cv::Range(0, frame_batch.size()), 
                             ParallelSketchProcessor(frame_batch, result_batch, this));

            // Write results
            for (const auto& result : result_batch) {
                writer.write(result);
                processed_frames++;
            }

            // Progress update
            if (processed_frames % (batch_size * 10) == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
                double progress = static_cast<double>(processed_frames) / total_frames * 100.0;
                double fps_current = static_cast<double>(processed_frames) / elapsed;
                std::cout << "Progress: " << progress << "% (" << processed_frames 
                         << "/" << total_frames << ") - " << fps_current << " FPS" << std::endl;
            }
        }

        cap.release();
        writer.release();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        double avg_fps = static_cast<double>(processed_frames) / total_time;
        
        std::cout << "Processing complete!" << std::endl;
        std::cout << "Total frames: " << processed_frames << std::endl;
        std::cout << "Total time: " << total_time << " seconds" << std::endl;
        std::cout << "Average FPS: " << avg_fps << std::endl;
    }

    // Pipeline processing with separate reader/processor/writer threads
    void process_video_pipeline(const std::string& input_path, const std::string& output_path) {
        cv::VideoCapture cap(input_path, cv::CAP_FFMPEG);
        if (!cap.isOpened()) {
            throw std::runtime_error("Cannot open video: " + input_path);
        }

        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);

        cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('H','2','6','4'), 
                              fps, cv::Size(width, height));

        // Start reader thread
        std::thread reader_thread(&VideoProcessor::frame_reader, this, std::ref(cap));

        // Start processor threads
        std::vector<std::thread> processor_threads;
        for (int i = 0; i < num_threads; i++) {
            processor_threads.emplace_back(&VideoProcessor::frame_processor, this);
        }

        // Writer thread (main thread)
        while (true) {
            cv::Mat processed_frame;
            {
                std::unique_lock<std::mutex> lock(processed_mutex);
                processed_cv.wait(lock, [this] { 
                    return !processed_queue.empty() || (finished.load() && frame_queue.empty()); 
                });
                
                if (processed_queue.empty() && finished.load()) break;
                if (processed_queue.empty()) continue;
                
                processed_frame = processed_queue.front();
                processed_queue.pop();
            }
            
            writer.write(processed_frame);
        }

        // Clean up threads
        reader_thread.join();
        for (auto& t : processor_threads) {
            t.join();
        }

        cap.release();
        writer.release();
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_video> <output_video> [threads] [use_gpu]" << std::endl;
        return -1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    int threads = (argc > 3) ? std::atoi(argv[3]) : std::thread::hardware_concurrency();
    bool use_gpu = (argc > 4) ? (std::string(argv[4]) == "true") : false;

    try {
        VideoProcessor processor(threads, use_gpu);
        
        std::cout << "Starting optimized video processing..." << std::endl;
        std::cout << "Input: " << input_path << std::endl;
        std::cout << "Output: " << output_path << std::endl;
        std::cout << "Threads: " << threads << std::endl;
        std::cout << "GPU: " << (use_gpu ? "Enabled" : "Disabled") << std::endl;

        // Use batch processing for better performance
        processor.process_video_batch(input_path, output_path);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
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
