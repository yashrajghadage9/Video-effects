import cv2
import numpy as np
import argparse

def apply_sketch_effect(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Invert and blur
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    # Blend with original grayscale
    inverted_blurred = 255 - blurred
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        sketched = apply_sketch_effect(frame)
        out.write(sketched)
    
    cap.release()
    out.release()
    print(f"Output saved to {output_path}")

def process_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        sketched = apply_sketch_effect(frame)
        cv2.imshow('Sketch Effect', sketched)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply sketch effect to video.")
    parser.add_argument("--input", help="Path to input video")
    parser.add_argument("--output", default="sketched_video.mp4", help="Path to output video")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for real-time sketch")
    args = parser.parse_args()
    
    if args.webcam:
        process_webcam()
    elif args.input:
        process_video(args.input, args.output)
    else:
        print("Provide --input or --webcam flag.")
