# Video Sketch Effect

A simple Python project to apply a sketch (pencil drawing) effect to videos using OpenCV. Inspired by sketch-based video tools like SketchVideo[1].

## Features
- Converts input videos to sketch-style output.
- Supports real-time processing via webcam.
- Easy to extend with AI models for advanced generation (e.g., from sketches to animated videos).

## Installation
1. Clone the repo: `git clone https://github.com/yourusername/VideoSketchEffect.git`
2. Install dependencies: `pip install -r requirements.txt`

## Usage
Run the script: `python sketch_video.py --input example_video.mp4 --output sketched_video.mp4`

For webcam mode: `python sketch_video.py --webcam`

## Example
Input: Regular video  
Output: Sketch-effect video (edges detected and inverted for pencil look).

## Contributing
Pull requests welcome! For major changes, open an issue first.

## License
MIT
