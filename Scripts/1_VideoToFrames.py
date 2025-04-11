"""
Preprocess Video Frames Script

This script performs the following steps:
  1. Iterates over each .avi file in the input directory.
  2. Loads the video frames (extracting one frame per second).
  3. Crops each frame to the region of interest and applies CLAHE enhancement.
  4. Saves the processed frames to an output directory (organized by video filename).

Requirements:
  - OpenCV
  - Python 3.x
"""

import os
import cv2

# Define input and output paths
video_path = 'NematodeAI/Data/C0105.MP4'      # Replace with your video path
output_dir =  video_path + '_processedFrames'  # Output directory for processed frames

# -------------------------------
# Helper Functions
# -------------------------------

def preprocess_frame(frame, clahe):
    """
    Convert the frame to grayscale, apply CLAHE, and convert back to 3-channel.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced_frame = clahe.apply(gray_frame)
    return cv2.cvtColor(enhanced_frame, cv2.COLOR_GRAY2BGR)

def load_video_frames(video_path):
    """
    Load video frames into a list of numpy arrays.
    Extracts one frame per second.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    frame_interval = fps                  # One frame per second
    frame_count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

def preprocess_video_frames(video_frames, clahe):
    """
    Preprocess video frames by cropping to the region of interest and applying CLAHE.
    Returns a list of processed frames in 3-channel format.
    """
    processed_frames = [
        preprocess_frame(frame, clahe) for frame in video_frames
    ]
    return processed_frames

# -------------------------------
# Main Processing Script
# -------------------------------

def main():
    os.makedirs(output_dir, exist_ok=True)
        
    print(f"Processing video: {video_path}")
    
    # Load frames (one frame per second)
    frames = load_video_frames(video_path)
    if not frames:
        print(f"WARNING: No frames extracted from video.")
        exit(1)
       
    # Create output directory using video filename
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Save each processed frame as an image file
    for idx, frame in enumerate(frames):
        output_filename = f"{base_name}_frame_{idx}.png"
        output_path = os.path.join(video_output_dir, output_filename)
        cv2.imwrite(output_path, frame)
        print(f"Saved preprocessed frame: {output_path}")

if __name__ == "__main__":
    main()
