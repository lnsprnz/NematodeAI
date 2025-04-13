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

def preprocess_video_frames(video_frames, window_left, window_right, clahe):
    """
    Preprocess video frames by cropping to the region of interest and applying CLAHE.
    Returns a list of processed frames in 3-channel format.
    """
    processed_frames = [
        preprocess_frame(frame[:, window_left:window_right, :], clahe) for frame in video_frames
    ]
    return processed_frames

# -------------------------------
# Main Processing Script
# -------------------------------

def main():
    # Define input and output directories
    video_dir = 'NematodeAI\Data\Videos DeadLiveCounting'      # Directory containing .avi files
    output_dir = 'NematodeAI\Data\Images DeadLiveCounting'     # Directory to save processed frames
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the region of interest (horizontal cropping)
    window_left, window_right = 400, 1520
    
    # Create a CLAHE object (for contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
    
    # Process each .avi file in the input directory
    for video_file in os.listdir(video_dir):
        if not video_file.lower().endswith('.avi'):
            continue
        
        video_path = os.path.join(video_dir, video_file)
        print(f"Processing video: {video_path}")
        
        # Load frames (one frame per second)
        raw_frames = load_video_frames(video_path)
        if not raw_frames:
            print(f"WARNING: No frames extracted from {video_file}.")
            continue
        
        # Preprocess the frames (crop and apply CLAHE)
        processed_frames = preprocess_video_frames(raw_frames, window_left, window_right, clahe)
        
        # Create a subdirectory for the current video
        base_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Save each processed frame as an image file
        for idx, frame in enumerate(processed_frames):
            output_filename = f"{base_name}_frame_{idx}.png"
            output_path = os.path.join(video_output_dir, output_filename)
            cv2.imwrite(output_path, frame)
            print(f"Saved preprocessed frame: {output_path}")

if __name__ == "__main__":
    main()
