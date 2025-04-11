import cv2 as cv
import os

# Define the input and output directories
input_dir = 'NematodeAI/Data/Images DeadLiveCounting/feste Kamera 1'
output_dir = 'NematodeAI/Data/Images DeadLiveCounting Edges'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Add other image formats if needed
        # Read the image
        img_path = os.path.join(input_dir, filename)

        print("Processing", img_path)

        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        
        # Perform edge detection
        edges = cv.Canny(img, 100, 200)
        
        # Save the result
        
        output_path = os.path.join(output_dir, filename)
        cv.imwrite(output_path, edges)

print("Edge detection completed and images saved to", output_dir)