""""
A script for preprocessing microscopy images of nematodes.
This script processes microscopy images by detecting and isolating circular regions 
containing nematodes. It applies several image processing techniques including:
- Watershed segmentation for initial object detection
- Hough Circle Transform to detect the circular well
- Cropping and masking to isolate the region of interest
- Optional CLAHE enhancement for contrast improvement
The script processes all images in a specified input directory and saves the 
processed images to an output directory with '_cropped' suffix.
Dependencies:
    - OpenCV (cv2)
    - NumPy
    - OS
Input:
    - Directory containing .jpg or .png microscopy images
Output:
    - Processed images saved in a new directory with '_cropped' suffix
    - Each output image is cropped and masked to the detected circular well
Example:
    If input directory is 'Data/Images', processed images will be saved in 'Data/Images_cropped'
"""

import os
import cv2
import numpy as np

# -------------------------------
# Helper Functions
# -------------------------------

def watershed (img):
    """""
    Apply watershed algorithm to an image.
    """""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    return sure_bg

def houghCircle (img, param1, param2, minRadius, maxRadius):
    """"
    Apply Hough Circle Transform to an image. Returns array of circles detected [centerx, centery, radius].
    """""
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=40,param2=20,minRadius=900,maxRadius=1100)
    return circles

def crop_image(img, x, y, r, tolerance):
    """
    Crop the image to the specified region of interest.
    """
    r = r + tolerance
    cropImage = img[y-r:y+r, x-r:x+r]
    return cropImage

def mask_image(img, r):
    """
    Apply mask to an image according to the circle detected.
    """
    mask = np.zeros_like(img)
    cv2.circle(mask, (mask.shape[1]//2, mask.shape[0]//2), r, (255, 255, 255), -1)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def clahe(img):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(gray)
    return img

def main():
    try:
        img_dir = os.path.abspath('NematodeAI/Data/C0105.MP4_processedFrames/C0105')
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Directory not found: {img_dir}")
            
        output_dir = img_dir + '_cropped'
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in os.listdir(img_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                try:
                    img_path = os.path.join(img_dir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if img is None:
                        print(f"Error reading image: {filename}")
                        continue
                        
                    # Apply watershed algorithm
                    watershed_img = watershed(img)
                    # Apply Hough Circle Transform
                    circles = houghCircle(watershed_img, 40, 20, 990, 1030)
                    
                    if circles is not None:
                        print(f"Circle detected in {filename}")
                        circles = np.uint16(np.around(circles))
                        max_circle = max(circles[0, :], key=lambda x: x[2])
                        x, y, r = max_circle
                        # Crop the image to the region of interest
                        cropped_img = crop_image(img, x, y, r, 10)
                        masked_image = mask_image(cropped_img, r)
                        masked_image = clahe(masked_image)
                        # Save the processed image
                        output_path = os.path.join(output_dir, filename)
                        cv2.imwrite(output_path, masked_image)
                    else:
                        print(f"No circles detected in {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()