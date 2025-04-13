import os
import cv2
import numpy as np

# -------------------------------
# Helper Functions
# -------------------------------

def watershed(img):
    """
    Apply watershed algorithm to an image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    return sure_bg

def houghCircle(img, param1, param2, minRadius, maxRadius):
    """
    Apply Hough Circle Transform to an image. Returns array of circles detected [centerx, centery, radius].
    """
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
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
    video_path = 'NematodeAI/Preprocessing/C0098.MP4'
    cap = cv2.VideoCapture(video_path)
    
    # Read first frame and detect circle
    ret, first_frame = cap.read()
    if not ret:
        return

    # Apply watershed algorithm
    watershed_img = watershed(first_frame)
    # Apply Hough Circle Transform
    circles = houghCircle(watershed_img, 40, 20, 990, 1030)
    if circles is not None:
        print("circle detected")
        circles = np.uint16(np.around(circles))
        max_circle = max(circles[0, :], key=lambda x: x[2])
        x, y, r = max_circle
    else:
        print("No circle detected")
        return

    # Get dimensions for output video (1024x1024)
    output_size = (1024, 1024)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(video_path).replace('Preprocessing', 'Processed')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output path
    output_path = os.path.join(output_dir, os.path.basename(video_path).replace('.MP4', '_cropped.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, output_size)

    # Process first frame
    cropped_img = crop_image(first_frame, x, y, r, 10)
    masked_image = mask_image(cropped_img, r)
    resized_image = cv2.resize(masked_image, output_size)
    out.write(resized_image)

    # Process remaining frames
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cropped_img = crop_image(frame, x, y, r, 10)
        masked_image = mask_image(cropped_img, r)
        resized_image = cv2.resize(masked_image, output_size)
        out.write(resized_image)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

    print(f"Finished processing {frame_count} frames")
    print(f"Saved video to: {output_path}")
    cap.release()
    out.release()

if __name__ == "__main__":
    main()