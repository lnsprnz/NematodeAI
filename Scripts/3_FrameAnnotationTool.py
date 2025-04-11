
"""
Frame Annotation Tool for Image Processing
This script provides a graphical interface for manually annotating points in images.
Users can click on specific locations in images to mark points with different labels (0, 1, 2).
The clicked coordinates along with their corresponding labels are saved to a CSV file.
Lables:
- 0: Bakground (B)
- 1: Juvenils (J)
- 2: Dauer Juvenils (DJ)
- 3: Adult (A)
Features:
- Support for multiple image formats (PNG, JPG, JPEG, BMP)
- Interactive point marking with different colors per label
- Backspace to undo last point
- ESC key to exit the annotation process
- Automatic saving of annotations to CSV
Usage:
1. Set IMAGE_DIR to the folder containing your images
2. Run the script
3. Click points on images for each label category
4. Close window when done with current label
5. Press ESC to exit completely
6. Results are saved to '_clicked_points.csv' in the image directory
Dependencies:
    - matplotlib
    - csv
    - os
    - pathlib
"""
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
from pathlib import Path

# Configuration
IMAGE_DIR = 'C:/Users/linus/NematodeAI/NematodeAI/Data/C0105.MP4_processedFrames/C0105_cropped'  # Change this to your image folder path
OUTPUT_CSV = IMAGE_DIR + '_clicked_points.csv'

# Supported image formats
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp']

clicked_data = []
exit_requested = False

def onclick(event, img_name, label):
    if event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata
        print(f"Clicked on {img_name} (Label {label}): ({x:.2f}, {y:.2f})")
        clicked_data.append((img_name, label, x, y))
        plt.plot(x, y, 'rx' if label == 0 else 'bx')
        plt.draw()

def onkey(event):
    global exit_requested
    if event.key == 'escape':
        print("Escape key pressed. Exiting labeling process.")
        exit_requested = True
        plt.close()

def clear_last_clicked_data(event):
    global exit_requested
    if len(clicked_data) > 0 & event.key == 'backspace':
        clicked_data.pop()
        plt.cla()
        for row in clicked_data:
            label = row[1]
            x, y = row[2], row[3]
            plt.plot(x, y, 'rx' if label == 0 else 'bx')
        plt.draw()

def process_images(image_dir):
    global exit_requested
    image_files = [f for f in sorted(Path(image_dir).iterdir()) if f.suffix.lower() in IMAGE_EXTENSIONS]
    for img_path in image_files:
        if exit_requested:
            break
        img_name = img_path.name
        for label in [0, 1, 2]:  # Only two sets/labels: 0 and 1
            if exit_requested:
                break
            img = mpimg.imread(img_path)
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.set_title(f"Click points for Label {label} on: {img_name}\nPress ESC to exit\nClose window when done")

            cid_click = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, img_name, label))
            cid_key = fig.canvas.mpl_connect('key_press_event', onkey)
            cid_key = fig.canvas.mpl_connect('key_press_event', lambda event: clear_last_clicked_data(event))
            plt.show()
            fig.canvas.mpl_disconnect(cid_click)
            fig.canvas.mpl_disconnect(cid_key)

def save_points_to_csv(output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label', 'x', 'y'])
        for row in clicked_data:
            writer.writerow(row)
    print(f"Saved clicked points to {output_csv}")

if __name__ == '__main__':
    process_images(IMAGE_DIR)
    save_points_to_csv(OUTPUT_CSV)
