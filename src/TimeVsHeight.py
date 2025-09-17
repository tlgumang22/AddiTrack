import cv2
import numpy as np
import pandas as pd
import os

def extract_line_length(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask = mask_red | mask_yellow

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 1:
        return None

    cnt = max(contours, key=cv2.contourArea)
    pts = cnt.reshape(-1, 2)
    if len(pts) < 2:
        return None

    max_dist = 0
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            dist = np.linalg.norm(pts[i] - pts[j])
            if dist > max_dist:
                max_dist = dist

    return max_dist


def process_folder(folder_path, first_height_real, voltage, current, feedrate):
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg','.png','.jpeg'))])

    if not image_files:
        print(f"No images found in {folder_path}")
        return pd.DataFrame()

    first_img_path = os.path.join(folder_path, image_files[0])
    first_pixel_height = extract_line_length(first_img_path)
    if first_pixel_height is None:
        print(f"Could not detect line in first image of {folder_path}")
        return pd.DataFrame()

    scale = first_height_real / first_pixel_height

    data = []
    time_sec = 1
    for fname in image_files:
        img_path = os.path.join(folder_path, fname)
        pixel_height = extract_line_length(img_path)
        if pixel_height is None:
            continue
        real_height = pixel_height * scale
        length = (feedrate / 60.0) * time_sec  # mm

        data.append([voltage, current, feedrate, length, time_sec, real_height])
        time_sec += 1

    df = pd.DataFrame(data, columns=["Voltage (V)", "Current (A)", "FeedRate (mm/min)",
                                     "Length (mm)", "Time (s)", "Height (mm)"])
    return df


def process_multiple_folders(folder_info_list, output_excel):
    all_data = []
    for folder_path, first_height_real, voltage, current, feedrate in folder_info_list:
        df = process_folder(folder_path, first_height_real, voltage, current, feedrate)
        if not df.empty:
            all_data.append(df)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_excel(output_excel, index=False)
        print(f"Saved results for all folders to {output_excel}")
    else:
        print("No data processed.")


# ================== Example usage ==================
folders_to_process = [
    # first real height , voltage , current , feed rate
    (r"C:\\Users\\dosiu\\OneDrive\\Desktop\\nk jain sir project\\Annotate Frames\\2", 1.9, 6, 14.5, 47),
    (r"C:\\Users\\dosiu\\OneDrive\\Desktop\\nk jain sir project\\Annotate Frames\\3", 1.8, 9, 14.5, 53),
    (r"C:\\Users\\dosiu\\OneDrive\\Desktop\\nk jain sir project\\Annotate Frames\\4", 1.8, 7, 14, 47),
    (r"C:\\Users\\dosiu\\OneDrive\\Desktop\\nk jain sir project\\Annotate Frames\\5", 1.9, 8, 14, 50),
    (r"C:\\Users\\dosiu\\OneDrive\\Desktop\\nk jain sir project\\Annotate Frames\\6", 1.8, 8, 14, 53),
    (r"C:\\Users\\dosiu\\OneDrive\\Desktop\\nk jain sir project\\Annotate Frames\\7", 1.9, 8, 15, 47),
    (r"C:\\Users\\dosiu\\OneDrive\\Desktop\\nk jain sir project\\Annotate Frames\\8", 1.8, 8, 15, 50),
    (r"C:\\Users\\dosiu\\OneDrive\\Desktop\\nk jain sir project\\Annotate Frames\\9", 1.9, 8, 15, 53),
]

process_multiple_folders(
    folders_to_process,
    r"C:\Users\dosiu\OneDrive\Desktop\python_vscode\BTP\data\all_folders_results.xlsx"
)
