import cv2
import os

video_path = "C:\\Users\\dosiu\\OneDrive\\Desktop\\nk jain sir project\\videos_single_layer_SS\\9_SS316-Current-15-Traverse-53-voltage-6-MS-Substrate-16mm-SOD.avi"

output_folder = "C:\\Users\\dosiu\\OneDrive\\Desktop\\nk jain sir project\\Normal Frames\\9"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % int(fps) == 0:
        cv2.imwrite(f"{output_folder}/frame_{saved_count:05d}.jpg", frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Extracted {saved_count} frames (1 per second).")
