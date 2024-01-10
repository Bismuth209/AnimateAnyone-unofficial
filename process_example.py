import cv2
import os
import shutil

def extract_first_frame(input_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(input_file)

    if not cap.isOpened():
        print(f"Error opening video file: {input_file}")
        return

    ret, frame = cap.read()

    if ret:
        output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + ".png")
        cv2.imwrite(output_file, frame)
        print(f"First frame saved to {output_file}")
    else:
        print(f"Error reading the first frame from {input_file}")

    cap.release()

def copy_video(input_file, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    target_file = os.path.join(target_folder, os.path.basename(input_file))

    shutil.copy(input_file, target_file)
    print(f"File copied to {target_file}")


output_folder = "/home/ubuntu/data/ubc_fashion//source_image"
copy_target_folder = "/home/ubuntu/data/ubc_fashion/driving/dwpose" 

video_name = "A1-Lv00GAzS.mp4"

input_video_1 = "/home/ubuntu/data/ubc_fashion/test/" + video_name
input_video_2 = "/home/ubuntu/data/ubc_fashion/test_dwpose/" + video_name

copy_video(input_video_2, copy_target_folder)
extract_first_frame(input_video_1, output_folder)
