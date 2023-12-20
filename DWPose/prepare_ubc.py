import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from dwpose_utils import DWposeDetector
from decord import VideoReader
from decord import cpu
from pathlib import Path

torch.set_grad_enabled(False)

def process_video(dwprocessor, video_path, output_video_path, detect_resolution):
    video_path = Path(video_path)
    complete = video_path.parent/f'{video_path.stem}.complete'
    if os.path.exists(complete): return
    video_path = str(video_path)
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()

    first_frame = vr[0].asnumpy()
    height, width, _ = first_frame.shape
    size = (width, height)
    
    # 创建视频写入器，使用原视频的帧率
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for idx in tqdm(range(len(vr)), desc=f"Processing {os.path.basename(video_path)}"):
        frame = vr[idx].asnumpy()
        detected_pose = process(dwprocessor, frame, detect_resolution)
        video_writer.write(detected_pose)

    video_writer.release()
    Path(complete).touch()

def process(dwprocessor, input_image, detect_resolution):
    if not isinstance(dwprocessor, DWposeDetector):
        dwprocessor = DWposeDetector()
    with torch.no_grad():
        detected_map = dwprocessor(input_image)
    return detected_map

nphases = int(os.environ['NPHASES'])
phase = int(os.environ['PHASE'])
dwprocessor = DWposeDetector()
print('loaded dwprocessor')
root = "/home/ubuntu/data/ubc_fasion/"
dataset_folder = root
sub_folders = ['train', 'test']
detect_resolution = 768

for sub_folder in sub_folders:
    path = os.path.join(dataset_folder, sub_folder)
    output_folder = os.path.join(dataset_folder, sub_folder + '_dwpose')
    print(output_folder)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for ix, video_name in tqdm(enumerate(os.listdir(path)), total=len(os.listdir(path)), desc=f"Processing {sub_folder}"):
        if ix%nphases != phase:
            continue
        video_path = os.path.join(path, video_name)
        output_video_path = os.path.join(output_folder, video_name.split('.')[0] + '.mp4')
        process_video(dwprocessor, video_path, output_video_path, detect_resolution)

"""
ubc.sh
"""