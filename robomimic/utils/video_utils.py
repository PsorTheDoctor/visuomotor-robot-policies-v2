import cv2
import numpy as np
from moviepy.editor import VideoFileClip


def cut_video(input_video, output_video):
    start_time = 7.75
    end_time = 8.5

    video_clip = VideoFileClip(input_video)
    cut_clip = video_clip.subclip(start_time, end_time)
    cut_clip.write_videofile(output_video, codec="libx264")
    video_clip.close()
    cut_clip.close()


def draw_video_shades(input_video):
    cap = cv2.VideoCapture(input_video)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        last_frame = np.array(frame, dtype=np.uint8)

    frames = np.array(frames, dtype=np.uint8)
    shadow = np.max(frames, axis=0)
    frames = np.array([shadow, last_frame], dtype=np.uint8)
    mean = np.mean(frames, axis=0)
    mean = mean.astype(np.uint8)

    cv2.imshow('', mean)
    cv2.waitKey(0)


cut_video()
draw_video_shades('out_of_vision.mp4')

