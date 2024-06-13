"""
Video processing utilities using OpenCV (cv2) and NumPy (np).

Functions:
- read_video(video_path: str) -> list[np.ndarray]: Reads a video file into frames.
- save_video(output_video_frames: list, output_video_path: str) -> None: Saves frames as a video.

Dependencies: cv2, np
"""

import cv2
import numpy as np


def read_video(
        video_path: str
) -> list[np.ndarray]:
    """
      Reads a video from the specified file path and returns a list of frames.

      Args:
          video_path (str): The path to the video file.

      Returns:
          list: A list of frames read from the video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def save_video(
        output_video_frames: list, output_video_path: str
) -> None:
    """
       Saves a list of frames to a video file at the specified path.

       Args:
           output_video_frames (list): A list of frames to be saved as a video.
           output_video_path (str): The path where the output video will be saved.

       Returns:
           None
    """
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps,
                          (output_video_frames[0].shape[1], output_video_frames[0].shape[0])
                          )
    for frame in output_video_frames:
        out.write(frame)
    out.release()
