"""
Imports necessary modules and functions for object tracking.

Imports:
- os.path: Standard library module for interacting with filesystem paths.
- pickle: Standard library module for serializing and deserializing Python objects.
- numpy as np: Third-party library for numerical operations.
- cv2: Third-party library for computer vision tasks.
- YOLO from ultralytics: Third-party class for object detection.
- Results from ultralytics.engine.results: Third-party class for handling detection results.
- sv from supervision: Third-party module for object detection supervision.
- get_bbox_width, get_center_of_bbox from utils: First-party functions for bounding box operations.
- Config: First-party module for configuration settings.
"""

import os.path
import pickle
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results
import supervision as sv
from utils import get_bbox_width, get_center_of_bbox
import Config
import pandas as pd


class Tracker:
    """
           A class for tracking objects in video frames using a specified model.

           Attributes:
               model (YOLO): The object detection model.
               tracker (sv.ByteTrack): The object tracker.
    """
    def __init__(
            self, model_path: str
    ):
        """
            Initializes the Tracker with a given model path.

            Args:
                model_path (str): The path to the object detection model.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(
            self, frames: list[np.ndarray]
    ) -> list[Results]:
        """
           Detects objects in a list of video frames using the model.

           Args:
               frames (list): A list of video frames.

           Returns:
               list: A list of detections for each frame.
        """
        batch_size = 20
        detections = []

        for frame in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(
                frames[frame:frame + batch_size], conf=Config.CONFIDENCE_FOR_PREDICT
            )
            detections += detection_batch
        return detections

    def get_object_tracks(
            self, frames: list[np.ndarray], read_from_stub=False, stub_path=None
    ) -> dict[str, list[dict]]:
        """
          Tracks objects in video frames and returns their tracks.

          Args:
              frames (list): A list of video frames.
              read_from_stub (bool): If True, read tracks from a stub file if it exists.
              stub_path (str, optional): The path to the stub file.

          Returns:
              dict: A dictionary containing the tracks for players, referees, and the ball.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as file:
                tracks = pickle.load(file)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for index_frame, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inverse = {val: key for key, val in cls_names.items()}

            # convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind, cls_id in enumerate(detection_supervision.class_id):
                if cls_names[cls_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inverse["player"]

            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inverse['player']:
                    tracks["players"][index_frame][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inverse['referee']:
                    tracks["referees"][index_frame][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inverse['ball']:
                    tracks["ball"][index_frame][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as file:
                pickle.dump(tracks, file)

        return tracks

    def draw_annotations(
            self, video_frames: list[np.array], tracks: dict[str, list[dict]], team_ball_control: list[int]
    ) -> list[np.array]:
        """
            Draws annotations on video frames based on tracking information.

            Args:
                video_frames (list): A list of video frames to be annotated.
                tracks (dict): A dictionary containing tracking information with keys
                'players', 'referees', and 'ball'.

            Returns:
                list: A list of video frames with annotations.
        """
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw Referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames

    @staticmethod
    def draw_team_ball_control(frame, frame_num, team_ball_control):
        """
        Draws team ball control statistics on the given frame.

        Args:
            frame (np.ndarray): The video frame to draw on.
            frame_num (int): The current frame number.
            team_ball_control (np.ndarray): Array indicating ball control by frame (1 for Team 1, 2 for Team 2).

        Returns:
            np.ndarray: The frame with the ball control statistics drawn on it.
        """
        # Draw a semi-transparent rectangle in the right bottom corner
        overlay = frame.copy()
        cv2.rectangle(overlay, (1100, 850), (1800, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Calculate ball control statistics
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_frames_count = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_frames_count = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        team1 = team_1_frames_count / (team_1_frames_count + team_2_frames_count)
        team2 = team_2_frames_count / (team_1_frames_count + team_2_frames_count)

        # Draw the text on the frame
        cv2.putText(frame, f"Possession Time - Team 1: {team1 * 100:.2f}%", (1150, 900),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Possession Time - Team 2: {team2 * 100:.2f}%", (1150, 950),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    @staticmethod
    def interpolate_ball_positions(ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate where the missing balls
        df_ball_positions = df_ball_positions.interpolate()

        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {'bbox': val}} for val in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    @staticmethod
    def draw_triangle(
            frame: np.ndarray, bbox: list[int], color: tuple[int, int, int]
    ) -> np.ndarray:
        """
           Draws a triangle on a frame based on the provided bounding box coordinates and color.

           Args:
               frame (np.ndarray): The input image frame where the triangle will be drawn.
               bbox (list[int]): List of integers representing the bounding box coordinates
                  [x1, y1, x2, y2].
               color (tuple[int, int, int]):: Tuple of integers representing the RGB color
                              values for the triangle.

           Returns:
               np.ndarray: The modified frame with the triangle drawn on it.
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    @staticmethod
    def draw_ellipse(
            frame: np.ndarray, bbox: list[int], color: tuple[int, int, int], track_id: int = None
    ) -> np.ndarray:
        """
            Draws an ellipse on the frame at the location specified by the bounding box.
            Optionally, a track ID can be added inside a rectangle near the ellipse.

            Args:
                frame (np.ndarray): The video frame on which to draw the ellipse.
                bbox (List[int]): The bounding box coordinates [x1, y1, x2, y2] where the
                    ellipse will be drawn.
                color (Tuple[int, int, int]): The color of the ellipse in BGR format.
                track_id (int, optional): The ID of the tracked object to be displayed.
                    Default is None.

            Returns:
                np.ndarray: The frame with the ellipse and optional track ID drawn on it.
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame


