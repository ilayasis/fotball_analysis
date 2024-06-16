"""
This module contains functions and classes for assigning the ball to the nearest player
based on their bounding boxes. It uses utility functions to calculate the center of
bounding boxes and measure distances, and it gets configuration parameters for the
assignment process.

Modules:
    utils: Provides utility functions `get_center_of_bbox` and `measure_distance`.
    config: Contains configuration parameters such as `MAX_DISTANCE_BETWEEN_PLAYER_AND_BALL`.
"""

from utils import get_center_of_bbox, measure_distance
import config


class PlayerBallAssigner:
    """
       A class to assign the ball to the nearest player based on their bounding boxes.

       Attributes:
           max_player_ball_distance (float): Maximum allowed distance between
                                             a player and the ball for assignment.
    """
    def __init__(self):
        """
               Initialize PlayerBallAssigner with maximum player-ball distance from the configuration.
        """
        self.max_player_ball_distance = config.MAX_DISTANCE_BETWEEN_PLAYER_AND_BALL

    def assign_ball_to_player(
            self, players, ball_bbox
    ) -> int:
        """
              Assign the ball to the nearest player if within the maximum distance.

              Args:
                  players (dict): Dictionary of players with their IDs and bounding boxes.
                      Each player is represented as {player_id:{'bbox':[x1, y1, x2, y2]}}.
                  ball_bbox (list[int]): Bounding box coordinates of the ball [x1, y1, x2, y2].

              Returns:
                int: The ID of the assigned player or -1 if no player is within
                      the maximum distance.
        """
        ball_pos = get_center_of_bbox(ball_bbox)

        min_dis = 100000
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            dis_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_pos)
            dis_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_pos)
            dis = min(dis_right, dis_left)

            if dis < self.max_player_ball_distance and dis < min_dis:
                min_dis = dis
                assigned_player = player_id

        return assigned_player
