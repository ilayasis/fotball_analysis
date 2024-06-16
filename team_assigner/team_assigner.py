"""
This module contains functions and classes related to team assignment.
It uses numpy for numerical operations and scikit-learn's KMeans for clustering.
"""

import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    """
        A class to assign teams to players based on their color in video frames.

        Attributes:
            team_colors (dict): Dictionary to store team colors with team IDs as keys.
            kmeans (KMeans): KMeans clustering model for color-based team assignment.
            player_team_dict (dict): Dictionary to store the assigned team for each player.
    """
    def __init__(self):
        """
               Initializes the TeamAssigner with empty team colors, None KMeans model,
               and an empty player-team assignment dictionary.
        """
        self.team_colors = {}
        self.kmeans = None
        self.player_team_dict = {}

    def get_player_color(
            self, frame:  np.ndarray, bbox: list[float]
    ) -> list[float]:
        """
           Extract the average color of the player's bounding box in the frame.

           Args:
               frame (np.ndarray): The video frame containing the player.
               bbox (list[float]): Bounding box coordinates of the player [x1, y1, x2, y2].

           Returns:
               list[float]: The average color (BGR) of the player's bounding box.
       """
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0: int(image.shape[0] / 2), :]

        kmeans = self.get_clustering_model(top_half_image)

        # get the cluster labels
        labels = kmeans.labels_

        # reshape the labels into the original image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # get player cluster
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0],
                            clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(
            self, frame:  np.ndarray, player_detections: dict[int, dict[str, list[float]]]
    ) -> None:
        """
           Assign team colors to detected players in the frame using KMeans clustering.

           Args:
               frame (np.ndarray): The video frame containing the players.
               player_detections (dict[int, dict[str, list[float]]]): Dictionary containing player
               detections with bounding boxes.
               Each player is represented as {player_id: {'bbox': [x1, y1, x2, y2]}}.
       """

        player_colors = []

        for _, player_detection in player_detections.items():

            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(
            self, frame:  np.ndarray, player_bbox: list[float], player_id: int
    ) -> int:
        """
           Determines the team of a player based on their color.

           Args:
               frame (np.ndarray): The video frame containing the player.
               player_bbox (list[float]): Bounding box coordinates of the player [x1, y1, x2, y2].
               player_id (int): The ID of the player.

           Returns:
               int: The team ID to which the player is assigned.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]

        team_id += 1

        if player_id == 91:  # for goalkeeper
            team_id = 1

        self.player_team_dict[player_id] = team_id

        return team_id

    @staticmethod
    def get_clustering_model(
            image: np.ndarray
    ) -> KMeans:

        """
               Generates a KMeans clustering model from the given image.

               Args:
                   image (np.ndarray): The image to cluster.

               Returns:
                   KMeans: The trained KMeans clustering model.
        """
        image_2d_format = image.reshape(-1, 3)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(image_2d_format)

        return kmeans
