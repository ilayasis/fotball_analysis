"""
Module for bounding box utilities.

Provides functions to calculate properties of bounding boxes.

Functions:
- get_center_of_bbox(bbox: list[int])
    -> tuple[int, int]: Calculates the center coordinates of a bounding box.

Usage:
Import the function into your script using:
    from .bbox_utils import get_center_of_bbox
"""
from typing import Union


def get_center_of_bbox(bbox: list[int]) -> tuple[int, int]:
    """Calculate the center coordinates of a bounding box.

    Args:
        bbox (list[int]): List of integers representing the bounding box coordinates
            [x1, y1, x2, y2].

    Returns:
        tuple[int, int]: Tuple containing the x and y coordinates of the center of the bbox.
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox: list[int]) -> int:
    """Calculate the width of a bounding box.

    Args:
        bbox (list[int]): List of integers representing the bounding box coordinates
            [x1, y1, x2, y2].

    Returns:
        int: Width of the bounding box.
    """
    return bbox[2] - bbox[0]


def measure_distance(
    point1: tuple[Union[int, float]], point2: tuple[Union[int, float]]
) -> float:
    """
        Calculate the Euclidean distance between two points in a 2-dimensional space.

        Args:
        - point1 (tuple): A tuple representing the coordinates of the first point (x1, y1).
        - point2 (tuple): A tuple representing the coordinates of the second point (x2, y2).

        Returns:
        - float: The Euclidean distance between point1 and point2.
    """
    return ((point1[0] - point2[0]) ** 2 + (point1[1]-point2[1]) ** 2) ** 0.5
