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
