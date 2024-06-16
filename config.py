"""
Configuration settings for the application.

Module-level constants:
- CONFIDENCE_FOR_PREDICT: Threshold confidence level for predictions.
- MAX_DISTANCE_BETWEEN_PLAYER_AND_BALL: Maximum allowable distance between
                                        player and ball in pixels.
"""

# Model Configuration
CONFIDENCE_FOR_PREDICT = 0.1

# Player-Ball Assignment Configuration
MAX_DISTANCE_BETWEEN_PLAYER_AND_BALL = 70  # Maximum allowable distance in pixels
