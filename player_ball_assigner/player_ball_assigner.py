from utils import get_center_of_bbox, measure_distance
import Config


class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = Config.MAX_DIS_BETWEEN_PLAYER_AND_BALL

    def assign_ball_to_player(
            self, players, ball_bbox
    ) -> int:
        ball_pos = get_center_of_bbox(ball_bbox)

        min_dis = 100000
        assigned_player = -1

        for player_id, player in players.item():
            player_bbox = player['bbox']

            dis_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_pos)
            dis_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_pos)
            dis = min(dis_right, dis_left)

            if dis < self.max_player_ball_distance and dis < min_dis:
                min_dis = dis
                assigned_player = player_id

        return assigned_player
