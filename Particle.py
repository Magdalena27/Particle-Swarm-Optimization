"""
@author: Magdalena Jaśkiewicz 233968
"""


class Particle:
    def __init__(self, position_vec, function, initial_velocity_vec):
        """
        Particle swarm algorithm particle
        :param position_vec: position vector
        :param function: function to count score
        :param initial_velocity_vec: velocity vector
        """
        self.position_vec = position_vec
        self.function = function
        self.velocity_vec = initial_velocity_vec
        self.score = self.count_score()
        self.historical_best_position = position_vec
        self.historical_best_score = self.count_score()

    def count_score(self):
        f = self.function
        pos = self.position_vec
        score = f(pos)
        self.score = score
        return score

