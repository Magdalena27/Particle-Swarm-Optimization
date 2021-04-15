"""
@author: Magdalena Jaśkiewicz 233968
"""
import numpy as np
import random

from MinMax import MinMax
from Particle import Particle


class PSO:
    def __init__(self, adaptation_function, number_of_particles, omega, fi1, fi2, n_dim, bounds, min_or_max,
                 number_of_iterations):
        """
        Particle swarm algorithm
        :param adaptation_function: the goal function
        :param number_of_particles: int
        :param omega: float - inertia parameter
        :param fi1: float - the parameter of the molecule's confidence in its own knowledge
        :param fi2: float - the parameter of the molecule's confidence in the swarm knowledge
        :param n_dim: int - number of dimensions
        :param bounds: vector of tuples - bounds for every dimension
        :param min_or_max: MinMax enum - minimum or maximum
        :param number_of_iterations: int
        """
        self.adaptation_function = adaptation_function
        self.number_of_particles = number_of_particles
        self.omega = omega
        self.fi1 = fi1
        self.fi2 = fi2
        self.n_dim = n_dim
        self.bounds = bounds
        self.min_or_max = min_or_max
        self.number_of_iterations = number_of_iterations
        self.best_particle_position = np.zeros(n_dim)
        self.best_particle_score = adaptation_function(self.best_particle_position)
        self.swarm = []

    def __initialize_swarm(self):
        swarm = []
        for particle in range(self.number_of_particles):
            position_vec = self.__init_position_vec()
            function = self.adaptation_function
            initial_velocity_vec = self.__init_velocity_vec()
            swarm.append(Particle(position_vec, function, initial_velocity_vec))
        self.swarm = swarm

    def __init_position_vec(self):
        vec = np.zeros(self.n_dim)
        for dim in range(self.n_dim):
            vec[dim] = random.uniform(self.bounds[dim][0], self.bounds[dim][1])
        return vec

    def __init_velocity_vec(self):
        vec = np.zeros(self.n_dim)
        for dim in range(self.n_dim):
            vec[dim] = random.uniform(0.0, self.bounds[dim][1]/2.0)
        return vec

    def __move_particles(self):
        swarm = self.swarm
        for particle in swarm:
            particle.position_vec = self.__check_bounds(particle.position_vec + particle.velocity_vec, particle)
        self.swarm = swarm

    def __check_bounds(self, position_vec, particle):
        for dim in range(self.n_dim):
            if position_vec[dim] < self.bounds[dim][0] or position_vec[dim] > self.bounds[dim][1]:
                position_vec[dim] = particle.position_vec[dim] - particle.velocity_vec[dim]
        return position_vec

    def __score_particles(self):
        swarm = self.swarm
        for particle in swarm:
            particle.count_score()
            if self.min_or_max == MinMax.MAX:
                if particle.historical_best_score < particle.score:
                    particle.historical_best_score = particle.score
                    particle.historical_best_position = particle.position_vec
                if self.best_particle_score < particle.score:
                    self.best_particle_score = particle.score
                    self.best_particle_position = particle.position_vec
            elif self.min_or_max == MinMax.MIN:
                if particle.historical_best_score > particle.score:
                    particle.historical_best_score = particle.score
                    particle.historical_best_position = particle.position_vec
                if self.best_particle_score > particle.score:
                    self.best_particle_score = particle.score
                    self.best_particle_position = particle.position_vec
        self.swarm = swarm

    def __actualize_velocity_of_particles(self):
        swarm = self.swarm
        omega = self.omega
        fi1 = self.fi1
        fi2 = self.fi2
        r1 = self.__draw_r()
        r2 = self.__draw_r()

        for particle in swarm:
            new_velocity = omega * particle.velocity_vec + fi1 * r1 * \
                           (particle.position_vec - particle.historical_best_position) + fi2 * r2 * \
                           (particle.position_vec - self.best_particle_position)
            particle.velocity_vec = new_velocity
        self.swarm = swarm

    def __draw_r(self):
        vec = np.zeros(self.n_dim)
        for dim in range(self.n_dim):
            vec[dim] = random.uniform(0.0, self.bounds[dim][1] / 2.0)
        return vec

    def perform_calculations(self):
        self.__initialize_swarm()
        for iter in range(self.number_of_iterations):
            self.__move_particles()
            self.__score_particles()
            self.__actualize_velocity_of_particles()

    def get_result(self):
        return self.best_particle_position, self.best_particle_score

