"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import norm as gaussian

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import sample_from_odometry
from tools.task import wrap_angle

import scipy

# particle filter (PF) approximate the distribution of posterior with a set of random state samples, belongs to this posterior
# this representation is approximate, but non-parametric - which is good because the distribution can be fidducult to parametrize (two hills for example)
# such a representation can model non-linear transformations


class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, beta, num_particles, global_localization):
        super(PF, self).__init__(initial_state, alphas, beta)
        # TODO add here specific class variables for the PF





        self.num_particles = num_particles
        self.global_localization = global_localization # if true - uniformly distribute particles over the field (x_min:x_max, y_min:y_max) with numpy.random.uniform

        self.weights = np.empty(self.num_particles)
        self.weights.fill(1./self.num_particles)

        # create a particle set, M - number of particles
        # particle_set = np.array([ [x,w],[x,w], ... ])


        # print('initial state',self._state.mu)
        particle_set = np.empty(self._state.mu.shape)
        for _ in range(self.num_particles-1):
            particle_set = np.append(particle_set, self._state.mu,axis=0)
        self.particle_set = particle_set.reshape(self.num_particles,3)
        # print('particle_set',self.particle_set)





        # Uniformly distribute the particles around the initial estimate at the beginning of the simulation.
        # with equal weights = 1







    def predict(self, u):
        # TODO Implement here the PF, perdiction part
        self._state_bar.mu = self.mu
        self._state_bar.Sigma = self.Sigma

        # for particle in self.particle_set:
        #     print(sample_from_odometry(particle, u, self._alphas))
        
        for idx,particle in enumerate(self.particle_set):
            self.particle_set[idx] = sample_from_odometry(particle, u, self._alphas)
        # print('particle_set', self.particle_set)

        self._state_bar = get_gaussian_statistics(self.particle_set)

    def update(self, z):
        # TODO implement correction step
        self._state.mu = self._state_bar.mu
        self._state.Sigma = self._state_bar.Sigma

        # UPDATE WEIGHTS. incorporate measurements into the weights. 
        stand_dev = np.sqrt(self._Q) # self._Q - observation noise var
        normal_r_v = scipy.stats.norm(scale=stand_dev) # randon variable distribution
        for idx,weight in enumerate(self.weights):
            innovation = z[0] - get_observation(self.particle_set[idx], z[1])[0]  # how far the observation from the expected measurement
            weight_update = normal_r_v.pdf(innovation) # pick the value of the pdf at innovation with zero mean
            # print('\ninnovation', innovation)
            # print('pdf value: ', weight_update)
            self.weights[idx] = weight_update # In the case of multiple observations at the same time: self.weights[idx] *= weight_update
        self.weights /= sum(self.weights) # to make them normalized (the sum equal to 1), stability

        # RESAMPLING
        # print('\nweights\n',self.weights)
        cumulative_sum = np.cumsum(self.weights) # array, distance between values bigger if the weight is bigger -> more chanses to get into the interval using random value generator
        cumulative_sum[-1] = 1. # to avoid errors 
        # print('cumulative_sum\n',cumulative_sum)
        indexes = np.searchsorted(cumulative_sum, np.random.random(self.num_particles))
        # print('indexes\n',indexes)
        # resample according to indexes (pick up only the best particles with highest weights)
        self.particle_set = self.particle_set[indexes]
        self.weights = self.weights[indexes]
        self.weights /= np.sum(self.weights) # to make them normalized (the sum equal to 1)


        self._state =  get_gaussian_statistics(self.particle_set)