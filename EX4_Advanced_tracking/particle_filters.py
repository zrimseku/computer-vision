import numpy as np
import cv2
import sympy as sp

from ex2_utils import get_patch, Tracker, create_epanechnik_kernel, extract_histogram
from ex4_utils import sample_gauss
from kalman_filter import derive_matrices


class ParticleFilterTracker(Tracker):

    def name(self):
        return "Particle Filter Tracker"

    def initialize(self, image, region):

        self.position = np.array([int(region[0] + region[2] / 2), int(region[1] + region[3] / 2)])
        w_odd = np.floor(region[2] / 2) * 2 + 1
        h_odd = np.floor(region[3] / 2) * 2 + 1
        self.size = np.array([w_odd, h_odd])

        self.kernel = create_epanechnik_kernel(*self.size, self.parameters.sigma_kernel)

        self.template = self.hist_from_patch(image, self.position)

        # NCV
        par = (1, self.parameters.q * np.min(self.size // 2), 1)
        self.Fi, self.Q, _, _ = derive_matrices(self.parameters.model_name, par)
        self.particles = np.zeros((self.parameters.n_particles, self.Q.shape[0]))
        self.particles[:, :2] = sample_gauss(self.position, self.Q[:2, :2], self.parameters.n_particles)
        self.weights = np.ones((self.parameters.n_particles, 1))

    def track(self, image):
        # resample particles
        weights_norm = self.weights / np.sum(self.weights)
        weights_cumsumed = np.cumsum(weights_norm)
        rand_samples = np.random.rand(self.parameters.n_particles, 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsumed)
        new_particles = self.particles[sampled_idxs.flatten(), :]

        # move particles with motion model
        self.particles = (self.Fi @ new_particles.T).T + sample_gauss(np.zeros(self.Q.shape[0]), self.Q,
                                                                      self.parameters.n_particles)

        # calculate new weights
        for i in range(self.parameters.n_particles):
            par_x = self.particles[i, 0]
            par_y = self.particles[i, 1]
            if par_x > image.shape[1] or par_x < 0 or par_y < 0 or par_y > image.shape[0]:  # if it is outside image
                self.weights[i] = 0
            else:
                hist = self.hist_from_patch(image, self.particles[i, :2])
                hist = hist / np.sum(hist)
                temp = self.template / np.sum(self.template)
                dist = np.linalg.norm(np.sqrt(hist) - np.sqrt(temp)) / np.sqrt(2)
                self.weights[i] = np.exp(-(dist**2 / self.parameters.sigma_probs**2) / 2)

        # calculate new position
        weights_norm = self.weights
        if np.sum(self.weights) != 0:
            weights_norm /= np.sum(self.weights)
        pos = weights_norm.T @ self.particles[:, :2]
        self.position = pos[0]

        # update template
        new_temp = self.hist_from_patch(image, self.position)
        self.template = self.parameters.alpha * new_temp + (1 - self.parameters.alpha) * self.template

        new_rec = [self.position[0] - self.size[0] / 2, self.position[1] - self.size[1] / 2, self.size[0], self.size[1]]

        return new_rec

    def hist_from_patch(self, image, position):
        patch, _ = get_patch(image, position, self.size)
        patch = patch.astype('float')
        if patch.shape[:2] == self.kernel.shape:
            for c in range(3):
                patch[:, :, c] *= self.kernel
        patch.astype('uint8')
        return extract_histogram(patch, self.parameters.nbins)

class PFParams():

    def __init__(self, n_particles=30, q=5, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='ncv'):
        self.n_particles = n_particles
        self.q = q
        self.sigma_probs = sigma_probs
        self.model_name = model_name
        self.sigma_kernel = sigma_kernel
        self.nbins = nbins
        self.alpha = alpha


# n_particles=70, q=15, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='ncv'
# _______________________________________
# On all sequences:
# All fails: 63
# Average speed: 56.04812408549122
# Average IOU: 0.5321791140310608

# n_particles=70, q=10, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='ncv'
# _______________________________________
# On all sequences:
# All fails: 51
# Average speed: 56.48483640857871
# Average IOU: 0.5406709246914388

# n_particles=70, q=5, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='ncv')
# _______________________________________
# On all sequences:
# All fails: 42
# Average speed: 55.26317661479613
# Average IOU: 0.5352002106760971

# n_particles=70, q=1, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='ncv'
# _______________________________________
# On all sequences:
# All fails: 32
# Average speed: 56.56
# Average IOU: 0.5203

# n_particles=70, q=0.5, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='ncv')
# _______________________________________
# On all sequences:
# All fails: 29
# Average speed: 55.118809892358286
# Average IOU: 0.5220171237583396

# n_particles=50, q=0.5, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='ncv')
# _______________________________________
# On all sequences:
# All fails: 37
# Average speed: 76.03353166337071
# Average IOU: 0.508476951564517

# n_particles=100, q=0.5, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='ncv')
# _______________________________________
# On all sequences:
# All fails: 27
# Average speed: 39.62288824373195
# Average IOU: 0.5236953275627355

# n_particles=120, q=0.5, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='ncv')
# _______________________________________
# On all sequences:
# All fails: 27
# Average speed: 32.97341832354467
# Average IOU: 0.5123771547367029

# n_particles=70, q=0.1, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='ncv')
# _______________________________________
# On all sequences:
# All fails: 42
# Average speed: 56.21292808587601
# Average IOU: 0.4839612316164131


# NCA
# n_particles=70, q=10, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='nca')
# _______________________________________
# On all sequences:
# All fails: 283
# Average speed: 53.58066733613666
# Average IOU: 0.49098443839517497

# n_particles=70, q=5, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='nca')
# _______________________________________
# On all sequences:
# All fails: 207
# Average speed: 51.76741115645182
# Average IOU: 0.49805280159029125

# n_particles=70, q=1, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='nca')
# _______________________________________
# On all sequences:
# All fails: 124
# Average speed: 52.859253245303044
# Average IOU: 0.5087049029244789

# n_particles=70, q=0.5, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='nca'
# _______________________________________
# On all sequences:
# All fails: 128
# Average speed: 52.33685758009373
# Average IOU: 0.5005989267691078


# n_particles=100, q=0.5, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='nca'):
# _______________________________________
# On all sequences:
# All fails: 69
# Average speed: 38.443464515495755
# Average IOU: 0.5161329657814667

# n_particles=70, q=0.1, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='nca')
# _______________________________________
# On all sequences:
# All fails: 124
# Average speed: 50.92385653597092
# Average IOU: 0.505191777560543


# RW
# n_particles=70, q=0.1, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='rw')
# _______________________________________
# On all sequences:
# All fails: 80
# Average speed: 55.73605238351457
# Average IOU: 0.4071963345144108

#  n_particles=70, q=0.5, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='rw'):
# _______________________________________
# On all sequences:
# All fails: 44
# Average speed: 56.709236294172285
# Average IOU: 0.46479733593066835

# n_particles=70, q=1, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='rw')
# _______________________________________
# On all sequences:
# All fails: 34
# Average speed: 53.210479333913874
# Average IOU: 0.4875736533119122

# n_particles=70, q=5, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='rw')
# _______________________________________
# On all sequences:
# All fails: 26
# Average speed: 56.06155817440742
# Average IOU: 0.5270395136481201

#  n_particles=70, q=10, sigma_kernel=1, nbins=16, alpha=0.02, model_name='rw')
# _______________________________________
# On all sequences:
# All fails: 32
# Average speed: 55.00408562574572
# Average IOU: 0.5345575370280771

# n_particles=70, q=10, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='rw')
# _______________________________________
# On all sequences:
# All fails: 31
# Average speed: 52.99460209062543
# Average IOU: 0.5271765149680719

# n_particles=70, q=20, sigma_kernel=1, sigma_probs=0.1, nbins=16, alpha=0.02, model_name='rw')
# _______________________________________
# On all sequences:
# All fails: 49
# Average speed: 57.390762879891305
# Average IOU: 0.5358842175327992

