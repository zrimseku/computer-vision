import numpy as np
import cv2

from ex2_utils import Tracker, extract_histogram, get_patch, create_epanechnik_kernel, backproject_histogram
from ms_modeseeking import find_mode


class MeanShiftTracker(Tracker):

    def initialize(self, image, region):

        self.position = np.array([int(region[0] + region[2] / 2), int(region[1] + region[3] / 2)])
        w_odd = np.floor(region[2]*self.parameters.enlarge_factor / 2) * 2 + 1
        h_odd = np.floor(region[3]*self.parameters.enlarge_factor / 2) * 2 + 1
        self.size = np.array([w_odd, h_odd])
        w_odd_org = np.floor(region[2] / 2) * 2 + 1
        h_odd_org = np.floor(region[3] / 2) * 2 + 1
        self.original_size = np.array([w_odd_org, h_odd_org])

        patch, _ = get_patch(image, self.position, self.size)
        patch = patch.astype('float')
        self.kernel = create_epanechnik_kernel(*self.size, self.parameters.sigma)
        for c in range(3):
            patch[:, :, c] *= self.kernel
        self.q = extract_histogram(patch, self.parameters.nbins)


    def track(self, image):

        step = 0
        ms_x = 1
        ms_y = 1
        # while abs(ms_x) + abs(ms_y) >= 1 and step < self.parameters.max_steps:
        while step < self.parameters.max_steps:
            patch, _ = get_patch(image, self.position, self.size)
            patch = patch.astype('float')
            for c in range(3):
                patch[:, :, c] *= self.kernel

            self.p = extract_histogram(patch, self.parameters.nbins)

            v = np.sqrt(self.q / (self.p + self.parameters.eps))
            weights = backproject_histogram(patch, v, self.parameters.nbins)

            sum_weights = np.sum(weights)

            span_x = np.arange(-int((self.size[0] - 1) / 2), int((self.size[0] + 1) / 2))
            span_y = np.arange(-int((self.size[1] - 1) / 2), int((self.size[1] + 1) / 2))
            x_i, y_i = np.meshgrid(span_x, span_y)

            ms_x = np.sum(x_i * weights) / sum_weights
            ms_y = np.sum(y_i * weights) / sum_weights

            self.position = self.position.astype('float')
            self.position += np.array([ms_x, ms_y])

            step += 1

        self.q = (1 - self.parameters.alpha) * self.q + self.parameters.alpha * self.p

        test = [self.position[0] - self.original_size[0]/2, self.position[1] - self.original_size[1]/2, self.original_size[0], self.original_size[1]]

        return test



class MSParams():

    def __init__(self, sigma=1, nbins=16, eps=0.01, enlarge_factor=1, alpha=0, max_steps=30):
        self.sigma = sigma
        self.nbins = nbins
        self.eps = eps
        self.enlarge_factor = enlarge_factor
        self.alpha = alpha
        self.max_steps = max_steps

