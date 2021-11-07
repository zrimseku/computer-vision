import numpy as np
import cv2

from ex2_utils import get_patch  # , Tracker
from ex3_utils import create_cosine_window, create_gauss_peak

from pytracking_toolkit_lite.utils.tracker import Tracker


class CorrelationFilterTracker(Tracker):

    def __init__(self):
        super().__init__()
        self.parameters = CFParams()

    def name(self):
        return "Correlation Filter Tracker"

    def initialize(self, image, region):

        self.position = np.array([int(region[0] + region[2] / 2), int(region[1] + region[3] / 2)])
        w_odd = np.floor(region[2] / 2) * 2 + 1
        h_odd = np.floor(region[3] / 2) * 2 + 1
        self.original_size = np.array([w_odd, h_odd])

        w_odd = np.floor(region[2] * self.parameters.enlarge_factor / 2) * 2 + 1
        h_odd = np.floor(region[3] * self.parameters.enlarge_factor / 2) * 2 + 1
        self.size = np.array([w_odd, h_odd])

        self.G = np.fft.fft2(create_gauss_peak(self.size, self.parameters.sigma))
        self.cos_window = create_cosine_window((int(self.size[0]), int(self.size[1])))

        patch, _ = get_patch(image, self.position, self.size)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch_fft = np.fft.fft2(patch * self.cos_window)

        self.H_conj = self.G * np.conjugate(patch_fft) / (patch_fft * np.conjugate(patch_fft) + self.parameters.lambda_)

    def track(self, image):

        # extract new patch
        patch, _ = get_patch(image, self.position, self.size)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch_fft = np.fft.fft2(patch * self.cos_window)

        # calculate response and update position
        response = np.real(np.fft.ifft2(self.H_conj * patch_fft))

        y, x = np.unravel_index(response.argmax(), response.shape)
        if x > self.size[0] / 2:
            x = x - self.size[0]
        if y > self.size[1] / 2:
            y = y - self.size[1]

        self.position += np.array([x, y]).astype('int')

        # update filter
        patch, _ = get_patch(image, self.position, self.size)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch_fft = np.fft.fft2(patch * self.cos_window)
        H_new = self.G * np.conjugate(patch_fft) / (patch_fft * np.conjugate(patch_fft) + self.parameters.lambda_)
        self.H_conj = (1 - self.parameters.alpha) * self.H_conj + self.parameters.alpha * H_new

        return [self.position[0] - self.original_size[0]/2, self.position[1] - self.original_size[1]/2,
                self.original_size[0], self.original_size[1]]


class CFParams():
    def __init__(self, sigma=2.5, alpha=0.1, lambda_=0.1, enlarge_factor=1.5):
        self.sigma = sigma
        self.alpha = alpha
        self.lambda_ = lambda_
        self.enlarge_factor = enlarge_factor

