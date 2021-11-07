import cv2
import torch

from siamfc.siamfc import TrackerSiamFC     # https://github.com/huanglianghua/siamfc-pytorch
from got10k.trackers import Tracker

import numpy as np


class LongTermSiamFC(Tracker):

    def __init__(self, net_path=None, sampling='uniform', sigma=0.1, nr_samples=10, threshold=0.45):
        super(LongTermSiamFC, self).__init__('SiamFC Long Term', True)
        self.tracker_short = TrackerSiamFC(net_path)
        self.sampling_method = sampling
        self.original_sigma = sigma
        self.nr_samples = nr_samples
        self.threshold = threshold
        self.target_visible = True
        self.samples = [[], []]

    def init(self, img, box):
        self.tracker_short.init(img, box)
        self.target_visible = True
        _, self.first_response = self.tracker_short.update(img)
        self.frames_lost = 0
        self.sigma = self.original_sigma * min(img.shape[:2]) / 2
        print(self.sigma)

    def update(self, image):

        if self.target_visible:
            bbox, response = self.tracker_short.update(image)

            self.target_visible = response / self.first_response > self.threshold

            if self.target_visible:
                return bbox, response

        prev_center = self.tracker_short.center
        self.tracker_short.center = self.find_target(image)

        # do another update with short term tracker, to get comparable results
        bbox, response = self.tracker_short.update(image)
        print(response / self.first_response)

        if response/self.first_response > self.threshold:
            self.target_visible = True
            self.frames_lost = 0
            return bbox, response
        else:
            self.tracker_short.center = prev_center
            self.frames_lost += 1
            return bbox, 0.

    @torch.no_grad()
    def find_target(self, image):
        self.tracker_short.net.eval()

        # sample patches
        im_h, im_w = image.shape[:2]
        target_h, target_w = self.tracker_short.target_sz
        target_h, target_w = int(target_h), int(target_w)

        sigma_gauss = self.sigma
        if self.sampling_method == 'gauss_growing':
            if sigma_gauss * 1.05**self.frames_lost < min(image.shape[:2]) / 2:
                sigma_gauss *= 1.05**self.frames_lost
            else:
                sigma_gauss = min(image.shape[:2]) / 2
        if self.sampling_method == 'uniform':
            sampled_x = np.random.randint(0, max(im_w - target_w, 1), self.nr_samples)
            sampled_y = np.random.randint(0, max(im_h - target_h, 1), self.nr_samples)
        elif self.sampling_method == 'gauss' or self.sampling_method == 'gauss_growing':
            sampled_x = np.random.normal(self.tracker_short.center[1], sigma_gauss, self.nr_samples)
            sampled_y = np.random.normal(self.tracker_short.center[0], sigma_gauss, self.nr_samples)
            sampled_x = np.maximum(np.minimum(sampled_x, im_w - target_w), 0)
            sampled_y = np.maximum(np.minimum(sampled_y, im_h - target_h), 0)
        else:
            print('Unknown sampling method')
        self.samples = (sampled_x, sampled_y)
        sampled_patches = [image[round(sampled_y[i]):round(sampled_y[i])+target_h,
                           round(sampled_x[i]):round(sampled_x[i])+target_w] for i in range(self.nr_samples)]

        x = [cv2.resize(patch, (self.tracker_short.cfg.instance_sz, self.tracker_short.cfg.instance_sz),
                           interpolation=cv2.INTER_LINEAR) for patch in sampled_patches]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.tracker_short.device).permute(0, 3, 1, 2).float()

        # find best patch
        x = self.tracker_short.net.backbone(x)
        responses = self.tracker_short.net.head(self.tracker_short.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        max_responses = np.amax(responses, axis=(1, 2))
        best_patch = np.argmax(max_responses)

        best_center = np.array([sampled_y[best_patch] + self.tracker_short.target_sz[0] / 2,
                               sampled_x[best_patch] + self.tracker_short.target_sz[1] / 2])

        return best_center


