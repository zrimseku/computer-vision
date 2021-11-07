import time
import os
import cv2

from sequence_utils import VOTSequence
from particle_filters import ParticleFilterTracker, PFParams


# set the path to directory where you have the sequences
dataset_path = 'sequences'  # TODO: set to the dataset path on your disk
sequence = 'polarbear'  # choose the sequence you want to test
sequences = os.listdir('sequences')
# sequences = ['polarbear']

all_fails = 0
fps = 0
IOU = 0
time_all = 0

for s in sequences:
    # visualization and setup parameters
    win_name = 'Tracking window'
    reinitialize = True
    show_gt = False
    video_delay = 1
    font = cv2.FONT_HERSHEY_PLAIN

    # create sequence object
    sequence = VOTSequence(dataset_path, s)
    init_frame = 0
    n_failures = 0

    # create parameters and tracker objects
    parameters = PFParams()
    tracker = ParticleFilterTracker(parameters)

    time_all = 0
    iou_avg = 0

    # initialize visualization window
    sequence.initialize_window(win_name)
    # tracking loop - goes over all frames in the video sequence
    frame_idx = 0
    while frame_idx < sequence.length():
        img = cv2.imread(sequence.frame(frame_idx))
        # initialize or track
        if frame_idx == init_frame:
            # initialize tracker (at the beginning of the sequence or after tracking failure)
            t = time.time()
            tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
            time_all += time.time() - t
            predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
        else:
            # track on current frame - predict bounding box
            t = time.time()
            predicted_bbox = tracker.track(img)
            time_all += time.time() - t

        # calculate overlap (needed to determine failure of a tracker)
        gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
        o = sequence.overlap(predicted_bbox, gt_bb)

        iou_avg += o

        # draw ground-truth and predicted bounding boxes, frame numbers and show image
        if show_gt:
            sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
        sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
        sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
        sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
        for i in range(tracker.parameters.n_particles):
            cv2.circle(img, (int(tracker.particles[i, 0]), int(tracker.particles[i, 1])), int(tracker.weights[i]),
                       (0, 255, 0))
        sequence.show_image(img, video_delay)

        if o > 0 or not reinitialize:
            # increase frame counter by 1
            frame_idx += 1
        else:
            # increase frame counter by 5 and set re-initialization to the next frame
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1

    all_fails += n_failures
    fps += sequence.length() / time_all
    iou_avg /= sequence.length()
    IOU += iou_avg
    print(s)
    print('Tracking speed: %.1f FPS' % (sequence.length() / time_all))
    print('Tracker failed %d times' % n_failures)
    print(f'Average IoU: {iou_avg}')

print('_______________________________________')
print('On all sequences:')
print(f'All fails: {all_fails}')
print(f'Average speed: {fps / len(sequences)}')
print(f'Average IOU: {IOU / len(sequences)}')

