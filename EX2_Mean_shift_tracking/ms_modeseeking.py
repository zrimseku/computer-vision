import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

from ex2_utils import generate_responses_1, generate_responses_2, get_patch


def find_mode(image, start_point, stop_criteria, kernel, return_points=False):
    x_prev = np.array(start_point)
    x_next = x_prev

    h = kernel.shape[1]
    w = kernel.shape[0]
    span_x = np.arange(-int((w - 1) / 2), int((w + 1) / 2))
    span_y = np.arange(-int((h - 1) / 2), int((h + 1) / 2))
    x_i, y_i = np.meshgrid(span_x, span_y)
    n_steps = 0

    if return_points:
        points = [start_point]

    while np.linalg.norm(x_prev - x_next) > stop_criteria or n_steps == 0:

        patch, _ = get_patch(image, x_next, kernel.shape)
        if np.sum(patch) == 0:
            # print('Stuck on a flat with all patch values equal to 0.')
            return x_prev, n_steps

        ms_x = np.sum(patch * x_i * kernel) / np.sum(patch * kernel)
        ms_y = np.sum(patch * y_i * kernel) / np.sum(patch * kernel)

        x_prev = x_next
        x_next = x_prev + np.array([ms_x, ms_y])
        n_steps += 1

        if return_points:
            points.append(x_next)

        if n_steps % 10000 == 0:
            print(f"Haven't reached desired distance in {n_steps} steps. Current point: {x_next}")
        if n_steps % 100000 == 0:
            break

    if return_points:
        return points, n_steps

    return x_next, n_steps


if __name__ == '__main__':

    image = generate_responses_1()
    # ind = np.unravel_index(np.argmax(image, axis=None), image.shape)
    # print(ind)
    kernel = np.ones([15, 15])
    start_point = [40, 40]

    point0, nst0 = find_mode(image, start_point, 0.1, kernel)
    print(point0, nst0)

    image = generate_responses_2()
    kernel = np.ones([15, 15])
    start_point = [90, 10]

    point, nst = find_mode(image, start_point, 0.1, kernel)
    print(point, nst)

    fig = plt.figure()
    ax = Axes3D(fig)
    response = generate_responses_2()
    response[int(point[1]), int(point[0])] = 0.002
    response[int(point[1])+1, int(point[0])] = 0.002
    x, y = np.meshgrid(np.arange(100), np.arange(100))
    surf = ax.plot_surface(x, y, response, cmap=matplotlib.cm.jet)
    fig.colorbar(surf)
    plt.show()

    plt.figure()
    points, nst = find_mode(image, start_point, 0.1, kernel, True)
    plt.imshow(generate_responses_2())
    for pt in points:
        plt.plot(pt[0], pt[1], 'ko', ms=0.8)
    plt.show()
