import numpy
import cv2
import matplotlib.pyplot as plt
import time

from EX1_Optical_flow.ex1_utils import *


def lucaskanade(im1, im2, N, normalize=True, check_reliability=False):
    sigma = 1

    if normalize:
        im1 = im1/255
        im2 = im2/255

    It = gausssmooth(im2 - im1, sigma)
    Ix1, Iy1 = gaussderiv(im1, sigma)
    Ix2, Iy2 = gaussderiv(im2, sigma)
    Ix = (Ix1 + Ix2) / 2
    Iy = (Iy1 + Iy2) / 2

    Ixx = np.multiply(Ix, Ix)
    Iyy = np.multiply(Iy, Iy)
    Ixy = np.multiply(Ix, Iy)
    Ixt = np.multiply(Ix, It)
    Iyt = np.multiply(Iy, It)

    kernel = np.ones([N, N])
    sum_Iyy = cv2.filter2D(Iyy, -1, kernel)
    sum_Ixy = cv2.filter2D(Ixy, -1, kernel)
    sum_Ixx = cv2.filter2D(Ixx, -1, kernel)
    sum_Ixt = cv2.filter2D(Ixt, -1, kernel)
    sum_Iyt = cv2.filter2D(Iyt, -1, kernel)

    D = np.multiply(sum_Ixx, sum_Iyy) - np.multiply(sum_Ixy, sum_Ixy)

    reliable = np.ones(np.shape(im1))
    if check_reliability:
        if 0 in D:
            reliable = np.where(D == 0, 0, 1)
    D = np.where(D == 0, 1e-10, D)

    u = (np.multiply(sum_Ixy, sum_Iyt) - np.multiply(sum_Iyy, sum_Ixt)) / D
    v = (np.multiply(sum_Ixy, sum_Ixt) - np.multiply(sum_Ixx, sum_Iyt)) / D

    if check_reliability:
        T = np.array([[sum_Ixx, sum_Ixy], [sum_Ixy, sum_Iyy]])
        eig = np.zeros([*np.shape(sum_Ixx), 2])
        for i in range(np.shape(sum_Ixx)[0]):
            for j in range(np.shape(sum_Ixx)[1]):
                eig[i, j, :] = np.linalg.eigvalsh(T[:, :, i, j])

        # eigenvalues must not be too small:
        if normalize:
            t = 1e-2
        else:
            t = 1e-5
        reliable = np.where(eig[:, :, 0] < t, 0, reliable)

        # ratio must not be too great
        eig[:, :, 0] = np.where(eig[:, :, 0] == 0, 1e-10, eig[:, :, 0])     # to not divide by 0
        ratio = eig[:, :, 1] / eig[:, :, 0]
        if normalize:
            t2 = 100
        else:
            t2 = 10000
        reliable = np.where(ratio > t2, 0, reliable)
        reliable = reliable.astype(np.uint8)
        return u*reliable, v*reliable, reliable

    return u, v


def hornschunck(im1, im2, n_iters, lmbd, normalize=True, initialize=None):
    sigma = 1

    if normalize:
        im1 = im1 / 255
        im2 = im2 / 255

    It = gausssmooth(im2 - im1, sigma)
    Ix1, Iy1 = gaussderiv(im1, sigma)
    Ix2, Iy2 = gaussderiv(im2, sigma)
    Ix = (Ix1 + Ix2) / 2
    Iy = (Iy1 + Iy2) / 2

    Ixx = np.multiply(Ix, Ix)
    Iyy = np.multiply(Iy, Iy)

    D = lmbd + Ixx + Iyy

    if initialize is None:
        u = np.zeros(np.shape(im1))
        v = np.zeros(np.shape(im1))
    else:
        u, v = initialize

    L = np.zeros([3, 3])
    L[[0, 2], 1] = 0.25
    L[1, [0, 2]] = 0.25
    for i in range(n_iters):
        u_a = cv2.filter2D(u, -1, L)
        v_a = cv2.filter2D(v, -1, L)
        P = Ix * u_a + Iy * v_a + It
        u = u_a - Ix * P / D
        v = v_a - Iy * P / D

    return u, v


def visualization_noise():
    im1 = np.random.rand(200, 200).astype(np.float32)
    im2 = im1.copy()
    im2 = rotate_image(im2, -1)
    U_lk, V_lk = lucaskanade(im1, im2, 3)
    U_hs, V_hs = hornschunck(im1, im2, 1000, 0.5)
    fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2, figsize=(5, 5))
    ax1_11.imshow(im1)
    ax1_12.imshow(im2)
    show_flow(U_lk, V_lk, ax1_21, type='angle')
    show_flow(U_lk, V_lk, ax1_22, type='field', set_aspect=True)
    plt.savefig('noise_lk.png')
    fig1.suptitle('Lucas-Kanade Optical Flow')
    fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2, figsize=(5,5))
    ax2_11.imshow(im1)
    ax2_12.imshow(im2)
    show_flow(U_hs, V_hs, ax2_21, type='angle')
    show_flow(U_hs, V_hs, ax2_22, type='field', set_aspect=True)
    plt.savefig('noise_hs.png')
    fig2.suptitle('Horn-Schunck Optical Flow')
    plt.show()


def show_on_image(im, ax, U, V):
    scaling = 0.1
    u = cv2.resize(gausssmooth(U, 1.5), (0, 0), fx=scaling, fy=scaling)
    v = cv2.resize(gausssmooth(V, 1.5), (0, 0), fx=scaling, fy=scaling)

    x_ = (np.array(list(range(1, u.shape[1] + 1))) - 0.5) / scaling
    y_ = (np.array(list(range(1, u.shape[0] + 1))) - 0.5) / scaling
    x, y = np.meshgrid(x_, y_)

    ax.imshow(im, cmap="gray")
    ax.quiver(x, y, -u * 5, v * 5)
    ax.set_aspect(1.)


def visualization2(im, U_lk, V_lk, U_hs, V_hs, title=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    show_on_image(im, ax1, U_lk, V_lk)

    if title:
        ax1.title.set_text('Lucas-Kanade')

    show_on_image(im, ax2, U_hs, V_hs)

    if title:
        ax2.title.set_text('Horn-Schunck')

    fig.show()


def determine_parameters(im1, im2):

    # Lukas-Kanade:
    neighborhood_sizes = [3, 5, 10, 15, 20, 50]

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 5.5))
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    for i in range(len(neighborhood_sizes)):
        U_lk, V_lk, _ = lucaskanade(im1, im2, neighborhood_sizes[i], True, True)
        show_on_image(im2, axes[i], U_lk, V_lk)
        axes[i].set_axis_off()
    fig.show()

    # Horn-Schunck:
    nr_iterations = [10, 100, 200, 300, 500, 1000]
    lmbdas = [0.01, 0.1, 0.25, 0.5, 10, 20]     # vse večje nardijo da se skala ne premika, cduni premiki pr trebuhu

    # nr of iterations:
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 5.5))
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    U_hs, V_hs = np.zeros(im1.shape), np.zeros(im1.shape)
    for i in range(len(neighborhood_sizes)):
        U_hs, V_hs = hornschunck(im1, im2, nr_iterations[i], 0.5, True)
        show_on_image(im2, axes[i], U_hs, V_hs)
        axes[i].set_axis_off()
    fig.show()

    # lambda:
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 5.5))
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    for i in range(len(neighborhood_sizes)):
        U_hs, V_hs = hornschunck(im1, im2, 300, lmbdas[i], True)
        show_on_image(im2, axes[i], U_hs, V_hs)
        axes[i].set_axis_off()
    fig.show()


def timing(im1, im2, show=False):
    t = time.time()
    ul, vl = lucaskanade(im1, im2, 15, True)
    t_lk = time.time() - t

    t = time.time()
    U_hs, V_hs = hornschunck(im1, im2, 300, 0.5, True)
    t_hs = time.time() - t

    if show:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))
        show_on_image(im2, ax1, U_hs, V_hs)


    t = time.time()
    U_hs_i, V_hs_i = hornschunck(im1, im2, 30, 0.5, True, initialize=(ul, vl))
    t_hs_i = time.time() - t + t_lk

    if show:
        show_on_image(im2, ax2, U_hs_i, V_hs_i)

    t = time.time()
    ul, vl, _ = lucaskanade(im1, im2, 15, True, True)
    t_lk_rel = time.time() - t

    t = time.time()
    U_hs_i, V_hs_i = hornschunck(im1, im2, 30, 0.5, True, initialize=(ul, vl))
    t_hs_i_rel = time.time() - t + t_lk_rel

    if show:
        show_on_image(im2, ax3, U_hs_i, V_hs_i)

        ax1.set_axis_off()
        ax2.set_axis_off()
        ax3.set_axis_off()

        fig.show()

    return np.array([t_lk, t_hs, t_hs_i, t_lk_rel, t_hs_i_rel])



if __name__ == "__main__":
    # TESTING ON NOISE
    visualization_noise()

    # TESTING ON ADDITIONAL IMAGES
    # show comparison if normalized or not:
    c1 = cv2.resize(cv2.imread('photos/candide4.png', cv2.IMREAD_GRAYSCALE)[:920, 420:1800], (300, 200))
    c2 = cv2.resize(cv2.imread('photos/candide5.png', cv2.IMREAD_GRAYSCALE)[:920, 420:1800], (300, 200))
    visualization2(c2, *lucaskanade(c1, c2, 3, False), *hornschunck(c1, c2, 500, 0.5, False), True)
    visualization2(c2, *lucaskanade(c1, c2, 15, True), *hornschunck(c1, c2, 500, 0.5, True))


    im1 = cv2.resize(cv2.imread('Exercise1-material/disparity/office2_left.png', cv2.IMREAD_GRAYSCALE), (300, 200))
    im2 = cv2.resize(cv2.imread('Exercise1-material/disparity/office2_right.png', cv2.IMREAD_GRAYSCALE), (300, 200))
    visualization2(im1, *lucaskanade(im1, im2, 3), *hornschunck(im1, im2, 500, 0.5))
    visualization2(im1, *lucaskanade(im1, im2, 15, True), *hornschunck(im1, im2, 500, 0.5, True))

    sh1 = cv2.resize(cv2.imread('photos/sharma1.png', cv2.IMREAD_GRAYSCALE), (300, 200))
    sh2 = cv2.resize(cv2.imread('photos/sharma2.png', cv2.IMREAD_GRAYSCALE), (300, 200))
    visualization2(sh2, *lucaskanade(sh1, sh2, 3), *hornschunck(sh1, sh2, 500, 0.5))
    visualization2(sh2, *lucaskanade(sh1, sh2, 15, True), *hornschunck(sh1, sh2, 500, 0.5, True))

    im1 = cv2.resize(cv2.imread('Exercise1-material/lab2/001.jpg', cv2.IMREAD_GRAYSCALE), (200, 200))
    im2 = cv2.resize(cv2.imread('Exercise1-material/lab2/002.jpg', cv2.IMREAD_GRAYSCALE), (200, 200))
    visualization2(im1, *lucaskanade(im1, im2, 3), *hornschunck(im1, im2, 1000, 0.5))
    visualization2(im1, *lucaskanade(im1, im2, 15, True), *hornschunck(im1, im2, 1000, 0.5, True))


    #  RELIABILITY
    images = [[im1, im2], [c1, c2], [sh1, sh2]]
    for i1, i2 in images:
        ul, vl, r = lucaskanade(i1, i2, 15, True, True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 5.5))
        show_on_image(i2, ax1, ul, vl)
        ax2.imshow(r*255, cmap='gray')
        ax1.set_axis_off()
        ax2.set_axis_off()
        fig.show()

    # RESULTS AT DIFFERENT PARAMETER VALUES
    determine_parameters(c1, c2)
    determine_parameters(sh1, sh2)
    determine_parameters(im1, im2)

    # TIMING THE METHODS
    tt = np.array([0, 0, 0, 0, 0])
    for i in range(100):
        for i1, i2 in images:
            tt = tt + timing(i1, i2)
    tt = tt / 300
    print(tt)

    # comparing output at HS time optimization
    timing(c1, c2, True)

