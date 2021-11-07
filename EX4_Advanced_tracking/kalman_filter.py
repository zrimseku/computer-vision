import sympy as sp
import numpy as np
import math
import matplotlib.pyplot as plt

from ex4_utils import kalman_step


def derive_matrices(model_name, par=None):
    T, q, r = sp.symbols('T q r')
    if model_name == 'ncv':
        F = sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        L = sp.Matrix([[0, 0], [0, 0], [1, 0], [0, 1]])
    elif model_name == 'rw':
        F = sp.Matrix([[0, 0], [0, 0]])
        L = sp.Matrix([[1, 0], [0, 1]])
    elif model_name == 'nca':
        F = sp.Matrix([[0, 0, 1, 0, T, 0], [0, 0, 0, 1, 0, T], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        L = sp.Matrix([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1]])
    else:
        print('Unknown model name')
    Fi = sp.exp(F*T)
    Q = sp.integrate((Fi*L)*q*(Fi*L).T, (T, 0, T))

    H = np.zeros((2, F.shape[0]), dtype=np.float32)
    H[0, 0] = 1
    H[1, 1] = 1
    R = r * sp.eye(2)

    if par is not None:
        Fi_fn = sp.lambdify((T, q, r), Fi, modules='numpy')
        Q_fn = sp.lambdify((T, q, r), Q, modules='numpy')
        R_fn = sp.lambdify((T, q, r), R, modules='numpy')
        Fi = Fi_fn(*par)
        Q = Q_fn(*par)
        R = R_fn(*par)

    return Fi, Q, H, R


def trajectory(N):
    v = np.linspace(5 * math.pi, 0, N)
    x = np.cos(v) * v
    y = np.sin(v) * v
    return x, y


def apply_kalman(x, y, A, H, Q, R):
    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()
    sx[0] = x[0]
    sy[0] = y[0]

    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
    state[0] = x[0]
    state[1] = y[0]
    P = np.eye(A.shape[0], dtype=np.float32)

    for j in range(1, x.size):
        state, P, _, _ = kalman_step(A, H, Q, R, np.reshape(np.array([x[j], y[j]]), (-1, 1)), np.reshape(state, (-1, 1)), P)
        sx[j] = state[0]
        sy[j] = state[1]

    return sx, sy


if __name__ == '__main__':
    # Derivations

    # NCV
    Fi_ncv, Q_ncv, H_ncv, R_ncv = derive_matrices('ncv')
    print('Nearly constant velocity:')
    print(f"Fi: {Fi_ncv}")
    print(f"Q: {Q_ncv}")
    print(f'H: {H_ncv}')
    print(f'R: {R_ncv}')
    print('________________________________________')

    # RW
    Fi_rw, Q_rw, H_rw, R_rw = derive_matrices('rw')
    print('Random walk:')
    print(f"Fi: {Fi_rw}")
    print(f"Q: {Q_rw}")
    print(f'H: {H_rw}')
    print(f'R: {R_rw}')
    print('________________________________________')

    # NCA
    Fi_nca, Q_nca, H_nca, R_nca = derive_matrices('nca')
    print("Nearly constant acceleration")
    print(f"Fi: {Fi_nca}")
    print(f"Q: {Q_nca}")
    print(f'H: {H_nca}')
    print(f'R: {R_nca}')

    # Applying Kalman filter
    # trajectory from instructions
    x1, y1 = trajectory(40)
    # square
    x2, y2 = np.array([-5, 5, 5, -5, -5]), np.array([5, 5, -5, -5, 5])
    # triangle
    x3, y3 = np.array([-2, 2, 0, -2]), np.array([0, 0, 3, 0])
    # star
    x4, y4 = np.zeros(21), np.zeros(21)
    angles = np.linspace(0, 2*math.pi, 21)
    x4[::2] = 10 * np.cos(angles[::2])
    x4[1::2] = 5 * np.cos(angles[1::2])
    y4[::2] = 10 * np.sin(angles[::2])
    y4[1::2] = 5 * np.sin(angles[1::2])

    parameters = [(1, 100, 1), (1, 5, 1), (1, 1, 1), (1, 1, 5), (1, 1, 100)]
    model_names = ['rw', 'ncv', 'nca']
    for x, y in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]:
        fig, axs = plt.subplots(len(model_names), len(parameters), figsize=(14, 8))

        for i, par in enumerate(parameters):
            for m, name in enumerate(model_names):

                A, Q, H, R = derive_matrices(name, par)
                sx, sy = apply_kalman(x, y, A, H, Q, R)

                axs[m, i].plot(x, y, '-o', label='Measurements')
                axs[m, i].plot(sx, sy, '-o', label='Filtered')
                axs[m, i].set_title(f'{model_names[m]}: q={par[1]}, r={par[2]}')

        fig.show()

