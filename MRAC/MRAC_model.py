import numpy as np

class MRACReferenceModel:
    def __init__(self, A_m, B_m):
        self.A_m = np.array(A_m)
        self.B_m = np.array(B_m)
        self.x_m = np.zeros((self.A_m.shape[0],))

    def reset(self, init=None):
        self.x_m = np.zeros_like(self.x_m) if init is None else init

    def step(self, r):
        self.x_m = self.A_m @ self.x_m + self.B_m @ r
        return self.x_m.copy()


class MRACPlainModel:
    def __init__(self, A, B, E):
        self.A = np.array(A)
        self.B = np.array(B)
        self.E = np.array(E)

    def predict(self, x, u, xi):
        return self.A @ x + self.B @ u + self.E @ xi


class MRACController:
    def __init__(self, state_dim, phi_dim, K=None, Gamma=None):
        self.K = np.array(K) if K is not None else np.zeros((1, state_dim))
        self.Gamma = np.eye(phi_dim) * 0.01 if Gamma is None else Gamma
        self.Theta = np.zeros((phi_dim, 1))

    def phi(self, x):
        return np.concatenate([x, x**2, np.tanh(x)])

    def get_control(self, x):
        x = np.array(x).reshape(-1)
        phi_x = self.phi(x).reshape(-1, 1)
        u = self.K @ x.reshape(-1, 1) + self.Theta.T @ phi_x
        return u.item(), phi_x

    def update(self, phi_x, error):
        error = np.array([[error]])
        self.Theta = self.Theta - self.Gamma @ phi_x @ error


class MRACErrorModel:
    """
    Computes tracking error:
    e(k) = x(k) - x_m(k)
    or   z(k) = x(k) - Π ξ(k)
    """
    def __init__(self, use_exosystem=False, Pi=None):
        self.use_exosystem = use_exosystem
        self.Pi = np.array(Pi) if Pi is not None else None

    def compute_error(self, x, ref):
        if self.use_exosystem:
            return x - self.Pi @ ref
        else:
            return x - ref


class MRACObserver:
    """
    Observer for estimating internal state or disturbance using:
    \hat{x}(k+1) = A \hat{x}(k) + B u(k) + L (e(k) - C \hat{x}(k))
    """
    def __init__(self, A, B, C, L):
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.L = np.array(L)
        self.x_hat = np.zeros((A.shape[0],))

    def reset(self, init=None):
        self.x_hat = np.zeros_like(self.x_hat) if init is None else init

    def step(self, u, e):
        e = np.array(e).reshape(-1)
        self.x_hat = self.A @ self.x_hat + self.B @ u + self.L @ (e - self.C @ self.x_hat)
        return self.x_hat.copy()
