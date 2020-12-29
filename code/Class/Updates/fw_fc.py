from Library.paths import *
from Library.import_library import *


class FW_FC_gaussian(object):
    def __init__(self, T, Sigma_inv, T_W, Sigma_W):
        self.T = T
        self.Sigma_inv = Sigma_inv
        self.T_W = T_W
        self.Sigma_W = Sigma_W

    def fw_fc(self):
        Sigma_star = inv(inv(self.Sigma_W) + self.Sigma_inv)
        T_star = Sigma_star.dot(
            inv(self.Sigma_W).dot(self.T_W) + self.Sigma_inv.dot(self.T))
        fW = T_star
        fC = Sigma_star
        return fW, fC


class FW_FC_binary(object):
    def __init__(self, T, Sigma_inv, K):
        self.T = T
        self.Sigma_inv = Sigma_inv
        self.K = K

    def fw_fc(self):
        fW = np.zeros((self.K))
        fC = np.zeros((self.K, self.K))

        if self.K == 1:
            arg_neg = -1/2 * (1-self.T)**2 * self.Sigma_inv
            arg_pos = -1/2 * (1+self.T)**2 * self.Sigma_inv
            Z = 1/2 * (exp(arg_neg) + exp(arg_pos))
            if Z != 0:
                fW = 1/2 * (exp(-1/2 * (1-self.T)**2 * self.Sigma_inv) -
                            exp(-1/2 * (1+self.T)**2 * self.Sigma_inv)) / Z
                fC = 1 - fW**2
        else:
            fW, fC = self.integrals_fW_fC()

        return fW, fC

    def configuration_W(self, n):
        W = np.array([int(x) for x in list('{0:0b}'.format(n).zfill(self.K))])
        return 2 * W - 1

    def integrals_fW_fC(self, threshold_zero=1e-8):
        (Z, W_avg, W_squared_avg) = 0, np.zeros(
            (self.K)), np.zeros((self.K, self.K))

        for i in range(2**self.K):
            W = self.configuration_W(i)
            argument = - 1/2 * ((W-self.T).transpose()
                                ).dot(self.Sigma_inv).dot(W-self.T)
            try:
                weight = exp(argument)
            except:
                weight = threshold_zero
            Z += weight
            W_avg += weight * W
            W_squared_avg += weight * \
                W.reshape(self.K, 1).dot(W.reshape(self.K, 1).transpose())

        Z /= (2**self.K)
        W_avg /= (2**self.K)
        W_squared_avg /= (2**self.K)

        if np.isnan(Z) or Z == 0 or not np.isfinite(Z):
            Z = 0
            f_W = np.zeros((self.K))
            f_C = np.zeros((self.K, self.K))

        if Z == 0 or Z < threshold_zero:
            (Z, fW, fC) = 0, np.zeros((self.K)), np.zeros((self.K, self.K))
        else:
            fW = W_avg/Z
            fC = W_squared_avg/Z - \
                fW.reshape(self.K, 1).dot(fW.reshape(self.K, 1).transpose())
        return (fW, fC)
