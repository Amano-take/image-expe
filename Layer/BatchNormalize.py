import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__))
import Adam
import MomentumSGD
import SAM

class BatchNormalize():
    eps = 1e-8

    def __init__(self, M, l2para, opt="Adam"):
        self.M = M
        self.l2lambda = l2para
        self.beta = np.zeros(self.M).reshape(self.M, 1)
        self.ganma = np.ones(self.M).reshape(self.M, 1)
        if(opt=="Adam"):
            self.betaAdam = Adam.Adam(self.beta)
            self.ganmaAdam = Adam.Adam(self.ganma)
        elif(opt=="MSGD"):
            self.betaAdam = MomentumSGD.MSGD(self.beta)
            self.ganmaAdam = MomentumSGD.MSGD(self.ganma)
        elif(opt == "SAM"):
            self.betasam = SAM.SAM(self.beta)
            self.ganmasam = SAM.SAM(self.ganma)
        elif(opt == "ADSGD"):
            self.betaAdam = MomentumSGD.ADSGD(self.beta)
            self.ganmaAdam = MomentumSGD.ADSGD(self.ganma)
        return 

    def prop(self, x):
        #x = B * M * 1
        self.x = x
        self.B, self.M, _ = x.shape
        self.microB = np.sum(x, axis=0) / x.shape[0]
        self.sigmaB = np.sum((x - self.microB) ** 2, axis=0) / x.shape[0]
        self.normalize_x = (x - self.microB) / np.sqrt(self.sigmaB + BatchNormalize.eps)
        self.y = self.ganma * self.normalize_x + self.beta
        # yをreturn
        return self.y

    def back(self, delta):
        delta_xi_head = delta * self.ganma
        delta_sigmaB2 = np.sum(delta_xi_head * (self.x.reshape(self.B, self.M).T - self.microB)
                                * (-1/2) * np.power(self.sigmaB + BatchNormalize.eps, -3/2), axis=1).reshape(self.M, 1)
        delta_microB = np.sum(delta_xi_head * (-1 / np.power(self.sigmaB + BatchNormalize.eps, 1/2)), axis=1).reshape(
            self.M, 1) + (-2) * delta_sigmaB2 * np.sum(self.x.reshape(self.B, self.M).T - self.microB, axis=1).reshape(self.M, 1)
        delta_a = delta_xi_head * np.power(self.sigmaB + BatchNormalize.eps, -1/2) + delta_sigmaB2 * 2 * (
            self.x.reshape(self.B, self.M).T - self.microB) / self.B + delta_microB / self.B
        delta_ganma = np.sum(
            delta * self.normalize_x.reshape(self.B, self.M).T, axis=1).reshape(self.M, 1)
        delta_beta = np.sum(delta, axis=1).reshape(self.M, 1)

        #更新
        self.beta = self.betaAdam.update(delta_beta)
        self.ganma = self.ganmaAdam.update(delta_ganma)
        
        return delta_a
    
    def test(self, x, beta, ganma):
        microB = np.sum(x, axis=0) / x.shape[0]
        sigmaB = np.sum((x - microB) ** 2, axis=0) / x.shape[0]
        y = ganma * np.power(sigmaB + BatchNormalize.eps, -1/2) * x + \
            (beta - ganma * microB * np.power(sigmaB + BatchNormalize.eps, -1/2))
        return y

    def update(self, beta, ganma):
        self.beta = beta
        self.ganma = ganma

    def SAM_back1(self, delta):
        delta_xi_head = delta * self.ganma
        delta_sigmaB2 = np.sum(delta_xi_head * (self.x.reshape(self.B, self.M).T - self.microB)
                                * (-1/2) * np.power(self.sigmaB + BatchNormalize.eps, -3/2), axis=1).reshape(self.M, 1)
        delta_microB = np.sum(delta_xi_head * (-1 / np.power(self.sigmaB + BatchNormalize.eps, 1/2)), axis=1).reshape(
            self.M, 1) + (-2) * delta_sigmaB2 * np.sum(self.x.reshape(self.B, self.M).T - self.microB, axis=1).reshape(self.M, 1)
        delta_a = delta_xi_head * np.power(self.sigmaB + BatchNormalize.eps, -1/2) + delta_sigmaB2 * 2 * (
            self.x.reshape(self.B, self.M).T - self.microB) / self.B + delta_microB / self.B
        delta_ganma = np.sum(
            delta * self.normalize_x.reshape(self.B, self.M).T, axis=1).reshape(self.M, 1)
        delta_beta = np.sum(delta, axis=1).reshape(self.M, 1)

        self.beta = self.betasam.calw(delta_beta)
        self.ganma = self.ganmasam.calw(delta_ganma)
        
        return delta_a
    
    def SAM_back2(self, delta):
        delta_xi_head = delta * self.ganma
        delta_sigmaB2 = np.sum(delta_xi_head * (self.x.reshape(self.B, self.M).T - self.microB)
                                * (-1/2) * np.power(self.sigmaB + BatchNormalize.eps, -3/2), axis=1).reshape(self.M, 1)
        delta_microB = np.sum(delta_xi_head * (-1 / np.power(self.sigmaB + BatchNormalize.eps, 1/2)), axis=1).reshape(
            self.M, 1) + (-2) * delta_sigmaB2 * np.sum(self.x.reshape(self.B, self.M).T - self.microB, axis=1).reshape(self.M, 1)
        delta_a = delta_xi_head * np.power(self.sigmaB + BatchNormalize.eps, -1/2) + delta_sigmaB2 * 2 * (
            self.x.reshape(self.B, self.M).T - self.microB) / self.B + delta_microB / self.B
        delta_ganma = np.sum(
            delta * self.normalize_x.reshape(self.B, self.M).T, axis=1).reshape(self.M, 1)
        delta_beta = np.sum(delta, axis=1).reshape(self.M, 1)

        self.beta = self.betasam.update(delta_beta)
        self.ganma = self.ganmasam.update(delta_ganma)

        return delta_a

    def back_ADSGD(self, delta, k):
        delta_xi_head = delta * self.ganma
        delta_sigmaB2 = np.sum(delta_xi_head * (self.x.reshape(self.B, self.M).T - self.microB)
                                * (-1/2) * np.power(self.sigmaB + BatchNormalize.eps, -3/2), axis=1).reshape(self.M, 1)
        delta_microB = np.sum(delta_xi_head * (-1 / np.power(self.sigmaB + BatchNormalize.eps, 1/2)), axis=1).reshape(
            self.M, 1) + (-2) * delta_sigmaB2 * np.sum(self.x.reshape(self.B, self.M).T - self.microB, axis=1).reshape(self.M, 1)
        delta_a = delta_xi_head * np.power(self.sigmaB + BatchNormalize.eps, -1/2) + delta_sigmaB2 * 2 * (
            self.x.reshape(self.B, self.M).T - self.microB) / self.B + delta_microB / self.B
        delta_ganma = np.sum(
            delta * self.normalize_x.reshape(self.B, self.M).T, axis=1).reshape(self.M, 1)
        delta_beta = np.sum(delta, axis=1).reshape(self.M, 1)

        #更新
        self.beta = self.betaAdam.update(delta_beta, k)
        self.ganma = self.ganmaAdam.update(delta_ganma, k)
        
        return delta_a

    def back_l2(self, delta):
        delta_xi_head = delta * self.ganma
        delta_sigmaB2 = np.sum(delta_xi_head * (self.x.reshape(self.B, self.M).T - self.microB)
                                * (-1/2) * np.power(self.sigmaB + BatchNormalize.eps, -3/2), axis=1).reshape(self.M, 1)
        delta_microB = np.sum(delta_xi_head * (-1 / np.power(self.sigmaB + BatchNormalize.eps, 1/2)), axis=1).reshape(
            self.M, 1) + (-2) * delta_sigmaB2 * np.sum(self.x.reshape(self.B, self.M).T - self.microB, axis=1).reshape(self.M, 1)
        delta_a = delta_xi_head * np.power(self.sigmaB + BatchNormalize.eps, -1/2) + delta_sigmaB2 * 2 * (
            self.x.reshape(self.B, self.M).T - self.microB) / self.B + delta_microB / self.B
        delta_ganma = np.sum(
            delta * self.normalize_x.reshape(self.B, self.M).T, axis=1).reshape(self.M, 1) + 2 * self.l2lambda * self.ganma
        delta_beta = np.sum(delta, axis=1).reshape(self.M, 1) + 2 * self.l2lambda * self.beta

        #更新
        self.beta = self.betaAdam.update(delta_beta)
        self.ganma = self.ganmaAdam.update(delta_ganma)
        
        return delta_a