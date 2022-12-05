def normalize_one(self, x):
        #x = B * 784 * 1
        microx = (np.sum(x, axis = 1) / x.shape[1]).reshape(x.shape[0], 1, 1)
        sigmax = (np.sum((x - microx) ** 2, axis = 1) / x.shape[0]).reshape(x.shape[0], 1, 1)
        return (x - microx) / np.sqrt(sigmax)