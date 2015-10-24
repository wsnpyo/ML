import numpy as np

class GMM:
    def __init__(self, K, DATA, epsilon = 1e-3):
        self.K = K
        self.DATA = np.array(DATA)
        self.epsilon = epsilon
        self.N, self.M = DATA.shape
        self.means = None
        self.covariances = None
        self.P_z = None

    def _init_parameters(self):
        means = []
        covariances = []
        Divisor = self.N//self.K
        for i in range(0, self.N, Divisor):
            means.append(np.mean(self.DATA[i:i+Divisor], axis = 0))
            covariances.append(np.cov(self.DATA[i:i+Divisor].T))
        self.means = np.array(means)
        self.covariances = np.array(covariances)
        self.P_z = np.ones((self.K, 1)) / self.K

    def print_status(self):
        print("=============== GMM STATUS ===============")
        print("### Properties of Data Set :\n###   N = %d\n###   M = %d\n###   K = %d" % (self.N, self.M, self.K))
        print("###   K Centroids = ")
        for i in range(self.K):
            print("###     " + str(self.means[i]))
        print("==========================================")

    def fit(self):
        self._init_parameters()

        old_likelihood = 0.0
        Densities = np.empty((self.N, self.K))
        P_zx = np.empty((self.N, self.K))

        while True:
            ''' E-step: (Calculate distribution P(z^i | x^i)) '''
            for i in range(self.N):
                for j in range(self.K):
                    Densities[i, j] = self.Norm_PDF(self.DATA[i], self.means[j], self.covariances[j])
                denominator = Densities[i].dot(self.P_z)
                for j in range(self.K):
                    P_zx[i, j] = Densities[i, j] * self.P_z[j] / denominator

            ''' Estimate likelihood '''
            likelihood = 0.0
            for i in range(self.N):
                likelihood += np.sum(np.log(Densities[i].dot(self.P_z)))
            print(likelihood - old_likelihood)
            if np.abs(likelihood - old_likelihood) < self.epsilon :
                break
            old_likelihood = likelihood

            ''' M-step: (Update the parameters to Maximize the likelihood function)'''
            for i in range(self.K):
                denominator = np.sum(P_zx[:, i])

                self.means[i] = P_zx[:, i].T.dot(self.DATA) / denominator

                diff = DATA - np.tile(self.means[i], (self.N, 1))
                self.covariances[i] = diff.T.dot(np.diag(P_zx[:, i]).dot(diff)) / denominator

                self.P_z[i] = denominator / self.N

    def Norm_PDF(self, x, mean, covariance):
        Centered = (x - mean).T
        # print(str(covariance) + '\n')
        Inv_Cov = np.linalg.inv(covariance)
        Det_Cov = np.linalg.det(covariance)
        return np.exp(-0.5 * (Centered.T.dot(Inv_Cov.dot(Centered)))) / np.sqrt(np.power(2 * np.pi, self.K) * Det_Cov)

### Test Case ###

group_a = np.random.normal(loc=(20.00, 14.00), scale=(4.0, 4.0), size=(1000, 2))
group_b = np.random.normal(loc=(15.00, 8.00), scale=(2.0, 2.0), size=(1000, 2))
group_c = np.random.normal(loc=(30.00, 40.00), scale=(2.0, 2.0), size=(1000, 2))
group_d = np.random.normal(loc=(25.00, 32.00), scale=(7.0, 7.0), size=(1000, 2))

DATA = np.concatenate((group_a, group_b, group_c, group_d))

S = GMM(4, DATA)
S.fit()
S.print_status()

