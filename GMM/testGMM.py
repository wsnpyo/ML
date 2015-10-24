import numpy as np
import matplotlib.pyplot as plt

from GMM import GMM

if __name__ == '__main__':
    group_a = np.random.normal(loc=(20.00, 14.00), scale=(4.0, 4.0), size=(1000, 2))
    group_b = np.random.normal(loc=(15.00, 8.00), scale=(2.0, 2.0), size=(1000, 2))
    group_c = np.random.normal(loc=(30.00, 40.00), scale=(2.0, 2.0), size=(1000, 2))
    group_d = np.random.normal(loc=(25.00, 32.00), scale=(7.0, 7.0), size=(1000, 2))
    group_e = np.random.normal(loc=(10.00, 32.00), scale=(7.0, 7.0), size=(1000, 2))

    DATA = np.concatenate((group_a, group_b, group_c, group_d, group_e))
    S = GMM(5, DATA, 1e-3)
    S.fit()
    S.print_status()

    testdata = np.random.rand(10000, 2)*50
    labels = S.Classify(testdata)

    plt.scatter(testdata[:, 0], testdata[:, 1], c=list(map(lambda i : {0:'b',1:'g',2:'r',3:'y',4:'k'}[i], labels)))
    plt.show()

