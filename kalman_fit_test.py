from kalman import kalman_fit, kalman_fit_batch, kalman_tau, plot_estim
import numpy as np
from time import time 

if __name__=='__main__':

    n_iter = 10
    n_samples = 1000

    #### Single

    y = np.random.randn(n_samples)
    t0 = time()
    state_means_filt, state_covariances_filt, _ = kalman_fit(y, n_iter=n_iter)
    print(f'Elapsed time (s): {time()-t0}, n_iter: {n_iter}, n_samples: {n_samples}')
    plot_estim(y, state_means_filt, state_covariances_filt, process=None)


    #### Batch

    # ys = [np.random.randn(1000) for i in range(4)]

    # t0 = time()
    # state_means_filt, state_covariances_filt = kalman_fit_batch(ys, n_iter=n_iter)
    # print(f'Elapsed time (s): {time()-t0}, n_iter: {n_iter}')
    # plot_estim(ys[0], state_means_filt[0], state_covariances_filt[0], process=None)

