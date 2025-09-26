from kalman import process_dynamics_tau, observ_dynamics, kalman_tau
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter


if __name__=='__main__':

    # Range of taus to try
    N_taus = 20
    N_test = 5000 # number of tests per taus
    high_bound=800
    low_bound=1
    taus = np.logspace(np.log10(high_bound), np.log10(low_bound), N_taus, endpoint=False)[::-1]

    # Fix mu (to be changed if needed)
    mu = 600

    # Fix other parameters
    T = 20
    sigma_q = 2 # process noise std
    Q = sigma_q**2
    sigma_r = 2 # obs noise std
    R = sigma_r**2
    C = 1

    
    # For every tau, generate a data sequence long of T timesteps and compute Kalman estimate
    T = 20
    mu_estimates = []
    mu_estimates_std = []
    mse = []
    rmse = []
    se = []
    for tau in taus:
        # Generate N_test sequences
        mus = []
        for i in range(N_test):
            # Generate observation sequence
            states, _, _ = process_dynamics_tau(tau, x_lim=mu, T=T, Q=Q, x0=np.array(ss.norm.rvs(mu, sigma_q, 1)), s0=Q) # here sampling x0 around the mean with std = sigma_q
            obs, _, _ = observ_dynamics(states, C=C, R=R, T=T)

            # Kalman estimate
            mu_hat, sigma_hat = kalman_tau(measurements=states, tau=tau, x_lim=mu, C=C, Q=Q, R=R, x0=np.array(ss.norm.rvs(mu, sigma_r, 1))) # here sampling x_hat_0 around the mean with std = sigma_r
            mus.append(mu_hat)

        # MSE and standard error of the mean
        mu_estimates.append(np.mean(mus))
        mu_estimates_std.append(np.std(mus))
        mse.append(((np.array(mus)-mu)**2).mean())  
        rmse.append(np.square(((np.array(mus)-mu)**2).mean()))            
        se.append(np.array(mus).std()/N_test)


    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(20, 8))

    # Plot true mean
    axs[0].axhline(y=mu, color='red', label='True mu')

    # Plot estimates as dots over a line
    axs[0].plot(taus, mu_estimates, label='Kalman mu_hat', color='black', linestyle='--', marker='o') 
    axs[0].fill_between(taus, np.array(mu_estimates)-np.array(mu_estimates_std), np.array(mu_estimates)+np.array(mu_estimates_std), color='black', alpha=0.2)
    axs[0].set_xlabel('taus')
    axs[0].set_ylabel('mu_hat (err=std)')
    axs[0].set_xscale('log')
    axs[0].legend()

    # Plot RMSE
    axs[1].plot(taus, rmse, label='RMSE', color='blue', linewidth=2, marker='o')
    axs[1].fill_between(taus, np.array(rmse)-np.array(se), np.array(rmse)+np.array(se), color='blue', alpha=0.2)
    axs[1].set_xscale('log')
    axs[1].set_xlabel('taus')
    axs[1].set_ylabel('RMSE (err=SEM)')

    # Add title to overall figure mentioning N_test
    axs[0].set_title(f'Simulations per tau value = {N_test}')
    plt.show()


        
