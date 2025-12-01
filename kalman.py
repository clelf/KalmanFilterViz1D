import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from tqdm import tqdm


def process_dynamics_A(A, B, Q, T, x0, s0):
    # Generate the process's states with noise of var Q
    process_state = np.zeros(T)
    process_state[0] = x0

    process_mean = np.zeros(T)
    process_mean[0] = x0
    process_var = np.zeros(T)
    process_var[0] = s0
    
    for t in range(1, T):
        process_state[t] = A * process_state[t-1] + np.random.normal(0, np.sqrt(Q))

        process_mean[t] = A * process_mean[t-1] + B
        process_var[t] = Q

    return process_state, process_mean, process_var


def process_dynamics_tau(tau, x_lim, Q, T, x0, s0):
    # Generate the process's states with noise of var Q
    process_state = np.zeros(T)
    process_state[0] = x0

    process_mean = np.zeros(T)
    process_mean[0] = x0
    process_var = np.zeros(T)
    process_var[0] = s0
    
    for t in range(1, T):
        process_state[t] = process_state[t-1] + 1 / tau * (x_lim - process_state[t-1]) + np.random.normal(0, np.sqrt(Q))

        process_mean[t] =  process_mean[t-1] + 1 / tau * (x_lim - process_mean[t-1])
        process_var[t] = Q 

    return process_state, process_mean, process_var



def observ_dynamics(process_state, C, R, T):
    # Generate observations (measurements) with noise of var R
    measurements = np.zeros(T)
    obs_mean = np.zeros(T)
    obs_var = np.zeros(T)

    for t in range(0, T):
        measurements[t] = C * process_state[t] + np.random.normal(0, np.sqrt(obs_var[t]))

        obs_mean[t] = C * process_state[t]
        obs_var[t] = R

    return measurements, obs_mean, obs_var

def kalman_A(measurements, A, B, C, Q, R, x0):
    T = len(measurements)

    # Kalman filter initialization
    x_est = np.zeros(T)  # Estimated states mean
    s_est = np.zeros(T)  # Estimated states variance
    x_est[0] = x0  # Start with the initial estimate
    s_est[0] = 0

    for t in range(1, T):
        # Prediction step
        x_pred = A * x_est[t-1] + B
        s_pred = A**2 * s_est[t-1] + Q

        # Update step (Kalman Gain)
        K = s_pred / (s_pred + R)
        x_est[t] = C * x_pred + K * (measurements[t] - x_pred)
        s_est[t] = (1 - K) * s_pred 

    return x_est, s_est

def kalman_tau(measurements, tau, x_lim, C, Q, R, x0):
    T = len(measurements)

    # Kalman filter initialization
    x_est = np.zeros(T)  # Estimated states mean
    s_est = np.zeros(T)  # Estimated states variance
    x_est[0] = x0  # Start with the initial estimate # TODO: or the mean of the last measurements ~ stationary value
    s_est[0] = 0

    for t in range(1, T):
        # Prediction step
        x_pred = x_est[t-1] + 1 / tau * (x_lim - x_est[t-1])
        s_pred = (1 - 1 / tau)**2 * s_est[t-1] + Q

        # Update step (Kalman Gain)
        K = s_pred / (s_pred + R)
        x_est[t] = C * x_pred + K * (measurements[t] - x_pred)
        s_est[t] = (1 - K) * s_pred 

    return x_est, s_est

def kalman_batch(ys, taus, mu_lims, C, Qs, Rs, x0s):
    y_hats, s_hats = [], []
    for y, tau, mu_lim, Q, R, y0 in zip(ys, taus, mu_lims, Qs, Rs, x0s):
        y_hat, s_hat = kalman_tau(y, tau, mu_lim, C, Q, R, y0)
        y_hats.append(y_hat)
        s_hats.append(s_hat)
    return np.stack([batch for batch in y_hats], axis=0), np.stack([batch for batch in s_hats], axis=0)


def kalman_fit(y, n_iter): # tau_init, b_init
    """
    Elapsed time (s): 1.15, n_iter: 5, n_samples: 1000
    Elapsed time (s): 2.28, n_iter: 10, n_samples: 1000
    """
    # Instead of using already computed parameters, estimate parameters by EM
    # Convert to A matrix format: x_t = x_t-1 + (b - x_t-1)/tau
    # A_init = np.array([[1.0 - 1.0/tau_init, b_init/tau_init], [0.0, 1.0]])

    kf = KalmanFilter(
        # transition_matrices=A_init,  # Start with random initial guess
        observation_matrices=np.array([[1.0, 0.0]]),  # Fixed - observe only x, not intercept
        n_dim_state=2,
        n_dim_obs=1  # 1D observations
    )

    # Fit parameters using EM algorithm - only estimate transition matrix and initial conditions
    kf_fitted = kf.em(y, n_iter=n_iter, em_vars='all')

    # Kalman filtering
    state_means_filt, state_covariances_filt = kf_fitted.filter(y)
    return state_means_filt[:,0], state_covariances_filt[:,0,0], kf_fitted


def kalman_fit_batch(ys, n_iter):
    y_hats, s_hats = [], []
    for y in tqdm(ys, desc="Samples fit to KF by EM"):
        y_hat, s_hat, _ = kalman_fit(y, n_iter)
        y_hats.append(y_hat)
        s_hats.append(s_hat)
    return np.stack([batch for batch in y_hats], axis=0), np.stack([batch for batch in s_hats], axis=0)



def plot_estim(obs, est_mu, est_var, process=None):
    """ process is not necessarily known --> can be set to None
    """
    plt.figure(figsize=(10, 6))
    if process is not None: 
        plt.plot(range(len(process)), process, label='True State', color='green', linewidth=2)
    plt.plot(range(len(obs)), obs, label='Noisy Measurements', color='red', linestyle='dotted')
    plt.plot(range(len(est_mu)), est_mu, label='Kalman Filter Estimate', color='blue', linewidth=2)
    plt.fill_between(range(len(est_var)), est_mu-np.sqrt(est_var), est_mu+np.sqrt(est_var), color='blue', alpha=0.2)
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.title('Kalman Filter: State Estimation Over Time')
    plt.legend()
    plt.show()

if __name__=='__main__':

    # Define system parameters
    A = 1  # State transition matrix
    B = 0 # B = x_lim / tau
    C = 1 # Observation model
    Q = 0.1  # Process noise covariance
    R = 1  # Measurement noise covariance
    x0 = 0.0  # Initial state mean
    T = 50  # Number of time steps

    # Generate data
    process,_,_ = process_dynamics_A(A, B, Q, T, x0)    
    obs,_,_ = observ_dynamics(process, C, Q, T)

    # Estimate with Kalman filter
    x_est, s_est = kalman_A(obs, A, B, C, Q, R, x0)
    
    # Plot the results
    plot_estim(process, obs, x_est, s_est)

    
