import numpy as np
import matplotlib.pyplot as plt



def process_dynamics(A, Q, T, x0, s0):
    # Generate the process's true states with noise of var Q
    true_state = np.zeros(T)
    true_state[0] = x0

    true_mean = np.zeros(T)
    true_mean[0] = x0
    true_var = np.zeros(T)
    true_var[0] = s0
    
    for t in range(1, T):
        true_mean[t] = A * true_mean[t-1]
        true_var[t] = Q
        true_state[t] = A * true_state[t-1] + np.random.normal(0, np.sqrt(true_var[t]))

    return true_state, true_mean, true_var

def observ_dynamics(true_state, R, T):
    # Generate observations (measurements) with noise of var R
    measurements = np.zeros(T)
    obs_mean = np.zeros(T)
    obs_var = np.zeros(T)

    for t in range(0, T):
        obs_mean[t] = true_state[t]
        obs_var[t] = R
        measurements[t] = obs_mean[t] + np.random.normal(0, np.sqrt(obs_var[t]))

    return measurements, obs_mean, obs_var

def kalman(measurements, A, Q, R, x0, s0):
    T = len(measurements)

    # Kalman filter initialization
    x_est = np.zeros(T)  # Estimated states mean
    s_est = np.zeros(T)  # Estimated states variance
    x_est[0] = x0  # Start with the initial estimate
    s_est[0] = s0

    for t in range(1, T):
        # Prediction step
        x_pred = A * x_est[t-1]
        s_pred = A**2 * s_est[t-1] + Q

        # Update step (Kalman Gain)
        K = s_pred / (s_pred + R)
        x_est[t] = x_pred + K * (measurements[t] - x_pred)
        s_est[t] = (1 - K) * s_pred 

    return x_est, s_est

def plot_estim(process, obs, est_mu, est_var):
    plt.figure(figsize=(10, 6))
    if process is not None: plt.plot(range(len(process)), process, label='True State', color='green', linewidth=2)
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
    Q = 0.1  # Process noise covariance
    R = 1  # Measurement noise covariance
    x0 = 0.0  # Initial state mean
    s0 = 2 # Initial state var
    T = 50  # Number of time steps

    # Generate data
    process,_,_ = process_dynamics(A, Q, T, x0, s0)    
    obs,_,_ = observ_dynamics(process, Q, T)

    # Estimate with Kalman filter
    x_est, s_est = kalman(obs, A, Q, R, x0, s0)
    
    # Plot the results
    plot_estim(process, obs, x_est, s_est)

    
