import numpy as np
import matplotlib.pyplot as plt

def generate_data(perfect_system=True, process_noise_variance=0.1, measurement_noise_variance=1.0, T=50, x0=0.0):
    """
    Generate true state and noisy measurements for a system that can be perfect (no process noise) or noisy
    """
    # Initialize
    true_states = np.zeros(T)
    true_states[0] = x0
    
    for t in range(1, T):
        if perfect_system:
            # Perfect system (no process noise)
            true_states[t] = A * true_states[t-1]
        else:
            # Noisy system (with process noise)
            process_noise = np.random.normal(0, np.sqrt(process_noise_variance))
            true_states[t] = A * true_states[t-1] + process_noise

    # Generate noisy measurements
    measurements = true_states + np.random.normal(0, np.sqrt(measurement_noise_variance), T)
    
    return true_states, measurements

def kalman_filter(measurements, kalman_gains=[None], A=1.0, Q=0.1, R=1.0, T=50, x0=0.0):
    """
    Apply the Kalman filter with specified Kalman gains
    """
    # Initialize state estimates for each Kalman gain scenario
    if kalman_gains != [None]:
        x_estimates = {f"Kalman gain = {K}": np.zeros(T) for K in kalman_gains if K is not None}
    else:
        x_estimates = {}
    x_estimates["Kalman filter"] = np.zeros(T)

    # State covariance 
    P = np.ones(T) * 10  # (large initial uncertainty)
    
    # Initialize state estimates for each Kalman gain scenario
    if kalman_gains != [None] :
        x_pred = {f"Kalman gain = {K}": np.zeros(T) for K in kalman_gains if K is not None}
    else:
        x_pred = {}
    x_pred["Kalman filter"] = np.zeros(T)

    for t in range(1, T):
        # Predict state
        # (We need process noise in the prediction: we sample it)
        process_noise = np.random.normal(0, np.sqrt(Q))
        x_pred["Kalman filter"][t] = A * x_estimates["Kalman filter"][t-1] + process_noise
        
        if kalman_gains != [None]:
            for K in kalman_gains:
                if K is not None:
                    x_pred[f"Kalman gain = {K}"][t] = A * x_estimates["Kalman filter"][t-1] + process_noise

        # Predict covariance
        P_pred = P[t-1] + Q
        
        # Kalman gain
        K_gain = P_pred / (P_pred + R)
        
        # Update state
        x_estimates["Kalman filter"][t] = x_pred["Kalman filter"][t] + K_gain * (measurements[t] - x_pred["Kalman filter"][t])
        
        # Update covariance
        P[t] = (1 - K_gain) * P_pred  # Update the uncertainty

        # Update for different Kalman gains if specified
        for K in kalman_gains:
            if K is not None:
                x_estimates[f"Kalman gain = {K}"][t] = x_pred[f"Kalman gain = {K}"][t] + K * (measurements[t] - x_pred[f"Kalman gain = {K}"][t])

    return x_estimates

def plot_results(true_states, measurements, x_estimates, title="", savepath=None):
    """
    Plot the true state, noisy measurements, and Kalman filter estimates.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(true_states, label='True state', color='black', linewidth=2)
    plt.plot(measurements, label='Noisy measurements', color='red', linestyle='dotted')

    if x_estimates is not None:
        for label, estimate in x_estimates.items():
            plt.plot(estimate, label=label, linewidth=2, linestyle='--')

    plt.xlabel('Time step')
    plt.ylabel('State value')
    plt.title(title)
    plt.legend()
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)

# Example of running the script with various configurations

# Parameters
A = 1.0  # State transition matrix
T = 50  # Number of time steps
x0 = 0.0  # Initial state
Q = 0.1 # Process noise covariance
R = 1.0 # Measurement noise covariance 


# 1. Perfect system
perfect_system = True
kalman_gains = [None]

# Generate data
true_states, measurements = generate_data(perfect_system, Q, R, T, x0)

# Apply Kalman filter with different Kalman gains
x_estimates = kalman_filter(true_states, measurements, Q, R, kalman_gains, A, Q, R, T, x0)

# Plot the results
plot_results(true_states, measurements, None, title="True state and measurements dynamics (perfect system)", savepath='/home/clevyfidel/clevyfidel/fig_kalman_perfect.png')
plot_results(true_states, measurements, x_estimates, title="Kalman filter estimate", savepath='/home/clevyfidel/clevyfidel/fig_kalman_perfect_filter.png')

# 2. Noisy system
perfect_system = False
kalman_gains = [None]

# Generate data
true_states, measurements = generate_data(perfect_system, process_noise_variance, measurement_noise_variance, T, x0)

# Apply Kalman filter with different Kalman gains
x_estimates = kalman_filter(true_states, measurements, process_noise_variance, measurement_noise_variance, kalman_gains, A, Q, R, T, x0)

# Plot the results
plot_results(true_states, measurements, None, title="True state and measurements dynamics (noisy system)", savepath='/home/clevyfidel/clevyfidel/fig_kalman_noisy.png')
plot_results(true_states, measurements, x_estimates, title="Kalman filter estimate", savepath='/home/clevyfidel/clevyfidel/fig_kalman_noisy_filter.png')

# 2b. Noisy system, different Kalman gain

# Choose Kalman gain values to compare (empty list means only the standard Kalman gain will be used)
kalman_gains = [0.1, 0.5, 0.9]  # Example: Compare small, medium, and large Kalman gains


# Apply Kalman filter with different Kalman gains
x_estimates = kalman_filter(true_states, measurements, process_noise_variance, measurement_noise_variance, kalman_gains, A, Q, R, T, x0)

# Plot the results
x_estimates.pop('Kalman filter')
plot_results(true_states, measurements, x_estimates, title="Effect of Kalman gain on state estimation", savepath='/home/clevyfidel/clevyfidel/fig_kalman_noisy_filter_gains.png')
