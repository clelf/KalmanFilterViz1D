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
    Fit a Kalman filter using EM and return FILTERED state estimates.
    
    Returns:
        y_hat: Filtered estimates. y_hat[t] is the state estimate after seeing y[0:t].
        s_hat: Corresponding state variances (not std).
        kf_fitted: The fitted Kalman filter object.
    
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

    # Kalman filtering - returns filtered state estimates
    # state_means_filt[t] is the state estimate after seeing y[0:t]
    state_means_filt, state_covariances_filt = kf_fitted.filter(y)
    
    return state_means_filt[:,0], state_covariances_filt[:,0,0], kf_fitted


def kalman_fit_predict(y, n_iter):
    """
    Fit a Kalman filter using EM and return ONE-STEP-AHEAD predictions.
    
    Wrapper around kalman_fit that computes predictions for the next timestep.
    After seeing observations up to y[t], predict y[t+1].
    
    Returns:
        y_hat: One-step-ahead predictions. y_hat[t] is the prediction for y[t+1] given y[0:t].
        s_hat: Corresponding prediction standard deviations.
        kf_fitted: The fitted Kalman filter object.
    """
    # Get filtered estimates
    state_means_filt, state_covariances_filt, kf_fitted = kalman_fit(y, n_iter)
    
    # Note: kalman_fit returns only the first component, we need full state for prediction
    # Re-run filter to get full state (or modify kalman_fit to return full state)
    full_state_means, full_state_covs = kf_fitted.filter(y)
    
    # Get fitted parameters for one-step-ahead prediction
    A = kf_fitted.transition_matrices
    H = kf_fitted.observation_matrices
    Q = kf_fitted.transition_covariance
    R = kf_fitted.observation_covariance
    
    # Handle None values
    if A is None:
        A = np.eye(2)
    if Q is None:
        Q = np.eye(2) * 0.1
    if R is None:
        R = np.array([[1.0]])
    
    T = len(y)
    y_hat = np.zeros(T)
    s_hat = np.zeros(T)
    
    # Compute one-step-ahead predictions
    # After filtering at time t (using y[0:t]), predict y[t+1]
    for t in range(T):
        x_next_pred = A @ full_state_means[t]
        P_next_pred = A @ full_state_covs[t] @ A.T + Q
        
        y_hat[t] = (H @ x_next_pred)[0]
        s_hat[t] = (H @ P_next_pred @ H.T + R)[0, 0]
    
    return y_hat, np.sqrt(s_hat), kf_fitted


def kalman_fit_batch(ys, n_iter, predict=False):
    """
    Batch version of kalman_fit.
    
    Parameters
    ----------
    ys : np.array
        2D array of observations, shape (N_samples, T)
    n_iter : int
        Number of EM iterations
    predict : bool
        If False, return filtered estimates (y_hat[t] is estimate after seeing y[0:t])
        If True, return one-step-ahead predictions (y_hat[t] predicts y[t+1])
    
    Returns
    -------
    y_hats : np.array
        Shape (N_samples, T)
    s_hats : np.array
        Shape (N_samples, T) of standard deviations
    """
    fit_func = kalman_fit_predict if predict else kalman_fit
    desc = "Samples fit to KF by EM" + (" (predict)" if predict else "")
    
    y_hats, s_hats = [], []
    for y in tqdm(ys, desc=desc):
        y_hat, s_hat, _ = fit_func(y, n_iter)
        y_hats.append(y_hat)
        s_hats.append(s_hat)
    return np.stack(y_hats, axis=0), np.stack(s_hats, axis=0)


# Convenience aliases for backward compatibility
def kalman_fit_predict_batch(ys, n_iter):
    """Batch version of kalman_fit_predict (returns one-step-ahead predictions)."""
    return kalman_fit_batch(ys, n_iter, predict=True)


def _fit_context_kfs(y, contexts, n_ctx, n_iter):
    """
    Helper function to fit separate Kalman filters for each context.
    
    Parameters
    ----------
    y : np.array
        1D array of observations
    contexts : np.array
        1D integer array indicating context at each timestep
    n_ctx : int
        Number of distinct contexts
    n_iter : int
        Number of EM iterations
    
    Returns
    -------
    kfs : list
        List of fitted KalmanFilter objects, one per context
    """
    kfs = []
    for c in range(n_ctx):
        ctx_mask = (contexts == c)
        y_ctx = y[ctx_mask]
        
        if len(y_ctx) > 2:  # Need enough observations to fit
            kf = KalmanFilter(
                observation_matrices=np.array([[1.0, 0.0]]),
                n_dim_state=2,
                n_dim_obs=1
            )
            kf_fitted = kf.em(y_ctx, n_iter=n_iter, em_vars='all')
            kfs.append(kf_fitted)
        else:
            # Fallback: use default parameters if too few observations
            kf = KalmanFilter(
                transition_matrices=np.array([[0.9, 0.1], [0.0, 1.0]]),
                observation_matrices=np.array([[1.0, 0.0]]),
                n_dim_state=2,
                n_dim_obs=1
            )
            kfs.append(kf)
    return kfs


def _get_kf_params(kf):
    """Helper to get KF parameters with defaults for None values."""
    A = kf.transition_matrices if kf.transition_matrices is not None else np.eye(2)
    Q = kf.transition_covariance if kf.transition_covariance is not None else np.eye(2) * 0.1
    H = kf.observation_matrices
    R = kf.observation_covariance if kf.observation_covariance is not None else np.array([[1.0]])
    return A, Q, H, R


def kalman_fit_context_aware(y, contexts, n_iter):
    """
    Context-aware Kalman filter that maintains separate state estimates for each context.
    Each context has its own Kalman filter that is only updated when that context is active.
    
    Parameters
    ----------
    y : np.array
        1D array of observations of length T
    contexts : np.array
        1D integer array of length T indicating which context (0, 1, ..., n_ctx-1) is active
    n_iter : int
        Number of EM iterations for fitting each Kalman filter
    
    Returns
    -------
    y_hat : np.array
        Filtered estimates of length T. y_hat[t] is the estimate after seeing y[t].
    s_hat : np.array
        Corresponding standard deviations of length T
    kfs : list
        List of fitted KalmanFilter objects, one per context
    """
    T = len(y)
    n_ctx = len(np.unique(contexts))
    
    # Fit a separate Kalman filter to each context's observations
    kfs = _fit_context_kfs(y, contexts, n_ctx, n_iter)
    
    # Now run the context-aware filter: maintain state for each context
    # and only update the active context at each timestep
    y_hat = np.zeros(T)
    s_hat = np.zeros(T)
    
    # State means and covariances for each context
    state_means = [np.zeros(2) for _ in range(n_ctx)]
    state_covs = [np.eye(2) * 1.0 for _ in range(n_ctx)]
    
    # Initialize states with initial state means from fitted KFs
    for c in range(n_ctx):
        if hasattr(kfs[c], 'initial_state_mean') and kfs[c].initial_state_mean is not None:
            state_means[c] = kfs[c].initial_state_mean.copy()
        if hasattr(kfs[c], 'initial_state_covariance') and kfs[c].initial_state_covariance is not None:
            state_covs[c] = kfs[c].initial_state_covariance.copy()
    
    for t in range(T):
        c = contexts[t]
        kf = kfs[c]
        
        # Get KF parameters with defaults
        A, Q, H, R = _get_kf_params(kf)
        
        # Prediction step for the active context (prior prediction for y[t])
        x_pred = A @ state_means[c]
        P_pred = A @ state_covs[c] @ A.T + Q
        
        # Update step with current observation y[t]
        y_obs = np.array([[y[t]]])
        innovation = y_obs - H @ x_pred.reshape(-1, 1)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        state_means[c] = (x_pred.reshape(-1, 1) + K @ innovation).flatten()
        state_covs[c] = (np.eye(2) - K @ H) @ P_pred
        
        # Store FILTERED estimate (after seeing y[t])
        y_hat[t] = (H @ state_means[c])[0]
        s_hat[t] = (H @ state_covs[c] @ H.T + R)[0, 0]
    
    return y_hat, np.sqrt(s_hat), kfs


def kalman_fit_context_aware_predict(y, contexts, n_iter):
    """
    Context-aware Kalman filter that returns ONE-STEP-AHEAD predictions.
    
    After seeing observations up to y[t], predict y[t+1].
    
    Parameters
    ----------
    y : np.array
        1D array of observations of length T
    contexts : np.array
        1D integer array of length T indicating which context (0, 1, ..., n_ctx-1) is active
    n_iter : int
        Number of EM iterations for fitting each Kalman filter
    
    Returns
    -------
    y_hat : np.array
        One-step-ahead predictions of length T. y_hat[t] predicts y[t+1] given y[0:t].
    s_hat : np.array
        Corresponding prediction standard deviations of length T
    kfs : list
        List of fitted KalmanFilter objects, one per context
    """
    T = len(y)
    n_ctx = len(np.unique(contexts))
    
    # Fit KFs per context (reuse the fitting logic)
    kfs = _fit_context_kfs(y, contexts, n_ctx, n_iter)
    
    # Run context-aware filter with one-step-ahead predictions
    y_hat = np.zeros(T)
    s_hat = np.zeros(T)
    
    state_means = [np.zeros(2) for _ in range(n_ctx)]
    state_covs = [np.eye(2) * 1.0 for _ in range(n_ctx)]
    
    for c in range(n_ctx):
        if hasattr(kfs[c], 'initial_state_mean') and kfs[c].initial_state_mean is not None:
            state_means[c] = kfs[c].initial_state_mean.copy()
        if hasattr(kfs[c], 'initial_state_covariance') and kfs[c].initial_state_covariance is not None:
            state_covs[c] = kfs[c].initial_state_covariance.copy()
    
    for t in range(T):
        c = contexts[t]
        kf = kfs[c]
        
        # Get KF parameters with defaults
        A, Q, H, R = _get_kf_params(kf)
        
        # Prediction step
        x_pred = A @ state_means[c]
        P_pred = A @ state_covs[c] @ A.T + Q
        
        # Update step with current observation y[t]
        y_obs = np.array([[y[t]]])
        innovation = y_obs - H @ x_pred.reshape(-1, 1)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        state_means[c] = (x_pred.reshape(-1, 1) + K @ innovation).flatten()
        state_covs[c] = (np.eye(2) - K @ H) @ P_pred
        
        # ONE-STEP-AHEAD PREDICTION: After updating with y[t], predict y[t+1]
        x_next_pred = A @ state_means[c]
        P_next_pred = A @ state_covs[c] @ A.T + Q
        
        y_hat[t] = (H @ x_next_pred)[0]
        s_hat[t] = (H @ P_next_pred @ H.T + R)[0, 0]
    
    return y_hat, np.sqrt(s_hat), kfs


def kalman_fit_context_aware_batch(ys, contexts_batch, n_iter, predict=False):
    """
    Batch version of context-aware Kalman filter.
    
    Parameters
    ----------
    ys : np.array
        2D array of observations, shape (N_samples, T)
    contexts_batch : np.array
        2D integer array of shape (N_samples, T) indicating contexts
    n_iter : int
        Number of EM iterations
    predict : bool
        If False, return filtered estimates (y_hat[t] is estimate after seeing y[0:t])
        If True, return one-step-ahead predictions (y_hat[t] predicts y[t+1])
    
    Returns
    -------
    y_hats : np.array
        Shape (N_samples, T)
    s_hats : np.array
        Shape (N_samples, T) of standard deviations
    """
    fit_func = kalman_fit_context_aware_predict if predict else kalman_fit_context_aware
    desc = "Context-aware KF by EM" + (" (predict)" if predict else "")
    
    y_hats, s_hats = [], []
    for y, contexts in tqdm(zip(ys, contexts_batch), desc=desc, total=len(ys)):
        y_hat, s_hat, _ = fit_func(y, contexts, n_iter)
        y_hats.append(y_hat)
        s_hats.append(s_hat)
    return np.stack(y_hats, axis=0), np.stack(s_hats, axis=0)


# Convenience alias for backward compatibility
def kalman_fit_context_aware_predict_batch(ys, contexts_batch, n_iter):
    """Batch version of kalman_fit_context_aware_predict (returns one-step-ahead predictions)."""
    return kalman_fit_context_aware_batch(ys, contexts_batch, n_iter, predict=True)


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

    
