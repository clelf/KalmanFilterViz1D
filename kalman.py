import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from tqdm import tqdm
# from hmmlearn import hmm


# Minimum observations needed for EM algorithm (n_dim_state + 1)
# EM requires at least this many observations to estimate covariance matrices without division by zero
MIN_OBS_FOR_EM = 3


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



def _get_kf_params(kf):
    """Helper to get KF parameters with defaults for None values."""
    A = kf.transition_matrices if kf.transition_matrices is not None else np.array([[0.99, 0.01], [0.0, 1.0]])
    Q = kf.transition_covariance if kf.transition_covariance is not None else np.eye(2) * 0.1
    H = kf.observation_matrices if kf.observation_matrices is not None else np.array([[1.0, 0.0]])
    R = kf.observation_covariance if kf.observation_covariance is not None else np.array([[1.0]])
    return A, Q, H, R


def _init_context_states(kfs):
    """Initialise per-context state means and covariances from fitted KF objects."""
    mus, Sigmas = [], []
    for kf in kfs:
        mus.append(
            kf.initial_state_mean.copy()
            if (hasattr(kf, 'initial_state_mean') and kf.initial_state_mean is not None)
            else np.zeros(2)
        )
        Sigmas.append(
            kf.initial_state_covariance.copy()
            if (hasattr(kf, 'initial_state_covariance') and kf.initial_state_covariance is not None)
            else np.eye(2)
        )
    return mus, Sigmas


def _aggregate_contexts(per_ctx_means, per_ctx_vars, lambda_t):
    """Combine per-context estimates via the law of total expectation/variance.

    Parameters
    ----------
    per_ctx_means : np.array, shape (C,)
    per_ctx_vars  : np.array, shape (C,)  — per-context variances (not stds)
    lambda_t      : np.array, shape (C,)  — context probabilities weights (sum to 1)

    Returns
    -------
    y_hat : float — weighted mean
    s_hat : float — weighted standard deviation (sqrt of total variance)
    """
    y_hat = np.sum(lambda_t * per_ctx_means)
    within_var = per_ctx_vars
    between_var = (per_ctx_means - y_hat) ** 2
    s_hat = np.sqrt(np.sum(lambda_t * (within_var + between_var)))
    return y_hat, s_hat



def _compute_marginal_context_probabilities(per_rule_probabilities, rule_probabilities):
    """Marginalise context probabilities over rules.

    Parameters
    ----------
    per_rule_probabilities : np.array, shape [R, T, C]
        Context responsibilities conditioned on each rule:
        per_rule_probabilities[r, t, c] = P(context=c | rule=r, t).
        For each (r, t), the values along the C axis should sum to 1.
    rule_probabilities : np.array, shape [T, R]
        Rule probabilities: rule_probabilities[t, r] = P(rule=r | t).
        For each t, values along the R axis should sum to 1.

    Returns
    -------
    marg_resp : np.array, shape [T, C]
        Marginalized context probabilities:
        marg_resp[t, c] = sum_r  P(rule=r | t) * P(context=c | rule=r, t).
    """
    # rule_probabilities : [T, R]
    # per_rule_probabilities : [R, T, C]
    # result : [T, C]
    marg_resp = np.einsum('tr,rtc->tc', rule_probabilities, per_rule_probabilities)
    # Alternative:
    # marg_resp = np.array([rule_probabilities[t] @ per_rule_probabilities[:, t, :] for t in range(T)])
    return marg_resp


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


def kalman_step(y_t, kf, state_mean_prev, state_cov_prev):
    """Run a single Kalman step (predict + update) for one observation y_t, given the previous state mean and covariance.
    Returns updated state mean and covariance after processing y_t.
    Note: using pykalman's filter_update, works for 1 context; use _kalman_step_custom to control the update step with lambda if more than 1 context.
    """
    state_mean_new, state_cov_new = kf.filter_update(state_mean_prev, state_cov_prev, y_t)
    return state_mean_new, state_cov_new


def kalman_fit(y, n_iter): # tau_init, b_init
    """
    Fit a Kalman filter using EM.
    
    Returns:
        kf_fitted: The fitted Kalman filter object.
    
    Elapsed time (s): 1.15, n_iter: 5, n_samples: 1000
    Elapsed time (s): 2.28, n_iter: 10, n_samples: 1000
    """
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

    return kf_fitted


def kalman_filtering(y, kf):
    """Filter a sequence of observations using a Kalman filter object already instantiated.
    
    state_means_filt[t] is the state estimate after seeing y[0:t]
    """
    state_means_filt, state_covariances_filt = kf.filter(y) # TODO: check dimensions
    
    return state_means_filt, state_covariances_filt


def kalman_fit_filter(y, n_iter): # tau_init, b_init
    """
    Fit a Kalman filter using EM AND return filtered state estimates associated with each observation of the sequence.
    
    Returns:
        state_means_filt: Filtered state mean estimates. y_hat[t] is the state estimate after seeing y[0:t].
        state_covariances_filt: Corresponding state covariance estimate.
        kf_fitted: The fitted Kalman filter object.
    
    Elapsed time (s): 1.15, n_iter: 5, n_samples: 1000
    Elapsed time (s): 2.28, n_iter: 10, n_samples: 1000
    """

    kf_fitted = kalman_fit(y, n_iter)
    state_means_filt, state_covariances_filt = kalman_filtering(y, kf_fitted)

    return state_means_filt, state_covariances_filt, kf_fitted



def kalman_online_fit_predict(y, n_iter):
    """
    Fit a Kalman filter using EM and return ONE-STEP-AHEAD predictions.
    
    Each prediction y_hat[t] is a one-step-ahead prediction for y[t+1],
    based on a Kalman filter whose parameters are fit using EM on observations
    y[0:t+1] (i.e., all observations up to and including y[t]).
    
    This ensures that prediction at time t+1 only uses information from times 0 to t
    (no data leakage from future observations).
    
    Parameters:
        y: 1D array of observations of length T
        n_iter: Number of EM iterations for fitting
    
    Returns:
        y_hat: 
            One-step-ahead predictions of length T (previously: length T-MIN_OBS_FOR_EM).
            y_hat[t] is the prediction for y[t] given y[0:t]. 
            (previously: y_hat[t] is the prediction for y[t+MIN_OBS_FOR_EM] given y[0:t+MIN_OBS_FOR_EM].)
        s_hat: 
            Corresponding prediction standard deviations of length T (previously: length T-MIN_OBS_FOR_EM).
        kf_fitted: 
            The last fitted Kalman filter object (fit on full sequence).
    
    Note:
        EM requires at least MIN_OBS_FOR_EM=3 observations (n_dim_state + 1) to estimate
        covariance matrices without division by zero. Therefore, predictions start at
        index MIN_OBS_FOR_EM-1, predicting y[MIN_OBS_FOR_EM:].
    """

    T = len(y)
    # y_hat = np.zeros(T - MIN_OBS_FOR_EM)
    # s_hat = np.zeros(T - MIN_OBS_FOR_EM)
    y_hat = np.ones(T) * np.nan
    s_hat = np.ones(T) * np.nan
    kf_fitted = None
    A, Q, H, R = None, None, None, None
    
    # Compute one-step-ahead predictions
    # To predict y[t+1], we fit KF on y[0:t+1] and use the last filtered state
    # Start at t = MIN_OBS_FOR_EM - 1 so we have at least MIN_OBS_FOR_EM observations
    for t in tqdm(range(MIN_OBS_FOR_EM - 1, T - 1), desc="Fitting current sample's individual timesteps", leave=False): # Last t == T-2 --> predict t == y[T-1] = y[-1]

        # # Output index offset by MIN_OBS_FOR_EM - 1
        # out_idx = t - (MIN_OBS_FOR_EM - 1)

        # If current obs is masked, skip EM and reuse last fitted parameters to run the filter (besides, this will only propagate the prediction step)
        # If not, update the KF with EM
        if not np.ma.is_masked(y[t]) or kf_fitted is None: # kf_fitted will not be None if the first masked observation happens after at least MIN_OBS_EM
            # Update KF parameters by fitting with EM based on observations y[0:t+1] (indices 0 to t inclusive) and filter
            kf_fitted = kalman_fit(y[:t + 1], n_iter)
        
        # Get fitted parameters for one-step-ahead prediction (with defaults for None)
        A, Q, H, R = _get_kf_params(kf_fitted)

        # Filter the sequence up to time t to get the last filtered state (index -1, which corresponds to time t)
        # state_means_filt, state_covariances_filt = kf_fitted.filter(y[:t + 1]) # TODO: should use .filter_update maybe here since only producing one filter step?
        state_means_filt, state_covariances_filt = kalman_filtering(y[:t + 1], kf_fitted) # TODO: should use .filter_update maybe here since only producing one filter step?

        # Alternative:
        # state_means_filt, state_covariances_filt = kf_fitted.filter_update(state_means_filt, state_covariances_filt, y[:t + 1]) # But need to initialize state_means_filt and state_covariances_filt first, which needs to take into account the fact that the first few observations might be masked
        
        # Use the LAST filtered state (index -1, which corresponds to time t) to predict y[t+1]
        x_last = state_means_filt[-1]  # Last filtered state mean
        P_last = state_covariances_filt[-1]  # Last filtered state covariance
        
        ### One-step-ahead prediction for y[t+1] using the KF equations + last fitted parameters:
        # One-step-ahead state prediction: x_{t+1|t} = A @ x_{t|t}
        # (see pykalman.standard filter_predict and filter_correct for verification)
        x_next_pred = A @ x_last
        P_next_pred = A @ P_last @ A.T + Q
        
        # One-step-ahead observation prediction: y_{t+1|t} = H @ x_{t+1|t}
        # y_hat[out_idx] = (H @ x_next_pred)[0]
        # s_hat[out_idx] = np.sqrt((H @ P_next_pred @ H.T + R)[0, 0])
        y_hat[t+1] = (H @ x_next_pred)[0]
        s_hat[t+1] = np.sqrt((H @ P_next_pred @ H.T + R)[0, 0])
    return y_hat, s_hat


def kalman_step_multicontext(mu, Sigma, y_t, A, Q, H, R, lam=1.0):
    """Single Kalman predict + prior context probability-weighted update step.

    Implements Eqs. 9–10 of the derivation document: the update collapses the
    two-component Gaussian mixture (updated / not-updated) into a single Gaussian
    by matching the mixture mean and variance.

    Note: n is the dimension of observations (=1 for 1D observations)

    Note: this aligns with how 

    Parameters
    ----------
    mu : np.array, shape (n,)
        Mean of state at time t given observations from times [0...t]
    Sigma : np.array, shape (n, n)
        Covariance of state at time t given observations from times
        [0...t]
    y_t : float
        Observation at time t.  If `observation` is a masked array and any of
        its values are masked, the observation will be ignored (the prediction step is still performed, but not the update step which uses the observation).
    A, Q, H, R : KF parameter matrices.
    lam : float
        Context probability weight in [0, 1]. lam=1 gives a standard Kalman update;
        lam=0 gives a prediction-only step (no measurement update).

    Returns
    -------
    mu_new : np.array, shape (n,) — updated state mean.
    Sigma_new : np.array, shape (n, n) — updated state covariance.

    Notes
    -----
    Mean update  (Eq. 9):  mu_new  = mu_pred + lam * K * eps
    Covar update (Eq. 10): Sigma_new = (I - lam*K*H) @ Sigma_pred          <- within-component
                                      + lam*(1-lam) * (K*eps) @ (K*eps)^T  <- between-component
    The between-component term is the extra contribution that arises from
    collapsing the two-Gaussian mixture into a single Gaussian and is missing
    from a naive "scaled-gain" Kalman update.
    """
    # Prediction step
    mu_pred = A @ mu
    Sigma_pred = A @ Sigma @ A.T + Q

    # Update step
    if not np.ma.is_masked(y_t):
        y_obs = np.array([[y_t]])
        innovation = y_obs - H @ mu_pred.reshape(-1, 1)   # ε = y - H * mu_pred, shape (m, 1)
        S = H @ Sigma_pred @ H.T + R
        K_std = Sigma_pred @ H.T @ np.linalg.inv(S)       # Standard (unscaled) Kalman gain, shape (n, m)

        K_eps = K_std @ innovation                         # K * ε, shape (n, 1)

        # Eq. 9: mean update scaled by responsibility
        mu_new = (mu_pred.reshape(-1, 1) + lam * K_eps).flatten()

        # Eq. 10: within-component variance + between-component variance
        Sigma_new = (np.eye(len(mu)) - lam * K_std @ H) @ Sigma_pred \
                + lam * (1.0 - lam) * K_eps @ K_eps.T

        return mu_new, Sigma_new
    else:
        # If the observation is masked, skip the update and return the prediction.
        return mu_pred, Sigma_pred


def kalman_fit_multicontext(y, contexts, n_ctx, n_iter):
    """
    Fit separate Kalman filters for each context using masked arrays.

    For each context c, the full-length observation sequence is passed to
    pykalman with all timesteps where ``contexts != c`` masked out.  pykalman
    runs the prediction step (advancing the state via A and Q) at *every*
    timestep but only performs a measurement update at unmasked ones. 

    Parameters
    ----------
    y : np.array
        1D array of observations of length T
    contexts : np.array
        1D integer array of shape (T,) indicating the active context at each
        timestep (values in [0, n_ctx-1])
    n_ctx : int
        Number of distinct contexts
    n_iter : int
        Number of EM iterations for each Kalman filter

    Returns
    -------
    kfs : list
        List of fitted KalmanFilter objects, one per context
    """
    kfs = []
    y_ctx_masked = [] # List of y with context-specific masks
    for c in range(n_ctx):
        # Mask every timestep where context c is NOT active.
        # pykalman skips the observation update at masked steps but still
        # advances the state, so Q is estimated and applied at the correct
        # 1-step timescale across gaps.
        y_masked = np.ma.array(y, mask=(contexts != c))
        y_ctx_masked.append(y_masked)

        n_active = int((contexts == c).sum())
        if n_active >= MIN_OBS_FOR_EM:
            kf = KalmanFilter(
                observation_matrices=np.array([[1.0, 0.0]]),
                n_dim_state=2,
                n_dim_obs=1,
            )
            kf_fitted = kf.em(y_masked, n_iter=n_iter, em_vars='all')
            kfs.append(kf_fitted)
        else:
            # Fallback: use default parameters if too few active observations
            kf = KalmanFilter(
                transition_matrices=np.array([[0.9, 0.1], [0.0, 1.0]]), # TODO: set this better!!!!!
                observation_matrices=np.array([[1.0, 0.0]]),
                n_dim_state=2,
                n_dim_obs=1,
            )
            kfs.append(kf)
    # return kfs, y_ctx_masked
    return kfs


def kalman_filtering_multicontext(y, prior_context_probabilities, kfs):
    """Filter a sequence of observations using Kalman filter objects already instantiated.
    
    Note: alternative skipping use of lambda 
    for c in range(n_ctx):
        states_means_filt[c], states_covariances_filt[c] = kfs[c].filter(y_ctx_masked[c])
    # TODO: verify that in case lambda=1, this algorithm returns the same as applying single ctx KF separately     

    Returns:
        states_means_filt : np.array [C,T]
            Filtered mean estimates for each context
        states_vars_filt : np.array [C,T]
            Filtered covariance estimates for each context
    """
    T = len(y)
    n_ctx = prior_context_probabilities.shape[1]  # [T, C]

    # Instantiate and initialize filtered observation mean and variance arrays
    states_means_filt,       states_vars_filt       = np.ones((T, n_ctx)) * np.nan, np.ones((T, n_ctx)) * np.nan # [T, C]
    states_means_filt[0, :], states_vars_filt[0, :] = _init_context_states(kfs)
    
    # Run all KFs on the full observation sequence in parallel. Each KF maintains its own state trajectory.
    for c in range(n_ctx):
        # Get parameters of current context
        A, Q, H, R = _get_kf_params(kfs[c])

        # Filter the sequence time step by time step
        for t in range(1, T):            
            states_means_filt[t, c], states_vars_filt[t, c] = kalman_step_multicontext(states_means_filt[t-1, c], states_vars_filt[t-1, c], y[t], A, Q, H, R, lam=prior_context_probabilities[t, c])

    return states_means_filt, states_vars_filt


def kalman_fit_filter_multicontext(y, prior_context_probabilities, n_iter):
    """
    Multi-context Kalman filter with soft context assignments (priors).
    
    Fits a separate KF per context using observations where that context is most
    probable, then runs all KFs in parallel and combines filtered states estimates weighted by λ.
    
    Parameters
    ----------
    y : np.array
        1D array of observations of length T
    prior_context_probabilities : np.array
        Context priors of shape [T, C] where C is number of contexts.
        prior_context_probabilities[t, c] = P(context=c | rule/design, time t)
        Used to weight KF updates. Each row should sum to 1.
    n_iter : int
        Number of EM iterations for fitting each Kalman filter
    
    Returns
    -------
    states_means_filt : np.array
        Weighted aggregate filtered estimates means [T]
    states_covariances_filt : np.array
        Weighted aggregate filtered estimates covariances [T]
    kfs : list
        List of fitted KalmanFilter objects, one per context
    """
    n_ctx = prior_context_probabilities.shape[1]  # [T, C]
    
    # Convert soft probabilities to hard assignments for fitting
    contexts = np.argmax(prior_context_probabilities, axis=1)  # [T] integer array
    
    # Fit a KF for each context using the existing helper
    kfs = kalman_fit_multicontext(y, contexts, n_ctx, n_iter)
    
    # Filter the sequence with all KFs in parallel and combine estimates weighted by context probabilities
    states_means_filt, states_vars_filt = kalman_filtering_multicontext(y, prior_context_probabilities, kfs)

    return states_means_filt, states_vars_filt, kfs
   

def kalman_online_fit_predict_multicontext(y, prior_context_probabilities, n_iter, return_per_ctx=False):
    """
    Multi-context Kalman filter returning one-step-ahead predictions.
    
    Like kalman_fit_multicontext but returns predictions for y[t+1] given y[0:t].
    
    For consistency with kalman_fit_predict (single-context), predictions start
    at index MIN_OBS_FOR_EM, so the output has length T - MIN_OBS_FOR_EM.
    This ensures both single-context and multi-context predictions align with
    y[MIN_OBS_FOR_EM:] for MSE computation.
    
    Parameters
    ----------
    y : np.array
        1D array of observations of length T
    prior_context_probabilities : np.array
        Context priors of shape [T, C]
    n_iter : int
        Number of EM iterations
    
    Returns
    -------
    y_hat : np.array
        One-step-ahead predictions of length T (previously: length T-MIN_OBS_FOR_EM).
        y_hat[t] is the prediction for y[t] given y[0:t]. 
        (previously: y_hat[t] is the prediction for y[t+MIN_OBS_FOR_EM] given y[0:t+MIN_OBS_FOR_EM].)
    s_hat : np.array
        Prediction standard deviations of length T (previously: length T-MIN_OBS_FOR_EM)..
    kfs : list
        Fitted KalmanFilter objects
    
    Note
    ----
    Although this function fits KFs once on all data (not refitting at each step like
    the single-context version), we still skip the first MIN_OBS_FOR_EM predictions
    for consistency. This ensures that both kalman_fit_predict and 
    kalman_fit_predict_multicontext return predictions aligned with y[MIN_OBS_FOR_EM:].
    """
    T = len(y)
    n_ctx = prior_context_probabilities.shape[1]
    
    # Output arrays aligned with y[MIN_OBS_FOR_EM:] for consistency with single-context version
    per_ctx_pred_means = np.ones((T, n_ctx)) * np.nan
    per_ctx_pred_vars = np.ones((T, n_ctx)) * np.nan
    # y_hat = np.zeros(T - MIN_OBS_FOR_EM)
    # s_hat = np.zeros(T - MIN_OBS_FOR_EM)
    y_hat = np.ones(T) * np.nan
    s_hat = np.ones(T) * np.nan

    kfs_fitted = [None for i in range(n_ctx)]
    A, Q, H, R = None, None, None, None
    
    # Get contexts as most probable contexts from context probabilities (used for fitting)
    contexts = np.argmax(prior_context_probabilities, axis=1)

    y_ctx_masked = [] # List of y with context-specific masks
    for c in range(n_ctx):
        y_ctx_masked.append(np.ma.array(y, mask=(contexts != c))) # Mask (mask==True) when y[t] does not belong to context c (ctx[t]!=c)
            
    # Process all timesteps, but only store predictions starting from MIN_OBS_FOR_EM.
    # Update each context KF weighted by the context's probability at time t.
    # This matches how each KF was *fitted*: kalman_fit_multicontext uses masked arrays so
    # KF c only received observation updates at timesteps where context c was active.
    for t in range(T - 1):
    # for t in range(MIN_OBS_FOR_EM - 1, T - 1): # --> first indices skipped context-specifically, see "if... continue" below

        # # Output index offset by MIN_OBS_FOR_EM - 1
        # out_idx = t - (MIN_OBS_FOR_EM - 1)

        for c in range(n_ctx):

            # Only start the process if number of unmasked observations for context c until and including t) >= MIN_OBS_FOR_EM, otherwise we won't have enough observations to fit the KF for context c. This is consistent with the single-context version where we also start predictions at index MIN_OBS_FOR_EM.
            if np.sum(~y_ctx_masked[c][:t+1].mask) < MIN_OBS_FOR_EM:
                continue

            # IF current obs is masked in current context, skip EM and reuse last fitted parameters to run the filter (besides, this will only propagate the prediction step)
            # If not, update the KF with EM
            if not np.ma.is_masked(y_ctx_masked[c][t]) or kfs_fitted[c] is None: 
                # EM-fit Kalman filter for current context on observations y[0:t+1] (indices 0 to t inclusive) and filter
                kfs_fitted[c] = kalman_fit(y_ctx_masked[c][:t+1], n_iter)   
            
            # Get fitted parameters for one-step-ahead prediction (with defaults for None)
            A, Q, H, R = _get_kf_params(kfs_fitted[c])
            state_means_filt, state_covariances_filt = kalman_filtering(y_ctx_masked[c][:t+1], kfs_fitted[c]) # NOTE: using this rather than below as no need to process BOTH contexts while we're only considering one atm
            # states_means_filt, states_covariances_filt = kalman_filtering_multicontext(y_ctx_masked[c][:t+1], prior_context_probabilities, kfs)
            # Or use: kalman_step_multicontext(per_ctx_state_means_filt, per_ctx_state_covariances_filt, y_ctx_masked[c][:t+1], A, Q, H, R, lam=prior_context_probabilities[t, c])
            
            # Use the LAST filtered state (index -1, which corresponds to time t) to predict y[t+1]
            x_last = state_means_filt[-1]  # Last filtered state mean
            P_last = state_covariances_filt[-1]  # Last filtered state covariance
            
            ### One-step-ahead prediction for y[t+1] using the KF equations + last fitted parameters:
            # One-step-ahead state prediction: x_{t+1|t} = A @ x_{t|t}
            x_next_pred = A @ x_last
            P_next_pred = A @ P_last @ A.T + Q 
            
            #  One-step-ahead observation prediction: y_{t+1|t} = H @ x_{t+1|t}
            per_ctx_pred_means[t+1, c] = (H @ x_next_pred)[0]
            per_ctx_pred_vars[t+1, c] = (H @ P_next_pred @ H.T + R)[0, 0]


        # Aggregate using context belief at time t (the last observed timestep).
        # y_hat[out_idx], s_hat[out_idx] = _aggregate_contexts(
        y_hat[t+1], s_hat[t+1] = _aggregate_contexts(
            per_ctx_pred_means[t+1,:], per_ctx_pred_vars[t+1,:], prior_context_probabilities[t, :]
        )

    if return_per_ctx:
        return y_hat, s_hat, kfs_fitted, per_ctx_pred_means, per_ctx_pred_vars
    else:
        return y_hat, s_hat, kfs_fitted


def kalman_fit_filter_multicontext_multirule(y, per_rule_prior_context_probabilities, prior_rule_probabilities, n_iter, return_per_ctx=False):
    """Multi-context Kalman filter with an additional rule layer.

    Extends :func:`kalman_fit_multicontext` by letting each rule carry its own
    set of context probabilities.  The marginalized context probability at time t
    is obtained by marginalising over rules::

        P(c | t) = sum_r  P(rule=r | t) * P(c | rule=r, t)

    These marginalized probabilities are then used identically to the
    ``probabilities`` argument of :func:`kalman_fit_multicontext`.

    Parameters
    ----------
    y : np.array, shape [T]
        Observation sequence.
    per_rule_probabilities : np.array, shape [R, T, C]
        Context responsibilities conditioned on each rule:
        per_rule_responsibilities[r, t, c] = P(context=c | rule=r, t).
    rule_responsibilities : np.array, shape [T, R]
        Rule probabilities: rule_responsibilities[t, r] = P(rule=r | t).
    n_iter : int
        Number of EM iterations.

    Returns
    -------
    y_hat : np.array, shape [T]
        Filtered state estimates (weighted by marginalized context probs).
    s_hat : np.array, shape [T]
        Filtered standard deviations.
    kfs : list
        Fitted KalmanFilter objects, one per context.
    marg_resp : np.array, shape [T, C]
        The marginalized context responsibilities used internally.
    """
    marg_ctx_prob = _compute_marginal_context_probabilities(per_rule_prior_context_probabilities, prior_rule_probabilities)
    if return_per_ctx:
        y_hat, s_hat, kfs, per_ctx_pred_means, per_ctx_pred_vars = kalman_fit_filter_multicontext(y, marg_ctx_prob, n_iter, return_per_ctx=return_per_ctx) # TODO: check if appropriate
        return y_hat, s_hat, kfs, marg_ctx_prob, per_ctx_pred_means, per_ctx_pred_vars
    else:
        y_hat, s_hat, kfs = kalman_fit_filter_multicontext(y, marg_ctx_prob, n_iter, return_per_ctx=return_per_ctx)
        return y_hat, s_hat, kfs, marg_ctx_prob


def kalman_online_fit_predict_multicontext_multirule(y, per_rule_prior_context_probabilities, prior_rule_probabilities, n_iter, return_per_ctx=False):
    """Multi-context KF with a rule layer, returning one-step-ahead predictions.

    Extends :func:`kalman_fit_predict_multicontext` by supporting R rules, each
    with its own context priors.  Marginalized context probabilities are
    computed by marginalising over rules::

        P(c | t) = sum_r  P(rule=r | t) * P(c | rule=r, t)

    These are then passed directly to :func:`kalman_fit_predict_multicontext`.

    Parameters
    ----------
    y : np.array, shape [T]
        Observation sequence.
    per_rule_prior_context_probabilities : np.array, shape [R, T, C]
        Context priors conditioned on each rule:
        per_rule_prior_context_probabilities[r, t, c] = P(context=c | rule=r, t).
    prior_rule_probabilities : np.array, shape [T, R]
        Rule priors: prior_rule_probabilities[t, r] = P(rule=r | t).
    n_iter : int
        Number of EM iterations.

    Returns
    -------
    y_hat : np.array, shape [T - MIN_OBS_FOR_EM]
        One-step-ahead predictions.  y_hat[t] predicts y[t + MIN_OBS_FOR_EM].
    s_hat : np.array, shape [T - MIN_OBS_FOR_EM]
        Prediction standard deviations.
    kfs : list
        Fitted KalmanFilter objects, one per context.
    marg_resp : np.array, shape [T, C]
        The marginalized context responsibilities used internally.
    """
    marg_ctx_prob = _compute_marginal_context_probabilities(per_rule_prior_context_probabilities, prior_rule_probabilities)
    if return_per_ctx:
        y_hat, s_hat, kfs, per_ctx_pred_means, per_ctx_pred_vars = kalman_online_fit_predict_multicontext(y, marg_ctx_prob, n_iter, return_per_ctx=return_per_ctx)
        return y_hat, s_hat, kfs, marg_ctx_prob, per_ctx_pred_means, per_ctx_pred_vars
    else:
        y_hat, s_hat, kfs = kalman_online_fit_predict_multicontext(y, marg_ctx_prob, n_iter, return_per_ctx=return_per_ctx)
        return y_hat, s_hat, kfs, marg_ctx_prob


############################

def likelihood_observation(y, mu, sigma):
    """Evaluate likelihood for an observation y of belonging to a normal distribution defined by my and sigma
    If y is an array, and mu and sigma are scalars, all observations of y will be evaluated under the same distribution defined by mu and sigma.
    If y is an array, but mu and sigma are arraus, each observation within y will be evaluated under the distribution defined by each value from mu and sigma, respectively.
    """
    likelihood = (1 / np.sqrt(2 * np.pi * sigma)) * np.exp(-0.5 * (y - mu)**2 / sigma)
    return likelihood

def likelihood_observation_cond_ctx(y, c, t, n_iter, prior_ctx_probs, prior_rule_probs=None):
    """Compute likelihood of observation conditional on belonging to a certain context c, at a certain time step t
    I.e. evaluate:
        P(y_t | ctx = c, y_{1:t-1}) = N(y_t; hat{y}_t^c, hat{Sigma}_t^c) 
        
    If y_hat, s_hat, kfs, per_ctx_pred_means, per_ctx_pred_vars = kalman_online_fit_predict_...()
    Then  hat{y}^c = per_ctx_pred_means[c],  hat{y}^c = per_ctx_pred_vars[c]

    NOTE: in the context of one-step-ahead prediction
    """
    if prior_rule_probs is None:
        _, _, _, per_ctx_pred_means, per_ctx_pred_vars = kalman_online_fit_predict_multicontext(y, prior_ctx_probs, n_iter, return_per_ctx=True)
    else:
        _, _, _, per_ctx_pred_means, per_ctx_pred_vars = kalman_online_fit_predict_multicontext_multirule(y, prior_ctx_probs, prior_rule_probs, n_iter, return_per_ctx=True)
    
    # For a single timestep t, evaluate Gaussian likelihood:
    likelihood = likelihood_observation(y=y[t], mu=per_ctx_pred_means[c][t], sigma=per_ctx_pred_vars[c][t])
    return likelihood













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

    
