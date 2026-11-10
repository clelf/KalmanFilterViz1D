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
        x_est[t] = C * x_pred + K * (measurements[t] - C * x_pred) # align with pykalman
        s_est[t] = (1 - K) * s_pred 

    return x_est, s_est



def _get_kf_params(kf):
    """Helper to get KF parameters with defaults for None values."""
    A = kf.transition_matrices if kf.transition_matrices is not None else np.array([[0.99, 0.01], [0.0, 1.0]])
    Q = kf.transition_covariance if kf.transition_covariance is not None else np.eye(2) * 0.1
    H = kf.observation_matrices if kf.observation_matrices is not None else np.array([[1.0, 1.0]]) # suggestion from Claude to 
    R = kf.observation_covariance if kf.observation_covariance is not None else np.array([[1.0]])
    return A, Q, H, R


def _init_context_states(y_ctx_masked, n_ctx):
    """
    Compute data-driven initial state estimates for each context,
    to be passed into KalmanFilter() BEFORE calling .em().

    Parameters
    ----------
    y_ctx_masked : list of masked arrays, one per context
    n_ctx : int

    Returns
    -------
    mus    : list of np.array shape (2,)  — initial state means
    Sigmas : list of np.array shape (2,2) — initial state covariances
    """
    mus, Sigmas = [], []
    for c in range(n_ctx):
        obs = y_ctx_masked[c].compressed()  # unmasked values for this context
        if len(obs) == 0:
            mus.append(np.zeros(2))
            Sigmas.append(np.eye(2))
        else:
            mu0 = np.array([obs[0], 0.0])
            sigma0 = np.eye(2) * np.var(obs)     
            sigma0[1, 1] = np.var(obs)           
            mus.append(mu0)
            Sigmas.append(sigma0)
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

    Zero-weight contexts are dropped rather than multiplied by 0, because a context that has not
    yet warmed up carries NaN and 0 * NaN = NaN would poison the sum: an inactive context would
    wipe out a valid prediction from the active one. A NaN in an *active* (non-zero weight)
    context still propagates, which is intended -- those timesteps are genuinely unpredictable.
    """
    lambda_t = np.asarray(lambda_t, dtype=float)
    per_ctx_means = np.asarray(per_ctx_means, dtype=float)
    per_ctx_vars = np.asarray(per_ctx_vars, dtype=float)

    active = lambda_t > 0
    y_hat = np.sum(np.where(active, lambda_t * per_ctx_means, 0.0))
    within_var = per_ctx_vars
    between_var = (per_ctx_means - y_hat) ** 2
    s_hat = np.sqrt(np.sum(np.where(active, lambda_t * (within_var + between_var), 0.0)))
    return y_hat, s_hat


def contexts_to_probabilities(contexts, n_ctx):
    """
    Convert context labels to one-hot responsibilities.
    
    This function allows to transform context labels obtained from generative models
    into responsibilities that are expected by the Kalman filter.
    
    Parameters
    ----------
    contexts : np.array
        1D integer array of shape (T,) with values in [0, n_ctx-1]
    n_ctx : int
        Number of contexts
    
    Returns
    -------
    responsibilities : np.array
        One-hot encoded array of shape (T, n_ctx) where each row sums to 1.0
    """
    T = len(contexts)
    probabilities = np.zeros((T, n_ctx))
    probabilities[np.arange(T), contexts.astype(int)] = 1.0
    return probabilities


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

def kalman_fit(y, n_iter, initial_state_mean=None, initial_state_covariance=None, em_vars='all', observation_noise=None):
    """
    Fit a Kalman filter using EM.
    
    Parameters
    ----------
    y : array-like
        Observations to fit on (may be a masked array).
    n_iter : int
        Number of EM iterations.
    initial_state_mean : np.array of shape (n_dim_state,), optional
        Warm-start for the initial state mean. If None, pykalman defaults to zeros.
    initial_state_covariance : np.array of shape (n_dim_state, n_dim_state), optional
        Warm-start for the initial state covariance. If None, pykalman defaults to identity.
    em_vars : str or list, optional
        Which parameters EM should estimate. When a warm-start is provided, you
        typically want to exclude 'initial_state_mean' and 'initial_state_covariance'
        so EM doesn't overwrite them.
    observation_noise : float or np.array, optional
        Fixed observation noise, as a VARIANCE (R = sigma_r**2), not a standard deviation:
        it is assigned straight to pykalman's `observation_covariance`. When provided, R is
        held at this value and excluded from EM. When None, R must be in `em_vars` for EM to
        estimate it -- otherwise pykalman silently leaves it at its default of 1.0.

    Elapsed time (s): 1.15, n_iter: 5, n_samples: 1000
    Elapsed time (s): 2.28, n_iter: 10, n_samples: 1000
    """
    obs_cov = None if observation_noise is None else np.array([[observation_noise]])
    
    # If observation_noise is fixed, remove it from EM variables
    if observation_noise is not None and em_vars == 'all':
        em_vars = ['transition_matrices', 'transition_covariance']
    elif observation_noise is not None and isinstance(em_vars, list) and 'observation_covariance' in em_vars:
        em_vars = [v for v in em_vars if v != 'observation_covariance']
    
    kf = KalmanFilter(
        observation_matrices=np.array([[1.0, 1.0]]),
        observation_covariance=obs_cov,
        initial_state_mean=initial_state_mean,       # None → pykalman uses zeros
        initial_state_covariance=initial_state_covariance,  # None → pykalman uses eye
        n_dim_state=2,
        n_dim_obs=1
    )
    kf_fitted = kf.em(y, n_iter=n_iter, em_vars=em_vars)
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
    Multi-context Kalman filter with probabilistic context responsibilities (priors).
    
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
   

def kalman_online_fit_predict_multicontext(y, prior_context_probabilities, n_iter, return_per_ctx=False, observation_noise=None, return_params=False):
    """
    Multi-context Kalman filter returning one-step-ahead predictions.
    
    Like kalman_fit_multicontext but returns predictions for y[t+1] given y[0:t], but re-fitting with EM at each new observation
    
    Parameters
    ----------
    y : np.array
        1D array of observations of length T
    prior_context_probabilities : np.array
        Context priors of shape [T, C]
    n_iter : int
        Number of EM iterations
    observation_noise : float or np.array, optional
        Fixed observation noise, as a VARIANCE (R = sigma_r**2), not a standard deviation.
        If provided, R is held at this value and excluded from EM. If None, EM estimates it.
    return_params : bool, optional
        If True, also return the per-timestep, per-context fitted KF parameters (A, Q, H, R)
        used to produce each prediction. See the `params` return below.

    Returns
    -------
    y_hat : np.array
        One-step-ahead predictions of length T (previously: length T-MIN_OBS_FOR_EM).
        y_hat[t] is the prediction for y[t] given y[0:t].
        (previously: y_hat[t] is the prediction for y[t+MIN_OBS_FOR_EM] given y[0:t+MIN_OBS_FOR_EM].)
    s_hat : np.array
        Prediction standard STANDARD DEVIATIONS of length T (sqrt of the total variance; see
        _aggregate_contexts).
    kfs : list
        Fitted KalmanFilter objects
    per_ctx_pred_means, per_ctx_pred_vars : np.array, shape [T, C]
        Only when return_per_ctx=True. NOTE the asymmetry with s_hat above:
        per_ctx_pred_vars holds VARIANCES (H P H' + R), not standard deviations. Callers that
        write these to disk under a `sigma_*` name must sqrt them first; callers that feed them
        to likelihood_observation (which is variance-parameterized) must not.
    params : dict, optional
        Only when return_params=True. Maps 'A', 'Q', 'H', 'R' to the fitted parameter
        trajectories, each an array indexed [t, c, ...] aligned with per_ctx_pred_means:
        params['X'][t, c] is the value of parameter X for context c that produced the
        one-step-ahead prediction of y[t] (fitted on y[0:t]). Shapes: A, Q -> [T, C, 2, 2];
        H -> [T, C, 1, 2]; R -> [T, C, 1, 1]. R is a VARIANCE. Entries are NaN wherever no
        prediction exists (a context has not yet reached MIN_OBS_FOR_EM observations), and are
        carried forward from the last fit at timesteps where that context is not re-fitted
        (masked observation), exactly like per_ctx_pred_means/vars.

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

    # Per-timestep, per-context fitted parameters (only stored/returned when requested). Same
    # NaN-then-fill / carry-forward semantics as per_ctx_pred_means: the value at [t+1, c] is
    # the parameter set that produced the prediction of y[t+1] for context c. Shapes follow the
    # fixed 2-state / 1-obs model (see kalman_fit): A, Q are 2x2, H is 1x2, R is 1x1.
    if return_params:
        param_A = np.full((T, n_ctx, 2, 2), np.nan)
        param_Q = np.full((T, n_ctx, 2, 2), np.nan)
        param_H = np.full((T, n_ctx, 1, 2), np.nan)
        param_R = np.full((T, n_ctx, 1, 1), np.nan)

    # EM must estimate the observation noise unless we are fixing it. Leaving
    # 'observation_covariance' out of em_vars while passing observation_noise=None means R is
    # neither supplied nor estimated, and pykalman silently holds it at its default of 1.0.
    em_vars = ['transition_matrices', 'transition_covariance']
    if observation_noise is None:
        em_vars = em_vars + ['observation_covariance']

    # Get contexts as most probable contexts from context probabilities (used for fitting)
    contexts = np.argmax(prior_context_probabilities, axis=1)

    # List of y with context-specific masks
    y_ctx_masked = [] # List of y with context-specific masks
    for c in range(n_ctx):
        y_ctx_masked.append(np.ma.array(y, mask=(contexts != c))) # Mask (mask==True) when y[t] does not belong to context c (ctx[t]!=c)
            
    # Initialize states
    mus, Sigmas = _init_context_states(y_ctx_masked, n_ctx)

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
                # kfs_fitted[c] = kalman_fit(y_ctx_masked[c][:t+1], n_iter) 
                kfs_fitted[c] = kalman_fit(
                    y_ctx_masked[c][:t+1],
                    n_iter,
                    initial_state_mean=mus[c],
                    initial_state_covariance=Sigmas[c],
                    em_vars=em_vars,
                    observation_noise=observation_noise
                )

                # Get fitted parameters
                A, Q, H, R = _get_kf_params(kfs_fitted[c])

                # Filter
                state_means_filt, state_covariances_filt = kalman_filtering(y_ctx_masked[c][:t+1], kfs_fitted[c]) # NOTE: using this rather than below as no need to process BOTH contexts while we're only considering one atm
                # states_means_filt, states_covariances_filt = kalman_filtering_multicontext(y_ctx_masked[c][:t+1], prior_context_probabilities, kfs)
                # Or use: kalman_step_multicontext(per_ctx_state_means_filt, per_ctx_state_covariances_filt, y_ctx_masked[c][:t+1], A, Q, H, R, lam=prior_context_probabilities[t, c])
            
                # Use the last filtered state (last time step at index -1, which corresponds to time t) to predict y[t+1]
                x_last = state_means_filt[-1]  # Last filtered state mean
                P_last = state_covariances_filt[-1]  # Last filtered state covariance
            
                ### One-step-ahead prediction for y[t+1] using the KF equations + last fitted parameters:
                # One-step-ahead state prediction: x_{t+1|t} = A @ x_{t|t}
                x_next_pred = A @ x_last
                P_next_pred = A @ P_last @ A.T + Q 
                
                #  One-step-ahead observation prediction: y_{t+1|t} = H @ x_{t+1|t}
                per_ctx_pred_means[t+1, c] = (H @ x_next_pred)[0]
                per_ctx_pred_vars[t+1, c] = (H @ P_next_pred @ H.T + R)[0, 0]

                # Record the parameters that produced this prediction (fit on y[0:t+1]).
                if return_params:
                    param_A[t+1, c] = A
                    param_Q[t+1, c] = Q
                    param_H[t+1, c] = H
                    param_R[t+1, c] = R

            else:
                # Re-use the last predicted value for this context:
                per_ctx_pred_means[t+1, c] = per_ctx_pred_means[t, c]
                per_ctx_pred_vars[t+1, c] = per_ctx_pred_vars[t, c]

                # Carry the last fitted parameters forward alongside the reused prediction.
                if return_params:
                    param_A[t+1, c] = param_A[t, c]
                    param_Q[t+1, c] = param_Q[t, c]
                    param_H[t+1, c] = param_H[t, c]
                    param_R[t+1, c] = param_R[t, c]

                pass


        # Aggregate using context belief at time t (the last observed timestep).
        # y_hat[out_idx], s_hat[out_idx] = _aggregate_contexts(
        y_hat[t+1], s_hat[t+1] = _aggregate_contexts(
            per_ctx_pred_means[t+1,:], per_ctx_pred_vars[t+1,:], prior_context_probabilities[t, :] # TODO - Note: prior prior_context_probabilities[t, :] could be predicted_context_probabilities(given last context)[t+1,:]
        )

    params = {'A': param_A, 'Q': param_Q, 'H': param_H, 'R': param_R} if return_params else None

    if return_per_ctx and return_params:
        return y_hat, s_hat, kfs_fitted, per_ctx_pred_means, per_ctx_pred_vars, params
    elif return_per_ctx:
        return y_hat, s_hat, kfs_fitted, per_ctx_pred_means, per_ctx_pred_vars
    elif return_params:
        return y_hat, s_hat, kfs_fitted, params
    else:
        return y_hat, s_hat, kfs_fitted



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





#### Work in progress ###

def _posterior_context_probabilities(y_t, prior_context_probabilities_t, pred_means, pred_vars):
    """
    Probabilities that y_t comes from each context once y_t is heard, lambda_tilde_t^c = p(c_t = c | y_{1:t})
    (Eq. 6' of derivations-tb-edited.md): Bayes' rule with the prior lambda_t^c = p(c_t = c | y_{1:t-1})
    and the likelihood N(y_t; pred_means[c], pred_vars[c]) of y_t under the prediction of context c.

    Computed from log-likelihoods shifted by their maximum over the contexts of non-zero prior, so that
    the likelihoods of tones far from every prediction do not all round to 0 (0 / 0). Contexts of zero
    prior keep a probability of exactly 0.

    Parameters
    ----------
    y_t : float
        Observation at tone t.
    prior_context_probabilities_t : np.array, shape (C,)
        lambda_t^c, probabilities of the contexts before y_t is heard (sum to 1).
    pred_means, pred_vars : np.array, shape (C,)
        Mean and VARIANCE (state variance + observation noise variance) of the prediction of y_t
        given y_{1:t-1}, for each context.

    Returns
    -------
    posterior_context_probabilities_t : np.array, shape (C,)
        lambda_tilde_t^c (sum to 1).
    """
    lam = np.asarray(prior_context_probabilities_t, dtype=float)
    log_likelihoods = -0.5 * np.log(2 * np.pi * pred_vars) - 0.5 * (y_t - pred_means) ** 2 / pred_vars

    is_possible_context = lam > 0
    log_likelihood_shift = np.max(log_likelihoods[is_possible_context])
    unnormalized_posterior = np.zeros(lam.size)
    unnormalized_posterior[is_possible_context] = lam[is_possible_context] \
        * np.exp(log_likelihoods[is_possible_context] - log_likelihood_shift)
    return unnormalized_posterior / np.sum(unnormalized_posterior)


def _init_online_kf_state(y_0, context_probabilities_0, prior_state_means, prior_state_vars,
                          tau_init, b_init, process_noise_init, observation_noise):
    """
    Filter state after the first tone, for the online estimation of derivations-tb-edited.md.

    The prior p(x_1^c) of each context is updated by y_1 weighted by the posterior context
    probabilities lambda_tilde_1 (Eqs. 6', 9-10, with the prior in place of the prediction). No
    transition (x_{t-1}, x_t) exists yet, so the running sums start at zero and the parameters are at
    their start values.

    Parameters
    ----------
    y_0 : float
        First observation.
    context_probabilities_0 : np.array, shape (C,)
        Context probabilities lambda_1 = p(c_1 = c) of the first tone, before it is heard.
    prior_state_means, prior_state_vars : np.array, shape (C,)
        Mean and VARIANCE of the prior p(x_1^c) of each context.
    tau_init, b_init : np.array, shape (C,)
        Start values of tau and b of each context.
    process_noise_init : np.array, shape (C,)
        Process noise VARIANCE (sigma_q**2) of each context: its start value if it is estimated,
        its fixed value otherwise.
    observation_noise : float
        Observation noise VARIANCE (sigma_r**2).

    Returns
    -------
    kf_state : dict
        'mu', 'var' : filtering mean and variance of x_t^c given y_{1:t}, shape (C,)
        'tau', 'b', 'process_noise' : parameters used to predict the next tone, shape (C,)
        'tau_numerator_sum', 'tau_denominator_sum' : running sums N and Z of Eq. 26, shape (C,)
        'b_sum' : running sum of Eq. 31, shape (C,)
        'process_noise_sum' : running sum of Eq. 34 (one per context), shape (C,)
        'n_transitions' : number of transitions (x_{t-1}, x_t) in the running sums of each context, shape (C,)
    """
    prior_state_means = np.asarray(prior_state_means, dtype=float)
    prior_state_vars = np.asarray(prior_state_vars, dtype=float)
    lam = _posterior_context_probabilities(
        y_0, context_probabilities_0, prior_state_means, prior_state_vars + observation_noise
    )
    n_ctx = lam.size

    gain = prior_state_vars / (prior_state_vars + observation_noise)
    innovation = y_0 - prior_state_means

    kf_state = {
        'mu': prior_state_means + lam * gain * innovation,
        'var': (1 - lam * gain) * prior_state_vars + lam * (1 - lam) * (gain * innovation) ** 2,
        'tau': np.asarray(tau_init, dtype=float),
        'b': np.asarray(b_init, dtype=float),
        'process_noise': np.asarray(process_noise_init, dtype=float),
        'tau_numerator_sum': np.zeros(n_ctx),
        'tau_denominator_sum': np.zeros(n_ctx),
        'b_sum': np.zeros(n_ctx),
        'process_noise_sum': np.zeros(n_ctx),
        'n_transitions': np.zeros(n_ctx, dtype=int),
    }
    return kf_state


def online_parameter_estimates(tau_numerator_sum, tau_denominator_sum, b_sum, process_noise_sum,
                               n_transitions, tau, b, process_noise, contexts_to_update,
                               estimate_process_noise, n_transitions_before_using_estimates):
    """
    Estimates of tau (Eq. 27), b (Eq. 31) and, optionally, the process noise variance (Eq. 34, one
    estimate per context) from the running sums of derivations-tb-edited.md.

    A context keeps its current values (tau, b, process_noise) if it is not in contexts_to_update, if
    its start-up is not over (fewer than n_transitions_before_using_estimates transitions in its sums,
    or none), or, for tau only, if the estimate is <= 1 (the decay 1 - 1/tau would be zero or negative).

    Parameters
    ----------
    tau_numerator_sum, tau_denominator_sum, b_sum, process_noise_sum : np.array, shape (C,)
        Running sums N and Z (Eq. 26), of Eq. 31 and of Eq. 34.
    n_transitions : np.array of int, shape (C,)
        Number of transitions (x_{t-1}, x_t) in the running sums of each context.
    tau, b, process_noise : np.array, shape (C,)
        Current values, kept where no estimate is used (process_noise is a VARIANCE).
    contexts_to_update : np.array of bool, shape (C,)
        Contexts whose estimates may be updated.
    estimate_process_noise : bool
        If False, process_noise is returned unchanged.
    n_transitions_before_using_estimates : int or np.array of int, shape (C,)
        Start-up rule, for all contexts or for each context.

    Returns
    -------
    new_tau, new_b, new_process_noise : np.array, shape (C,)
    """
    uses_estimates = contexts_to_update & (n_transitions >= n_transitions_before_using_estimates) \
        & (n_transitions > 0)

    tau_estimate = np.divide(tau_numerator_sum, tau_denominator_sum, out=np.array(tau, dtype=float),
                             where=uses_estimates)
    new_tau = np.where(tau_estimate > 1, tau_estimate, tau)
    new_b = np.divide(b_sum, n_transitions, out=np.array(b, dtype=float), where=uses_estimates)

    if estimate_process_noise:
        new_process_noise = np.divide(process_noise_sum, n_transitions,
                                      out=np.array(process_noise, dtype=float), where=uses_estimates)
    else:
        new_process_noise = process_noise
    return new_tau, new_b, new_process_noise


def kalman_online_step_multicontext(y_t, context_probabilities_t, kf_state, observation_noise,
                                    estimate_process_noise, n_transitions_before_using_estimates,
                                    contexts_moving_t):
    """
    One tone of the online estimation of derivations-tb-edited.md, for every context: prediction of
    y_t, posterior context probabilities, update, moments of the pair (x_{t-1}, x_t) given y_{1:t},
    and new estimates of tau, b and, optionally, the process noise.

    A context that does not move at tone t (contexts_moving_t False, e.g. the deviant process of
    kalman_full.py outside the deviant's position) keeps its state as prediction (no transition), is
    still updated with its context probability, and adds no term to its running sums: its parameters
    and its number of transitions are unchanged.

    Parameters
    ----------
    y_t : float
        Observation at tone t.
    context_probabilities_t : np.array, shape (C,)
        Context probabilities lambda_t^c = p(c_t = c | y_{1:t-1}) of tone t, before y_t is heard.
        The update uses the posterior lambda_tilde_t^c computed from them (Eq. 6').
    kf_state : dict
        Filter state after tone t-1 (see _init_online_kf_state). Not modified.
    observation_noise : float
        Observation noise VARIANCE (sigma_r**2).
    estimate_process_noise : bool
        If True, the process noise variance of each context is estimated (Eqs. 34-36, one estimate
        per context). If False, it stays at kf_state['process_noise'].
    n_transitions_before_using_estimates : int or np.array of int, shape (C,)
        Start-up rule, for all contexts or for each context: the parameters of a context stay at
        their start values until this number of transitions is in its running sums; from then on,
        the estimates are used.
    contexts_moving_t : np.array of bool, shape (C,)
        Contexts whose state moves (transition of Eq. 1) between tone t-1 and tone t.

    Returns
    -------
    pred_means, pred_vars : np.array, shape (C,)
        Mean and VARIANCE of the prediction of y_t given y_{1:t-1}, for each context.
    new_kf_state : dict
        Filter state after tone t. A new dict, so that kf_state can be the common start of
        several hypotheses (e.g. the deviant positions m of kalman_full.py).
    """
    tau, b, process_noise = kf_state['tau'], kf_state['b'], kf_state['process_noise']
    mu_prev, var_prev = kf_state['mu'], kf_state['var']

    # Prediction (Eq. 6); a context that does not move keeps its state
    decay = 1 - 1 / tau
    mu_pred = np.where(contexts_moving_t, mu_prev + (b - mu_prev) / tau, mu_prev)
    var_pred = np.where(contexts_moving_t, decay ** 2 * var_prev + process_noise, var_prev)
    pred_means = mu_pred
    pred_vars = var_pred + observation_noise

    # Probabilities that y_t comes from each context once y_t is heard (Eq. 6'): lambda_tilde_t,
    # used by the update and by the moments of the pair below (Eqs. 9-10, 25)
    lam = _posterior_context_probabilities(y_t, context_probabilities_t, pred_means, pred_vars)

    # Update weighted by the posterior context probability, merged into one Gaussian (Eqs. 9-10)
    gain = var_pred / (var_pred + observation_noise)
    innovation = y_t - mu_pred
    mu = mu_pred + lam * gain * innovation
    var = (1 - lam * gain) * var_pred + lam * (1 - lam) * (gain * innovation) ** 2

    # Moments of x_{t-1} revised by y_t and covariance of (x_{t-1}, x_t), given y_{1:t} (Eq. 25).
    # smoother_gain is J_t of Eq. 23; this form avoids the division by decay of the equivalent
    # (1 - process_noise / var_pred) / decay of Eq. 25'
    smoother_gain = decay * var_prev / var_pred
    mu_prev_revised = mu_prev + smoother_gain * lam * gain * innovation
    cov_pair = (1 - lam * gain) * decay * var_prev \
        + lam * (1 - lam) * smoother_gain * (gain * innovation) ** 2
    var_prev_revised = (1 - lam * gain * decay * smoother_gain) * var_prev \
        + lam * (1 - lam) * (smoother_gain * gain * innovation) ** 2

    # Running sums of tau (Eqs. 26-27) and b (Eq. 31), one term per transition of each moving context;
    # each new term uses the parameters of this tone
    n_transitions = kf_state['n_transitions'] + contexts_moving_t
    tau_numerator_term = (b - mu_prev_revised) ** 2 + var_prev_revised
    tau_denominator_term = (mu - mu_prev_revised) * (b - mu_prev_revised) - cov_pair + var_prev_revised
    b_term = tau * mu + (1 - tau) * mu_prev_revised
    tau_numerator_sum = kf_state['tau_numerator_sum'] + np.where(contexts_moving_t, tau_numerator_term, 0)
    tau_denominator_sum = kf_state['tau_denominator_sum'] + np.where(contexts_moving_t, tau_denominator_term, 0)
    b_sum = kf_state['b_sum'] + np.where(contexts_moving_t, b_term, 0)

    # Optional: process noise variance, one estimate per context (Eqs. 34-36 without the sum over contexts)
    if estimate_process_noise:
        expected_squared_process_noise = (mu - decay * mu_prev_revised - b / tau) ** 2 \
            + var - 2 * decay * cov_pair + decay ** 2 * var_prev_revised
        process_noise_sum = kf_state['process_noise_sum'] \
            + np.where(contexts_moving_t, expected_squared_process_noise, 0)
    else:
        process_noise_sum = kf_state['process_noise_sum']

    new_tau, new_b, new_process_noise = online_parameter_estimates(
        tau_numerator_sum, tau_denominator_sum, b_sum, process_noise_sum, n_transitions,
        tau, b, process_noise, contexts_moving_t, estimate_process_noise,
        n_transitions_before_using_estimates
    )

    new_kf_state = {
        'mu': mu,
        'var': var,
        'tau': new_tau,
        'b': new_b,
        'process_noise': new_process_noise,
        'tau_numerator_sum': tau_numerator_sum,
        'tau_denominator_sum': tau_denominator_sum,
        'b_sum': b_sum,
        'process_noise_sum': process_noise_sum,
        'n_transitions': n_transitions,
    }
    return pred_means, pred_vars, new_kf_state


def kalman_online_fit_predict_multicontext_alt(y, prior_context_probabilities, observation_noise,
                                               prior_state_means, prior_state_vars, tau_init, b_init,
                                               n_transitions_before_using_estimates,
                                               process_noise=None, process_noise_init=None,
                                               return_per_ctx=False, return_params=False):
    """
    Multi-context Kalman filter returning one-step-ahead predictions, with tau and b (and optionally
    the process noise) estimated online in a single forward pass (derivations-tb-edited.md).

    Alternative to kalman_online_fit_predict_multicontext, which refits by EM on the whole history
    at every tone. Here each context has a 1-D state, x_t = x_{t-1} + (b - x_{t-1}) / tau + process
    noise; at every tone every context is predicted, the predictions are combined with weights
    lambda_t^c = prior_context_probabilities[t, c] (Eq. 2), then every context is updated with the
    posterior weight lambda_tilde_t^c computed from lambda_t^c and y_t (Eq. 6'; no argmax, no masks).
    Past tones are not read again: the estimates come from running sums.

    Parameters
    ----------
    y : np.array
        1D array of observations of length T
    prior_context_probabilities : np.array
        Context probabilities of shape [T, C]: prior_context_probabilities[t, c] = p(c_t = c | y[0:t]),
        the probability that y[t] comes from context c before y[t] is heard.
    observation_noise : float
        Observation noise as a VARIANCE (R = sigma_r**2), not a standard deviation. Known, not estimated.
    prior_state_means, prior_state_vars : np.array, shape [C]
        Mean and VARIANCE of the prior p(x_1^c) of each context.
    tau_init, b_init : np.array, shape [C]
        Start values of tau and b of each context.
    n_transitions_before_using_estimates : int
        Start-up rule: the filter uses the start values until this number of transitions is in the
        running sums, then the estimates. Estimates used from the first tones make tau and b move
        away from their true values (see kalman_online_step_multicontext).
    process_noise : np.array, shape [C], or None
        Process noise as a VARIANCE (sigma_q**2) for each context. If given, it is used as is.
        If None, it is estimated (one estimate per context), starting from process_noise_init.
    process_noise_init : np.array, shape [C], or None
        Start value of the process noise VARIANCE; only used when process_noise is None.
    return_per_ctx : bool
        If True, also return the per-context predictions.
    return_params : bool
        If True, also return the per-tone, per-context parameters used for each prediction.

    Returns
    -------
    y_hat : np.array
        One-step-ahead predictions of length T: y_hat[t] is the prediction for y[t] given y[0:t].
        At t = 0: the priors of the contexts, combined with prior_context_probabilities[0].
    s_hat : np.array
        Prediction STANDARD DEVIATIONS of length T (sqrt of the total variance; see _aggregate_contexts).
    kf_state : dict
        Filter state after the last tone (see _init_online_kf_state).
    per_ctx_pred_means, per_ctx_pred_vars : np.array, shape [T, C]
        Only when return_per_ctx=True. per_ctx_pred_vars holds VARIANCES, as in
        kalman_online_fit_predict_multicontext. At t = 0: the prior of each context.
    params : dict, optional
        Only when return_params=True. Maps 'tau', 'b', 'process_noise' to arrays [T, C]:
        params['X'][t, c] is the value of X used to predict y[t] for context c ('process_noise' is
        a VARIANCE). NaN at t = 0, where the prediction is the prior.
    """
    T = len(y)
    n_ctx = prior_context_probabilities.shape[1]

    # Process noise: fixed if given, otherwise estimated from its start value
    estimate_process_noise = process_noise is None
    if estimate_process_noise:
        process_noise_start = process_noise_init
    else:
        process_noise_start = process_noise

    per_ctx_pred_means = np.ones((T, n_ctx)) * np.nan
    per_ctx_pred_vars = np.ones((T, n_ctx)) * np.nan
    y_hat = np.ones(T) * np.nan
    s_hat = np.ones(T) * np.nan

    if return_params:
        param_tau = np.full((T, n_ctx), np.nan)
        param_b = np.full((T, n_ctx), np.nan)
        param_process_noise = np.full((T, n_ctx), np.nan)

    # First tone: the prediction is the prior; then the prior is updated by y[0]
    per_ctx_pred_means[0] = prior_state_means
    per_ctx_pred_vars[0] = np.asarray(prior_state_vars, dtype=float) + observation_noise
    y_hat[0], s_hat[0] = _aggregate_contexts(
        per_ctx_pred_means[0, :], per_ctx_pred_vars[0, :], prior_context_probabilities[0, :]
    )
    kf_state = _init_online_kf_state(
        y[0], prior_context_probabilities[0], prior_state_means, prior_state_vars,
        tau_init, b_init, process_noise_start, observation_noise
    )

    # Every context moves at every tone
    contexts_moving = np.ones(n_ctx, dtype=bool)

    for t in range(1, T):
        # Record the parameters that produce the prediction of y[t]
        if return_params:
            param_tau[t] = kf_state['tau']
            param_b[t] = kf_state['b']
            param_process_noise[t] = kf_state['process_noise']

        per_ctx_pred_means[t], per_ctx_pred_vars[t], kf_state = kalman_online_step_multicontext(
            y[t], prior_context_probabilities[t], kf_state, observation_noise,
            estimate_process_noise, n_transitions_before_using_estimates, contexts_moving
        )

        # Combine the predictions of the contexts with the probabilities of the contexts of y[t] before
        # y[t] is heard (Eq. 2)
        y_hat[t], s_hat[t] = _aggregate_contexts(
            per_ctx_pred_means[t, :], per_ctx_pred_vars[t, :], prior_context_probabilities[t, :]
        )

    params = {'tau': param_tau, 'b': param_b, 'process_noise': param_process_noise} if return_params else None

    if return_per_ctx and return_params:
        return y_hat, s_hat, kf_state, per_ctx_pred_means, per_ctx_pred_vars, params
    elif return_per_ctx:
        return y_hat, s_hat, kf_state, per_ctx_pred_means, per_ctx_pred_vars
    elif return_params:
        return y_hat, s_hat, kf_state, params
    else:
        return y_hat, s_hat, kf_state










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

    
