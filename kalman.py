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
    state_means_filt, state_covariances_filt = kf_fitted.filter(y) # TODO: check dimensions
    
    # return state_means_filt[:,0], state_covariances_filt[:,0,0], kf_fitted # "first component" = ?
    return state_means_filt, state_covariances_filt, kf_fitted


def kalman_fit_predict(y, n_iter):
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
        y_hat: One-step-ahead predictions of length T-MIN_OBS_FOR_EM. 
               y_hat[t] is the prediction for y[t+MIN_OBS_FOR_EM] given y[0:t+MIN_OBS_FOR_EM].
        s_hat: Corresponding prediction standard deviations of length T-MIN_OBS_FOR_EM.
        kf_fitted: The last fitted Kalman filter object (fit on full sequence).
    
    Note:
        EM requires at least MIN_OBS_FOR_EM=3 observations (n_dim_state + 1) to estimate
        covariance matrices without division by zero. Therefore, predictions start at
        index MIN_OBS_FOR_EM-1, predicting y[MIN_OBS_FOR_EM:].
    """

    T = len(y)
    y_hat = np.zeros(T - MIN_OBS_FOR_EM)
    s_hat = np.zeros(T - MIN_OBS_FOR_EM)
    kf_fitted = None
    
    # Compute one-step-ahead predictions
    # To predict y[t+1], we fit KF on y[0:t+1] and use the last filtered state
    # Start at t = MIN_OBS_FOR_EM - 1 so we have at least MIN_OBS_FOR_EM observations
    for t in tqdm(range(MIN_OBS_FOR_EM - 1, T - 1), desc="Fitting current sample's individual timesteps", leave=False):
        # Output index offset by MIN_OBS_FOR_EM - 1
        out_idx = t - (MIN_OBS_FOR_EM - 1)
              
        # Fit Kalman filter on observations y[0:t+1] (indices 0 to t inclusive)
        state_means_filt, state_covariances_filt, kf_fitted = kalman_fit(y[:t + 1], n_iter)
        
        # Get fitted parameters for one-step-ahead prediction (with defaults for None)
        A, Q, H, R = _get_kf_params(kf_fitted)
        
        # Use the LAST filtered state (index -1, which corresponds to time t)
        # to predict y[t+1]
        x_last = state_means_filt[-1]  # Last filtered state mean
        P_last = state_covariances_filt[-1]  # Last filtered state covariance
        
        # One-step-ahead prediction: x_{t+1|t} = A @ x_{t|t}
        x_next_pred = A @ x_last
        P_next_pred = A @ P_last @ A.T + Q
        
        # Observation prediction: y_{t+1|t} = H @ x_{t+1|t}
        y_hat[out_idx] = (H @ x_next_pred)[0]
        s_hat[out_idx] = np.sqrt((H @ P_next_pred @ H.T + R)[0, 0])
    
    return y_hat, s_hat, kf_fitted



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
                  Output shape: (N_samples, T)
        If True, return one-step-ahead predictions (y_hat[t] predicts y[t+1])
                  Output shape: (N_samples, T-1)
    
    Returns
    -------
    y_hats : np.array
        Shape (N_samples, T) if predict=False, (N_samples, T-1) if predict=True
    s_hats : np.array
        Shape (N_samples, T) if predict=False, (N_samples, T-1) if predict=True (standard deviations)
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

def kalman_fit_multicontext(y, responsibilities, n_iter):
    """
    Multi-context Kalman filter with soft context assignments (responsibilities).
    
    Fits a separate KF per context using observations where that context is most
    responsible, then runs all KFs in parallel and combines estimates weighted by λ.
    
    Parameters
    ----------
    y : np.array
        1D array of observations of length T
    responsibilities : np.array
        Context probabilities of shape [T, C] where C is number of contexts.
        responsibilities[t, c] = P(context=c | observation at time t)
        Each row should sum to 1.
    n_iter : int
        Number of EM iterations for fitting each Kalman filter
    
    Returns
    -------
    y_hat : np.array
        Weighted aggregate filtered estimates [T]
    s_hat : np.array
        Weighted aggregate standard deviations [T]
    kfs : list
        List of fitted KalmanFilter objects, one per context
    
    Notes
    -----
    The key insight is that we separate FITTING from INFERENCE:
    - Fitting: Each KF is fit on observations where its context is dominant (argmax)
    - Inference: All KFs run on ALL observations, outputs weighted by λ
    """
    T = len(y)
    n_ctx = responsibilities.shape[1]  # [T, C]
    
    # Step 1: Convert soft responsibilities to hard assignments for fitting
    contexts = np.argmax(responsibilities, axis=1)  # [T] integer array
    
    # Step 2: Fit a KF for each context using the existing helper
    kfs = _fit_context_kfs(y, contexts, n_ctx, n_iter)
    
    # Step 3: Run all KFs on the full observation sequence in parallel
    # Each KF maintains its own state trajectory
    per_ctx_means = np.zeros((n_ctx, T))  # [C, T]
    per_ctx_vars = np.zeros((n_ctx, T))   # [C, T]
    
    for c in range(n_ctx):
        kf = kfs[c]
        A, Q, H, R = _get_kf_params(kf)
        
        # Initialize state for this context
        if hasattr(kf, 'initial_state_mean') and kf.initial_state_mean is not None:
            mu = kf.initial_state_mean.copy()
        else:
            mu = np.zeros(2)
        if hasattr(kf, 'initial_state_covariance') and kf.initial_state_covariance is not None:
            Sigma = kf.initial_state_covariance.copy()
        else:
            Sigma = np.eye(2)
        
        for t in range(T):
            # Predict
            mu_pred = A @ mu
            Sigma_pred = A @ Sigma @ A.T + Q
            
            # Update with observation
            y_obs = np.array([[y[t]]])
            innovation = y_obs - H @ mu_pred.reshape(-1, 1)
            S = H @ Sigma_pred @ H.T + R
            K = Sigma_pred @ H.T @ np.linalg.inv(S)
            
            mu = (mu_pred.reshape(-1, 1) + K @ innovation).flatten()
            Sigma = (np.eye(2) - K @ H) @ Sigma_pred
            
            # Store filtered observation estimate
            per_ctx_means[c, t] = (H @ mu)[0]
            per_ctx_vars[c, t] = (H @ Sigma @ H.T + R)[0, 0]
    
    # Step 4: Aggregate across contexts weighted by responsibilities
    # y_hat[t] = Σ_c λ[t,c] * μ^c[t]
    # s_hat[t] = sqrt( Σ_c λ[t,c] * (σ^c[t]² + (μ^c[t] - y_hat[t])²) )  # Law of total variance
    y_hat = np.zeros(T)
    s_hat = np.zeros(T)
    
    for t in range(T):
        lambda_t = responsibilities[t, :]  # [C]
        
        # Weighted mean
        y_hat[t] = np.sum(lambda_t * per_ctx_means[:, t])
        
        # Weighted variance (law of total variance)
        within_var = per_ctx_vars[:, t]
        between_var = (per_ctx_means[:, t] - y_hat[t]) ** 2
        total_var = np.sum(lambda_t * (within_var + between_var))
        s_hat[t] = np.sqrt(total_var)
    
    return y_hat, s_hat, kfs



def kalman_fit_predict_multicontext(y, responsibilities, n_iter):
    """
    Multi-context Kalman filter returning ONE-STEP-AHEAD predictions.
    
    Like kalman_fit_multicontext but returns predictions for y[t+1] given y[0:t].
    
    For consistency with kalman_fit_predict (single-context), predictions start
    at index MIN_OBS_FOR_EM, so the output has length T - MIN_OBS_FOR_EM.
    This ensures both single-context and multi-context predictions align with
    y[MIN_OBS_FOR_EM:] for MSE computation.
    
    Parameters
    ----------
    y : np.array
        1D array of observations of length T
    responsibilities : np.array
        Context probabilities of shape [T, C]
    n_iter : int
        Number of EM iterations
    
    Returns
    -------
    y_hat : np.array
        One-step-ahead predictions of length T - MIN_OBS_FOR_EM.
        y_hat[t] is the prediction for y[t + MIN_OBS_FOR_EM] given y[0:t + MIN_OBS_FOR_EM].
    s_hat : np.array
        Prediction standard deviations of length T - MIN_OBS_FOR_EM.
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
    n_ctx = responsibilities.shape[1]
    
    # Output arrays aligned with y[MIN_OBS_FOR_EM:] for consistency with single-context version
    y_hat = np.zeros(T - MIN_OBS_FOR_EM)
    s_hat = np.zeros(T - MIN_OBS_FOR_EM)
    
    # Convert soft responsibilities to hard assignments for fitting
    contexts = np.argmax(responsibilities, axis=1)
    
    # Fit KFs using the existing helper
    kfs = _fit_context_kfs(y, contexts, n_ctx, n_iter)
    
    # Per-context state tracking
    mus = []
    Sigmas = []
    for c in range(n_ctx):
        kf = kfs[c]
        if hasattr(kf, 'initial_state_mean') and kf.initial_state_mean is not None:
            mus.append(kf.initial_state_mean.copy())
        else:
            mus.append(np.zeros(2))
        if hasattr(kf, 'initial_state_covariance') and kf.initial_state_covariance is not None:
            Sigmas.append(kf.initial_state_covariance.copy())
        else:
            Sigmas.append(np.eye(2))
    
    # Process all timesteps, but only store predictions starting from MIN_OBS_FOR_EM
    for t in range(T - 1):
        # Update all contexts with current observation y[t]
        per_ctx_pred_means = np.zeros(n_ctx)
        per_ctx_pred_vars = np.zeros(n_ctx)
        
        for c in range(n_ctx):
            kf = kfs[c]
            A, Q, H, R = _get_kf_params(kf)
            
            # Predict
            mu_pred = A @ mus[c]
            Sigma_pred = A @ Sigmas[c] @ A.T + Q
            
            # Update with y[t]
            y_obs = np.array([[y[t]]])
            innovation = y_obs - H @ mu_pred.reshape(-1, 1)
            S = H @ Sigma_pred @ H.T + R
            K = Sigma_pred @ H.T @ np.linalg.inv(S)
            
            mus[c] = (mu_pred.reshape(-1, 1) + K @ innovation).flatten()
            Sigmas[c] = (np.eye(2) - K @ H) @ Sigma_pred
            
            # ONE-STEP-AHEAD: predict y[t+1] from updated state
            mu_next = A @ mus[c]
            Sigma_next = A @ Sigmas[c] @ A.T + Q
            
            per_ctx_pred_means[c] = (H @ mu_next)[0]
            per_ctx_pred_vars[c] = (H @ Sigma_next @ H.T + R)[0, 0]
        
        # Only store predictions for t+1 >= MIN_OBS_FOR_EM (i.e., t >= MIN_OBS_FOR_EM - 1)
        if t >= MIN_OBS_FOR_EM - 1:
            out_idx = t - (MIN_OBS_FOR_EM - 1)
            
            # Aggregate using responsibilities at t+1
            lambda_next = responsibilities[t + 1, :]
            
            y_hat[out_idx] = np.sum(lambda_next * per_ctx_pred_means)
            within_var = per_ctx_pred_vars
            between_var = (per_ctx_pred_means - y_hat[out_idx]) ** 2
            s_hat[out_idx] = np.sqrt(np.sum(lambda_next * (within_var + between_var)))
    
    return y_hat, s_hat, kfs

def kalman_fit_multicontext_batch(ys, responsibilities_batch, n_iter, predict=False):
    """
    Batch version of multi-context Kalman filter.
    
    Parameters
    ----------
    ys : np.array
        2D array of observations, shape [N_samples, T]
    responsibilities_batch : np.array
        3D array of context probabilities, shape [N_samples, T, C]
    n_iter : int
        Number of EM iterations
    predict : bool
        If False: return filtered estimates [N_samples, T]
        If True: return one-step-ahead predictions [N_samples, T-1]
    
    Returns
    -------
    y_hats : np.array
        Filtered/predicted estimates
    s_hats : np.array
        Standard deviations
    """
    fit_func = kalman_fit_predict_multicontext if predict else kalman_fit_multicontext
    desc = "Multi-context KF by EM" + (" (predict)" if predict else "")
    
    y_hats, s_hats = [], []
    for y, resp in tqdm(zip(ys, responsibilities_batch), desc=desc, total=len(ys)):
        y_hat, s_hat, _ = fit_func(y, resp, n_iter)
        y_hats.append(y_hat)
        s_hats.append(s_hat)
    return np.stack(y_hats, axis=0), np.stack(s_hats, axis=0)


def kalman_fit_predict_multicontext_batch(ys, responsibilities_batch, n_iter):
    """Batch version of kalman_fit_predict_multicontext (returns one-step-ahead predictions)."""
    return kalman_fit_multicontext_batch(ys, responsibilities_batch, n_iter, predict=True)







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
        
        if len(y_ctx) >= MIN_OBS_FOR_EM:  # Need enough observations for EM
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
    A = kf.transition_matrices if kf.transition_matrices is not None else np.array([[0.99, 0.01], [0.0, 1.0]])
    Q = kf.transition_covariance if kf.transition_covariance is not None else np.eye(2) * 0.1
    H = kf.observation_matrices if kf.observation_matrices is not None else np.array([[1.0, 0.0]])
    R = kf.observation_covariance if kf.observation_covariance is not None else np.array([[1.0]])
    return A, Q, H, R


# def hmm_context_inference(y_batch, n_ctx, n_iter=100):
#     """
#     HMM-based context inference - the discrete analog of Kalman filtering.
    
#     Returns responsibilities (soft context assignments) that can be used
#     as input to kalman_fit_predict_multicontext.
    
#     Parameters
#     ----------
#     y_batch : np.array
#         Observations of shape (N_samples, T)
#     n_ctx : int
#         Number of contexts
#     n_iter : int
#         Number of EM iterations for HMM fitting
    
#     Returns
#     -------
#     responsibilities_batch : np.array
#         Context probabilities of shape (N_samples, T, n_ctx)
#     predicted_contexts : np.array
#         Hard context predictions of shape (N_samples, T)
#     """
#     responsibilities_batch = []
#     predicted_contexts = []
    
#     for y in y_batch:
#         # Fit HMM to this sequence
#         model = hmm.GaussianHMM(
#             n_components=n_ctx, 
#             covariance_type="diag",  # or "full" 
#             n_iter=n_iter
#         )
#         model.fit(y.reshape(-1, 1))
        
#         # Get soft assignments (responsibilities)
#         resp = model.predict_proba(y.reshape(-1, 1))  # Shape: (T, n_ctx)
#         responsibilities_batch.append(resp)
        
#         # Get hard predictions (Viterbi)
#         pred = model.predict(y.reshape(-1, 1))  # Shape: (T,)
#         predicted_contexts.append(pred)
    
#     return np.array(responsibilities_batch), np.array(predicted_contexts)

# def hmm_kf_benchmark(y_batch, n_ctx, n_iter_hmm=100, n_iter_kf=5):
#     """
#     Two-stage benchmark:
#     1. HMM infers context probabilities from observations
#     2. Multi-context KF uses inferred contexts to estimate states
    
#     This is the realistic benchmark for your model that jointly learns both.
#     """
#     # Stage 1: Infer contexts with HMM
#     responsibilities_batch, _ = hmm_context_inference(y_batch, n_ctx, n_iter_hmm)
    
#     # Stage 2: Use inferred contexts for KF
#     mu_pred, sigma_pred = kalman_fit_predict_multicontext_batch(
#         y_batch, responsibilities_batch, n_iter=n_iter_kf
#     )
    
#     return mu_pred, sigma_pred, responsibilities_batch


# def kalman_fit_context_aware(y, contexts, n_iter):
#     """
#     Context-aware Kalman filter that maintains separate state estimates for each context.
#     Each context has its own Kalman filter that is only updated when that context is active.
    
#     Parameters
#     ----------
#     y : np.array
#         1D array of observations of length T
#     contexts : np.array
#         1D integer array of length T indicating which context (0, 1, ..., n_ctx-1) is active
#     n_iter : int
#         Number of EM iterations for fitting each Kalman filter
    
#     Returns
#     -------
#     y_hat : np.array
#         Filtered estimates of length T. y_hat[t] is the estimate after seeing y[t].
#     s_hat : np.array
#         Corresponding standard deviations of length T
#     kfs : list
#         List of fitted KalmanFilter objects, one per context
#     """
#     T = len(y)
#     n_ctx = len(np.unique(contexts))
    
#     # Fit a separate Kalman filter to each context's observations
#     kfs = _fit_context_kfs(y, contexts, n_ctx, n_iter)
    
#     # Now run the context-aware filter: maintain state for each context
#     # and only update the active context at each timestep
#     y_hat = np.zeros(T)
#     s_hat = np.zeros(T)
    
#     # State means and covariances for each context
#     state_means = [np.zeros(2) for _ in range(n_ctx)]
#     state_covs = [np.eye(2) * 1.0 for _ in range(n_ctx)]
    
#     # Initialize states with initial state means from fitted KFs
#     for c in range(n_ctx):
#         if hasattr(kfs[c], 'initial_state_mean') and kfs[c].initial_state_mean is not None:
#             state_means[c] = kfs[c].initial_state_mean.copy()
#         if hasattr(kfs[c], 'initial_state_covariance') and kfs[c].initial_state_covariance is not None:
#             state_covs[c] = kfs[c].initial_state_covariance.copy()
    
#     for t in range(T):
#         c = contexts[t]
#         kf = kfs[c]
        
#         # Get KF parameters with defaults
#         A, Q, H, R = _get_kf_params(kf)
        
#         # Prediction step for the active context (prior prediction for y[t])
#         x_pred = A @ state_means[c]
#         P_pred = A @ state_covs[c] @ A.T + Q
        
#         # Update step with current observation y[t]
#         y_obs = np.array([[y[t]]])
#         innovation = y_obs - H @ x_pred.reshape(-1, 1)
#         S = H @ P_pred @ H.T + R
#         K = P_pred @ H.T @ np.linalg.inv(S)
        
#         state_means[c] = (x_pred.reshape(-1, 1) + K @ innovation).flatten()
#         state_covs[c] = (np.eye(2) - K @ H) @ P_pred
        
#         # Store FILTERED estimate (after seeing y[t])
#         y_hat[t] = (H @ state_means[c])[0]
#         s_hat[t] = (H @ state_covs[c] @ H.T + R)[0, 0]
    
#     return y_hat, np.sqrt(s_hat), kfs


# def kalman_fit_context_aware_predict(y, contexts, n_iter, min_obs_per_ctx=5, refit_interval=10):
#     """
#     Context-aware Kalman filter that returns ONE-STEP-AHEAD predictions.
    
#     Each prediction y_hat[t] is a one-step-ahead prediction for y[t+1],
#     based on Kalman filters whose parameters are fit using EM on observations
#     y[0:t+1] (i.e., all observations up to and including y[t]).
    
#     This ensures that prediction at time t+1 only uses information from times 0 to t
#     (no data leakage from future observations).
    
#     Parameters
#     ----------
#     y : np.array
#         1D array of observations of length T
#     contexts : np.array
#         1D integer array of length T indicating which context (0, 1, ..., n_ctx-1) is active
#     n_iter : int
#         Number of EM iterations for fitting each Kalman filter
    
#     Returns
#     -------
#     y_hat : np.array
#         One-step-ahead predictions of length T-1. y_hat[t] predicts y[t+1] given y[0:t].
#     s_hat : np.array
#         Corresponding prediction standard deviations of length T-1
#     kfs : list
#         List of fitted KalmanFilter objects, one per context (fit on full sequence)
#     """
#     T = len(y)
#     n_ctx = len(np.unique(contexts))
    
#     y_hat = np.zeros(T - 1)
#     s_hat = np.zeros(T - 1)
    
#     # State means and covariances for each context
#     state_means = [np.zeros(2) for _ in range(n_ctx)]
#     state_covs = [np.eye(2) * 1.0 for _ in range(n_ctx)]
    
#     # Current fitted KF parameters for each context (start with defaults)
#     kfs = [None] * n_ctx
#     A_default = np.array([[0.99, 0.01], [0.0, 1.0]])
#     Q_default = np.eye(2) * 0.1
#     H_default = np.array([[1.0, 0.0]])
#     R_default = np.array([[1.0]])
    
#     # Track observation counts per context
#     obs_counts = [0] * n_ctx
#     # Track cumulative observations per context for mean fallback
#     obs_sums = [0.0] * n_ctx
    
#     # Store when we last refit each context
#     last_refit = [0] * n_ctx
    
#     for t in range(T - 1):
#         # Update observation counts for current context
#         c_current = contexts[t]
#         obs_counts[c_current] += 1
#         obs_sums[c_current] += y[t]
        
#         # Fit KFs on observations y[0:t+1] using contexts[0:t+1]
#         kfs = _fit_context_kfs(y[:t + 1], contexts[:t + 1], n_ctx, n_iter)
#         for c in range(n_ctx):
#             last_refit[c] = t
#             # Re-initialize states with fitted initial conditions
#             if kfs[c] is not None:
#                 if hasattr(kfs[c], 'initial_state_mean') and kfs[c].initial_state_mean is not None:
#                     state_means[c] = kfs[c].initial_state_mean.copy()
#                 if hasattr(kfs[c], 'initial_state_covariance') and kfs[c].initial_state_covariance is not None:
#                     state_covs[c] = kfs[c].initial_state_covariance.copy()
        
#         # Get current context
#         c = contexts[t]
#         kf = kfs[c]
        
#         # Get KF parameters (use defaults if not yet fitted)
#         if kf is not None:
#             A, Q, H, R = _get_kf_params(kf)
#         else:
#             A, Q, H, R = A_default, Q_default, H_default, R_default
        
#         # Kalman filter predict step (prior)
#         x_pred = A @ state_means[c]
#         P_pred = A @ state_covs[c] @ A.T + Q
        
#         # Kalman filter update step with observation y[t]
#         y_obs = np.array([[y[t]]])
#         innovation = y_obs - H @ x_pred.reshape(-1, 1)
#         S = H @ P_pred @ H.T + R
#         K = P_pred @ H.T @ np.linalg.inv(S)
        
#         state_means[c] = (x_pred.reshape(-1, 1) + K @ innovation).flatten()
#         state_covs[c] = (np.eye(2) - K @ H) @ P_pred
        
#         # ONE-STEP-AHEAD PREDICTION: After updating with y[t], predict y[t+1]
#         # We use the NEXT context (contexts[t+1]) for the prediction
#         c_next = contexts[t + 1] if t + 1 < T else c
#         kf_next = kfs[c_next]
        
#         if kf_next is not None:
#             A_next, Q_next, H_next, R_next = _get_kf_params(kf_next)
#         else:
#             A_next, Q_next, H_next, R_next = A_default, Q_default, H_default, R_default
        
#         # Predict next state and observation
#         x_next_pred = A_next @ state_means[c_next]
#         P_next_pred = A_next @ state_covs[c_next] @ A_next.T + Q_next
        
#         y_hat[t] = (H_next @ x_next_pred)[0]
#         s_hat[t] = np.sqrt((H_next @ P_next_pred @ H_next.T + R_next)[0, 0])
    
#     # Final fit on full data (for returning)
#     kfs = _fit_context_kfs(y, contexts, n_ctx, n_iter)
    
#     return y_hat, s_hat, kfs


# def kalman_fit_context_aware_batch(ys, contexts_batch, n_iter, predict=False):
#     """
#     Batch version of context-aware Kalman filter.
    
#     Parameters
#     ----------
#     ys : np.array
#         2D array of observations, shape (N_samples, T)
#     contexts_batch : np.array
#         2D integer array of shape (N_samples, T) indicating contexts
#     n_iter : int
#         Number of EM iterations
#     predict : bool
#         If False, return filtered estimates (y_hat[t] is estimate after seeing y[0:t])
#                   Output shape: (N_samples, T)
#         If True, return one-step-ahead predictions (y_hat[t] predicts y[t+1])
#                   Output shape: (N_samples, T-1)
    
#     Returns
#     -------
#     y_hats : np.array
#         Shape (N_samples, T) if predict=False, (N_samples, T-1) if predict=True
#     s_hats : np.array
#         Shape (N_samples, T) if predict=False, (N_samples, T-1) if predict=True (standard deviations)
#     """
#     fit_func = kalman_fit_context_aware_predict if predict else kalman_fit_context_aware
#     desc = "Context-aware KF by EM" + (" (predict)" if predict else "")
    
#     y_hats, s_hats = [], []
#     for y, contexts in tqdm(zip(ys, contexts_batch), desc=desc, total=len(ys)):
#         y_hat, s_hat, _ = fit_func(y, contexts, n_iter)
#         y_hats.append(y_hat)
#         s_hats.append(s_hat)
#     return np.stack(y_hats, axis=0), np.stack(s_hats, axis=0)


# # Convenience alias for backward compatibility
# def kalman_fit_context_aware_predict_batch(ys, contexts_batch, n_iter):
#     """Batch version of kalman_fit_context_aware_predict (returns one-step-ahead predictions)."""
#     return kalman_fit_context_aware_batch(ys, contexts_batch, n_iter, predict=True)





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

    
