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
    
    # Step 3: Run all KFs on the full observation sequence in parallel.
    # Each KF maintains its own state trajectory.
    per_ctx_means = np.zeros((n_ctx, T))  # [C, T]
    per_ctx_vars = np.zeros((n_ctx, T))   # [C, T]

    mus, Sigmas = _init_context_states(kfs)

    for t in range(T):
        for c in range(n_ctx):
            A, Q, H, R = _get_kf_params(kfs[c])
            # Responsibility-gated update: scale the Kalman gain by λ[t,c].
            # This matches the masked-array EM fitting where KF c only received
            # observation updates when context c was dominant.
            mus[c], Sigmas[c], _, _ = _kalman_step(mus[c], Sigmas[c], y[t], A, Q, H, R, lam=responsibilities[t, c])
            # Store filtered observation estimate
            per_ctx_means[c, t] = (H @ mus[c])[0]
            per_ctx_vars[c, t] = (H @ Sigmas[c] @ H.T + R)[0, 0]

    # Step 4: Aggregate across contexts weighted by responsibilities.
    # Uses law of total expectation/variance via _aggregate_contexts.
    y_hat = np.zeros(T)
    s_hat = np.zeros(T)
    for t in range(T):
        y_hat[t], s_hat[t] = _aggregate_contexts(per_ctx_means[:, t], per_ctx_vars[:, t], responsibilities[t, :])

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
    kalman_fit_predict_multicontext
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
    mus, Sigmas = _init_context_states(kfs)

    # Process all timesteps, but only store predictions starting from MIN_OBS_FOR_EM.
    # Update each context KF weighted by its responsibility at time t.
    # This matches how each KF was *fitted*: _fit_context_kfs uses masked arrays so
    # KF c only received observation updates at timesteps where context c was active.
    # Using full updates for all KFs here (regardless of context) would contaminate
    # "sleeping" KFs with wrong-context observations, biasing their state estimates.
    for t in range(T - 1):
        per_ctx_pred_means = np.zeros(n_ctx)
        per_ctx_pred_vars = np.zeros(n_ctx)

        for c in range(n_ctx):
            A, Q, H, R = _get_kf_params(kfs[c])
            # Responsibility-gated update: λ=1 → standard update; λ=0 → predict only.
            # This keeps the inference consistent with the masked-array EM fitting.
            mus[c], Sigmas[c], _, _ = _kalman_step(mus[c], Sigmas[c], y[t], A, Q, H, R, lam=responsibilities[t, c])

            # ONE-STEP-AHEAD: predict y[t+1] from updated state
            mu_next = A @ mus[c]
            Sigma_next = A @ Sigmas[c] @ A.T + Q
            per_ctx_pred_means[c] = (H @ mu_next)[0]
            per_ctx_pred_vars[c] = (H @ Sigma_next @ H.T + R)[0, 0]

        # Only store predictions for t+1 >= MIN_OBS_FOR_EM (i.e., t >= MIN_OBS_FOR_EM - 1)
        if t >= MIN_OBS_FOR_EM - 1:
            out_idx = t - (MIN_OBS_FOR_EM - 1)
            # Aggregate using context belief at time t (the last observed timestep).
            # Using responsibilities[t+1] would be an oracle leak: it would use the
            # true ground-truth context at the *future* timestep we are predicting.
            # responsibilities[t] is causally correct: it reflects what we know at t.
            y_hat[out_idx], s_hat[out_idx] = _aggregate_contexts(
                per_ctx_pred_means, per_ctx_pred_vars, responsibilities[t, :]
            )

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
    Fit separate Kalman filters for each context using masked arrays.

    For each context c, the full-length observation sequence is passed to
    pykalman with all timesteps where ``contexts != c`` masked out.  pykalman
    runs the prediction step (advancing the state via A and Q) at *every*
    timestep but only performs a measurement update at unmasked ones.

    This is the correct way to handle the temporal gaps between same-context
    blocks: the latent process is assumed to have continued evolving during
    the gap, we simply did not observe it.  Consequently EM estimates Q as a
    genuine 1-step transition covariance rather than an inflated mixture of
    multi-step variances (which would happen if non-contiguous observations
    were concatenated and presented as consecutive).

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
    for c in range(n_ctx):
        # Mask every timestep where context c is NOT active.
        # pykalman skips the observation update at masked steps but still
        # advances the state, so Q is estimated and applied at the correct
        # 1-step timescale across gaps.
        y_masked = np.ma.array(y, mask=(contexts != c))

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
                transition_matrices=np.array([[0.9, 0.1], [0.0, 1.0]]),
                observation_matrices=np.array([[1.0, 0.0]]),
                n_dim_state=2,
                n_dim_obs=1,
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


def _kalman_step(mu, Sigma, y_t, A, Q, H, R, lam=1.0):
    """Single Kalman predict + responsibility-gated update step.

    Implements Eqs. 9–10 of the derivation document: the update collapses the
    two-component Gaussian mixture (updated / not-updated) into a single Gaussian
    by matching the mixture mean and variance.

    Parameters
    ----------
    mu : np.array, shape (n,)
        Current state mean.
    Sigma : np.array, shape (n, n)
        Current state covariance.
    y_t : float
        Current scalar observation.
    A, Q, H, R : KF parameter matrices.
    lam : float
        Responsibility weight in [0, 1]. lam=1 gives a standard Kalman update;
        lam=0 gives a prediction-only step (no measurement update).

    Returns
    -------
    mu_new : np.array, shape (n,) — updated state mean.
    Sigma_new : np.array, shape (n, n) — updated state covariance.
    mu_pred : np.array, shape (n,) — predicted mean (before measurement update).
    Sigma_pred : np.array, shape (n, n) — predicted covariance (before update).

    Notes
    -----
    Mean update  (Eq. 9):  mu_new  = mu_pred + lam * K * eps
    Covar update (Eq. 10): Sigma_new = (I - lam*K*H) @ Sigma_pred          <- within-component
                                      + lam*(1-lam) * (K*eps) @ (K*eps)^T  <- between-component
    The between-component term is the extra contribution that arises from
    collapsing the two-Gaussian mixture into a single Gaussian and is missing
    from a naive "scaled-gain" Kalman update.
    """
    mu_pred = A @ mu
    Sigma_pred = A @ Sigma @ A.T + Q

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

    return mu_new, Sigma_new, mu_pred, Sigma_pred


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
    lambda_t      : np.array, shape (C,)  — responsibility weights (sum to 1)

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

    
