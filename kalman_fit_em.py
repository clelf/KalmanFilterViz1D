import numpy as np
from tqdm import tqdm
from pykalman import KalmanFilter
from .kalman_core import _aggregate_contexts


# Minimum observations needed for EM algorithm (n_dim_state + 1)
# EM requires at least this many observations to estimate covariance matrices without division by zero
MIN_OBS_FOR_EM = 3


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



# For N_ctx > 1
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


# For N_ctx == 1
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




###### Never used

# (N_ctx==1) Single Kalman predict + update step using Pykalman
def kalman_step(y_t, kf, state_mean_prev, state_cov_prev):
    """Run a single Kalman step (predict + update) for one observation y_t, given the previous state mean and covariance.
    Returns updated state mean and covariance after processing y_t.
    Note: using pykalman's filter_update, works for 1 context; use _kalman_step_custom to control the update step with lambda if more than 1 context.
    """
    state_mean_new, state_cov_new = kf.filter_update(state_mean_prev, state_cov_prev, y_t)
    return state_mean_new, state_cov_new


# (N_ctx>1) Single Kalman predict + update step, manual implementation as alternative to using Pykalman
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

# (N_ctx>1) Without filtering
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

# (N_ctx>1) Without fitting
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

# (N_ctx>1) Fitting and filtering, but fitting once for the whole sequence (not recursive)
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
   
