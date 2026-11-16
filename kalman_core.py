import numpy as np


def likelihood_observation(y, mu, sigma):
    """Evaluate likelihood for an observation y of belonging to a normal distribution defined by my and sigma
    If y is an array, and mu and sigma are scalars, all observations of y will be evaluated under the same distribution defined by mu and sigma.
    If y is an array, but mu and sigma are arraus, each observation within y will be evaluated under the distribution defined by each value from mu and sigma, respectively.
    """
    likelihood = (1 / np.sqrt(2 * np.pi * sigma)) * np.exp(-0.5 * (y - mu)**2 / sigma)
    return likelihood


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

