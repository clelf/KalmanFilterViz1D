import numpy as np
from .kalman_core import _aggregate_contexts


# Single step filtering
def kalman_online_step_multicontext(y_t, context_probabilities_t, kf_state, observation_noise,
                                    estimate_process_noise, n_transitions_before_using_estimates,
                                    contexts_updating_t):
    """
    One tone of the online estimation of derivations-tb-edited.md, for every context: prediction of
    y_t, posterior context probabilities, update, moments of the pair (x_{t-1}, x_t) given y_{1:t},
    and new estimates of tau, b and, optionally, the process noise.

    A context that is not updating at tone t (contexts_updating_t False, e.g. the deviant process of
    kalman_full.py outside the deviant's position) keeps its state as prediction (no transition), still
    has its state revised by the observation with its context probability, and adds no term to its running sums: its parameters
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
    contexts_updating_t : np.array of bool, shape (C,)
        Contexts that follow their transition (Eq. 1) between tone t-1 and tone t.

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

    # Prediction (Eq. 6); a context that is not updating keeps its state
    decay = 1 - 1 / tau
    mu_pred = np.where(contexts_updating_t, mu_prev + (b - mu_prev) / tau, mu_prev)
    var_pred = np.where(contexts_updating_t, decay ** 2 * var_prev + process_noise, var_prev)
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

    # Running sums of tau (Eqs. 26-27) and b (Eq. 31), one term per transition of each context being updated;
    # each new term uses the parameters of this tone
    n_transitions = kf_state['n_transitions'] + contexts_updating_t
    tau_numerator_term = (b - mu_prev_revised) ** 2 + var_prev_revised
    tau_denominator_term = (mu - mu_prev_revised) * (b - mu_prev_revised) - cov_pair + var_prev_revised
    b_term = tau * mu + (1 - tau) * mu_prev_revised
    tau_numerator_sum = kf_state['tau_numerator_sum'] + np.where(contexts_updating_t, tau_numerator_term, 0)
    tau_denominator_sum = kf_state['tau_denominator_sum'] + np.where(contexts_updating_t, tau_denominator_term, 0)
    b_sum = kf_state['b_sum'] + np.where(contexts_updating_t, b_term, 0)

    # Optional: process noise variance, one estimate per context (Eqs. 34-36 without the sum over contexts)
    if estimate_process_noise:
        expected_squared_process_noise = (mu - decay * mu_prev_revised - b / tau) ** 2 \
            + var - 2 * decay * cov_pair + decay ** 2 * var_prev_revised
        process_noise_sum = kf_state['process_noise_sum'] \
            + np.where(contexts_updating_t, expected_squared_process_noise, 0)
    else:
        process_noise_sum = kf_state['process_noise_sum']

    new_tau, new_b, new_process_noise = online_parameter_estimates(
        tau_numerator_sum, tau_denominator_sum, b_sum, process_noise_sum, n_transitions,
        tau, b, process_noise, contexts_updating_t, estimate_process_noise,
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


# Online fit-and-filter over a whole sequence. Not used in HAudIn. 
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

    # Every context is updated at every tone
    contexts_updating = np.ones(n_ctx, dtype=bool)

    for t in range(1, T):
        # Record the parameters that produce the prediction of y[t]
        if return_params:
            param_tau[t] = kf_state['tau']
            param_b[t] = kf_state['b']
            param_process_noise[t] = kf_state['process_noise']

        per_ctx_pred_means[t], per_ctx_pred_vars[t], kf_state = kalman_online_step_multicontext(
            y[t], prior_context_probabilities[t], kf_state, observation_noise,
            estimate_process_noise, n_transitions_before_using_estimates, contexts_updating
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

