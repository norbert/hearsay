__all__ = ['normalize_baseline',
           'normalize_spikes',
           'moving_average']


import numpy as np


def normalize_baseline(signals, beta=1, epsilon=0.01,
                       mode='online', function=None):
    if function is None:
        function = np.sum
    if mode == 'online':
        norm_signals = _normalize_baseline_online(signals, beta, epsilon, function)
    elif mode == 'offline':
        norm_signals = _normalize_baseline_offline(signals, beta, epsilon, function)
    else:
        raise ValueError
    return norm_signals


def _normalize_baseline_online(signals, beta, epsilon, function):
    e = epsilon
    norm_signals = [[((signal[i] + e) / (function(signal[0:(i + 1)], 0) + e))
                    ** beta for i in range(len(signal))] for signal in signals]
    return np.array(norm_signals)


def _normalize_baseline_offline(signals, beta, epsilon, function):
    e = epsilon
    norm = function(signals, 1)[:, None]
    norm_signals = ((signals + e) / (norm + e)) ** beta
    return norm_signals


def normalize_spikes(signals, alpha=1.2):
    norm_signals = (np.abs(signals[:, 1:] - signals[:, 0:-1])) ** alpha
    return norm_signals


def moving_average(signals, n=1):
    cs = np.cumsum(signals, 1)
    cs[:, n:] = cs[:, n:] - cs[:, :-n]
    norm_signals = cs[:, (n - 1):] / n
    return norm_signals
