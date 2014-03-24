__all__ = ['normalize_baseline',
           'normalize_spikes',
           'moving_average']


import numpy as np


def normalize_baseline(signals, beta=1, mode='online', epsilon=0.01):
    if mode == 'online':
        return _normalize_baseline_online(signals, beta, epsilon)
    elif mode == 'offline':
        return _normalize_baseline_offline(signals, beta, epsilon)
    else:
        raise ValueError


def _normalize_baseline_online(signals, beta=1, epsilon=0.01):
    e = epsilon
    norm_values = [[((signal[i] + e) / (np.sum(signal[0:(i + 1)]) + e))
                    ** beta for i in range(len(signal))] for signal in signals]
    return np.array(norm_values)


def _normalize_baseline_offline(signals, beta=1, epsilon=0.01):
    e = epsilon
    norm = np.sum(signals, 1)[:, None]
    norm_values = ((signals + e) / (norm + e)) ** beta
    return norm_values


def normalize_spikes(signal, alpha=1.2):
    values = (np.abs(signal[:, 1:] - signal[:, 0:-1])) ** alpha
    return values


def moving_average(signal, n=1):
    cs = np.cumsum(signal, 1)
    cs[:, n:] = cs[:, n:] - cs[:, :-n]
    return cs[:, (n - 1):] / n
