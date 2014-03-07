__all__ = ['detect_stream',
           'detect',
           'dist_to_reference',
           'dist',
           'prob_class']


import numpy as np


def detect_stream(s_inf, N_obs, R_pos, R_neg, gamma=1, theta=1, D_req=1):
    """Algorithm 1.1

    Perform online binary classification on the infinite stream s_inf using
    sets of positive and negative reference signals R_pos and R_neg.
    """

    consecutive_detections = 0
    s = []
    i = -1

    def update_observation(s_inf, N_obs):
        try:
            s_i = s_inf.next()
            s.append(s_i)
        except StopIteration as e:
            return
        if len(s) > N_obs:
            del s[:-N_obs]
        return s

    while True:
        i += 1
        s = update_observation(s_inf, N_obs)
        if s is None:
            return
        elif len(s) < N_obs:
            continue
        s_ = np.array(s)
        result = detect(s_, R_pos, R_neg, gamma, theta)
        if result:
            consecutive_detections += 1
            if consecutive_detections >= D_req:
                return i
        else:
            consecutive_detections = 0


def detect(s, R_pos, R_neg, gamma=1, theta=1):
    """Algorithm 1.2

    Perform binary classification on the signal s using sets of positive and
    negative reference signals R_pos and R_neg.
    """

    pos_dists = dist_to_reference(s, R_pos)
    neg_dists = dist_to_reference(s, R_neg)
    ratio = prob_class(pos_dists, gamma) / prob_class(neg_dists, gamma)
    if theta is not None:
        return ratio > theta
    else:
        return ratio


def dist_to_reference(s, r):
    """Algorithm 2

    Compute the minimum distance between s and all pieces of r of the same
    length as s.
    """

    N_obs = s.shape[0]
    N_ref = r.shape[1]
    min_dists = None
    for i in range(N_ref - N_obs + 1):
        dists = dist(r[:, i:(i + N_obs)], s)
        if min_dists is not None:
            min_dists = np.fmin(min_dists, dists)
        else:
            min_dists = dists
    return min_dists


def dist(s, t):
    """Algorithm 3

    Compute the distance between two signals s and t of the same length.
    """

    return np.sum((s - t) ** 2, 1)


def prob_class(Dists, gamma=1):
    """Algorithm 4

    Using the distances Dists of an observation to the reference signals of a
    certain class, compute a number proportional to the probability that the
    observation belongs to that class.
    """

    return np.sum(np.exp(Dists * -gamma))
