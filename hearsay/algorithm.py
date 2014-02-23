__all__ = ['DetectStream', 'Detect', 'DistToReference', 'Dist', 'ProbClass']


import math


import numpy as np


def DetectStream(s_inf, N_obs, R_pos, R_neg, gamma=1, theta=1, D_req=1):
    """Algorithm 1.1

    Perform online binary classification on the infinite stream s_inf using
    sets of positive and negative reference signals R_pos and R_neg.
    """

    ConsecutiveDetections = 0
    s = []
    i = -1

    def UpdateObservation(s_inf, N_obs):
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
        s = UpdateObservation(s_inf, N_obs)
        if s is None:
            return
        elif len(s) < N_obs:
            continue
        s_ = np.array(s)
        R = Detect(s_, R_pos, R_neg, gamma, theta)
        if R:
            ConsecutiveDetections += 1
            if ConsecutiveDetections >= D_req:
                return i
        else:
            ConsecutiveDetections = 0


def Detect(s, R_pos, R_neg, gamma=1, theta=1):
    """Algorithm 1.2

    Perform binary classification on the signal s using sets of positive and
    negative reference signals R_pos and R_neg.
    """

    PosDists = DistToReference(s, R_pos)
    NegDists = DistToReference(s, R_neg)
    R = ProbClass(PosDists, gamma) / ProbClass(NegDists, gamma)
    return R >= theta


def DistToReference(s, R):
    """Algorithm 2

    Compute the minimum distance between s and all pieces of r of the same
    length as s.
    """

    N_obs = s.shape[0]
    N_ref = R.shape[1]
    MinDists = None
    for i in range(N_ref - N_obs + 1):
        D = Dist(R[:, i:(i + N_obs)], s)
        if MinDists is not None:
            MinDists = np.fmin(MinDists, D)
        else:
            MinDists = D
    return MinDists


def Dist(s, t):
    """Algorithm 3

    Compute the distance between two signals s and t of the same length.
    """
    D = np.sum((s - t) ** 2, 1)
    return D


def ProbClass(Dists, gamma=1):
    """Algorithm 4

    Using the distances Dists of an observation to the reference signals of a
    certain class, compute a number proportional to the probability that the
    observation belongs to that class.
    """
    P = np.sum(np.exp(Dists * -gamma))
    return P
