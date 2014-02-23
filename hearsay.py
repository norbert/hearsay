__all__ = ['Detect', 'DistToReference', 'Dist', 'ProbClass']


import math


def Detect(s_inf, N_obs, R_pos, R_neg, gamma=1, theta=1, D_req=1):
    """Algorithm 1

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
        PosDists, NegDists = [], []
        for r in R_pos:
            PosDists.append(DistToReference(s, r))
        for r in R_neg:
            NegDists.append(DistToReference(s, r))
        R = ProbClass(PosDists, gamma) / ProbClass(NegDists, gamma)
        if R >= theta:
            ConsecutiveDetections += 1
            if ConsecutiveDetections >= D_req:
                return i
        else:
            ConsecutiveDetections = 0


def DistToReference(s, r):
    """Algorithm 2

    Compute the minimum distance between s and all pieces of r of the same
    length as s.
    """

    N_obs = len(s)
    N_ref = len(r)
    MinDist = float('inf')
    for i in range(N_ref - N_obs + 1):
        MinDist = min(MinDist, Dist(r[i:(i + N_obs)], s))
    return MinDist


def Dist(s, t):
    """Algorithm 3

    Compute the distance between two signals s and t of the same length.
    """

    D = 0
    for i in range(len(s)):
        D += (s[i] - t[i]) ** 2
    return D


def ProbClass(Dists, gamma=1):
    """Algorithm 4

    Using the distances Dists of an observation to the reference signals of a
    certain class, compute a number proportional to the probability that the
    observation belongs to that class.
    """

    P = 0
    for i in range(len(Dists)):
        P += math.exp(-gamma * Dists[i])
    return P
