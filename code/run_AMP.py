from Library.paths import *
from Class.approximate_message_passing import Teacher, ApproximateMessagePassing


if __name__ == "__main__":
    K = 2
    weights_distrib = "gaussian"
    channel = "sign-sign"
    N = 1000
    alpha = 2.25
    Delta = 0
    seed = True

    TEACHER = Teacher(N=N, alpha=alpha, K=K,
                      weights_distrib=weights_distrib,
                      channel=channel, Delta=Delta,
                      seed=seed, verbose=False
                      )

    AMP = ApproximateMessagePassing(N=N, alpha=alpha, K=K,
                                    weights_distrib=weights_distrib,
                                    channel=channel, Delta=Delta,
                                    seed=seed, verbose=True
                                    )

    AMP.initialization(TEACHER)
    AMP.AMP_iteration()
