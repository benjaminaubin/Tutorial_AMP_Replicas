import argparse
from Library.paths import *
from Class.approximate_message_passing import Teacher, ApproximateMessagePassing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AMP')
    parser.add_argument('--K', type=int, default=2)
    parser.add_argument('--weights', type=str, default="gaussian")
    parser.add_argument('--channel', type=str, default="sign-sign")
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--alpha', type=float, default=2.25)
    parser.add_argument('--Delta', type=float, default=0)
    parser.add_argument('--seed', action='store_true')
    args = parser.parse_args()

    TEACHER = Teacher(N=args.N, alpha=args.alpha, K=args.K,
                      weights_distrib=args.weights,
                      channel=args.channel, Delta=args.Delta,
                      seed=args.seed, verbose=False
                      )

    AMP = ApproximateMessagePassing(N=args.N, alpha=args.alpha, K=args.K,
                                    weights_distrib=args.weights,
                                    channel=args.channel, Delta=args.Delta,
                                    seed=args.seed, verbose=True
                                    )

    AMP.initialization(TEACHER)
    AMP.AMP_iteration()
