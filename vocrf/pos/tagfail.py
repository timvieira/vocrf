"""
Run POS tagging experiments.
"""
from __future__ import division

import pylab as pl
from argparse import ArgumentParser
from arsenal import colors
from arsenal.timer import timeit
from arsenal.iterextras import groupby2

from vocrf.activeset import ActiveSet
from vocrf.pos.conllu import CoNLL_U
from vocrf.util import prefix_closure, fixed_order_contexts
from vocrf.pos.instance import Instance


def main():
    p = ArgumentParser()
    p.add_argument('--initial-order', type=int, default=1)
    p.add_argument('--max-order', type=int)
    p.add_argument('--inner-iterations', type=int, required=True)
    p.add_argument('--outer-iterations', type=int, required=True)
    p.add_argument('--C', type=float, required=True)
    p.add_argument('--budget', type=int, required=True)
    p.add_argument('--context-count', type=int)
    p.add_argument('--results', type=str)
    p.add_argument('--lang', type=str, required=True)
    p.add_argument('--tag-type', type=str, required=True, choices=('upos', 'xpos', 'mtag'))
    p.add_argument('--quick', action='store_true')
    p.add_argument('--baseline', action='store_true', help='Will take the last char subst closure.')
    p.add_argument('--profile', choices=('yep', 'cprofile'))
    p.add_argument('--dump')

    args = p.parse_args()

    from arsenal.profiling import profiler
    with profiler(args.profile):
        _main(args)


def contexts_by_count(corpus, max_order, threshold):
    "Find n-grams in `corpus` with count >= `threshold` and order <= `max_order`."
    C = set()
    for n in xrange(1, max_order+2):    # +2 because order=0 is size 1; order=1 size 2.
        C.update(k for k,v in corpus.tag_ngram_counts(n=n).iteritems() if v >= threshold)
    return C


def _main(args):
    with timeit('load data'):
        corpus = CoNLL_U('data/UD/{lang}/UD_{lang}'.format(lang=args.lang), tag_type=args.tag_type)

    if args.quick:
        corpus.train = corpus.train[:100]
        corpus.dev = corpus.train[:0]

    allowed_contexts = None
    if args.context_count is not None:
        print 'context count filter threshold %s' % args.context_count

        max_order = args.initial_order + args.outer_iterations,
        if args.max_order is not None:
            max_order = args.max_order

        allowed_contexts = contexts_by_count(corpus, max_order, args.context_count)
        print 'allowed_contexts:', len(allowed_contexts)

        B = groupby2(allowed_contexts, len)
        print '(sizes %s)' % (', '.join('%s: %s' % (z, len(B[z])) for z in sorted(B)))

        if 0:
            # things that survived the threshold.
            for k, v in B.items():
                if k >= 10:   # context size >= 10
                    print
                    print k
                    for vv in v:
                        print '-'.join(vv)
            pl.plot(B.keys(), map(len, B.values()))
            pl.show()

        if 0:
            max_order = args.outer_iterations
            C = {}
            for n in xrange(1, max_order+1):   # initial order + num iters
                C.update(corpus.tag_ngram_counts(n=n))
            pl.scatter(map(len, C.keys()), C.values(), lw=0, alpha=0.5)
            pl.show()

    elif args.max_order is not None:
        allowed_contexts = prefix_closure(fixed_order_contexts(corpus.Y, order=args.max_order))
        print 'allowed_contexts:', len(allowed_contexts)

    A = ActiveSet(corpus,
                  Y = corpus.Y,
                  train = corpus.make_instances('train', Instance),
                  dev = corpus.make_instances('dev', Instance),
                  group_budget = args.budget,
                  regularizer = args.C,
                  outer_iterations = args.outer_iterations,
                  inner_iterations = args.inner_iterations,
                  initial_contexts = fixed_order_contexts(corpus.Y, args.initial_order),
                  allowed_contexts = allowed_contexts,
                  no_failure_arcs = args.baseline,
                  dump = args.dump)

    A.active_set()


if __name__ == '__main__':
    main()
