"""
Run citation segmentation experiments.
"""


import numpy as np
from argparse import ArgumentParser

from arsenal import iterview
from arsenal.alphabet import Alphabet
from arsenal.nlp.evaluation import F1
from arsenal.nlp.annotation import fromSGML, extract_contiguous

from vocrf.activeset import ActiveSet
from vocrf.util import prefix_closure, fixed_order_contexts
from vocrf.pos.instance import Instance
from vocrf.pos.data import Dataset


class CoraCitations(Dataset):

    def __init__(self, filename):
        self.Y = Alphabet()
        data = list(fromSGML(filename, linegrouper="<NEW.*?>", bioencoding=False))
        np.random.shuffle(data)
        super(CoraCitations, self).__init__(train = data[len(data)//5:],
                                            dev = data[:len(data)//5],
                                            test = [])
        self.train = self.make_instances('train', Instance)
        self.dev = self.make_instances('dev', Instance)

    def evaluate(self, predict, data, name, verbosity=1):
        if not data:
            return
        if verbosity:
            print()
            print('Phrase-based F1:', name)
        f1 = F1()
        for i, x in enumerate(iterview(data, msg='Eval %s' % name)):
            pred = extract_contiguous(predict(x))
            gold = extract_contiguous(self.Y.lookup_many(x.tags))
            # (i,begin,end) uniquely identifies the span
            for (label, begins, ends) in gold:
                f1.add_relevant(label, (i, begins, ends))
            for (label, begins, ends) in pred:
                f1.add_retrieved(label, (i, begins, ends))
        if verbosity:
            print()
        return f1.scores(verbose=verbosity >= 1)


def main():
    p = ArgumentParser()
    p.add_argument('--initial-order', type=int, default=1)
    p.add_argument('--max-order', type=int)
    p.add_argument('--inner-iterations', type=int, required=True)
    p.add_argument('--outer-iterations', type=int, required=True)
    p.add_argument('--C', type=float, required=True)
    p.add_argument('--budget', type=int, required=True)
    p.add_argument('--quick', action='store_true')

    args = p.parse_args()

    corpus = CoraCitations('data/cora.txt')

    if args.quick:
        corpus.train = corpus.train[:100]
        corpus.dev = []

    allowed_contexts = None
    if args.max_order is not None:
        allowed_contexts = prefix_closure(fixed_order_contexts(corpus.Y, order=args.max_order))
        print('allowed_contexts:', len(allowed_contexts))

    A = ActiveSet(corpus,
                  Y = corpus.Y,
                  train = corpus.train,
                  dev = corpus.dev,
                  group_budget = args.budget,
                  regularizer = args.C,
                  outer_iterations = args.outer_iterations,
                  inner_iterations = args.inner_iterations,
                  initial_contexts = fixed_order_contexts(corpus.Y, args.initial_order),
                  allowed_contexts = allowed_contexts)

    A.active_set()


if __name__ == '__main__':
    main()
