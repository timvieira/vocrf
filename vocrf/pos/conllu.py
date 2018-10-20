from __future__ import division

import os
from collections import Counter
from vocrf.util import read_file
from vocrf.pos.data import Dataset
from arsenal import colors, iterview
from arsenal.nlp.evaluation import F1


class CoNLL_U(Dataset):
    "Load tagging portion of the universal dependencies in CoNLLu format."

    def __init__(self, base, tag_type='upos'):
        self.base = base
        self.tag_type = tag_type
        train, dev, test = self._read(base)
        super(CoNLL_U, self).__init__(train, dev, test)

    def __repr__(self):
        return '%s(%s, train/dev/test=%s/%s/%s' % (self.__class__.__name__,
                                                   self.base,
                                                   len(self.train),
                                                   len(self.dev),
                                                   len(self.test))

    def _read(self, base):
        "Read in all train/dev/test files in directory `base`. File must end in '.conllu'."
        data = train, dev, test = [], [], []
        for fname in os.listdir(base):
            if not fname.endswith('.conllu'):
                continue
            f = os.path.join(base, fname)
            if 'train' in fname:
                train.extend(self._read_file(f))
            elif 'dev' in fname:
                dev.extend(self._read_file(f))
            elif 'test' in fname:
                test.extend(self._read_file(f))
        return data

    def _read_file(self, fname):
        """Read an individual universal CoNLL file (i.e., file extension '.conllu').

        See https://universaldependencies.github.io/docs/format.html for a
        description of the file formate.

        """

        for sentence in read_file(fname):
            s = []
            for [_, form, _, upos, xpos, mfeats, _, _, _, _] in sentence:
                # which tag type is our target?
                if self.tag_type == 'xpos':
                    tag = xpos
                elif self.tag_type == 'mtag':
                    tag = '|'.join(['Pos=' + upos, mfeats])
                else:
                    tag = upos
                s.append((tag, form))
            yield s

    def error_classifications(self, x, t, freq_threshold=5):
        "Token context classifications. Used in `evaluate`'s accuracy breakdown."

        # ordinary accuracy
        yield 'overall'

        # token frequence (rare, oov, frequent)
        freq = self.V_freq.get(x.tokens[t], 0)
        if freq == 0:
            yield 'oov'
        elif freq <= freq_threshold:
            yield 'rare'
        else:
            yield 'freq'

        # unseen context: the combination of previous and next words were never
        # observed in the training set.
        if 0 < t < len(x.tags)-1:   # not BOS or EOS
            p = x.tokens[t-1]
            n = x.tokens[t+1]
            if not self.V_freq.get(p) and not self.V_freq.get(n):
                yield 'oov-ctx'

    def evaluate(self, predict, data, msg, verbosity=2):
        "Run predict `predict` function on data."

        if not data:
            return float('nan'), []

        ff = F1()

        correct = Counter()
        total = Counter()

        for ii, x in enumerate(iterview(data, colors.blue % 'Eval (%s)' % msg)):

            y = predict(x)
            gold = self.Y.lookup_many(x.tags)

            for t, (got, want) in enumerate(zip(y, gold)):
                if verbosity >= 2:
                    ff.report(instance=(ii, t), prediction=got, target=want)
                for c in self.error_classifications(x, t):
                    if got == want:
                        correct[c] += 1
                    total[c] += 1

        #print 'sentences:', len(data), 'tokens:', total['overall']

        c = 'overall'
        acc = '%s: %.2f' % (colors.light_yellow % c, 100 * correct[c] / total[c])
        other = total.keys()
        other.remove(c)
        breakdown = ', '.join('%s: %.2f' % (c, 100 * correct[c] / total[c])  for c in sorted(other))

        print('%s (%s)' % (acc, breakdown))

        if verbosity >= 2:
            print('F1 breakdown')
            print('============')
            ff.scores()

        return correct['overall'] / total['overall']


def main():
    "Load data, function used for testing."
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('directory',
                   help='CoNLLu POS dataset root directory.')
    args = p.parse_args()
    c = CoNLL_U(args.directory)
    print(c)


if __name__ == '__main__':
    main()
