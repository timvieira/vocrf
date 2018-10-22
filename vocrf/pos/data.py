from arsenal.alphabet import Alphabet
from collections import Counter, defaultdict
from vocrf.util import prefixes, suffixes, ngram_counts
from arsenal import iterview


class Dataset(object):

    def __init__(self, train, dev, test):
        self.train = train
        self.dev = dev
        self.test = test
        # indexes will be populated by `_index`.
        self.Y = Alphabet()          # tag set
        self.V = Alphabet()          # vocabulary
        self.V_freq = Counter()      # token unigram counts
        self.V2Y = defaultdict(set)  # tag dictionary
        self.prefixes = Counter()
        self.suffixes = Counter()
        self._index(self.train)

    def _index(self, data):
        "frequency tables, etc."
        for sentence in data:
            for y, w in sentence:
                self.Y.add(y)
                self.V.add(w)
                self.V2Y[w].add(y)
                self.V_freq[w] += 1
                for prefix in prefixes(w):
                    self.prefixes[prefix] += 1
                for suffix in suffixes(w):
                    self.suffixes[suffix] += 1

    def make_instances(self, fold, cls):
        "Convert tuples in data `fold` to instances of `cls`."
        data = []
        for x in iterview(getattr(self, fold), msg='Features (%s)' % fold):
            tags, tokens = list(zip(*x))
            data.append(cls(tokens, self.Y.map(tags), self))
        return data

    def tag_ngram_counts(self, n):
        "Returns tag ngram count for subsequences of length n."

#        Y = self.Y

        def tag_sequences():
            """Iterate over tag sequence (as `str` instead of `int`, which is how they are
            stored.).

            """
            for e in self.train:
                y, _ = list(zip(*e))
#                assert all(isinstance(yy, int) for yy in y), y
#                yield tuple(Y.lookup_many(y))
                yield y

        return ngram_counts(tag_sequences(), n)
