import numpy as np
from vocrf.pos.features import token_padded, letter_pattern, has_digit
from vocrf.sparse import SparseBinaryVector
from vocrf.util import prefixes, suffixes

# Feature hashing range
MAGIC = 2**18


class Instance(object):

    def __init__(self, tokens, tags, corpus):
        self.tokens = tokens
        self.tags = np.array(tags, dtype=np.int32)
        self.N = len(tokens)

        # extract token properties
        self.properties = {}
        for t in xrange(self.N):

            w   = token_padded(self.tokens, t)
            n   = token_padded(self.tokens, t+1)
            p   = token_padded(self.tokens, t-1)
            nn  = token_padded(self.tokens, t+2)
            pp  = token_padded(self.tokens, t-2)
            nnn = token_padded(self.tokens, t+3)
            ppp = token_padded(self.tokens, t-2)

            P = [('prefix', x) for x in prefixes(w, n=4, frequency=corpus.prefixes, threshold=5)]
            S = [('suffix', x) for x in suffixes(w, n=4, frequency=corpus.suffixes, threshold=5)]
            shape = letter_pattern(w)

            F = [
                'bias',
                ('-1', p),
                ('0', w),
                ('+1', n),
                ('+2', nn),
                ('-2', pp),
                ('+3', nnn),
                ('-3', ppp),
                ('-1 & 0', (p, w)),
                ('0 & +1', (w, n)),
                ('-1 & +1', (p, n)),
                ('has_digit', has_digit(w)),
                ('allcaps', w.isupper()),
                ('shape', shape),
                ('all_lower', w == w.lower())
            ]
            F += P
            F += S

            F = [abs(hash(x)) % MAGIC for x in F]
            F.sort()
            self.properties[t] = SparseBinaryVector(F)
