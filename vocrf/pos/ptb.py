import re, os, gzip
from arsenal.alphabet import Alphabet
from collections import defaultdict as dd
re_tagged = re.compile('(\S+)/(\S+)')

PUNC = set([":", ",", ".", "#", "$", ")", "(", "``", "''"])
class PTB(object):
    "Load the POS-tagged Penn Treebank."

    def __init__(self, base, coarse=True):
        self.base = base
        self.coarse = coarse
        self.Y = Alphabet()   # tag set
        self.V, self.V_freq = Alphabet(), {} # vocabulary
        self.V2Y, self.Y2V = dd(set), dd(set)
        self.train, self.dev, self.test = [], [], []
        self.prefixes, self.suffixes = {}, {}
        self.prefix2int, self.suffix2int = {}, {}

        # Read data and create standard splits according to
        # http://aclweb.org/aclwiki/index.php?title=POS_Tagging_(State_of_the_art)
        #
        # train split [0,18]
        for sectionid in xrange(19):
            read = self.read_section(sectionid)
            for sentence in read:
                #for tag, word in sentence:
                #    if tag == self.Y["BAD"]:
                #        break

                self.train.append(sentence)
                for y, w in sentence:
                    self.V.add(w)
                    self.V2Y[w].add(self.Y.lookup(y))
                    self.Y2V[self.Y.lookup(y)].add(w)
                    if w not in self.V_freq:
                        self.V_freq[w] = 0
                    self.V_freq[w] += 1
                    for prefix in self.extract_prefixes(w):
                        if prefix not in self.prefixes:
                            self.prefixes[prefix] = 0
                        self.prefixes[prefix] += 1
                    for suffix in self.extract_suffixes(w):
                        if suffix not in self.suffixes:
                            self.suffixes[suffix] = 0
                        self.suffixes[suffix] += 1

        # dev split [19,21]
        for sectionid in xrange(19, 22):
            read = self.read_section(sectionid)

            for sentence in read:
                #for tag, word in sentence:
                #    if tag == self.Y["BAD"]:
                #        break

                self.dev.append(sentence)

        # test split [22,24]
        for sectionid in xrange(22, 25):
            #for tag, word in sentence:
            #    if tag == self.Y["BAD"]:
            #        break

            self.test.extend(self.read_section(sectionid))
        self.Y.freeze()

    def extract_prefixes(self, w, n=10):
        """ gets prefixes up to length n """
        prefixes = []
        for i in xrange(1, min(len(w)+1, n+1)):
            segment = w[:i]
            if segment not in self.prefix2int:
                self.prefix2int[segment] = len(self.prefix2int)
            prefixes.append(w[:i])
        return prefixes

    def extract_suffixes(self, w, n=10):
        """ gets suffixes up to lenght n """
        suffixes = []
        for i in xrange(1, min(len(w)+1, n+1)):
            segment = w[-i:]
            if segment not in self.suffix2int:
                self.suffix2int[segment] = len(self.suffix2int)
            suffixes.append(w[-i:])
        return suffixes

    def tag_bigrams(self):
        """ extract all tag bigrams """
        bigram2count = {}
        for sentence in self.train:
            for (tag1, _), (tag2, _) in zip(sentence, sentence[1:]):
                key = (tag1, tag2)
                if key not in bigram2count:
                    bigram2count[key] = 0
                bigram2count[key] += 1
        return bigram2count

    def read_section(self, sectionid):
        "Read a section number `sectionid` from the PTB."
        root = os.path.join(self.base, str(sectionid).zfill(2))
        for fname in os.listdir(root):
            if not fname.endswith('pos.gz'):
                continue
            with gzip.open(os.path.join(root, fname), 'rb') as f:
                for chunk in f.read().split('======================================'):
                    if chunk.strip():
                        if self.coarse:
                            # STUPID BIO ENCODING
                            #yield [(self.Y["NNP"] if "NNP" in y else self.Y["OTHER"], w) for w, y in re_tagged.findall(chunk)]
                            # Note: clean up punc reduction
                            yield [(self.Y["PUNC"] if y in PUNC else self.Y[y[0]], w) for w, y in re_tagged.findall(chunk)]
                        else:
                            # TODO: what to do able bars in the tags?
                            # FIND OUT AND CLEAN UP
                            yield [(self.Y["PUNC"] if y in PUNC else self.Y[y.split("|")[0]] if "|" in y else self.Y[y], w) for w, y in re_tagged.findall(chunk)]

    def pp(self, sentence):
        "Pretty print."
        return ' '.join('%s/%s' % (w, self.Y.lookup(t)) for (t, w) in sentence)


if __name__ == '__main__':
    ptb = PTB('data/Penn-Treebank-2/tagged/wsj')
    print ptb.pp(ptb.train[5])
