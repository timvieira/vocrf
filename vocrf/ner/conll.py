

import os
from vocrf.pos.data import Dataset


class CoNLL(Dataset):
    " Load NER data in the CoNLL format "

    def __init__(self, base):
        self.base = base
        train, dev, test = self._read(base)
        super(CoNLL, self).__init__(train, dev, test)

    def __repr__(self):
        return '%s(%s, train/dev/test=%s/%s/%s' % (self.__class__.__name__,
                                                   self.base,
                                                   len(self.train),
                                                   len(self.dev),
                                                   len(self.test))

    def _read(self, base):
        "Read in all train/dev/test files in directory `base`. "
        data = train, dev, test = [], [], []
        for fname in os.listdir(base):
            f = os.path.join(base, fname)
            if 'train' in fname:
                train.extend(self._read_file(f))
            elif 'dev' in fname:
                dev.extend(self._read_file(f))
            elif 'test' in fname:
                test.extend(self._read_file(f))
        return data

    def _read_file(self, fname):
        "Read an individual universal CoNLL file"

        pass

    def evaluate(self, predict, data, msg, verbosity=2):
        " Runs the CoNLL perl script "
        pass


def main():
    "Load data, function used for testing."
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('directory',
                   help='CoNLL NER dataset root directory.')
    args = p.parse_args()
    print(args)
    c = CoNLL(args.directory)
    print(c)


if __name__ == '__main__':
    main()
