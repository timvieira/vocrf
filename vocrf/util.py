# -*- coding: utf-8 -*-
"""
Misc utilities.
"""

import codecs
import itertools
from collections import Counter, defaultdict
from arsenal import iterview
from arsenal.maths import compare


def prefix_closure(C):
    """
    Compute prefix closure of `C`, including epsilon.

      >>> prefix_closure(['AA', 'ABB'])
      ['', 'A', 'AA', 'AB', 'ABB']

    If you pass in strings, you get strings back

      >>> prefix_closure(['B', 'A', 'BB'])
      ['', 'A', 'B', 'BB']

    If you pass in tuples, you get tuples back.

      >>> prefix_closure([('B',), ('A',), ('B','B')])
      [(), ('A',), ('B',), ('B', 'B')]

    """
    P = {z[:p] for z in C for p in range(len(z)+1)}
    return list(sorted(P))


def last_char_sub_closure(sigma, C):
    """Take the closure of `C` under last character substitution from the alphabet
    `sigma`.

    >>> last_char_sub_closure('ABC', [()])
    [('A',), ('B',), ('C',)]

    >>> last_char_sub_closure('ABC', [('A',)])
    [('A',), ('B',), ('C',)]

    >>> last_char_sub_closure('ABC', [('A','B')])
    [('A', 'A'), ('A', 'B'), ('A', 'C')]

    """
    return list(sorted({c[:-1] + (a,) for c in C for a in sigma}))


def longest_suffix_in(s, S):
    """
    The longest suffix of `s` that is in `S`.

    >>> longest_suffix_in('abcde', ['e', 'de'])
    'de'

    >>> longest_suffix_in('abcde', [''])
    ''

    """

    if s in S:
        return s
    elif not s:
        raise KeyError('no suffixes found')
    else:
        return longest_suffix_in(s[1:], S)


def groups(C):
    """Create groups structure based on `C`. Returns a flat version of the
    tree-structured group.

    Groups are tree-structured. For each string c ∈ C, (C is assumed to be
    prefix-closed), we have a group G[c] that consists of all strings c' ∈ C
    that have c as a proper prefix.

    To double check we understand, the extreme points are

      G[ε] = C
      G[c] = {c} if c has no extensions in C

    Some examples:

      For doctests to look nice, use this helper function,
      >>> G = lambda C: {k: list(sorted(v)) for k, v in groups(prefix_closure(C)).items()}

      >>> G(['abc'])
      {'': ['', 'a', 'ab', 'abc'], 'a': ['a', 'ab', 'abc'], 'ab': ['ab', 'abc'], 'abc': ['abc']}

      >>> G(['abc', 'bc'])
      {'': ['', 'a', 'ab', 'abc', 'b', 'bc'], 'a': ['a', 'ab', 'abc'], 'ab': ['ab', 'abc'], 'abc': ['abc'], 'b': ['b', 'bc'], 'bc': ['bc']}

    Vieira, Cotterell and Eisner (2016) use *proper* descendants in the prefix
    tree of `C` because it's a closer fit to minimizing the number of *states*
    in the DFA. They made this choice because the implementation lacked failure
    arcs. With failure arcs it makes more sense to minimize the arcs in the DFA,
    which is the construction in Cotterell and Eisner (2015).

    """

    G = defaultdict(set)
    for c in C:
        for p in prefixes(c):
            G[p].add(c)
    return G


def prefixes(w, n=None, frequency=None, threshold=5):
    """Extract prefixes of word `w` with length <= n.

    If frequency table is available, we'll only return prefixes with frequence
    >= `threshold`.

    >>> prefixes('abc')
    ['', 'a', 'ab', 'abc']

    >>> prefixes('abc', n=2)
    ['', 'a', 'ab']

    """
    if n is None:
        n = len(w)
    x = [w[:i] for i in range(min(len(w), n)+1)]
    if frequency is not None:
        x = frequency_filter(x, frequency, threshold)
    return x


def frequency_filter(X, frequency, threshold):
    """Remove elements with `frequency` lower than a given `threshold`.

    >>> freq = {'a': 2, 'aa': 1}
    >>> frequency_filter(['a', 'aa', 'aaa', 'a'], freq, threshold=1)
    ['a', 'aa', 'a']

    """
    return [x for x in X if frequency.get(x, 0) >= threshold]


def suffixes(w, n=None, frequency=None, threshold=5):
    """Extract suffixes of word `w` with length <= n.

    If frequency table is available, we'll only return suffixes with frequence
    >= `threshold`.

    """
    if n is None:
        n = len(w)
    x = [w[-i:] for i in range(1, min(len(w)+1, n+1))]
    if frequency is not None:
        x = frequency_filter(x, frequency, threshold)
    return x


def ngram_counts(data, n):
    """
    Count `n`-grams in `data`.

    >>> ngram_counts(['abc', 'aa', 'ab', 'aba'], n=2)
    Counter({'ab': 3, 'bc': 1, 'aa': 1, 'ba': 1})

    """
    counts = Counter()
    for s in data:
        m = len(s)
        for i in range(m):
            if i+n > m:
                continue
            w = s[i:i+n]
            counts[w] += 1
    return counts


def fixed_order_contexts(sigma, order):
    """
    Create tag set for fixed order model over tag alphabet `sigma`.

    >>> list(map(''.join, fixed_order_contexts('AB', order=3)))    # doctest: +NORMALIZE_WHITESPACE
    ['AAAA', 'AAAB', 'AABA', 'AABB',
     'ABAA', 'ABAB', 'ABBA', 'ABBB',
     'BAAA', 'BAAB', 'BABA', 'BABB',
     'BBAA', 'BBAB', 'BBBA', 'BBBB']

    """
    return list(sorted(itertools.product(*(sigma,)*(order+1))))


def fdcheck(func, w, g, keys = None, eps = 1e-5):
    """
    Finite-difference check.

    Returns `arsenal.maths.compare` instance.

    - `func`: zero argument function, which references `w` in caller's scope.
    - `w`: parameters.
    - `g`: gradient estimate to compare against
    - `keys`: dimensions to check
    - `eps`: perturbation size

    """
    if keys is None:
        if hasattr(w, 'keys'):
            keys = list(w.keys())
        else:
            keys = list(range(len(w)))
    fd = {}
    for key in iterview(keys):
        was = w[key]
        w[key] = was + eps
        b = func()
        w[key] = was - eps
        a = func()
        w[key] = was
        fd[key] = (b-a) / (2*eps)

    return compare([fd[k] for k in keys],
                   [g[k] for k in keys])


def read_file(filename, *args, **kwargs):
    "Read `sep`-delimited files that are split into chunks by empty lines."
    with codecs.open(filename, 'rb', encoding='utf-8') as f:
        for x in _read_file(f, *args, **kwargs):
            yield x


def _read_file(lines, sep='\t', comment='#'):
    r"""Read `sep`-delimited files that are split into chunks by empty lines. Will
    skip lines starting with the comment symbol.

    >>> list(_read_file('''
    ... # comment
    ... 1 a
    ... # comment in a block
    ... 1 b
    ...
    ...
    ... # comment
    ... 2
    ...
    ... 3'''.split('\n'), sep=None))
    [[['1', 'a'], ['1', 'b']], [['2']], [['3']]]

    """
    sentence = []
    for line in lines:
        line = line.rstrip()
        # skip commented lines.
        if comment is not None and line.startswith(comment):
            continue
        # empty line marks the end of a sentence.
        if not line:
            if sentence:
                yield sentence
            sentence = []
            continue
        # parse line.
        sentence.append(line.split(sep))
    # Emit lingering sentences. This case triggers when there is no newline at
    # the end of the file.
    if sentence:
        yield sentence
