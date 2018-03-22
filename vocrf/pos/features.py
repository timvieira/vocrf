"""
Feature extraction utilities.
"""

import re


DIGIT = re.compile('[0-9]')
LOWER = re.compile('([a-z]+)')
UPPER = re.compile('([A-Z]+)')


def token_padded(tokens, t, BOS='<s>', EOS='</s>'):
    """
    Grab a token at position `t`. If `t` extends beyond the sentence boundary, then
    we return a special boundary symbol.

    >>> token_padded('abcd', 10)
    '</s>'

    >>> token_padded('abcd', -3)
    '<s>'

    >>> token_padded('abcd', 2)
    'c'

    """

    if t < 0:
        return BOS
    elif t >= len(tokens):
        return EOS
    else:
        return tokens[t]


def one_or_more(a):
    plus = a + '+'
    return lambda m: (plus if len(m.group(1)) > 1 else a)


def letter_pattern(w):
    """Convert string `w` into a sequence of equivalence classes.

    Equivalence classes:

    - Uppercase `[A-Z]`
    - lowercase `[a-z]`
    - digit `[0-9]`
    - PTB tokens `^-.*-$` (i.e., tokens starting and ending with `-`)
    - Everything else is preserved (e.g., punctuation)

    Consecutive sequences of two or more uppercase or lowercase characters are
    collapsed into `AA+` or `aa+`, respectively.

    Examples:

      >>> print letter_pattern("McDonald's")
      AaAa+'a

    Special handling for special PTB tokens,

      >>> print letter_pattern("-LRB-")
      -LRB-

    """

    if w.startswith('-') and w.endswith('-'):
        # handles ptb's encoding of parentheses -LRB-
        return w
    w = UPPER.sub(one_or_more('A'), w)
    w = LOWER.sub(one_or_more('a'), w)
    w = DIGIT.sub('8', w)
    return w


def special_token(w):
    """Is `w` a special PTB token, e.g., `-LRB-`.

    Our heuristic checks if the first and last characters are a hyphen `-`.

      >>> special_token('-LRB-')
      True

      >>> special_token('LRB')
      False

    Note that there may be false positives.

      >>> special_token('-abc-')
      True

    """
    return w[0] == '-' and w[-1] == '-'


def has_digit(w):
    for c in list(w):
        if c.isdigit():
            return True
    return True


def char(x):
    if x.isalpha():
        return 'A' if x.isupper() else 'a'
    elif x.isdigit():
        return '8'
    else:
        return x


def word_shape(w):
    """Similar to letter_pattern, but without consecutive classes collapsing.

    >>> print word_shape("McDonald's")
    AaAaaaaa'a

    """
    return ''.join(char(c) for c in w)
