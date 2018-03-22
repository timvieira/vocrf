Variable-order CRFs
===================

**Install**

We assume that you are running the
[Anaconda](https://www.continuum.io/downloads) Python distribution. This code uses Python 2.

Type the following in your terminal and you should be good to go!

    $ make
    $ make test

To grab some POS tagging data run, e.g., English UD data,

    $ make data/UD/English

Here is an example invocation for training a model

    $ python vocrf/pos/tagfail.py --tag-type upos --lang English \
        --C .01 --budget 1500 \
        --inner-iterations 1 --outer-iterations 2 \
        --initial-order 0 --max-order 0 --context-count 5 \
        --dump /tmp/foo

(To quickly test the model on some real data, pass in the `--quick` flag, which
will run on a small subset of data.)


**Publications**

* Tim Vieira\*, Ryan Cotterell\*, and Jason Eisner.
  [Speed-Accuracy Tradeoffs in Tagging with Variable-Order CRFs and Structured Sparsity](http://timvieira.github.com/doc/2016-emnlp-vocrf.pdf).
  EMNLP 2016.

* Tim Vieira\*, Ryan Cotterell\*, and Jason Eisner.
  [Forward-Backward with Failure Arcs: Faster Inference for Variable-Order Conditional Random Fields](http://timvieira.github.io/doc/2018-draft-vocrf2.pdf).
  In preparation.
