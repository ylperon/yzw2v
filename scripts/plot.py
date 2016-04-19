#!/usr/bin/env python
# coding=utf-8

import argparse

import matplotlib
matplotlib.use('Agg')

import matplotlib.pylab as pl
pl.style.use('ggplot')


def _parse_options():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        '--word2vec',
        required=True,
        metavar='FILE',
        dest='word2vec_path',
    )
    p.add_argument(
        '--yzw2v',
        required=True,
        metavar='FILE',
        dest='yzw2v_path',
    )
    p.add_argument(
        '-o', '--output',
        required=True,
        metavar='FILE',
        dest='output_path',
    )
    p.add_argument(
        '-t', '--title',
        default='',
        metavar='FILE',
        dest='graph_title',
    )
    return p.parse_args()


def _load(path):
    times = []
    threads = []
    # expected format:
    # thread_count \t time_1 \t time_2 \t time_3 ...
    with open(path, 'r') as in_:
        for input_line in in_:
            cols = input_line.rstrip('\n').split('\t')
            threads.append(int(cols[0]))
            times.append(min((float(x) for x in cols[1:])))

    return times, threads


def _plot(word2vec_threads, word2vec_times, yzw2v_threads, yzw2v_times, path, title):
    pl.figure()
    pl.plot(word2vec_times, word2vec_threads, label='word2vec')
    pl.plot(yzw2v_times, yzw2v_threads, label='yzw2v')
    pl.xlabel('thread count')
    pl.ylabel('training time (sec.)')
    pl.legend(loc='upper right')
    pl.title(title)

    pl.savefig(path)


def _main(args):
    word2vec_threads, word2vec_times = _load(args.word2vec_path)
    yzw2v_threads, yzw2v_times = _load(args.yzw2v_path)
    _plot(word2vec_threads, word2vec_times, yzw2v_threads, yzw2v_times,
          args.output_path, args.graph_title)


if '__main__' == __name__:
    args = _parse_options()
    _main(args)
