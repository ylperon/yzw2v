#!/usr/bin/env python
# coding=utf8

import argparse
import subprocess
import sys
import time


def _parse_options():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    w2v = p.add_argument_group('w2v')
    w2v.add_argument(
        '--size',
        metavar='INT',
        dest='w2v_size',
        default="100",
    )
    w2v.add_argument(
        '--train',
        metavar='FILE',
        dest='w2v_train',
        required=True,
    )
    w2v.add_argument(
        '--save-vocab',
        metavar='FILE',
        dest='w2v_save_vocab',
    )
    w2v.add_argument(
        '--read-vocab',
        metavar='FILE',
        dest='w2v_read_vocab',
    )
    w2v.add_argument(
        '--binary',
        dest='w2v_binary',
        default='1',
    )
    w2v.add_argument(
        '--alpha',
        metavar='FLOAT',
        dest='w2v_alpha',
        default='0.05',
    )
    w2v.add_argument(
        '--output',
        metavar='FILE',
        dest='w2v_output',
    )
    w2v.add_argument(
        '--window',
        metavar='INT',
        dest='w2v_window',
        default='5',
    )
    w2v.add_argument(
        '--sample',
        metavar='FLOAT',
        dest='w2v_sample',
        default='0.005',
    )
    w2v.add_argument(
        '--hs',
        dest='w2v_hs',
        default='0',
    )
    w2v.add_argument(
        '--negative',
        metavar='INT',
        dest='w2v_negative',
        default='5',
    )
    w2v.add_argument(
        '--iter',
        metavar='INT',
        dest='w2v_iter',
        default='5',
    )
    w2v.add_argument(
        '--min-count',
        metavar='INT',
        dest='w2v_min_count',
        default='5',
    )
    pp = p.add_argument_group('params')
    pp.add_argument(
        '--min-thread-count',
        metavar='INT',
        dest='min_thread_count',
        required=True,
        type=int
    )
    pp.add_argument(
        '--max-thread-count',
        metavar='INT',
        dest='max_thread_count',
        required=True,
        type=int,
    )
    pp.add_argument(
        '--binary-path',
        metavar='FILE',
        dest='binary_path',
        required=True,
    )
    pp.add_argument(
        '--type',
        metavar='TYPE',
        dest='binary_type',
        choices=['yzw2v', 'word2vec'],
    )
    pp.add_argument(
        '--repeat',
        metavar='INT',
        dest='repeat_count',
        type=int,
    )
    p.add_argument(
        '--table',
        metavar='FILE',
        dest='table_path',
        required=True,
    )
    return p.parse_args()


def _make_yzw2v_cmd(args, thread_count):
    cmd = [args.binary_path, '--threads', str(thread_count)]
    if args.w2v_size:
        cmd.extend(['--size', args.w2v_size])
    if args.w2v_train:
        cmd.extend(['--train', args.w2v_train])
    if args.w2v_save_vocab:
        cmd.extend(['--save-vocab', args.w2v_save_vocab])
    if args.w2v_read_vocab:
        cmd.extend(['--read-vocab', args.w2v_read_vocab])
    if args.w2v_binary:
        cmd.extend(['--binary', args.w2v_binary])
    if args.w2v_alpha:
        cmd.extend(['--alpha', args.w2v_alpha])
    if args.w2v_output:
        cmd.extend(['--output', args.w2v_output])
    if args.w2v_window:
        cmd.extend(['--window', args.w2v_window])
    if args.w2v_sample:
        cmd.extend(['--sample', args.w2v_sample])
    if '0' != args.w2v_hs:
        cmd.extend(['--hs'])
    if args.w2v_negative:
        cmd.extend(['--negative', args.w2v_negative])
    if args.w2v_iter:
        cmd.extend(['--iter', args.w2v_iter])
    if args.w2v_min_count:
        cmd.extend(['--min-count', args.w2v_min_count])
    return cmd


def _make_word2vec_cmd(args, thread_count):
    cmd = [args.binary_path, '-threads', str(thread_count)]
    if args.w2v_size:
        cmd.extend(['-size', args.w2v_size])
    if args.w2v_train:
        cmd.extend(['-train', args.w2v_train])
    if args.w2v_save_vocab:
        cmd.extend(['-save-vocab', args.w2v_save_vocab])
    if args.w2v_read_vocab:
        cmd.extend(['-read-vocab', args.w2v_read_vocab])
    if args.w2v_binary:
        cmd.extend(['-binary', args.w2v_binary])
    if args.w2v_alpha:
        cmd.extend(['-alpha', args.w2v_alpha])
    if args.w2v_output:
        cmd.extend(['-output', args.w2v_output])
    if args.w2v_window:
        cmd.extend(['-window', args.w2v_window])
    if args.w2v_sample:
        cmd.extend(['-sample', args.w2v_sample])
    if args.w2v_hs:
        cmd.extend(['-hs', args.w2v_hs])
    if args.w2v_negative:
        cmd.extend(['-negative', args.w2v_negative])
    if args.w2v_iter:
        cmd.extend(['-iter', args.w2v_iter])
    if args.w2v_min_count:
        cmd.extend(['-min-count', args.w2v_min_count])
    return cmd


def _make_cmd(args, thread_count):
    assert args.binary_type in ('yzw2v', 'word2vec')
    if 'yzw2v' == args.binary_type:
        return _make_yzw2v_cmd(args, thread_count)

    return _make_word2vec_cmd(args.thread_count, thread_count)


def _time_cmd(cmd):
    start = time.time()
    subprocess.check_call(cmd)
    elapsed = time.time() - start
    return elapsed


def _main(args):
    with open(args.table_path, 'w') as out:
        for thread_count in range(args.min_thread_count, args.max_thread_count + 1):
            cmd = _make_cmd(args, thread_count)
            for _ in range(args.repeat_count):
                sys.stderr.write('cmd: {}\n'.format(cmd))
                sys.stderr.flush()
                t = _time_cmd(cmd)
                out.write('\t{}'.format(t))
                out.flush()


if '__main__' == __name__:
    args = _parse_options()
    _main(args)
