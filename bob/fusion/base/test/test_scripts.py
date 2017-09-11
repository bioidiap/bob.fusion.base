#!/usr/bin/env python

import os
import shutil
import tempfile
import numpy
from nose.tools import assert_raises
from ..script.bob_fuse import main as bob_fuse
from ..script.bob_fusion_decision_boundary import main as decision_boundary
from bob.io.base.test_utils import datafile
from bob.measure.load import load_score

train_files = [datafile("scores-train-1", 'bob.fusion.base', 'test/data'),
               datafile("scores-train-2", 'bob.fusion.base', 'test/data')]
eval_files = [datafile("scores-eval-1", 'bob.fusion.base', 'test/data'),
              datafile("scores-eval-2", 'bob.fusion.base', 'test/data')]
fused_train_files = [
    datafile('scores-fused-train', 'bob.fusion.base', 'test/data'),
    datafile('scores-fused-train-licit', 'bob.fusion.base', 'test/data'),
    datafile('scores-fused-train-spoof', 'bob.fusion.base', 'test/data')]
fused_eval_files = [
    datafile('scores-fused-eval', 'bob.fusion.base', 'test/data'),
    datafile('scores-fused-eval-licit', 'bob.fusion.base', 'test/data'),
    datafile('scores-fused-eval-spoof', 'bob.fusion.base', 'test/data')]


def compare_scores(path1, path2):
    score1 = load_score(path1)
    score2 = load_score(path2)
    for i, name in enumerate(('claimed_id', 'real_id', 'test_label', 'score')):
        if i == 3:
            assert numpy.allclose(score1[name], score2[name])
        else:
            assert all(score1[name] == score2[name])


def test_bob_fuse():
    tmpdir = tempfile.mkdtemp(prefix='bob_')
    try:
        fused_train_file = os.path.join(tmpdir, 'scores-train')
        fused_eval_file = os.path.join(tmpdir, 'scores-eval')
        cmd = ['-s', tmpdir, '-t'] + train_files + ['-e'] + eval_files + \
            ['-T', fused_train_file, '-E', fused_eval_file, '-a', 'llr']
        bob_fuse(cmd)
        compare_scores(fused_train_file, fused_train_files[0])
        compare_scores(fused_train_file + '-licit', fused_train_files[1])
        compare_scores(fused_train_file + '-spoof', fused_train_files[2])
        compare_scores(fused_eval_file, fused_eval_files[0])
        compare_scores(fused_eval_file + '-licit', fused_eval_files[1])
        compare_scores(fused_eval_file + '-spoof', fused_eval_files[2])
        bob_fuse(cmd)
        compare_scores(fused_train_file, fused_train_files[0])
        compare_scores(fused_train_file + '-licit', fused_train_files[1])
        compare_scores(fused_train_file + '-spoof', fused_train_files[2])
        compare_scores(fused_eval_file, fused_eval_files[0])
        compare_scores(fused_eval_file + '-licit', fused_eval_files[1])
        compare_scores(fused_eval_file + '-spoof', fused_eval_files[2])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_bob_fuse_train_only():
    tmpdir = tempfile.mkdtemp(prefix='bob_')
    try:
        fused_train_file = os.path.join(tmpdir, 'scores-train')
        cmd = ['-s', tmpdir, '-t'] + train_files + \
            ['-T', fused_train_file, '-a', 'llr']
        bob_fuse(cmd)
        compare_scores(fused_train_file, fused_train_files[0])
        compare_scores(fused_train_file + '-licit', fused_train_files[1])
        compare_scores(fused_train_file + '-spoof', fused_train_files[2])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_bob_fuse_with_dev():
    tmpdir = tempfile.mkdtemp(prefix='bob_')
    try:
        fused_train_file = os.path.join(tmpdir, 'scores-train')
        cmd = ['-s', tmpdir, '-t'] + train_files + ['-d'] + \
            train_files + ['-T', fused_train_file, '-a', 'llr']
        bob_fuse(cmd)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_bob_fuse_inconsistent():
    tmpdir = tempfile.mkdtemp(prefix='bob_')
    try:
        fused_train_file = os.path.join(tmpdir, 'scores-train')
        # make inconsistency
        wrong_train2 = os.path.join(tmpdir, 'scores-train-2')
        with open(wrong_train2, 'w') as f1, open(train_files[1]) as f2:
            lines = f2.readlines()
            temp = lines[0].split()
            temp = (temp[0], temp[1], 'temp1_id', temp[3], temp[4])
            lines[0] = ' '.join(temp) + '\n'
            f1.writelines(lines)

        cmd = ['-s', tmpdir, '-t'] + train_files[0:1] + \
            [wrong_train2] + ['-T', fused_train_file, '-a', 'llr']
        assert_raises(Exception, bob_fuse, cmd)

        # make inconsistency
        wrong_train2 = os.path.join(tmpdir, 'scores-train-1')
        with open(wrong_train2, 'w') as f1, open(train_files[0]) as f2:
            lines = f2.readlines()
            temp = lines[5].split()
            print(temp)
            temp = (temp[0], '200', temp[2], temp[3])
            lines[5] = ' '.join(temp) + '\n'
            f1.writelines(lines)

        # this should not raise an error
        cmd = ['-s', tmpdir, '-t'] + train_files[0:1] + [wrong_train2] + \
            ['-T', fused_train_file, '-a', 'llr', '--skip-check']
        bob_fuse(cmd)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_decision_boundary():
    tmpdir = tempfile.mkdtemp(prefix='bob_')
    try:
        fused_train_file = os.path.join(tmpdir, 'scores-train')
        cmd = ['-s', tmpdir, '-t'] + train_files + \
            ['-T', fused_train_file, '-a', 'llr']
        bob_fuse(cmd)
        # test plot
        model_file = os.path.join(tmpdir, 'Model.pkl')
        output = os.path.join(tmpdir, 'scatter.pdf')
        cmd = ['-e'] + eval_files + ['-m', model_file, '-t', '0', '-o', output]
        decision_boundary(cmd)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_decision_boundary_grouping():
    tmpdir = tempfile.mkdtemp(prefix='bob_')
    try:
        fused_train_file = os.path.join(tmpdir, 'scores-train')
        cmd = ['-s', tmpdir, '-t'] + train_files + \
            ['-T', fused_train_file, '-a', 'llr']
        bob_fuse(cmd)
        # test plot
        model_file = os.path.join(tmpdir, 'Model.pkl')
        output = os.path.join(tmpdir, 'scatter.pdf')

        cmd = ['-e'] + eval_files + ['-m', model_file, '-t', '0', '-o', output]
        cmd += ['-G', 'random', '-g', '50']
        decision_boundary(cmd)

        cmd = ['-e'] + eval_files + ['-m', model_file, '-t', '0', '-o', output]
        cmd += ['-G', 'kmeans', '-g', '50']
        decision_boundary(cmd)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
