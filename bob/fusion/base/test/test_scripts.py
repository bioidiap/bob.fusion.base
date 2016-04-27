#!/usr/bin/env python

import os
import shutil
import tempfile

from bob.fusion.base.script.bob_fuse import main as bob_fuse
from bob.fusion.base.script.plot_fusion_decision_boundary import main as plot_fusion_decision_boundary
from bob.io.base.test_utils import datafile

dev_files = [datafile("scores-dev-1", 'bob.fusion.base'),
             datafile("scores-dev-2", 'bob.fusion.base')]
eval_files = [datafile("scores-eval-1", 'bob.fusion.base'),
              datafile("scores-eval-2", 'bob.fusion.base')]


def test_scripts():

  tmpdir = tempfile.mkdtemp()
  try:
    fused_dev_file = os.path.join(tmpdir, 'scores-dev')
    fused_eval_file = os.path.join(tmpdir, 'scores-eval')

    # test normally
    cmd = ['-i'] + dev_files + ['-o', fused_dev_file, '-a', 'llr']
    bob_fuse(cmd)

    cmd = ['-i'] + dev_files + ['-I'] + eval_files + ['-o', fused_dev_file, '-O', fused_eval_file, '-a', 'llr']
    bob_fuse(cmd)

    # make inconsistency
    wrong_dev2 = os.path.join(tmpdir, 'scores-dev-2')
    with open(wrong_dev2, 'w') as f1, open(dev_files[1]) as f2:
      lines = f2.readlines()
      temp = lines[0].split()
      temp = (temp[0], 'temp1_id', temp[2], temp[3])
      lines[0] = ' '.join(temp) + '\n'
      f1.writelines(lines)

    cmd = ['-i'] + dev_files[0:1] + [wrong_dev2] + ['-o', fused_dev_file, '-a', 'llr']
    try:
      bob_fuse(cmd)
    except AssertionError:
      pass
    else:
      raise Exception('An AssertionError should have been raised.')

    # this should not raise an error
    cmd = ['-i'] + dev_files[0:1] + [wrong_dev2] + ['-o', fused_dev_file, '-a', 'llr', '--skip-check']
    bob_fuse(cmd)

    # test plot
    model_file = os.path.join(tmpdir, 'Model.pkl')
    output = os.path.join(tmpdir, 'scatter.pdf')
    cmd = dev_files + [model_file, '-o', output]
    plot_fusion_decision_boundary(cmd)

  finally:
    shutil.rmtree(tmpdir)
