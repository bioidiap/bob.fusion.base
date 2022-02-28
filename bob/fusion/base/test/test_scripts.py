#!/usr/bin/env python

import os

import numpy

from click.testing import CliRunner

from bob.bio.base.score import load_score
from bob.extension.scripts.click_helper import assert_click_runner_result
from bob.io.base.test_utils import datafile

from ..script.boundary import boundary
from ..script.fuse import fuse

train_files = [
    datafile("scores-train-1", "bob.fusion.base", "test/data"),
    datafile("scores-train-2", "bob.fusion.base", "test/data"),
]
eval_files = [
    datafile("scores-eval-1", "bob.fusion.base", "test/data"),
    datafile("scores-eval-2", "bob.fusion.base", "test/data"),
]
fused_train_files = [
    datafile("scores-fused-train", "bob.fusion.base", "test/data"),
    datafile("scores-fused-train-licit", "bob.fusion.base", "test/data"),
    datafile("scores-fused-train-spoof", "bob.fusion.base", "test/data"),
]
fused_eval_files = [
    datafile("scores-fused-eval", "bob.fusion.base", "test/data"),
    datafile("scores-fused-eval-licit", "bob.fusion.base", "test/data"),
    datafile("scores-fused-eval-spoof", "bob.fusion.base", "test/data"),
]

REGENERATE_TEST_FILES = False


def compare_scores(path1, path2):
    score1 = load_score(path1)
    score2 = load_score(path2)
    for i, name in enumerate(("claimed_id", "real_id", "test_label", "score")):
        if i == 3:
            assert numpy.allclose(score1[name], score2[name])
        else:
            assert all(score1[name] == score2[name])


def test_fuse():
    runner = CliRunner()
    with runner.isolated_filesystem():
        fused_train_file = os.path.join("fusion_result", "scores-train")
        fused_eval_file = os.path.join("fusion_result", "scores-eval")

        # Test with training
        cmd = [x for xy in zip(train_files, eval_files) for x in xy] + [
            "-g",
            "train",
            "-g",
            "eval",
            "-a",
            "llr",
        ]
        result = runner.invoke(fuse, cmd)
        assert_click_runner_result(result)

        if REGENERATE_TEST_FILES:
            for data_file, new_file in zip(
                fused_train_files + fused_eval_files,
                [
                    fused_train_file,
                    fused_train_file + "-licit",
                    fused_train_file + "-spoof",
                    fused_eval_file,
                    fused_eval_file + "-licit",
                    fused_eval_file + "-spoof",
                ],
            ):
                with open(data_file, "w") as f1, open(new_file, "r") as f2:
                    f1.write(f2.read())

        compare_scores(fused_train_file, fused_train_files[0])
        compare_scores(fused_train_file + "-licit", fused_train_files[1])
        compare_scores(fused_train_file + "-spoof", fused_train_files[2])
        compare_scores(fused_eval_file, fused_eval_files[0])
        compare_scores(fused_eval_file + "-licit", fused_eval_files[1])
        compare_scores(fused_eval_file + "-spoof", fused_eval_files[2])

        # Test without training
        cmd = eval_files + [
            "-g",
            "eval",
            "-a",
            "llr",
            "-m",
            "fusion_result/Model.pkl",
        ]
        result = runner.invoke(fuse, cmd)
        assert_click_runner_result(result)
        compare_scores(fused_train_file, fused_train_files[0])
        compare_scores(fused_train_file + "-licit", fused_train_files[1])
        compare_scores(fused_train_file + "-spoof", fused_train_files[2])
        compare_scores(fused_eval_file, fused_eval_files[0])
        compare_scores(fused_eval_file + "-licit", fused_eval_files[1])
        compare_scores(fused_eval_file + "-spoof", fused_eval_files[2])


def test_fuse_train_only():
    runner = CliRunner()
    with runner.isolated_filesystem():
        fused_train_file = os.path.join("fusion_result", "scores-train")
        cmd = train_files + ["-g", "train", "-a", "llr"]
        result = runner.invoke(fuse, cmd)
        assert_click_runner_result(result)
        compare_scores(fused_train_file, fused_train_files[0])
        compare_scores(fused_train_file + "-licit", fused_train_files[1])
        compare_scores(fused_train_file + "-spoof", fused_train_files[2])


def test_fuse_with_dev():
    runner = CliRunner()
    with runner.isolated_filesystem():
        cmd = (
            train_files
            + train_files
            + ["-g", "train", "-g", "dev", "-a", "llr"]
        )
        result = runner.invoke(fuse, cmd)
        assert_click_runner_result(result)


def test_fuse_inconsistent():
    runner = CliRunner()
    with runner.isolated_filesystem():
        # make inconsistency
        wrong_train2 = "scores-train-2"
        with open(wrong_train2, "w") as f1, open(train_files[1]) as f2:
            lines = f2.readlines()
            temp = lines[0].split()
            temp = (temp[0], temp[1], "temp1_id", temp[3], temp[4])
            lines[0] = " ".join(temp) + "\n"
            f1.writelines(lines)

        cmd = train_files[0:1] + [wrong_train2] + ["-g", "train", "-a", "llr"]
        result = runner.invoke(fuse, cmd)
        assert_click_runner_result(result, exit_code=1)
        assert isinstance(result.exception, ValueError)

        # make inconsistency
        wrong_train2 = "scores-train-1"
        with open(wrong_train2, "w") as f1, open(train_files[0]) as f2:
            lines = f2.readlines()
            temp = lines[5].split()
            temp = (temp[0], "200", temp[2], temp[3])
            lines[5] = " ".join(temp) + "\n"
            f1.writelines(lines)

        cmd = (
            train_files[0:1]
            + [wrong_train2]
            + ["-g", "train", "-a", "llr", "--skip-check"]
        )
        result = runner.invoke(fuse, cmd)
        assert_click_runner_result(result)
        assert not result.exception, result.exception


def test_boundary():
    runner = CliRunner()
    with runner.isolated_filesystem():
        cmd = train_files + ["-g", "train", "-a", "llr"]
        result = runner.invoke(fuse, cmd)
        assert_click_runner_result(result)

        model_file = "fusion_result/Model.pkl"
        cmd = eval_files + ["-m", model_file, "-t", "0"]
        result = runner.invoke(boundary, cmd)
        assert_click_runner_result(result)


def test_boundary_grouping():
    runner = CliRunner()
    with runner.isolated_filesystem():
        cmd = train_files + ["-g", "train", "-a", "llr"]
        result = runner.invoke(fuse, cmd)
        assert_click_runner_result(result)

        model_file = "fusion_result/Model.pkl"
        cmd1 = eval_files + ["-m", model_file, "-t", "0"]

        cmd = cmd1 + ["-G", "random", "-g", "50"]
        result = runner.invoke(boundary, cmd)
        assert_click_runner_result(result)

        cmd = cmd1 + ["-G", "kmeans", "-g", "50"]
        result = runner.invoke(boundary, cmd)
        assert_click_runner_result(result)
