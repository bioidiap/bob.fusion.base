#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from setuptools import dist, setup

dist.Distribution(dict(setup_requires=["bob.extension"]))

from bob.extension.utils import find_packages, load_requirements

install_requires = load_requirements()

setup(
    name="bob.fusion.base",
    version=open("version.txt").read().rstrip(),
    description="Score fusion in biometric and pad experiments",
    url="https://gitlab.idiap.ch/bob/bob.fusion.base",
    license="GPLv3",
    author="Amir Mohammadi",
    author_email="amir.mohammadi@idiap.ch",
    keywords="bob, score fusion, evaluation, biometric",
    long_description=open("README.rst").read(),
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    entry_points={
        # main entry for bob fusion cli
        "bob.cli": [
            "fusion = bob.fusion.base.script.fusion:fusion",
        ],
        # bob fusion scripts
        "bob.fusion.cli": [
            "fuse = bob.fusion.base.script.fuse:fuse",
            "resource = bob.fusion.base.script.resource:resource",
            "boundary = bob.fusion.base.script.boundary:boundary",
        ],
        "bob.fusion.algorithm": [
            "mean         = bob.fusion.base.config.algorithm.mean:algorithm",
            "mean-tanh    = bob.fusion.base.config.algorithm.mean:algorithm_tanh",
            "llr          = bob.fusion.base.config.algorithm.llr_skl:algorithm",
            "llr-tanh     = bob.fusion.base.config.algorithm.llr_skl:algorithm_tanh",
            "plr-2        = bob.fusion.base.config.algorithm.plr_2:algorithm",
            "plr-2-tanh   = bob.fusion.base.config.algorithm.plr_2:algorithm_tanh",
            "plr-3        = bob.fusion.base.config.algorithm.plr_3:algorithm",
            "plr-3-tanh   = bob.fusion.base.config.algorithm.plr_3:algorithm_tanh",
            "gmm          = bob.fusion.base.config.algorithm.gmm:algorithm",
            "gmm-tanh     = bob.fusion.base.config.algorithm.gmm:algorithm_tanh",
        ],
    },
    classifiers=[
        "Framework :: Bob",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
