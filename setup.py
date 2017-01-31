#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from setuptools import setup, dist
dist.Distribution(dict(setup_requires=['bob.extension']))

from bob.extension.utils import load_requirements, find_packages
install_requires = load_requirements()

setup(

    name='bob.fusion.base',
    version=open("version.txt").read().rstrip(),
    description='Score fusion in biometric experiments',

    url='https://gitlab.idiap.ch/bob/bob.fusion.base',
    license='GPLv3',
    author='Amir Mohammadi',
    author_email='amir.mohammadi@idiap.ch',
    keywords='bob, score fusion, evaluation, biometric',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    install_requires=install_requires,

    entry_points={

      'console_scripts': [
        'bob_fuse.py     = bob.fusion.base.script.bob_fuse:main',
        'bob_fusion_decision_boundary.py = bob.fusion.base.script.bob_fusion_decision_boundary:main',
      ],

      'bob.fusion.algorithm': [
        'mean        = bob.fusion.base.config.algorithm.mean:algorithm',
        'llr         = bob.fusion.base.config.algorithm.llr:algorithm',
        'llr-skl     = bob.fusion.base.config.algorithm.llr_skl:algorithm',
        'plr-2       = bob.fusion.base.config.algorithm.plr_2:algorithm',
        'mlp         = bob.fusion.base.config.algorithm.mlp:algorithm',
        'gmm         = bob.fusion.base.config.algorithm.gmm:algorithm',
      ],

        },

    classifiers=[
      'Framework :: Bob',
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
