.. vim: set fileencoding=utf-8 :
.. Amir Mohammadi <amir.mohammadi@idiap.ch>

.. image:: http://img.shields.io/badge/docs-stable-yellow.png
   :target: http://pythonhosted.org/bob.fusion.base/index.html
.. image:: http://img.shields.io/badge/docs-latest-orange.png
   :target: https://www.idiap.ch/software/bob/docs/latest/bioidiap/bob.fusion.base/master/index.html
.. image:: http://travis-ci.org/bioidiap/bob.fusion.base.svg?branch=master
   :target: https://travis-ci.org/bioidiap/bob.fusion.base?branch=master
.. image:: https://coveralls.io/repos/bioidiap/bob.fusion.base/badge.svg?branch=master
   :target: https://coveralls.io/r/bioidiap/bob.fusion.base?branch=master
.. image:: https://img.shields.io/badge/github-master-0000c0.png
   :target: https://github.com/bioidiap/bob.fusion.base/tree/master
.. image:: http://img.shields.io/pypi/v/bob.fusion.base.png
   :target: https://pypi.python.org/pypi/bob.fusion.base
.. image:: http://img.shields.io/pypi/dm/bob.fusion.base.png
   :target: https://pypi.python.org/pypi/bob.fusion.base

====================================================================
 Scripts to run score fusion in biometric recognition experiments
====================================================================

This package is part of the ``bob.fusion`` packages, which allow to run comparable and reproducible score fusion in biometric recognition experiments.

This package contains basic functionality to run score fusion in biometric recognition experiments.
It provides a generic ``./bin/fuse.py`` script that takes several parameters, including:

* A list of score files
* A classification algorithm
* A list of preprocessors.

All these steps of the score fusion in biometric recognition system are given as configuration files.


Installation
------------
To create your own working package using one or more of the ``bob.fusion`` packages, please follow the `Installation Instructions <http://pythonhosted.org/bob.fusion.base/installation.html>`__ of the ``bob.fusion`` packages.

To install this package -- alone or together with other `Packages of Bob <https://github.com/idiap/bob/wiki/Packages>`_ -- please read the `Installation Instructions <https://github.com/idiap/bob/wiki/Installation>`__.
For Bob_ to be able to work properly, some dependent packages are required to be installed.
Please make sure that you have read the `Dependencies <https://github.com/idiap/bob/wiki/Dependencies>`_ for your operating system.

Documentation
-------------
For further documentation on this package, please read the `Stable Version <http://pythonhosted.org/bob.fusion.base/index.html>`_ or the `Latest Version <https://www.idiap.ch/software/bob/docs/latest/bioidiap/bob.fusion.base/master/index.html>`_ of the documentation.
For a list of tutorials on this or the other packages ob Bob_, or information on submitting issues, asking questions and starting discussions, please visit its website.

.. _bob: https://www.idiap.ch/software/bob
