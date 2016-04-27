.. vim: set fileencoding=utf-8 :
.. author: Amir Mohammadi <amir.mohammadi@idiap.ch>

.. _bob.fusion.base.fusion:


==========================================================
Running Score Fusion in Biometric Recognition Experiments
==========================================================

Each score fusion experiment requires at least a classifier.
The ``bob.fusion.base`` package itself implements three such classifiers: ``MLP``, ``LLR`` and ``WeightedSum``.
You can also use any class as a classifier that implements a ``fit(X[, y])`` and a ``decision_function(X)`` method.
An example is `sklearn.linear_model.LogisticRegression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.

You can also use pre-processors to pre-process your data. The pre-processor class should implement a ``fit_transform(X[, y])`` and a ``transform(X[, y, copy])`` methods. An example is `sklearn.preprocessing.StandardScaler <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_.

Using ``sklearn`` classifiers and pre-processors you can implement different fusion algorithms. Please take a look at some examples in ``bob.fusion.base.config.algorithm`` to see how it is done.

There are two scripts available in the package as of now and they work for verification scenarios.
Spoofing and Anti-spoofing scenarios are not considered yet **in the scripts** however the Python API allows for easy extension.

Also take a look at the scripts ``bob_fuse.py`` and ``plot_fusion_decision_boundary.py`` and use them for your actual fusion experiments.

.. include:: links.rst
