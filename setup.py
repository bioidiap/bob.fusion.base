#!/usr/bin/env python
# Amir Mohammadi <amir.mohammadi@idiap.ch>
# Mon 21 Mar 08:18:08 2016 CEST
#
# Copyright (C) Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# This file contains the python (distutils/setuptools) instructions so your
# package can be installed on **any** host system. It defines some basic
# information like the package name for instance, or its homepage.
#
# It also defines which other packages this python package depends on and that
# are required for this package's operation. The python subsystem will make
# sure all dependent packages are installed or will install them for you upon
# the installation of this package.
#
# The 'buildout' system we use here will go further and wrap this package in
# such a way to create an isolated python working environment. Buildout will
# make sure that dependencies which are not yet installed do get installed, but
# **without** requiring administrative privileges on the host system. This
# allows you to test your package with new python dependencies w/o requiring
# administrative interventions.


from setuptools import setup, dist
dist.Distribution(dict(setup_requires=['bob.extension']))

from bob.extension.utils import load_requirements, find_packages
install_requires = load_requirements()

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    # This is the basic information about your project. Modify all this
    # information before releasing code publicly.
    name='bob.fusion.base',
    version=open("version.txt").read().rstrip(),
    description='Basic fusion implementations',

    url='https://www.github.com/bioidiap/bob.fusion.base',
    license='GPLv3',
    author='Amir Mohammadi',
    author_email='amir.mohammadi@idiap.ch',
    keywords='bob, fusion',

    # If you have a better, long description of your package, place it on the
    # 'doc' directory and then hook it here
    long_description=open('README.rst').read(),

    # This line is required for any distutils based packaging.
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    # This line defines which packages should be installed when you "install"
    # this package. All packages that are mentioned here, but are not installed
    # on the current system will be installed locally and only visible to the
    # scripts of this package. Don't worry - You won't need administrative
    # privileges when using buildout.
    install_requires=install_requires,

    # Your project should be called something like 'bob.<foo>' or
    # 'bob.<foo>.<bar>'. To implement this correctly and still get all your
    # packages to be imported w/o problems, you need to implement namespaces
    # on the various levels of the package and declare them here. See more
    # about this here:
    # http://peak.telecommunity.com/DevCenter/setuptools#namespace-packages
    #
    # Our database packages are good examples of namespace implementations
    # using several layers. You can check them out here:
    # https://github.com/idiap/bob/wiki/Satellite-Packages


    # This entry defines which scripts you will have inside the 'bin' directory
    # once you install the package (or run 'bin/buildout'). The order of each
    # entry under 'console_scripts' is like this:
    #   script-name-at-bin-directory = module.at.your.library:function
    #
    # The module.at.your.library is the python file within your library, using
    # the python syntax for directories (i.e., a '.' instead of '/' or '\').
    # This syntax also omits the '.py' extension of the filename. So, a file
    # installed under 'example/foo.py' that contains a function which
    # implements the 'main()' function of particular script you want to have
    # should be referred as 'example.foo:main'.
    #
    # In this simple example we will create a single program that will print
    # the version of bob.
    # entry_points={

    #   # scripts should be declared using this entry:
    #   'console_scripts': [
    #     'verify.py         = bob.fusion.base.script.verify:main',
    #     'resources.py      = bob.fusion.base.script.resources:resources',
    #     'databases.py      = bob.fusion.base.script.resources:databases',
    #     'evaluate.py       = bob.fusion.base.script.evaluate:main',
    #     'collect_results.py = bob.fusion.base.script.collect_results:main',
    #     'grid_search.py    = bob.fusion.base.script.grid_search:main',
    #     'preprocess.py     = bob.fusion.base.script.preprocess:main',
    #     'extract.py        = bob.fusion.base.script.extract:main',
    #     'enroll.py         = bob.fusion.base.script.enroll:main',
    #     'score.py          = bob.fusion.base.script.score:main',
    #     'fusion_llr.py     = bob.fusion.base.script.fusion_llr:main',
    #   ],

    #   'bob.bio.database': [
    #     # for test purposes only
    #     'dummy             = bob.fusion.base.test.dummy.database:database',
    #   ],

    #   'bob.bio.preprocessor': [
    #     # for test purposes only
    #     'dummy             = bob.fusion.base.test.dummy.preprocessor:preprocessor',
    #   ],

    #   'bob.bio.extractor': [
    #     # for test purposes only
    #     'dummy             = bob.fusion.base.test.dummy.extractor:extractor',
    #     'linearize         = bob.fusion.base.config.extractor.linearize:extractor',
    #   ],

    #   'bob.bio.algorithm': [
    #     # for test purposes only
    #     'dummy             = bob.fusion.base.test.dummy.algorithm:algorithm',
    #     'distance-euclidean = bob.fusion.base.config.algorithm.distance_euclidean:algorithm',
    #     'distance-cosine   = bob.fusion.base.config.algorithm.distance_cosine:algorithm',
    #     'pca               = bob.fusion.base.config.algorithm.pca:algorithm',
    #     'lda               = bob.fusion.base.config.algorithm.lda:algorithm',
    #     'pca+lda           = bob.fusion.base.config.algorithm.pca_lda:algorithm',
    #     'plda              = bob.fusion.base.config.algorithm.plda:algorithm',
    #     'pca+plda          = bob.fusion.base.config.algorithm.pca_plda:algorithm',
    #     'bic               = bob.fusion.base.config.algorithm.bic:algorithm',
    #   ],

    #   'bob.bio.grid': [
    #     'local-p4          = bob.fusion.base.config.grid.local:grid',
    #     'local-p8          = bob.fusion.base.config.grid.local:grid_p8',
    #     'local-p16         = bob.fusion.base.config.grid.local:grid_p16',
    #     'grid              = bob.fusion.base.config.grid.grid:grid',
    #     'demanding         = bob.fusion.base.config.grid.demanding:grid',
    #   ],
    #     },

    # Classifiers are important if you plan to distribute this package through
    # PyPI. You can find the complete list of classifiers that are valid and
    # useful here (http://pypi.python.org/pypi?%3Aaction=list_classifiers).
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
