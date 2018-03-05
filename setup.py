#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

try:
    import sphinx_rtd_theme
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
except ImportError:
    pass

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    "jsmin",
    "termcolor",
    "progressbar",
    "jsonmerge",
    "numpy",
    "pandas",
]

test_requirements = []
version  = '0.2.6'
setup(
    name             = 'mlearn',
    version          =  version,
    description      = "mlearn",
    long_description = readme + '\n\n' + history,
    author           = "Yacine Haddad",
    author_email     = 'yhaddad@cern.ch',
    url              = 'https://github.com/yhaddad/vbf-mlearn',
    download_url     = 'https://github.com/yhaddad/vbf-mlearn/tarball/' + version,
    packages = [
        'mlearn',
    ],
    package_dir          = {'mlearn': 'mlearn'},
    include_package_data = True,
    #scripts              = ['heppi-draw','macros/makeplotcard.py'],
    #install_requires     = requirements,
    license     = "ISCL",
    zip_safe    = False,
    keywords    = ['vbf_mlearn','vbf','plotting', 'pyROOT', 'ROOT', 'HEP', 'CERN'],
    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Utilities',
        'Programming Language :: Python :: 2.7'
    ],
    test_suite    = 'tests',
    tests_require = test_requirements
)
