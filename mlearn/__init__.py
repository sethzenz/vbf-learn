# -*- coding: utf-8 -*-

__author__  = 'Yacine Haddad'
__email__   = 'yhaddad@cern.ch'
__version__ = '0.2.6'

# == plotting ==
import matplotlib
import matplotlib.pyplot as plt

# == numpy    ==
import numpy as np
import root_numpy
import pandas as pd # more about Pandas in short time

# == sklearn ==
from sklearn.model_selection  import train_test_split
from sklearn.model_selection  import cross_val_score
from sklearn                   import model_selection, metrics
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import BaggingClassifier
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.ensemble          import AdaBoostClassifier
from sklearn.ensemble          import GradientBoostingClassifier
from sklearn.metrics           import roc_curve, auc, roc_auc_score
from sklearn.model_selection   import learning_curve
from sklearn.model_selection   import validation_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets          import make_classification
import sklearn.model_selection   as grid_search

# == root    ==
import ROOT
# == utils   ==
from termcolor    import colored
from jsmin        import jsmin
from pprint       import pprint
import glob, json, os, re, logging, collections, math, parser
from progressbar import ProgressBar, Bar, Percentage, ETA
from collections import OrderedDict
from tabulate     import tabulate


from scipy.stats._distn_infrastructure import rv_continuous
from scipy.stats  import norm
from scipy.special import erf
