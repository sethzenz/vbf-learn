
# VBF traning kit

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/git@github.com:yhaddad/vbf-learn.git/master)


## installation
You can install this tools in your local machine or on SWAN using the following commands
```bash
git clone git@github.com:yhaddad/vbf-learn.git
cd vbf-learn
pip install . 
```
You need to have pip and ROOT installed in your machine as well sklean, pandas and numpy packages.
I recomend using SWAN if you don't have the right setup on your local machine. You can lunch a terminal directly though your prowser and install the missing packages (ROOT environement will be already set), you can just start a terminal session and run the previous commands. You might also have add `--user` to the `pip install .` command if required.


## usage 

You can produce the numpy (h5) files by running this example :

``` python
from mlearn import classify
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd

# Reading and converting root trees as set in the configuration file
clf = classify.train(config="config/train_vbf_data.json")
clf.read_cfg()
clf.book_data()
data_all =  pd.concat((clf.df_sig,clf.df_bkg,clf.df_data))

# Adding new variables
data_all['avg_pt' ] = (data_all.dipho_leadPt  + data_all.dipho_subleadPt )/2.0
data_all['max_id' ] = np.maximum(data_all.dipho_leadIDMVA,  data_all.dipho_subleadIDMVA )
data_all['min_id' ] = np.minimum(data_all.dipho_leadIDMVA,  data_all.dipho_subleadIDMVA )
data_all['min_eta'] = np.minimum(np.abs(data_all.dipho_leadEta ),  np.abs(data_all.dipho_subleadEta))
data_all['max_eta'] = np.maximum(np.abs(data_all.dipho_leadEta ),  np.abs(data_all.dipho_subleadEta))

data_all['pass_id'] = (data_all.min_id>-0.2)
data_all['fail_id'] = (data_all.min_id<-0.4)

data_all['Y'] = np.zeros(data_all.shape[0])
data_all.loc[( data_all['sample'] == 'vbf' ), 'Y'] = 1*np.ones(data_all[(data_all['sample'] == 'vbf' )].shape[0])
data_all.loc[( data_all['sample'] == 'data'), 'Y'] = 2*np.ones(data_all[(data_all['sample'] == 'data')].shape[0])

sample_sm =((data_all['sample'] == 'ggh'  ) |
            (data_all['sample'] == 'dipho') |
            (data_all['sample'] == 'zee'  ) )

data_all['Z'] = np.zeros(data_all.shape[0])
data_all.loc[sample_sm, 'Z'] = np.ones(data_all[sample_sm].shape[0])

# save data into hdf5 files to be used later on the training : look into example directory
data_all.to_hdf('data/hgg-trees-moriond-with-sigmaEoE-2017.h5'   , 'results_table', mode='w', format='table')
```

this produces an '.h5' files that contains numpy (or pandas) arrays that can be used directly withing sklearn.
The configuration files given in `config/` are based on ROOT trees produced flashgg VBF dumper. You need just to update the path to different input files and you're ready to go.

An example of training can be found in `examples`


