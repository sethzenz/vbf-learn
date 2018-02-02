from __init__ import *

logging.basicConfig(format=colored('%(levelname)s:', attrs=['bold'])
                    + colored('%(name)s:', 'blue') + ' %(message)s')
logger = logging.getLogger('ML')
logger.setLevel(level=logging.INFO)

class feature():

    def __init__(self, name="", options={}):
        self.__template__ = {
            "name"     : "",
            "title"    : "",
            "spectator": False
        }
        self.__dict__ = self.__template__
        self.__dict__.update(options)
        self.name = name

    def __str__(self):
        if self.spectator:
            return colored(" -- feature :: %18s -- %12s" % (self.name, "spectator"), "red" )
        else:
            return colored(" -- feature :: %18s -- %12s" % (self.name, "training" ), "green" )
class options (object):
    """
    object type containing Heppi options:
    * ratio_range : the range in the ratio plots
    * ratio_plot  : make the ratio plot
    * legend      : list of lignes that you want to
                    be displayed as legend on your plot
    * treename    : Gloabl tree name :
    """

    def __init__(self, options={}):
        self.__template__ = {
            "weight_branch": "weight",
        }
        self.__dict__ = self.__template__
        self.__dict__.update(options)

    def __str__(self):
        string = " -- Heppi options :\n"
        for opt in self.__dict__:
            string += "    + %15s : %20s \n" % (opt, str(self.__dict__[opt]))
        return string


class sample  (object):
    """
    object type for sample and options :
    * files : represents the name of the input files
              you can give a string or a combination
              of string to match the files name.
              You can declare a single string or an array
              if you want to combine many files all together.
    * cut   : string cut appied only on this sample
    * title : string title that will be displayed in the plot legend
    * class : the index for one of the calsses for classficiation
    * type  :
        * data       :
        * signal     :
        * background :
        * spectator  :
    """
    def __init__(self, name="", options = {}):
        self.__template__ = {
            "files"  : "",
            "title"  : "",
            "cut"    : [],
            "type"   : "",
            "class"  : "",
            "label"  : "",
            "kfactor": 1.0,
        }
        self.__dict__  = self.__template__
        self.__dict__.update(options)
        self.name      = name
        self.label     = self.label.lower()
        self.root_tree = ROOT.TChain()

    def __str__(self):
        return " -- sample :: %20s %12i" % (self.name, self.root_tree.GetEntries())



class train():

    def __init__(self, config):
        logger.info( "  ")
        logger.info( "  ______  _______________  ")
        logger.info( "  ___   |/  /__  /___    | ")
        logger.info( "  __  /|_/ /__  / __  /| | ")
        logger.info( "  _  /  / / _  /___  ___ | ")
        logger.info( "  /_/  /_/  /_____/_/  |_| ")
        logger.info( "  author : yhaddad@cern.ch ")
        logger.info( "  ")
        self.configfile = config
        self.samples    = {}
        self.selection  = {}
        self.features   = OrderedDict()
        self.opts     = None
        self.df_sig   = pd.DataFrame()
        self.df_class = pd.DataFrame()
        self.df_bkg   = pd.DataFrame()
        self.df_data  = pd.DataFrame()

    def book_data(self, selection=[], skim=-1):
        branches = self.features.keys()
        logger.info(' -- selection :\t \n %10s ' % '\t \n '.join(self.selection + selection) )
        for p,sam in self.samples.items():
            chain_ = ROOT.TChain(p)
            for fi in sam.files:
                chain_.Add(fi)
            np_chain_ = root_numpy.tree2array( chain_,
                                               branches=branches,
                                               selection='&&'.join(self.selection + selection + sam.cut ) )
            # to avoid bias in the resampling
            np.random.seed(42)
            np.random.shuffle(np_chain_)
            if skim > 0:
                # calculate new weights for the selection
                _fraction_ = (np_chain_['weight'].sum()/np_chain_[:skim]['weight'].sum())
                np_chain_['weight'] = np_chain_['weight']*_fraction_
                np_chain_ = np_chain_[:skim]
                logger.info(' -- process: %10s -- events: %7s -- weight scale : %1.3f' % (p, len(np_chain_), _fraction_) )
            else:
                logger.info(' -- process: %10s -- events: %7s -- cut : %20s' % (p, len(np_chain_), sam.cut ) )
            # Converting structured array to regular
            if sam.type == 'signal':
                if self.df_sig.shape == (0, 0):
                    self.df_sig = pd.DataFrame(np_chain_)
                    _sample_ = np.chararray(np_chain_.shape[0],itemsize=5)
                    _sample_[:] = p
                    self.df_sig['sample'] = _sample_
                else:
                    _temp_ =  pd.DataFrame(np_chain_)
                    _sample_ = np.chararray(np_chain_.shape[0],itemsize=5)
                    _sample_[:] = p
                    _temp_['sample'] = _sample_
                    self.df_sig.append(pd.DataFrame(np_chain_), ignore_index=True)

            elif sam.type == 'background':
                if self.df_bkg.shape == (0, 0):
                    self.df_bkg = pd.DataFrame(np_chain_)
                    _sample_ = np.chararray(np_chain_.shape[0],itemsize=5)
                    _sample_[:] = p
                    self.df_bkg['sample'] = _sample_
                else:
                    df_ = pd.DataFrame(np_chain_)
                    _sample_ = np.chararray(np_chain_.shape[0],itemsize=5)
                    _sample_[:] = p
                    df_['sample'] = _sample_
                    self.df_bkg = pd.concat([self.df_bkg, df_])
            elif sam.type == 'class':
                if self.df_class.shape == (0, 0):
                    self.df_class = pd.DataFrame(np_chain_)
                    _sample_ = np.chararray(np_chain_.shape[0],itemsize=5)
                    _sample_[:] = p
                    self.df_class['sample'] = _sample_
                else:
                    df_ = pd.DataFrame(np_chain_)
                    _sample_ = np.chararray(np_chain_.shape[0],itemsize=5)
                    _sample_[:] = p
                    df_['sample'] = _sample_
                    self.df_class = pd.concat([self.df_class, df_])
            elif sam.type == 'data':
                if self.df_data.shape == (0, 0):
                    self.df_data = pd.DataFrame(np_chain_)
                    _sample_ = np.chararray(np_chain_.shape[0],itemsize=5)
                    _sample_[:] = p
                    self.df_data['sample'] = _sample_
                else:
                    df_ = pd.DataFrame(np_chain_)
                    _sample_ = np.chararray(np_chain_.shape[0],itemsize=5)
                    _sample_[:] = p
                    df_['sample'] = _sample_
                    self.df_data = pd.concat([self.df_class, df_])
            else:
                print '[warning] type not defined'


    def read_cfg(self):
        _config_ = None
        with open(self.configfile) as f:
            _config_ = json.loads(
                jsmin(f.read()), object_pairs_hook=OrderedDict)

        for key in _config_:
            if 'features' in key.lower():
                for var in _config_[key]:
                    v = feature(var, _config_[key][var])
                    self.features[v.name] = v
                    logger.info(v)
            if 'samples' in key.lower():
                for p in _config_[key]:
                    self.samples[p] = sample(p, _config_[key][p])

            if "selection" in key.lower():
                self.selection = _config_[key]
            if "options" in key.lower():
                opt = options(_config_[key])
                self.opts = opt

    def correlations(self, data,label, **kwds):
        """
        Calculate pairwise correlation between features.
        Extra arguments are passed on to DataFrame.corr()
        bg  = df.y < 0.5
        sig = df.y > 0.5
        """
        corrmat = data.corr(**kwds)

        fig, ax1 = plt.subplots(ncols=1, figsize=(6,5))

        opts = {'cmap': plt.get_cmap("RdBu"),
                'vmin': -1, 'vmax': +1}
        heatmap1 = ax1.pcolor(corrmat, **opts)
        plt.colorbar( heatmap1, ax=ax1 )

        ax1.set_title( "Correlations" )

        labels = corrmat.columns.values
        for ax in (ax1,):
            # shift location of ticks to center of the bins
            ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
            ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
            ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
            ax.set_yticklabels(labels, minor=False)
        plt.tight_layout()
        plt.savefig('plots/correlation_' + self.version + '_' + label + '.pdf')
    
    def compare_train_test(self, clf,
                           x_train, y_train, w_train,
                           x_test , y_test , w_test ,
                           label, title='', **hist_prams):
        bins = 50
        hist_params = {
            "bins"  : bins
        }


        decisions = []
        weights   = []
        for x,y,w in ((x_train, y_train, w_train), (x_test, y_test, w_test)):
            d1_ = clf.predict_proba(x[y>0.5])[:,1].ravel()
            d2_ = clf.predict_proba(x[y<0.5])[:,1].ravel()
            w1_ = np.array(w[y>0.5])
            w2_ = np.array(w[y<0.5])
            decisions += [d1_, d2_]
            weights   += [w1_, w2_]

        low  = min(np.min(d) for d in decisions)
        high = max(np.max(d) for d in decisions)
        low_high = (low,high)
        plt.figure(figsize=(5,4.5))
        plt.title(title)
        plt.hist(decisions[0],weights=weights[0],
                 color='r', alpha=0.5, range=low_high,
                 histtype='stepfilled', normed=True,
                 label='S (train)', **hist_params)
        plt.hist(decisions[1], weights=weights[1],
                 color='b', alpha=0.5, range=low_high,
                 histtype='stepfilled', normed=True,
                 label='B (train)', **hist_params)

        hist, bins = np.histogram( decisions[2],weights=weights[2],
                                   bins=bins, range=low_high, normed=True)
        scale = len(decisions[2]) / sum(hist)
        err = np.sqrt(hist * scale) / scale

        width  = (bins[1  ] - bins[0 ])
        center = (bins[:-1] + bins[1:]) / 2
        plt.errorbar(center, hist, yerr=err,
                     c='r', label='S (test)',
                     fmt='o', capthick=0, ms=3,ls='None')

        hist, bins = np.histogram( decisions[3],weights=weights[3],
                                   bins=bins, range=low_high, normed=True)
        scale = len(decisions[2]) / sum(hist)
        err = np.sqrt(hist * scale) / scale

        plt.errorbar(center, hist,
                     yerr=err, c='b', label='B (test)',
                     fmt='o', capthick=0, ms=3,ls='None')

        plt.xlabel ("$O_{classifier}$")
        plt.ylabel ("$(1/n) dn/dO_{classifier}$")
        plt.legend (loc='best')
        # plt.yscale ('log')
        plt.savefig('plots/over_fit_check_' + label + '.pdf')

    def export_classfier(self):
        # to be completed
        pass
