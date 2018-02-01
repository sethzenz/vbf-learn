from __init__ import *

logging.basicConfig(format=colored('%(levelname)s:', attrs=['bold'])
                    + colored('%(name)s:', 'blue') + ' %(message)s')
logger = logging.getLogger('ML')
logger.setLevel(level=logging.INFO)


def plot_features(features=[],  data = pd.DataFrame(), weights = None,  log = False, label = ''):
    hist_params = { 'normed'  : True,
                    'bins'    : 60,
                    'alpha'   : 0.4,
                    'histtype':'stepfilled'}
    _nloop_ = 1+(len(features)/9) if abs(len(features)/9 - len(features)/9.0)>0 else len(features)/9
    for c in range(_nloop_):
        plt.figure( figsize=(12,12) )
        _n_ = abs(len(features)-9*c) % 9 if (len(features)-9*c) < 9 else 9
        for n in range(_n_):
            ax = plt.subplot(3,3,n+1)
            min_value, max_value = np.percentile(data[features[(c*9)+n]], [1, 99.9])
            hs,_,_ = plt.hist(data[features[(c*9)+n]][data.Y == 1], range=(min_value, max_value), label='sig', **hist_params)
            hb,_,_ = plt.hist(data[features[(c*9)+n]][data.Y == 0], range=(min_value, max_value), label='bkg', **hist_params)
            plt.legend(loc='best')
            plt.title(features[(c*9)+n])
            plt.xlim([min_value, max_value])
            if log :  ax.set_yscale('log', nonposx='clip')
            plt.ylim([0, 1.2*max(max(hs),max(hb))])
        plt.savefig('plots/features_histogram_'+label+'_' + str(c) + '.pdf')
    # return pl

def dump_rocs(self, label, fpr=[], tpr=[], thresholds=[]):
    data_roc = pd.DataFrame(np.array([fpr,tpr,thresholds]).T, columns=list(['fpr','tpr','thresholds']))
    data_roc.to_csv('data_roc_' + label +'_'+self.version + '.csv', sep=';')

def plot_rocs(rocs = {}, dump=True, range=[[0,1],[0,1]], label=''):
    plt.figure(figsize=(5,4.5))
    for k, spine in plt.gca().spines.items():
        spine.set_zorder(10)
    plt.gca().xaxis.grid(which='major', color='0.7' , linestyle='--',dashes=(5,1),zorder=0)
    plt.gca().yaxis.grid(which='major', color='0.7' , linestyle='--',dashes=(5,1),zorder=0)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    for name,roc in rocs.items():
        fpr, tpr, thr = roc
        roc_auc_ = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=name+'(area = %0.2f)'%(roc_auc_), zorder=5)
        if dump : self.dump_rocs( name ,fpr,tpr,thr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title ('ROC curves')
    plt.legend(loc='best')
    plt.grid()
    plt.xlim(range[0])
    plt.ylim(range[1])
    plt.savefig('plots/roc_'+self.version+'.pdf')
    plt.show()



def validation_curve(self,clfs, train, test, label=''):
    X_test, y_test   = test
    X_train, y_train = train
    plt.figure(figsize=(5,4))
    for n,clf in enumerate(clfs):
        test_score  = np.empty(len(clf.estimators_))
        train_score = np.empty(len(clf.estimators_))
        for i, pred in enumerate(clf.staged_decision_function(X_test)):
            test_score[i] = 1-roc_auc_score(y_test, pred)
        for i, pred in enumerate(clf.staged_decision_function(X_train)):
            train_score[i] = 1-roc_auc_score(y_train, pred)
        best_iter = np.argmin(test_score)
        learn = clf.get_params()[ 'learning_rate']
        depth = clf.get_params()[ 'max_depth'    ]
        test_line = plt.plot(test_score , label=' learn=%.1f depth=%i (%.2f)'%(learn,depth, test_score[best_iter]))

        colour = test_line[-1].get_color()
        plt.plot(train_score, '--', color=colour)
        plt.xlabel("Number of boosting iterations")
        plt.ylabel("1 - area under ROC")

        plt.axvline(x=best_iter, color=colour)

    plt.legend (loc='best')
    plt.savefig('plots/validation_curve_' + label + '.pdf' )
    plt.show()
def parameter_validation_curve( self,estimator, X, y, param_name, param_range,
                            ylim=(0, 1.1), cv=5, n_jobs=-1, scoring=None):
    estimator_name = type(estimator).__name__
    plt.figure(figsize=(5,4))
    plt.title("Validation curves for %s on %s"
              % (param_name, estimator_name))
    plt.ylim(*ylim); plt.grid()
    plt.xlim(min(param_range), max(param_range))
    plt.xlabel(param_name)
    plt.ylabel("Score")

    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name, param_range,
        cv=cv, n_jobs=n_jobs, scoring=scoring)

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores , axis=1)
    plt.semilogx(param_range, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.semilogx(param_range, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")
    logger.info(" -- Best test score: {:.4f}".format(test_scores_mean[-1]))
    plt.savefig('plots/scanned_' + param_name  + '_' + self.version + '.pdf')
def plot_learning_curve(self,estimator, title, X, y, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5),
                        scoring=None, ax=None, xlabel=True):
    if ax is None:
        print ax
        plt.figure(figsize=(4,3))

    if xlabel:
        ax.set_xlabel("Training examples")

    ax.set_ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores , axis=1)
    train_scores_std  = np.std (train_scores , axis=1)
    test_scores_mean  = np.mean(test_scores  , axis=1)
    test_scores_std   = np.std (test_scores  , axis=1)

    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

    ax.set_title(title)
    ax.set_ylim([0.65, 1.0])
    # plt.legend(loc="best")
    #plt.savefig('learning_curve_' + self.version + '_' + label + '.pdf')
    return plt

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
def variable_importance(self, clf, features):
    plt.figure(figsize=(10, 5))
    ordering      = np.argsort(clf.feature_importances_)[::-1]
    importances   = clf.feature_importances_[ordering]
    feature_names = features.columns[ordering]
    #feature_names = [p.title for _,p in clf.features.items() ]

    x = np.arange(len(feature_names))
    plt.bar(x, importances)
    plt.xticks(x + 0.5, feature_names, rotation=90, fontsize=15);
def compare_train_test_new( self, clf,
                            x_train, y_train , w_train,
                            x_test , y_test  , w_test ,
                            label  , title='', **hist_prams):

    plt.figure(figsize=(4,4))
    plt.title(title)

    bins = 50
    decisions = []
    weights   = []
    for X,y,w in ((x_train, y_train, w_train), (x_test, y_test, w_test)):
        d1 = clf.decision_function(X[y>0.5]).ravel()
        d2 = clf.decision_function(X[y<0.5]).ravel()
        decisions += [d1, d2]
        w1_ = np.array(w[y>0.5])
        w2_ = np.array(w[y<0.5])
        weights   += [w1_, w2_]

    low  = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)

    plt.hist(decisions[0],weights=weights[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='S (train)')
    plt.hist(decisions[1],weights=weights[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='B (train)')

    hist, bins = np.histogram(decisions[2],weights=weights[2],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err   = np.sqrt(hist * scale) / scale

    width  = (bins[1  ] - bins[0 ])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, c='r', label='S (test)', fmt='o', capthick=0, ms=3,ls='None')

    hist, bins = np.histogram(decisions[3],weights=weights[3],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err   = np.sqrt(hist * scale) / scale
    plt.errorbar(center, hist, yerr=err, c='b', label='B (test)', fmt='o', capthick=0, ms=3,ls='None')

    plt.xlabel("BDT output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    plt.savefig('plots/test.pdf')
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
    plt.errorbar(center, hist, yerr=err, c='r', label='S (test)', fmt='o', capthick=0, ms=3,ls='None')

    hist, bins = np.histogram( decisions[3],weights=weights[3],
                               bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, c='b', label='B (test)', fmt='o', capthick=0, ms=3,ls='None')

    plt.xlabel ("$O_{classifier}$")
    plt.ylabel ("$(1/n) dn/dO_{classifier}$")
    plt.legend (loc='best')
    # plt.yscale ('log')
    plt.savefig('plots/over_fit_check_' + label + '.pdf')

def AMSScore(self,s,b,breg=0):
    """
    statitical segnificance optimised for ML
    the breg is chosen to be 0 for the moment
    in the HiggsML from Kaggle they set that to
    10 to relax the optimisation, but I'm not
    quite sure if this will be the right thing
    to do in our case here.
    """
    return  np.sqrt (2.*( (s + b + breg)*np.log(1.+s/(b+breg))-s))

def export_classfier(self):
    # to be completed
    pass

def testing_gbc(self):
    pass
