#!/usr/bin/env python
"""
Created : 26-03-2019
Last Modified : Thu 28 Mar 2019 09:48:19 PM EDT
Created By : Enrique D. Angola
"""
from sklearn.model_selection import cross_val_score
import numpy as np
from matplotlib import pylab as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
import pdb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold


class Analysis():
    """
    Analysis object - allows ML modelling

    """

    def __init__(self,reader,n_jobs = 16):
        self.reader = reader
        self.njobs = n_jobs

    def relativefreq_binned_df(self,groupby=None,labels=None):
        data = self.reader.data
        valueData = (data.groupby([groupby])[labels]
                     .value_counts(normalize=True)
                     .rename('relative percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values(labels)) 

        return valueData



    def linear_classifiers(self,alpha='auto',C='auto',gridC = np.logspace(-1,2,50),\
            grid=np.logspace(1,5,50),learningCurveOn=False,relevanceOn=True):
        """
        Computes scores for regression using simple algorithms such as

        Parameters
        ----------
        alpha: int
            alpha parameter for ridge regression, if alpha = 'auto'
            uses grid cross validation to find optimal alpha.
        grid: numpy array or list
            grid of alpha values to evaluate for grid cross validation

        Returns
        -------
        scores: Dictionary
            scores for the tested regressors

        """
        #make sure to standardize data
        self.reader.predictors = preprocessing.scale(self.reader.predictors)

        # find alpha for ridge regression using cross validation
        if C == 'auto':
            C = self.gridsearchCV(method=linear_model.LogisticRegression(solver='lbfgs'),\
                    parameter = 'C', grid=gridC)

        #define regressors
        classifiers = {'lr':linear_model.LogisticRegression(C=C,solver='lbfgs')}

        #use cross validation to score regressors
        scores = self._score_classifiers(classifiers)

        if relevanceOn:
            model = linear_model.LogisticRegression(solver='lbfgs')
            model.fit(self.reader.predictors,self.reader.target)
            print(np.std(self.reader.predictors,0)*model.coef_)


        if learningCurveOn:
            self.learning_curve(classifiers)



        return scores

    def ensemble_classifiers(self,treesR='auto',trees='auto',maxFeatures='auto',learningRate=0.1,\
            grids=range(10,200,10),gridR=range(10,200,10),scoreOn=False,relevanceOn = False,learningCurveOn=False):

        """
        computes scores for regression using ensemble algorithms. random forest
        and AdaBoost. This method also plots feature importance.

        Parameters
        ----------
        treesR: int
            number of trees for random forest algorithm, if set to 'auto'
            it finds the value using grid search cross validation.
        trees: int
            number of trees for AdaBoost, if set to 'auto' it find the value
            using grid search cross validation
        maxFeatures: int
            maximum number of features to consider for each split when using
            random forests.
        learningRate: float
            learning rate for the AdaBoost algorithm.
        grid: Numpy array or list
            grid for the grid search cross validation to find trees for AdaBoost
        gridR: Numpy array or list
            grid for the grid search cross validation to find trees for random forest.

        Returns
        -------
         scores: Dictionary
            scores for the tested classifiers

       """

        from sklearn import ensemble

        #find optimal number of estimators for AdaBoost
        if trees == 'auto':
            trees = self.gridsearchCV(method=ensemble.AdaBoostClassifier(learning_rate=learningRate),\
                    parameter='n_estimators',grid=grids)
        if treesR == 'auto':
            treesR = self.gridsearchCV(method=ensemble.RandomForestClassifier(max_features=\
                    maxFeatures),parameter='n_estimators',grid=gridR)


        #define classifiers
        classifiers = {'randomf':ensemble.RandomForestClassifier(treesR,max_features=maxFeatures,n_jobs=self.njobs),\
                'boost':ensemble.AdaBoostClassifier(learning_rate= \
                learningRate,n_estimators=trees)}
        classifiersTwo = {'randomf':ensemble.RandomForestClassifier(treesR,max_features=maxFeatures,n_jobs=self.njobs)}


                #plot feature importance
        if relevanceOn:
            self._plot_feature_importance(classifiersTwo)

        if scoreOn:
            scores = self._score_classifiers(classifiers)

            return scores
        if learningCurveOn:
            self.learning_curve(classifiers)

    def _plot_feature_importance(self,classifiers):
        """
        This privated method plots the feature importance
        for ensemble methods

        Parameters
        ----------
        classifiers: dictionary
            classifiers used

        Returns
        -------
        None

        """

        for name,clf in classifiers.items():
            clf.fit(self.reader.predictors,self.reader.target)
            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1]
          #  if name == 'boost':
         #       std = np.std([tree[0].feature_importances_ for tree in clf.estimators_], axis=0)
        #    else:
            std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

            plt.figure()
            plt.title("Feature Relevance for " + name)
            plt.bar(range(self.reader.predictors.shape[1]),importances[indices],color='r',yerr=std[indices]\
                    , align='center')
            plt.xticks(range(self.reader.predictors.shape[1]),indices)
            plt.xlim([-1, self.reader.predictors.shape[1]])
            plt.show()



    def _predict_proba(self,splits=6,classifiers=None):

        cv = StratifiedKFold(n_splits=splits)
        #iterate through classifiers and predict probabiliies
        for name, clf in classifiers.items():
            pass
        return x

    def learning_curve(self,regressors=None,cv=2,trainSizes = np.linspace(.325,1.0,5)):


        for name,estimator in regressors.items():
            title = 'Learning Curves (%s)' %name
            self.plot_learning_curve(estimator,title,(0,1.01),cv,trainSizes)
            plt.show()


    def plot_learning_curve(self,estimator=None, title=None, ylim=None, cv=None,train_sizes=None,scoring='f1'):

        """
        Generate a simple plot of the test and training learning curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - An object to be used as a cross-validation generator.
              - An iterable yielding train/test splits.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
        """

        from sklearn.model_selection import learning_curve
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(estimator, self.reader.predictors, self.reader.target, \
                cv=cv, n_jobs=self.njobs, train_sizes=train_sizes,scoring=scoring)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt

    def _score_classifiers(self,regressors=None,cv=3):

        """
        This private method uses k-fold cross-validation to score the regressors.

        Parameters
        ----------
        regressors: Dictionary
            regressors used
        scoring: str
            metric to use for the scoring (look at scikitlearn documentation)
        cv: int
            how many k-folds for cross-validation

        Returns
        -------
        scores: dictionary
            mean scores for the regressors

        """

        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import cross_val_predict
        scores = {}
        confusion = {}
        for name, clf in regressors.items():
            scoresTmp = cross_val_score(clf,self.reader.predictors,self.reader.target,cv=cv,scoring='f1')
            predictTmp = cross_val_predict(clf,self.reader.predictors,self.reader.target,cv=cv)
            confusion[name] = confusion_matrix(self.reader.target,predictTmp)
            scores[name] = np.mean(scoresTmp)

        return confusion,scores

    def gridsearchCV(self,method=None,parameter = 'alpha',\
            grid=None,scoring = 'f1'):
        """
        Performs grid search cross-validation. it also plots results
        for the iterations.

        Parameters
        ----------
        method: scikitlearn regressor object
            regressor to find parameter for.
        parameter: str
            parameter to find
        scorer: int
            metric to use for scoring (look at scikitlearn documentation)
        grid: numpy array or list
            grid of values for parameter to evaluate

        Returns
        -------
        grid.best_params_[parameter]: float
            best value for parameter found by the algorithm.


        """

        from sklearn.model_selection import GridSearchCV as GSCV

        params = {parameter:grid}
        grid = GSCV(method, params,n_jobs=self.njobs,cv=3,scoring=scoring)
        grid.fit(self.reader.predictors,self.reader.target)
        results = grid.cv_results_

        plt.figure()
        plt.plot(params[parameter],results['mean_test_score'],'*')
        plt.plot(params[parameter],results['mean_test_score'],'-')
        plt.ylabel('Score')
        plt.xlabel(parameter)
        plt.show()
        print(parameter)


        print('best parameter for %s is %d'%(parameter,grid.best_params_[parameter]))

        return grid.best_params_[parameter]


    def plot_confusion_matrix(self,cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        import itertools
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

