from typing import Tuple, Dict

import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from scipy.stats import mode
import matplotlib.pyplot as plt
import lightgbm as lgb


class LightGBMIrisClassifier(object):
    
    def __init__(self):
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "num_leaves": 10,
            "max_depth": 10,
            "learning_rate": 0.01,
            "bagging_fraction": 0.9, 
            "feature_fraction": 0.9,
            "bagging_freq": 5,
            "bagging_seed": 42,
            "verbosity": -1,
        }
        
        other_params = {
            'fold': 5
        }
        
        self.lg_train, self.lg_test = self._get_sets()
        self.training_output = self.train_classifier(
            params=params,
            **other_params,
        )
        self._best_model = self.training_output['cvbooster']
        self.pred = self._predict()
        
    def _predict(self) -> np.ndarray:
        return mode(
            np.argmax(
                self._best_model.predict(self.lg_test.get_data()), 
                axis=2
            ),
            axis=0
        ).mode.reshape(-1)
            
    def run_procedure(self):
        self.pred = self._predict()
        self._plot_confusion_matrix()
        print(f'Accuracy is {self._get_accuracy()}')
        
    def _get_sets(self, seed: int = 42) -> Tuple[lgb.Dataset]:
        iris = datasets.load_iris()
        le = preprocessing.LabelEncoder()
        y_label = le.fit_transform(iris.target)
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data , y_label, test_size=0.30, random_state=seed, stratify=y_label)
        return (
            lgb.Dataset(X_train, y_train, 
                        feature_name=iris.feature_names,
                        free_raw_data=False,
                        ), 
            lgb.Dataset(X_test, y_test, 
                        feature_name=iris.feature_names,
                        free_raw_data=False,
                        ).construct()
        )
        
        
    def train_classifier(self, params: Dict, fold: int, **kwargs):
        return lgb.cv(
            params=params,
            nfold=fold,
            num_boost_round=5000,
            shuffle=True,
            stratified=True,
            train_set=self.lg_train, 
            early_stopping_rounds=100, 
            verbose_eval=200,
            return_cvbooster=True,
            **kwargs,
        )
        
    def _plot_confusion_matrix(
        self,
        normalize=False,
        title=None,
        cmap=plt.cm.Blues
    ):
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
        
        test_data = self.lg_test.get_label()
        cm = confusion_matrix(test_data, self.pred)
        classes = unique_labels(test_data, self.pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes, 
            yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label'
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax
    
    def _get_accuracy(self) -> float:
        cm = confusion_matrix(
            self.lg_test.get_label(), 
            self.pred
        )
        return np.diagonal(cm).sum() / cm.sum()