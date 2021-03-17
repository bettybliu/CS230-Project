"""
Baseline models without cross validation
"""
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import argparse
import os
import time
import seaborn as sn
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


model_factory = {"knn": KNeighborsClassifier(n_neighbors=7), # k=5% training set size (n=144)
        "elasticnet": LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.9), # generate sparse solutions
        "randomforest": RandomForestClassifier(n_estimators=10, random_state=4)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=list(model_factory.keys()), type=str)
    return parser.parse_args()


def load_data():
    """
    Load train and test data

    :return: X_train, y_train, X_test, y_test
    """
    adata = ad.read_h5ad('data/normct_all.h5ad')
    train = adata[adata.obs['split']=='train']
    test = adata[adata.obs['split']=='test']
    return train.X, train.obs['sleflare'].to_numpy(), test.X, test.obs['sleflare'].to_numpy()

def run(model):
    """
    Run model and plot confusion matrix

    :param model: the model object
    :return: None
    """
    # train model
    X_train, y_train, X_test, y_test = load_data()
    model.fit(X_train, y_train)

    # run model on test set
    y_score = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    y_true = LabelBinarizer().fit_transform(y_test)

    # calculate accuracy
    metrics = {}
    metrics['acc'] = np.sum(y_pred == y_test) / y_pred.shape[0]

    # calculate AUROC
    #   compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    metrics['auroc'] = roc_auc

    #   compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #   plot a specific class or the micro average
    f1, ax1 = plt.subplots()
    cls = 1
    ax1.plot(fpr[cls], tpr[cls], label=f'AUC = {roc_auc[cls]:.2f}',
            lw=2)

    #   plot random chance line
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           xlabel=f'False Positive Rate (Positive Label {cls})',
           ylabel=f'True Positive Rate (Positive Label {cls})',
           title="Receiver operating characteristic")
    ax1.legend(loc='lower right')

    # calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=['inactiveSLE', 'activeSLE', 'healthy'],
                         columns=['inactiveSLE', 'activeSLE', 'healthy'])

    # plot confusion matrix
    f2, ax2 = plt.subplots()
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax2)

    return metrics, f1, f2


def main():
    args = parse_args()
    model = model_factory[args.model]
    metrics, roc, cm = run(model)

    # write to output
    if not os.path.exists(f'runs/{args.model}'):
        os.mkdir(f'runs/{args.model}')
    with open(f'runs/{args.model}/{time.strftime("%Y%m%d_%H%M%S")}_metrics.txt', 'w') as outfile:
        outfile.write(str(metrics))
    roc.savefig(f'runs/{args.model}/{time.strftime("%Y%m%d_%H%M%S")}_ROC.png', dpi=300)
    cm.savefig(f'runs/{args.model}/{time.strftime("%Y%m%d_%H%M%S")}_cm.png', dpi=300)

if __name__ == "__main__":
    main()