import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import argparse
import os
import time
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


model_factory = {"knn": KNeighborsClassifier(n_neighbors=9), # k=5% training set size (10fold CV)
        "elasticnet": LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.9), # generate sparse solutions
        "randomforest": RandomForestClassifier(n_estimators=500, random_state=1)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=list(model_factory.keys()), type=str)
    return parser.parse_args()


def load_data():
    adata = ad.read_h5ad('data/normct_all.h5ad')
    return adata.X, adata.obs['sleflare'].to_numpy()


def clf_and_plot(model, X, y):
    """
    Run the classifier and plot ROC curves

    :param model: classifier object
    :param X: input data
    :param y: data labels
    :return: accs: array of prediction accuracies for each fold, fig: figure object
    """
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=10)
    classifier = model

    tprs = []
    aucs = []
    accs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 6))
    for j, (train, test) in enumerate(cv.split(X, y)):
        # Fit data to model
        classifier.fit(X[train], y[train])
        y_score = classifier.predict_proba(X[test])
        y_pred = classifier.predict(X[test])
        y_true = LabelBinarizer().fit_transform(y[test])

        # Compute accuracy
        acc = np.sum(y_pred == y[test]) / y_pred.shape[0]
        accs.append(acc)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # plot a specific class or the micro average
        cls = 1
        ax.plot(fpr[cls], tpr[cls], label=f'ROC fold {j} (AUC = {roc_auc[cls]:.2f})',
                lw=1, alpha=0.3)
        interp_tpr = np.interp(mean_fpr, fpr[cls], tpr[cls])  # interpolate
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc[cls])

    # plot random chance line
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    # plot macro average
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    # plot standard deviation shade
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           xlabel=f'False Positive Rate (Positive Label {cls})',
           ylabel=f'True Positive Rate (Positive Label {cls})',
           title="Receiver operating characteristic")
    ax.legend(loc='lower right')
    # plt.show()
    return accs, fig


def main():
    args = parse_args()
    model = model_factory[args.model]
    X,y = load_data()
    accs, fig = clf_and_plot(model, X, y)

    # write to output
    if not os.path.exists(f'runs/{args.model}'):
        os.mkdir(f'runs/{args.model}')
    np.savetxt(f'runs/{args.model}/{time.strftime("%Y%m%d_%H%M%S")}_accs.txt', accs)
    fig.savefig(f'runs/{args.model}/{time.strftime("%Y%m%d_%H%M%S")}_ROC.png', dpi=300)


if __name__ == "__main__":
    main()