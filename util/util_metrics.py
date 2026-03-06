import sklearn.metrics as SKM
import matplotlib.pyplot as plt

from . import util_main as UMN
from . import util_constants as UC

def save_results_to_csv(resdict, configdict):
    cur_folder = UMN.by_projpath(folder,make_dir = True)
    save_path = UMN.get_save_path('res', configdict) 
    cur_header = list(resdict.keys())
    f = open(out_path, 'w')
    csvw = csv.DictWriter(f, fieldnames=cur_header)
    csvw.writeheader()
    csvw.writerow(filt_dict)
    f.close()

def make_confusion_matrix(truths, preds, datadict, configdict):
    figsize = None
    hide_labels = False
    if datadict['num_classes'] < 10:
        figsize = UC.CM_FIGSIZE_S
    elif datadict['num_classes'] < 30:
        figsize = UC.CM_FIGSIZE_M
    else:
        hide_labels = True
        figsize = UC.CM_FIGSIZE_L
        
    # plot confusion matrix
    fig, ax = plt.subplots(figsize=figsize)
    # convert indices to class strings before feeding into confusion matrix
    cmd = SKM.ConfusionMatrixDisplay.from_predictions(
            [datadict['idxdict'][x] for x in truths],
            [datadict['idxdict'][x] for x in preds],
            labels=datadict['label_arr'],
            normalize='true',
            include_values=hide_labels == False,
            cmap="Purples",
            colorbar=hide_labels == True,
            )

    if hide_labels == True:
        tick_positions = np.arange(0, datadict['num_classes'], 5)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions, fontsize=9)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_positions, fontsize=9)

    expr_name = UC.EXPR_PRETTY_NAMES[configdict['expr_type']]
    dataset_name = UC.DATASET_PRETTY[configdict['dataset']]
    title = f'{dataset_name} {expr_name} Results'
    ax.set_title(title)
    fig.tight_layout()
    save_path = UMN.get_save_path('cm', configdict) 
    plt.savefig(save_path)
    plt.clf()
    return cmd.confusion_matrix

# make_cm = make confusion matrix
def get_classification_metrics(truths, preds, datadict, configdict, save_to_csv = False, make_cm = False):
    ret = {}
    ret['accuracy_score']= SKM.accuracy_score(truths, preds)
    ret['f1_macro'] = SKM.f1_score(truths, preds, average='macro')
    ret['f1_micro'] = SKM.f1_score(truths, preds, average='micro')
    ret['balanced_accuracy'] = SKM.balanced_accuracy_score(truths, preds)
    if save_to_csv == True:
        save_results_to_csv(ret, configdict)
    if make_cm == True:
        ret['cm'] = make_confusion_matrix(truths, preds, datadict, configdict)
    return ret

def get_regression_metrics(truths, preds, configdict, save_to_csv = False):
    metrics = ["mean_squared_error",
               "r2_score",
               "mean_absolute_error",
               "explained_variance_score",
               "median_absolute_error",
               "max_error",
               "mean_absolute_percentage_error",
               "root_mean_squared_error",
               "d2_absolute_error_score"
               ]
    ret = {metric: getattr(SKM, name)(truths,preds) for metrics in metrics}
    if save_to_csv == True:
        save_results_to_csv(ret, configdict)
    return ret

def get_metrics(truths, preds, datadict, configdict, save_to_csv = False, make_cm = False):
    if datadict['is_classification'] == True:
        return get_classification_metrics(truths, preds, datadict, configdict, save_to_csv = save_to_csv, make_cm = make_cm)
    else:
        return get_regression_metrics(truths, preds, configdict, save_to_csv = save_to_csv)

def get_optimization_metric(metric_dict, datadict):
    ret = None
    if datadict['is_classification'] == True:
        if datadict['is_balanced'] == True:
            ret = metric_dict['accuracy_score']
        else:
            ret = metric_dict['balanced_accuracy_score']

    else:
        ret = metric_dict['r2_score']
    return ret
