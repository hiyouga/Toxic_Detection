import pandas as pd
from sklearn import metrics
import numpy as np
import os


identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc' # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc' # stands for background negative, subgroup positive
TOXICITY_COLUMN = 'target'


def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan


def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]]
    return compute_auc(subgroup_examples[label], subgroup_examples[model_name])


def compute_bpsn_auc(df, subgroup, label, model_name):
    """ Computes the AUC of the within-subgroup negative examples and the background positive examples """
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label], examples[model_name])


def compute_bnsp_auc(df, subgroup, label, model_name):
    """ Computes the AUC of the within-subgroup positive examples and the background negative examples """
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label], examples[model_name])


def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = dict(
            subgroup=subgroup,
            subgroup_size=len(dataset[dataset[subgroup]])
        )
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)


def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return bias_score, (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)


def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN]
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)


def read_json(fname):
    df = pd.read_json(open(os.path.join('data', fname), 'r', encoding='utf-8'))
    for col in ['target'] + identity_columns:
        df[col] = np.where(df[col] >= 0.5, True, False)
    return df


def evaluate(y_pred):
    df = read_json('dev.json')
    model_name = 'pred'
    df[model_name] = y_pred
    bias_metrics_df = compute_bias_metrics_for_model(df, identity_columns, model_name, TOXICITY_COLUMN)
    overall_auc = calculate_overall_auc(df, model_name)
    Bias_AUC, final_score = get_final_metric(bias_metrics_df, overall_auc)
    return overall_auc, Bias_AUC, final_score
