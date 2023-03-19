import numpy as np
from munkres import Munkres
from sklearn.metrics import cohen_kappa_score

def Kappa(true_label, pred):
    true, pred = check_clustering(true_label), check_clustering(pred)
    pred_aligned = label_map(true, pred)
    return cohen_kappa_score(true, pred_aligned)

def check_clustering(labels):
    labels = np.array(labels)

    #check whether labels count from 0 and are continuous
    k_set = np.unique(labels)
    label_max, label_min = max(k_set), min(k_set)
    if label_min != 0 or (label_max + 1) != len(k_set):
        labels_resort = np.zeros_like(labels)
        for i in range(len(k_set)):
          idx = labels == k_set[i]
          labels_resort[idx] = i
        return labels_resort
    return labels
    
def label_map(true_label, pred):
    #calculate the cost matrix
    n = len(true_label)
    k_true = max(true_label) + 1
    k_pred = max(pred) + 1
    cost_matrix = np.zeros([k_pred, k_true], int)
    for j in range(k_true):
        idx_true = (true_label == j).astype(int)
        col_val = map(map_cost,
                      np.tile(pred, (k_pred, 1)),
                      [i for i in range(k_pred)],
                      np.tile(idx_true, (k_pred, 1)))
        cost_matrix[:, j] = list(col_val)
    count = 0
    if k_pred < k_true:
        while k_pred + count < k_true:
            cost_matrix = np.concatenate((cost_matrix, np.repeat(0, k_true).reshape(1, -1)), 0)
            count += 1
    elif k_pred > k_true:
        while k_true + count < k_pred:
            cost_matrix = np.concatenate((cost_matrix, np.repeat(0, k_pred).reshape(-1, 1)), 1)
            count += 1
    assert cost_matrix.shape[0] == max(k_pred, k_true) and cost_matrix.shape[1] == max(k_pred, k_true)

    #solve
    solve = Munkres()
    solution_map = solve.compute(cost_matrix)
    pred_to_true = {k : v for (k, v) in solution_map} #pred_label : true_label
    pred_aligned = [pred_to_true[i] for i in pred]
    return pred_aligned

def map_cost(pred, k, idx_true):
    idx_pred = (pred == k).astype(int)
    union_num = sum((idx_pred +  idx_true) > 0)
    intersection_num = sum((idx_pred +  idx_true) == 2)
    return union_num - intersection_num