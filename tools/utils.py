import os
import random
import torch
import yaml

from collections import defaultdict
import numpy as np
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_data(embedding_file, device, keys=None):
    data = load_file(embedding_file)
    keys = data.keys() if keys is None else keys
    return (
        data[k].to(device, dtype=torch.bool) if 'mask' in k 
        else data[k].to(device, dtype=torch.float32) for k in keys
    )


def f1_score(y_true, y_pred):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * recalls * precisions / (recalls + precisions)
    return np.max(f1_scores[~np.isnan(f1_scores)]).item()


def binarize(y, thres=3):
    """Given threshold, binarize the ratings.
    """
    y[y< thres] = 0
    y[y>=thres] = 1
    return y.astype(int)


def shuffle(x, y):
    idxs = np.arange(x.shape[0])
    np.random.shuffle(idxs)
    return x[idxs], y[idxs]


def test(model, X_TEST, Y_TEST, loss, mode='regression'):
    """
    Test function with two modes: 'classification' and 'regression'.
    - classification: ndcg, auc, train_loss, recall, precision, f1
    - regression: mae, rmse, ndcg
    """
    test_pred = model.predict(X_TEST)
    metrics = {}

    if mode == 'classification':
        auc = roc_auc_score(Y_TEST, test_pred)
        ndcg = ndcg_func(model, X_TEST, Y_TEST, top_k_list=[5, 10, 30, 50])
        recall = recall_func(model, X_TEST, Y_TEST, top_k_list=[5, 10, 30, 50])
        precision = precision_func(model, X_TEST, Y_TEST, top_k_list=[5, 10, 30, 50])
        # For F1, need binary predictions
        if hasattr(model, "predict_proba"):
            pred_labels = (test_pred >= 0.5).astype(int)
        else:
            pred_labels = (test_pred >= 0.5).astype(int)
        f1 = f1_score(Y_TEST, pred_labels)
        metrics['auc'] = round(auc.tolist(), 5)
        metrics['train_loss'] = round(loss.tolist(), 5)
        ndcg = {f'ndcg@{k}': round(np.mean(ndcg[k]).tolist(), 5) for k in ndcg}
        recall = {f'recall@{k}': round(np.mean(recall[k]).tolist(), 5) for k in recall}
        precision = {f'precision@{k}': round(np.mean(precision[k]).tolist(), 5) for k in precision}
        metrics.update(ndcg)
        metrics.update(recall)
        metrics.update(precision)
        metrics['f1'] = round(f1.tolist(), 5)
    elif mode == 'regression':
        mae = mean_absolute_error(Y_TEST, test_pred)
        rmse = np.sqrt(mean_squared_error(Y_TEST, test_pred))
        r2 = r2_score(Y_TEST, test_pred)
        # ndcg = ndcg_func(model, X_TEST, Y_TEST, top_k_list=[5, 10, 30, 50])
        metrics['mae'] = round(mae, 5)
        metrics['rmse'] = round(rmse, 5)
        metrics['r2'] = round(r2, 5)
        # ndcg = {f'ndcg@{k}': round(np.mean(ndcg[k]).tolist(), 5) for k in ndcg}
        # metrics.update(ndcg)
    else:
        raise ValueError("mode must be 'classification' or 'regression'")

    print(",".join([f"{key} {metrics[key]}" for key in metrics]))
    return metrics


def rating_mat_to_sample(mat):
    row, col = np.nonzero(mat)
    y = mat[row,col]
    x = np.concatenate([row.reshape(-1,1), col.reshape(-1,1)], axis=1)
    return x, y


def ndcg_func(model, x_te, y_te, top_k_list = [5, 10]):
    """Evaluate nDCG@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_te[:,0])
    all_tr_idx = np.arange(len(x_te))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_idx = all_tr_idx[x_te[:,0] == uid]
        x_u = x_te[u_idx]
        y_u = y_te[u_idx]

        pred_u = model.predict(x_u).flatten()

        for top_k in top_k_list:
            pred_top_k = np.argsort(-pred_u)[:top_k]
            count = y_u[pred_top_k].sum()

            log2_iplus1 = np.log2(1+np.arange(1,top_k+1))

            dcg_k = y_u[pred_top_k] / log2_iplus1

            best_dcg_k = y_u[np.argsort(-y_u)][:top_k] / log2_iplus1

            if np.sum(best_dcg_k) == 0:
                ndcg_k = 1
            else:
                ndcg_k = np.sum(dcg_k) / np.sum(best_dcg_k)

            result_map["ndcg_{}".format(top_k)].append(ndcg_k)

    return result_map


def recall_func(model, x_te, y_te, top_k_list = [5, 10]):
    """Evaluate nDCG@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_te[:,0])
    all_tr_idx = np.arange(len(x_te))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_idx = all_tr_idx[x_te[:,0] == uid]
        x_u = x_te[u_idx]
        y_u = y_te[u_idx]
        pred_u = model.predict(x_u).flatten()
#         print(len(pred_u))
        for top_k in top_k_list:
            pred_top_k = np.argsort(-pred_u)[:top_k]
            recall = np.sum(y_u[pred_top_k]) / max(1, sum(y_u))# / log2_iplus1
            result_map["recall_{}".format(top_k)].append(recall)

    return result_map


def precision_func(model, x_te, y_te, top_k_list = [5, 10]):
    """Evaluate nDCG@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_te[:,0])
    all_tr_idx = np.arange(len(x_te))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_idx = all_tr_idx[x_te[:,0] == uid]
        x_u = x_te[u_idx]
        y_u = y_te[u_idx]
        pred_u = model.predict(x_u).flatten()
#         print(len(pred_u))
        for top_k in top_k_list:
            pred_top_k = np.argsort(-pred_u)[:top_k]
#             print(len(pred_top_k))
            count = y_u[pred_top_k].sum()

#             log2_iplus1 = np.log2(1+np.arange(1,top_k+1))

            recall = np.sum(y_u[pred_top_k]) / top_k # / log2_iplus1

#             best_dcg_k = y_u[np.argsort(-y_u)][:top_k] / log2_iplus1

#             if np.sum(best_dcg_k) == 0:
#                 ndcg_k = 1
#             else:
#                 ndcg_k = np.sum(dcg_k) / np.sum(best_dcg_k)

            result_map["precision_{}".format(top_k)].append(recall)

    return result_map


def check_nan_in_model(state_dict):
    for name, param in state_dict.items():
        if torch.isnan(param).any():
            print(f"NaN found in {name}")
            return True
    return False


class EarlyStopping:
    """
    For reproducing the results in PURL.
    """
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.loss_min = 1e9
        self.early_stop = False
        self.counter = 0
        self.state_dict = None

    def __call__(self, loss, state_dict=None):
        relative_loss_div = (self.loss_min-loss)/(self.loss_min+1e-10)
        if self.delta < relative_loss_div:
            self.loss_min = loss
            self.counter = 0
            self.state_dict = state_dict
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return v


def drop_params(argv, params=[]):
    if params == []:
        return argv
    idxs = [[argv.index(p), argv.index(p)+1] for p in params]
    idxs = sum(idxs, [])
    return [arg for i, arg in enumerate(argv) if i not in idxs]


def save_metrics(args, metrics):
    with open(f"{args.output_dir}/performance.yaml", "w") as perf_file:
        yaml.dump(metrics, perf_file)
    with open(f"{args.output_dir}/config.yaml", "w") as config_file:
        yaml.dump(vars(args), config_file)


def refine_dict(data):
    _data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            if v.squeeze().ndim == 0:
                _data[k] = v.item()
            else:
                _data[k] = v.tolist()
        elif isinstance(v, np.number):
            _data[k] = v.item()
        else:
            _data[k] = v
    return _data


if __name__ == "__main__":
    data = {
        "a": np.array(1.0),
        "b": np.array(2.0),
        "c": 3.0,
        "d": np.array([1,2,3]),
        "e": np.float64(1.0),
        "f": np.int64(10),
    }
    print(refine_dict(data))