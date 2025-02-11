import sys
import random
import itertools
from time import time

import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from model.NFM import NFM
from parser.parser_nfm import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_nfm import DataLoaderNFM


def evaluate_batch(model, dataloader, user_ids, Ks):
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    n_users = len(user_ids)
    n_items = dataloader.n_items
    item_ids = list(range(n_items))
    user_idx_map = dict(zip(user_ids, range(n_users)))

    feature_values = dataloader.generate_test_batch(user_ids)
    with torch.no_grad():
        scores = model(feature_values, is_train=False)              # (batch_size)

    rows = [user_idx_map[u] for u in np.repeat(user_ids, n_items).tolist()]
    cols = item_ids * n_users
    score_matrix = torch.Tensor(sp.coo_matrix((scores, (rows, cols)), shape=(n_users, n_items)).todense())

    user_ids = np.array(user_ids)
    item_ids = np.array(item_ids)
    metrics_dict = calc_metrics_at_k(score_matrix, train_user_dict, test_user_dict, user_ids, item_ids, Ks)

    score_matrix = score_matrix.numpy()
    return score_matrix, metrics_dict


def evaluate_mp(model, dataloader, Ks, num_processes, device):
    test_batch_size = dataloader.test_batch_size
    test_user_dict = dataloader.test_user_dict

    model.eval()
    model.to("cpu")

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]

    pool = mp.Pool(num_processes)
    res = pool.starmap(evaluate_batch, [(model, dataloader, batch_user, Ks) for batch_user in user_ids_batches])
    pool.close()

    score_matrix = np.concatenate([r[0] for r in res], axis=0)
    metrics_dict = {k: {} for k in Ks}
    for k in Ks:
        for m in ['precision', 'recall', 'ndcg']:
            metrics_dict[k][m] = np.concatenate([r[1][k][m] for r in res]).mean()

    torch.cuda.empty_cache()
    model.to(device)
    return score_matrix, metrics_dict


def evaluate(model, dataloader, Ks, num_processes, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]

    n_users = len(user_ids)
    n_items = dataloader.n_items
    item_ids = list(range(n_items))
    user_idx_map = dict(zip(user_ids, range(n_users)))

    cf_users = []
    cf_items = []
    cf_scores = []

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user in user_ids_batches:
            feature_values = dataloader.generate_test_batch(batch_user)
            feature_values = feature_values.to(device)

            with torch.no_grad():
                batch_scores = model(feature_values, is_train=False)            # (batch_size)

            cf_users.extend(np.repeat(batch_user, n_items).tolist())
            cf_items.extend(item_ids * len(batch_user))
            cf_scores.append(batch_scores.cpu())
            pbar.update(1)

    rows = [user_idx_map[u] for u in cf_users]
    cols = cf_items
    cf_scores = torch.cat(cf_scores)
    cf_score_matrix = torch.Tensor(sp.coo_matrix((cf_scores, (rows, cols)), shape=(n_users, n_items)).todense())

    user_ids = np.array(user_ids)
    item_ids = np.array(item_ids)
    metrics_dict = calc_metrics_at_k(cf_score_matrix, train_user_dict, test_user_dict, user_ids, item_ids, Ks)

    cf_score_matrix = cf_score_matrix.numpy()
    for k in Ks:
        for m in ['precision', 'recall', 'ndcg']:
            metrics_dict[k][m] = metrics_dict[k][m].mean()
    return cf_score_matrix, metrics_dict


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoaderNFM(args, logging)
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # construct model & optimizer
    model = NFM(args, data.n_users, data.n_items, data.n_entities, user_pre_embed, item_pre_embed)
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # initialize metrics
    best_epoch = -1
    best_recall = 0

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in Ks}

    num_processes = args.test_cores
    if num_processes and num_processes > 1:
        evaluate_func = evaluate_mp
    else:
        evaluate_func = evaluate

    # train model
    for epoch in range(1, args.n_epoch + 1):
        model.train()

        # train cf
        time1 = time()
        total_loss = 0
        n_batch = data.n_cf_train // data.train_batch_size + 1

        for iter in range(1, n_batch + 1):
            time2 = time()
            pos_feature_values, neg_feature_values = data.generate_train_batch(data.train_user_dict)
            pos_feature_values = pos_feature_values.to(device)
            neg_feature_values = neg_feature_values.to(device)
            batch_loss = model(pos_feature_values, neg_feature_values, is_train=True)

            if np.isnan(batch_loss.cpu().detach().numpy()):
                logging.info('ERROR: Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_batch))
                sys.exit()

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss.item()

            if (iter % args.print_every) == 0:
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_batch, time() - time2, batch_loss.item(), total_loss / iter))
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_batch, time() - time1, total_loss / n_batch))

        # evaluate cf
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time3 = time()
            _, metrics_dict = evaluate_func(model, data, Ks, num_processes, device)
            logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                epoch, time() - time3, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))

            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall', 'ndcg']:
                    metrics_list[k][m].append(metrics_dict[k][m])
            best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], args.stopping_steps)

            if should_stop:
                break

            if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

    # save metrics
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx']
    for k in Ks:
        for m in ['precision', 'recall', 'ndcg']:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append('{}@{}'.format(m, k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)

    # print best metrics
    best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
    logging.info('Best CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        int(best_metrics['epoch_idx']), best_metrics['precision@{}'.format(k_min)], best_metrics['precision@{}'.format(k_max)], best_metrics['recall@{}'.format(k_min)], best_metrics['recall@{}'.format(k_max)], best_metrics['ndcg@{}'.format(k_min)], best_metrics['ndcg@{}'.format(k_max)]))


def predict(args):
    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoaderNFM(args, logging)

    # load model
    model = NFM(args, data.n_users, data.n_items, data.n_entities)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    # predict
    num_processes = args.test_cores
    if num_processes and num_processes > 1:
        evaluate_func = evaluate_mp
    else:
        evaluate_func = evaluate

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    cf_scores, metrics_dict = evaluate_func(model, data, Ks, num_processes, device)
    np.save(args.save_dir + 'cf_scores.npy', cf_scores)
    print('CF Evaluation: Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))



if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    args = parse_nfm_args()
    train(args)
    # predict(args)


