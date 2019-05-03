'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import utility.metrics as metrics
from utility.parser import parse_args
import multiprocessing
import heapq
import numpy as np

from utility.loader_bprmf import BPRMF_loader

from utility.loader_cke import CKE_loader
from utility.loader_nfm import NFM_loader
from utility.loader_kgat import KGAT_loader
from utility.loader_cfkg import CFKG_loader

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)

if args.model_type == 'bprmf':
    data_generator = BPRMF_loader(args=args, path=args.data_path + args.dataset)
    batch_test_flag = False

elif args.model_type == 'cke':
    data_generator = CKE_loader(args=args, path=args.data_path + args.dataset)
    batch_test_flag = False

elif args.model_type in ['cfkg']:
    data_generator = CFKG_loader(args=args, path=args.data_path + args.dataset)
    batch_test_flag = True

elif args.model_type in ['fm','nfm']:
    data_generator = NFM_loader(args=args, path=args.data_path + args.dataset)
    batch_test_flag = True

elif args.model_type in ['kgat']:
    data_generator = KGAT_loader(args=args, path=args.data_path + args.dataset)
    batch_test_flag = False


USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator.train_user_dict[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_generator.test_user_dict[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    # # .......checking.......
    # try:
    #     assert len(user_pos_test) != 0
    # except Exception:
    #     print(u)
    #     print(training_items)
    #     print(user_pos_test)
    #     exit()
    # # .......checking.......

    return get_performance(user_pos_test, r, auc, Ks)


def test(sess, model, users_to_test, drop_flag=False, batch_test_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)

    if args.model_type in ['ripple']:

        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE // 20
    elif args.model_type in ['fm', 'nfm']:
        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE
    else:
        u_batch_size = BATCH_SIZE * 2
        i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        if batch_test_flag:

            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)

                feed_dict = data_generator.generate_test_feed_dict(model=model,
                                                                   user_batch=user_batch,
                                                                   item_batch=item_batch,
                                                                   drop_flag=drop_flag)
                i_rate_batch = model.eval(sess, feed_dict=feed_dict)
                i_rate_batch = i_rate_batch.reshape((-1, len(item_batch)))

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            item_batch = range(ITEM_NUM)
            feed_dict = data_generator.generate_test_feed_dict(model=model,
                                                               user_batch=user_batch,
                                                               item_batch=item_batch,
                                                               drop_flag=drop_flag)
            rate_batch = model.eval(sess, feed_dict=feed_dict)
            rate_batch = rate_batch.reshape((-1, len(item_batch)))

        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users


    assert count == n_test_users
    pool.close()
    return result