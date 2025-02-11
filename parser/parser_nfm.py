import argparse


def parse_nfm_args():
    parser = argparse.ArgumentParser(description="Run NFM.")

    parser.add_argument('--seed', type=int, default=2019,
                        help='Random seed.')

    parser.add_argument('--model_type', nargs='?', default='nfm',
                        help='Specify a model type from {fm, nfm}.')

    parser.add_argument('--data_name', nargs='?', default='amazon-book',
                        help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
    parser.add_argument('--data_dir', nargs='?', default='datasets/',
                        help='Input data path.')

    parser.add_argument('--use_pretrain', type=int, default=1,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/',
                        help='Path of learned embeddings.')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth',
                        help='Path of stored model.')

    parser.add_argument('--embed_dim', type=int, default=64,
                        help='User / entity Embedding size.')
    parser.add_argument('--hidden_dim_list', nargs='?', default='[64, 32, 16]',
                        help='Output sizes of every hidden layer.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]',
                        help='Dropout probability w.r.t. message dropout for bi-interaction layer and each hidden layer. 0: no dropout.')
    parser.add_argument('--l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating l2 loss.')

    parser.add_argument('--train_batch_size', type=int, default=1024,
                        help='Train batch size.')
    parser.add_argument('--test_batch_size', type=int, default=50,
                        help='Test batch size.')
    parser.add_argument('--test_cores', type=int, default=32,
                        help='Number of cores when evaluating.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=1000,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=10,
                        help='Number of epoch for early stopping')

    parser.add_argument('--print_every', type=int, default=1,
                        help='Iter interval of printing loss.')
    parser.add_argument('--evaluate_every', type=int, default=10,
                        help='Epoch interval of evaluating CF.')

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Calculate metric@K when evaluating.')

    args = parser.parse_args()

    save_dir = 'trained_model/NFM/{}/{}_embed-dim{}_{}_lr{}_pretrain{}/'.format(
        args.data_name, args.model_type, args.embed_dim,
        '-'.join([str(i) for i in eval(args.hidden_dim_list)]), args.lr, args.use_pretrain)
    args.save_dir = save_dir

    return args


