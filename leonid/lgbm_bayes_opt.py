"""
Bayesian hyperparameter optimization [https://github.com/fmfn/BayesianOptimization]

"""

__author__ = "Leonid Sinev"

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
# from tqdm import tqdm
# import gc
try:
    import cPickle as pickle
except:
    import pickle
# import time

def lgb_evaluate(max_depth,
                 num_leaves,
                 bagging_fraction,
                 feature_fraction,
                 bagging_freq):
    params = {
        "objective": "regression",
        "metric": "rmse",
        # "num_leaves": 361,  # 40
        # 'max_depth': 21,
        "learning_rate": 0.04,  # 0.005
        # "bagging_fraction": 0.7,
        # "feature_fraction": 0.4,  # 0.6
        # "bagging_freq": 5,
        "verbosity": -1,
        'num_threads': 4,
        "seed": random_state
    }

    params['max_depth'] = int(max_depth)
    params['num_leaves'] = int(num_leaves)
    params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
    params['feature_fraction'] = max(feature_fraction, 0)
    params['bagging_freq'] = int(bagging_freq)

    cv_result = lgb.cv(params, lgtrain, num_boost_round=num_rounds, nfold=5,
                       seed=random_state,
                       verbose_eval=20,
                       stratified=False, #have to add, because of objective regression
                       early_stopping_rounds=50)

    return -cv_result['rmse-mean'][-1]


def prepare_data():
    FEATURES_PATH = '../features/'
    data_store = f'{FEATURES_PATH}crowded_features_v2_boosted_data_store.pkl'
    if os.path.isfile(data_store):
        print(f'loading data from pickle file {data_store}')
        with open(os.path.abspath(data_store), 'rb') as f:
            total_df, target, train_idx_rng, test_idx_rng = pickle.load(f, encoding='bytes')
            print('total_df:', type(total_df), total_df.shape)
            print('target:', type(target), target.shape)
            print('train_idx_rng:', type(train_idx_rng), 'start:', train_idx_rng.start,
                  'stop:', train_idx_rng.stop, 'step:', train_idx_rng.step)
            print('test_idx_rng:', type(test_idx_rng), 'start:', test_idx_rng.start,
                  'stop:', test_idx_rng.stop, 'step:', test_idx_rng.step)

    train_dataset = lgb.Dataset(total_df.iloc[train_idx_rng],
                                label=np.log1p(target))

    return train_dataset


if __name__ == '__main__':
    lgtrain = prepare_data()

    num_rounds = 100
    random_state = 2018
    num_iter = 50
    init_points = 5
    # params = {
    #     'eta': 0.1,
    #     'silent': 1,
    #     'eval_metric': 'mae',
    #     'verbose_eval': True,
    #     'seed': random_state
    # }

    lgbBO = BayesianOptimization(lgb_evaluate, {'max_depth': (8, 23),
                                                'num_leaves': (360, 2048),
                                                'bagging_fraction': (0.7, 0.9),
                                                'feature_fraction': (0.1, 0.99),
                                                'bagging_freq': (7, 10),
                                                })
    # lgbBO.explore({'max_depth': [5, 8, 13, 21],
    #                'num_leaves': [40, 80, 361, 512],
    #                })
    lgbBO.maximize(init_points=init_points, n_iter=num_iter)

    # Finally, we take a look at the final results.
    print(lgbBO.res['max'])
    print(lgbBO.res['all'])
