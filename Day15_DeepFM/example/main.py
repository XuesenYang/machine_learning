import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import config
from metrics import gini_norm
from DataReader import FeatureDictionary, DataParser
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
sys.path.append("..")
from DeepFM import DeepFM

gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)


def _load_data(application='classification'):
    """
    dfTrain:  原始的训练集, index注意有没有带['id']和['target']
    dfTest:   原始的测试集
    X_train:  训练集特征向量
    y_train:  训练集标签
    X_test:   测试集特征向量
    ids_test: 测试集标签
    CATEGORICAL_COLS:  离散特征集合
    NUMERIC_COLS:      连续特征集合
    IGNORE_COLS:       需要筛除的特征集合
    """
    if application == 'classification':
        data = pd.read_csv("./data/Sonar.csv")
        data.rename(columns={'label':'target'}, inplace = True)
        X = data.drop(['target'],axis=1)
        x_id = pd.DataFrame(list(range(0, X.shape[0])), columns=['id'])
        X = pd.concat([x_id, X], axis=1, ignore_index=False)
        NUMERIC_COLS = X.columns.tolist()
        Y = data['target']
        samplesNumber = X.shape[0]
        # traim:test = 7:3
        data = pd.concat([X, Y], axis=1, ignore_index=False)
        dfTrain = data.iloc[:int(0.7*samplesNumber),:]
        dfTest = data.iloc[int(0.7*samplesNumber):,:]
        X_train = X.iloc[:int(0.7*samplesNumber),:].values
        X_test = X.iloc[int(0.7*samplesNumber):,:].values
        y_train = Y.iloc[:int(0.7*samplesNumber)].values
        y_test = Y.iloc[int(0.7*samplesNumber):].values
        IGNORE_COLS = []
        CATEGORICAL_COLS = []
    else:
        data = pd.read_csv("./data/HousePrice.csv")
        data = data.drop(['Alley','MasVnrType','FireplaceQu','GarageType','GarageFinish','GarageQual',
                          'GarageCond','PoolQC','Fence','MiscFeature','BsmtCond','BsmtExposure',
                          'BsmtFinType1','BsmtFinType2','Heating','HeatingQC','BsmtQual',
                          'Electrical'],axis=1)
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        data[['LotFrontage','MasVnrArea','GarageYrBlt']] = imputer.fit_transform(data[['LotFrontage','MasVnrArea','GarageYrBlt']])
        text_clos = ['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
             'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
            'ExterQual','ExterCond','Foundation','CentralAir','KitchenQual','Functional','PavedDrive','SaleType','SaleCondition']
        LabelEncoder_cols = data.loc[:, text_clos]
        LabelEncoder_cols = LabelEncoder_cols.values   
        data = data.drop(text_clos, axis=1)
        for i in range(LabelEncoder_cols.shape[1]):
            LabelEncoder_cols[:,i] = LabelEncoder().fit_transform(LabelEncoder_cols[:,i])
        LabelEncoder_cols = pd.DataFrame(LabelEncoder_cols)
        # binary_cols = LabelEncoder_cols.columns
        data = pd.concat([data, LabelEncoder_cols], axis=1, ignore_index=False)
        data.rename(columns={'Id':'id','SalePrice':'target'}, inplace = True)
        X = data.drop(['target'],axis=1)
        NUMERIC_COLS = X.columns.tolist()
        Y = data['target']
        samplesNumber = X.shape[0]
        # traim:test = 7:3
        data = pd.concat([X, Y], axis=1, ignore_index=False)
        dfTrain = data.iloc[:int(0.7*samplesNumber),:]
        dfTest = data.iloc[int(0.7*samplesNumber):,:]
        X_train = X.iloc[:int(0.7*samplesNumber),:].values
        X_test = X.iloc[int(0.7*samplesNumber):,:].values
        y_train = Y.iloc[:int(0.7*samplesNumber),:].values
        y_test = Y.iloc[int(0.7*samplesNumber):,:].values
        IGNORE_COLS = []
        CATEGORICAL_COLS = []
    return dfTrain, dfTest, X_train, y_train, X_test, y_test, IGNORE_COLS, NUMERIC_COLS, CATEGORICAL_COLS


def _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params, NUMERIC_COLS, IGNORE_COLS, application='classification'):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=NUMERIC_COLS,
                           ignore_cols=IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest, has_label=True)

    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])

    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
    _get = lambda x, l: [x[i] for i in l]
    results_cv = np.zeros(len(folds), dtype=float)
    results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)

        y_train_meta[valid_idx,0] = dfm.predict(Xi_valid_, Xv_valid_)
        y_test_meta[:,0] += dfm.predict(Xi_test, Xv_test)
        
        if application == 'classification':
            results_cv[i] = roc_auc_score(y_valid_, y_train_meta[valid_idx])
        elif application == 'regression':
            results_cv[i] = np.sqrt(mean_squared_error(y_valid_, y_train_meta[valid_idx]))
        else:
            results_cv[i] = gini_norm(y_valid_, y_train_meta[valid_idx])
        results_epoch_train[i] = dfm.train_result
        results_epoch_valid[i] = dfm.valid_result

    y_test_meta /= float(len(folds))

    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("%s: rmse/accuracy/gini is %.4f (std is %.4f)"%(clf_str, results_cv.mean(), results_cv.std()))
    filename = "%s_Mean%.5f.csv"%(clf_str, results_cv.mean())
    _make_submission(ids_test, y_test_meta, filename)

    _plot_fig(results_epoch_train, results_epoch_valid, clf_str, application)

    return y_train_meta, y_test_meta


def _make_submission(ids, y_pred, filename="submission.csv"):
    pd.DataFrame({"id": ids, "target": y_pred.flatten()}).to_csv(
        os.path.join("./output", filename), index=False, float_format="%.5f")


def _plot_fig(train_results, valid_results, model_name, application):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d"%(i+1))
        legends.append("valid-%d"%(i+1))
    plt.xlabel("Epoch")
    if application == 'classification':
        plt.ylabel("accuracy value")
    elif application == 'regression':
        plt.ylabel("RMSE value")
    else:
        plt.ylabel("Normalized Gini")
    plt.title("%s"%model_name)
    plt.legend(legends)
    plt.savefig("./fig/%s.png"%model_name)
    plt.close()


# load data
application='regression'
# 'regression' , 'classification'
dfTrain, dfTest, X_train, y_train, X_test, y_test, IGNORE_COLS, NUMERIC_COLS, CATEGORICAL_COLS = _load_data(application)

# folds
folds = list(StratifiedKFold(n_splits=3, shuffle=True,
                             random_state=2019).split(X_train, y_train))


# ------------------ DeepFM Model ------------------
# params

dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 100,
    "batch_size": 30,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": gini_norm,
    "random_seed": config.RANDOM_SEED
}
if application== 'classification':
    dfm_params[' loss_type'] = "logloss",
    dfm_params[' eval_metric'] = roc_auc_score
else:
    dfm_params[' loss_type'] = "mse",
    dfm_params[' eval_metric'] = mean_squared_error
    
y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params,NUMERIC_COLS, IGNORE_COLS, application)

# ------------------ FM Model ------------------
fm_params = dfm_params.copy()
fm_params["use_deep"] = False
y_train_fm, y_test_fm = _run_base_model_dfm(dfTrain, dfTest, folds, fm_params, NUMERIC_COLS, IGNORE_COLS, application)


# ------------------ DNN Model ------------------
dnn_params = dfm_params.copy()
dnn_params["use_fm"] = False
y_train_dnn, y_test_dnn = _run_base_model_dfm(dfTrain, dfTest, folds, dnn_params, NUMERIC_COLS, IGNORE_COLS, application)
