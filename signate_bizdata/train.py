"""モデルの学習スクリプト"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, f1_score, classification_report
import lightgbm as lgb
import joblib

from utils.common import seed_everything, load_data


# 学習に使用しないカラムを定義
EXCLUDE_COLS = [
    '企業ID',           # ID（リーク）
    '企業名',           # 固有名詞
    '企業概要',         # テキスト（未処理）
    '組織図',           # テキスト（未処理）
    '今後のDX展望',     # テキスト（未処理）
    '業界', 
    '上場種別', 
    '特徴', 
    # '業界_label',
    '業界_target_enc',
    '業界_freq',
    '業界_count',
    '業界_grouped', 
    '業界_grouped_label',
    '従業員数_mean_by_業界',
    '従業員数_std_by_業界',
    '従業員数_rank_in_業界',
    '従業員数_diff_from_業界_mean',
    '売上_mean_by_業界',
    '売上_std_by_業界',
    '売上_rank_in_業界',
    '売上_diff_from_業界_mean',
    '総資産_mean_by_業界',
    '総資産_std_by_業界',
    '総資産_rank_in_業界',
    '総資産_diff_from_業界_mean',
    '営業利益_mean_by_業界',
    '営業利益_std_by_業界',
    '営業利益_rank_in_業界',
    '営業利益_diff_from_業界_mean',
    '資本金_mean_by_業界',
    '資本金_std_by_業界',
    '資本金_rank_in_業界',
    '資本金_diff_from_業界_mean',
    '特徴_is_missing',
    '特徴_count',
    '特徴_target_enc',
    'BtoB_target_enc',
    'BtoC_target_enc',
    'CtoC_target_enc',
    '特徴_pattern_id',
    'is_BtoB_only',
    'is_BtoB_with_others',
    'is_not_BtoB',
    '業界_BtoB_ratio',
    '業界_BtoC_ratio',
    '業界_CtoC_ratio',
    '従業員数_x_BtoB',
    '従業員数_x_BtoC',
    '従業員数_x_CtoC',
    '売上_x_BtoB',
    '売上_x_BtoC',
    '売上_x_CtoC',
]


def find_best_threshold(y_true: np.ndarray, 
                       y_pred_proba: np.ndarray,
                       search_range: tuple = (0.1, 0.9),
                       step: float = 0.01) -> tuple:
    """F1スコアを最大化する閾値を探索
    
    Args:
        y_true: 正解ラベル
        y_pred_proba: 予測確率
        search_range: 探索範囲（最小値、最大値）
        step: 探索ステップ
        
    Returns:
        (最適閾値, 最大F1スコア)
    """
    best_threshold = 0.5
    best_f1 = 0.0
    
    thresholds = np.arange(search_range[0], search_range[1], step)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1, thresholds, f1_scores


def get_feature_columns(train: pd.DataFrame, 
                       target_col: str,
                       exclude_cols: list = None) -> list:
    """学習に使用する特徴量カラムを取得
    
    Args:
        train: 学習データ
        target_col: 目的変数のカラム名
        exclude_cols: 除外するカラムのリスト
        
    Returns:
        使用する特徴量のリスト
    """
    if exclude_cols is None:
        exclude_cols = EXCLUDE_COLS
    
    # 目的変数と除外カラムを除いた特徴量を取得
    all_exclude = exclude_cols + [target_col]
    features = [col for col in train.columns if col not in all_exclude]
    
    print(f"\n=== 特徴量選択 ===")
    print(f"全カラム数: {len(train.columns)}")
    print(f"除外カラム数: {len(all_exclude)}")
    print(f"使用特徴量数: {len(features)}")
    print(f"\n除外されたカラム:")
    for col in all_exclude:
        print(f"  - {col}")
    
    return features


def train_model(train_path: str, 
               target_col: str,
               n_splits: int = 5,
               seed: int = 42,
               exclude_cols: list = None):
    """モデルの学習
    
    Args:
        train_path (str): 学習データのパス
        target_col (str): 目的変数のカラム名
        n_splits (int): CVのfold数
        seed (int): 乱数シード
        exclude_cols (list): 除外するカラムのリスト（Noneの場合はEXCLUDE_COLSを使用）
    """
    seed_everything(seed)
    
    # データの読み込み
    train = load_data(train_path)
    print(f'Train shape: {train.shape}')
    
    # 特徴量とターゲットの分離
    features = get_feature_columns(train, target_col, exclude_cols)
    X = train[features]
    y = train[target_col]
    
    print(f'\n=== モデル学習開始 ===')
    print(f'特徴量数: {len(features)}')
    print(f'サンプル数: {len(train)}')
    print(f'目的変数の分布: {y.value_counts().to_dict()}')
    
    # クロスバリデーション（分類問題なのでStratifiedKFoldを使用）
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    models = []
    oof_predictions = np.zeros(len(train))
    fold_thresholds = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        print(f'\n===== Fold {fold + 1} / {n_splits} =====')
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        print(f'Train samples: {len(X_train)}, Valid samples: {len(X_valid)}')
        print(f'Train positive rate: {y_train.mean():.3f}, Valid positive rate: {y_valid.mean():.3f}')
        
        # LightGBMの学習
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
        
        params = {
            'objective': 'binary',  # 二値分類
            'metric': 'binary_logloss',  # 学習時はloglossを使用
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'random_state': seed,
            'verbose': -1,
            'is_unbalance': True  # 不均衡データの場合
            # 'learning_rate': 0.108106,
            # 'num_leaves': 183,
            # 'max_depth': 3,
            # 'min_child_samples': 28,
            # 'subsample': 0.7489,
            # 'subsample_freq': 1,
            # 'colsample_bytree': 0.7825,
            # 'reg_alpha': 0.000910,
            # 'reg_lambda': 0.000000,
            # 'min_split_gain': 0.216622,
            # 'random_state': seed,
            # 'verbose': -1,
            # 'is_unbalance': True
        }
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train, lgb_valid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # 予測（確率）
        oof_predictions[valid_idx] = model.predict(X_valid, num_iteration=model.best_iteration)
        
        # 最適な閾値を探索（このfoldのvalidデータで）
        best_threshold, best_f1, _, _ = find_best_threshold(
            y_valid.values, 
            oof_predictions[valid_idx],
            search_range=(0.1, 0.9),
            step=0.01
        )
        fold_thresholds.append(best_threshold)
        
        # スコア計算
        fold_auc = roc_auc_score(y_valid, oof_predictions[valid_idx])
        y_pred_binary = (oof_predictions[valid_idx] >= best_threshold).astype(int)
        fold_f1 = f1_score(y_valid, y_pred_binary)
        
        print(f'Fold {fold + 1} AUC: {fold_auc:.5f}')
        print(f'Fold {fold + 1} F1 Score: {fold_f1:.5f} (threshold={best_threshold:.3f})')
        
        # モデルの保存
        models.append(model)
        joblib.dump(model, f'models/lgb_fold{fold}.pkl')
    
    # CV全体のスコア計算
    # 1. 全foldの平均閾値を使用
    avg_threshold = np.mean(fold_thresholds)
    
    # 2. OOF全体で最適閾値を再探索
    best_threshold_overall, best_f1_overall, thresholds_all, f1_scores_all = find_best_threshold(
        y.values, 
        oof_predictions,
        search_range=(0.1, 0.9),
        step=0.01
    )
    
    cv_auc = roc_auc_score(y, oof_predictions)
    y_pred_avg = (oof_predictions >= avg_threshold).astype(int)
    y_pred_optimal = (oof_predictions >= best_threshold_overall).astype(int)
    
    cv_f1_avg = f1_score(y, y_pred_avg)
    cv_f1_optimal = f1_score(y, y_pred_optimal)
    
    print(f'\n===== CV Score =====')
    print(f'AUC: {cv_auc:.5f}')
    print(f'\n--- F1 Score (平均閾値={avg_threshold:.3f}) ---')
    print(f'F1 Score: {cv_f1_avg:.5f}')
    print(f'\n--- F1 Score (最適閾値={best_threshold_overall:.3f}) ---')
    print(f'F1 Score: {cv_f1_optimal:.5f}')
    
    print(f'\n===== Classification Report (最適閾値) =====')
    print(classification_report(y, y_pred_optimal, target_names=['Class 0', 'Class 1']))
    
    # 閾値情報を保存
    threshold_info = {
        'fold_thresholds': fold_thresholds,
        'avg_threshold': avg_threshold,
        'optimal_threshold': best_threshold_overall,
        'cv_f1_avg': cv_f1_avg,
        'cv_f1_optimal': cv_f1_optimal
    }
    joblib.dump(threshold_info, 'models/threshold_info.pkl')
    
    # OOF予測の保存
    oof_df = pd.DataFrame({
        '企業ID': train['企業ID'],  # IDも一緒に保存
        'target': y,
        'prediction_proba': oof_predictions,
        'prediction_avg_threshold': y_pred_avg,
        'prediction_optimal_threshold': y_pred_optimal
    })
    oof_df.to_csv('models/oof_predictions.csv', index=False)
    print(f'\nOOF predictions saved to models/oof_predictions.csv')
    
    # 閾値とF1スコアの関係を保存
    threshold_curve = pd.DataFrame({
        'threshold': thresholds_all,
        'f1_score': f1_scores_all
    })
    threshold_curve.to_csv('models/threshold_f1_curve.csv', index=False)
    
    # 特徴量重要度の保存
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': np.mean([model.feature_importance() for model in models], axis=0)
    }).sort_values('importance', ascending=False)
    feature_importance.to_csv('models/feature_importance.csv', index=False)
    
    print('\n===== Top 20 Features =====')
    for i, row in feature_importance.head(20).iterrows():
        print(f"{row['feature']:40s} {row['importance']:8.1f}")
    
    return models, oof_predictions, feature_importance, threshold_info


if __name__ == '__main__':
    # パラメータの設定
    TRAIN_PATH = 'data/processed/train_processed.csv'  # 前処理済みデータ
    TARGET_COL = '購入フラグ'
    N_SPLITS = 5
    SEED = 525193
    
    # 追加で除外したいカラムがあれば指定（Noneの場合はEXCLUDE_COLSを使用）
    # CUSTOM_EXCLUDE = EXCLUDE_COLS + ['上場種別']  # 例：上場種別も除外したい場合
    CUSTOM_EXCLUDE = None
    
    # 学習の実行
    models, oof_preds, feature_importance, threshold_info = train_model(
        TRAIN_PATH, 
        TARGET_COL, 
        N_SPLITS, 
        SEED,
        exclude_cols=CUSTOM_EXCLUDE
    )
    
    print("\n===== 学習完了 =====")
    print(f"モデル保存先: models/lgb_fold*.pkl")
    print(f"OOF予測: models/oof_predictions.csv")
    print(f"特徴量重要度: models/feature_importance.csv")
    print(f"閾値情報: models/threshold_info.pkl")
    print(f"閾値-F1曲線: models/threshold_f1_curve.csv")
    print(f"\n推奨閾値: {threshold_info['optimal_threshold']:.3f}")
    print(f"この閾値でのF1スコア: {threshold_info['cv_f1_optimal']:.5f}")

