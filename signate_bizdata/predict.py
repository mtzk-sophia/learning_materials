"""予測スクリプト"""

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from utils.common import load_data, save_submission
from train import EXCLUDE_COLS, get_feature_columns


def predict(test_path: str,
           sample_submission_path: str,
           model_dir: str = 'models',
           n_models: int = 5,
           exclude_cols: list = None,
           use_optimal_threshold: bool = True):
    """テストデータの予測
    
    Args:
        test_path (str): テストデータのパス
        sample_submission_path (str): サンプル提出ファイルのパス
        model_dir (str): モデルの保存ディレクトリ
        n_models (int): 使用するモデル数
        exclude_cols (list): 除外するカラムのリスト
        use_optimal_threshold (bool): 最適閾値を使用するか（Falseの場合は0.5）
    """
    # データの読み込み
    test = load_data(test_path)
    print(f'Test shape: {test.shape}')
    
    # 特徴量の選択（trainと同じカラムを除外）
    if exclude_cols is None:
        exclude_cols = EXCLUDE_COLS
    
    # 目的変数はテストデータにないので、除外リストから削除
    target_col = '購入フラグ'
    if target_col in test.columns:
        features = get_feature_columns(test, target_col, exclude_cols)
    else:
        # テストデータには目的変数がないので、除外カラムのみ除外
        features = [col for col in test.columns if col not in exclude_cols]
        print(f"\n=== 特徴量選択 ===")
        print(f"全カラム数: {len(test.columns)}")
        print(f"除外カラム数: {len(exclude_cols)}")
        print(f"使用特徴量数: {len(features)}")
    
    X_test = test[features]
    
    # モデルの読み込みと予測
    predictions = []
    for fold in range(n_models):
        model_path = os.path.join(model_dir, f'lgb_fold{fold}.pkl')
        if not os.path.exists(model_path):
            print(f'Warning: {model_path} not found, skipping...')
            continue
        
        model = joblib.load(model_path)
        pred = model.predict(X_test, num_iteration=model.best_iteration)
        predictions.append(pred)
        print(f'Fold {fold} prediction done (mean: {pred.mean():.4f}, std: {pred.std():.4f})')
    
    if len(predictions) == 0:
        raise ValueError("No models found for prediction!")
    
    # アンサンブル（平均）
    final_predictions_proba = np.mean(predictions, axis=0)
    
    # 閾値の読み込み
    threshold_path = os.path.join(model_dir, 'threshold_info.pkl')
    if use_optimal_threshold and os.path.exists(threshold_path):
        threshold_info = joblib.load(threshold_path)
        threshold = threshold_info['optimal_threshold']
        print(f'\n学習時の最適閾値を使用: {threshold:.3f}')
        print(f'CV F1スコア: {threshold_info["cv_f1_optimal"]:.5f}')
    else:
        threshold = 0.5
        print(f'\nデフォルト閾値を使用: {threshold:.3f}')
    
    # 確率から0/1に変換
    final_predictions = (final_predictions_proba >= threshold).astype(int)
    
    print(f'\n=== Prediction Probabilities ===')
    print(f'Mean: {final_predictions_proba.mean():.4f}')
    print(f'Std: {final_predictions_proba.std():.4f}')
    print(f'Min: {final_predictions_proba.min():.4f}')
    print(f'Max: {final_predictions_proba.max():.4f}')
    
    print(f'\n=== Binary Predictions (threshold={threshold:.3f}) ===')
    print(f'Class 0: {(final_predictions == 0).sum()} ({(final_predictions == 0).sum() / len(final_predictions) * 100:.1f}%)')
    print(f'Class 1: {(final_predictions == 1).sum()} ({(final_predictions == 1).sum() / len(final_predictions) * 100:.1f}%)')
    
    # 提出ファイルの作成
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'submissions/submission_{timestamp}.csv'
    save_submission(final_predictions, sample_submission_path, output_path)
    
    # 確率値も別ファイルに保存（デバッグ用）
    proba_output_path = f'submissions/submission_proba_{timestamp}.csv'
    save_submission(final_predictions_proba, sample_submission_path, proba_output_path)
    print(f'\n確率値も保存: {proba_output_path}')
    
    return final_predictions, final_predictions_proba, threshold


if __name__ == '__main__':
    # パラメータの設定
    TEST_PATH = 'data/processed/test_processed.csv'  # 前処理済みデータ
    SAMPLE_SUBMISSION_PATH = 'data/raw/sample_submit.csv'
    MODEL_DIR = 'models'
    N_MODELS = 5
    
    # 追加で除外したいカラムがあれば指定（trainと同じにする）
    CUSTOM_EXCLUDE = None
    
    # 最適閾値を使用するか（Falseの場合は0.5を使用）
    USE_OPTIMAL_THRESHOLD = True
    
    # 予測の実行
    predictions, predictions_proba, threshold = predict(
        TEST_PATH, 
        SAMPLE_SUBMISSION_PATH, 
        MODEL_DIR, 
        N_MODELS,
        exclude_cols=CUSTOM_EXCLUDE,
        use_optimal_threshold=USE_OPTIMAL_THRESHOLD
    )
    
    print("\n===== 予測完了 =====")
    print(f"提出ファイル: submissions/submission_*.csv")
    print(f"使用した閾値: {threshold:.3f}")
    print(f"予測クラス1の割合: {predictions.mean():.3f}")

