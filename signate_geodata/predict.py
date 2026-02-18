"""
不動産価格予測モデル - SIGNATE 地理空間データ活用コンペティション
評価指標: MAPE (Mean Absolute Percentage Error)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# パス設定
DATA_DIR = '/home/ubuntu/signate/geospatial_data_challenge_2nd/data/raw'
OUTPUT_DIR = '/home/ubuntu/signate/geospatial_data_challenge_2nd/data/submit'

def load_data():
    """データの読み込み"""
    print("データ読み込み中...")
    train = pd.read_csv(f'{DATA_DIR}/train.csv')
    test = pd.read_csv(f'{DATA_DIR}/test.csv')
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    return train, test

def mape(y_true, y_pred):
    """MAPE計算"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mape_lgb(y_pred, data):
    """LightGBM用MAPEメトリック"""
    y_true = data.get_label()
    score = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return 'mape', score, False

def preprocess_features(train, test):
    """特徴量の前処理"""
    print("特徴量前処理中...")
    
    # ターゲット変数
    target = 'money_room'
    y_train = train[target].values
    
    # 不要なカラムを削除
    drop_cols = [target, 'id']
    
    # 日付関連のカラムを特定
    date_cols = [col for col in train.columns if 'date' in col.lower()]
    
    # テキスト系カラム（多くのユニーク値を持つ可能性）
    text_cols = ['building_name', 'building_name_ruby', 'homes_building_name', 
                 'homes_building_name_ruby', 'full_address', 'unit_name', 'name_ruby',
                 'addr2_name', 'addr3_name', 'rosen_name1', 'eki_name1', 'bus_stop1',
                 'rosen_name2', 'eki_name2', 'bus_stop2', 'traffic_other', 'traffic_car',
                 'school_ele_name', 'school_jun_name', 'reform_exterior_other',
                 'reform_common_area', 'reform_place', 'reform_place_other',
                 'reform_wet_area', 'reform_wet_area_other', 'reform_interior',
                 'reform_interior_other', 'reform_etc', 'renovation_etc',
                 'money_sonota_str1', 'money_sonota_str2', 'money_sonota_str3',
                 'parking_memo', 'empty_contents', 'land_seigen', 'est_other_name',
                 'building_tag_id', 'unit_tag_id', 'statuses']
    
    # 使用するカラム
    feature_cols = [col for col in train.columns 
                    if col not in drop_cols + date_cols + text_cols]
    
    # 日付からの特徴量抽出
    for df in [train, test]:
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if df[col].notna().any():
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    feature_cols.extend([f'{col}_year', f'{col}_month'])
    
    # 重複削除
    feature_cols = list(set(feature_cols))
    
    # カテゴリカル・数値カラムの分類
    cat_cols = []
    num_cols = []
    
    for col in feature_cols:
        if col in train.columns and col in test.columns:
            if train[col].dtype == 'object':
                cat_cols.append(col)
            else:
                num_cols.append(col)
    
    # target_ymから年月を抽出
    for df in [train, test]:
        df['year'] = df['target_ym'] // 100
        df['month'] = df['target_ym'] % 100
    
    if 'year' not in num_cols:
        num_cols.extend(['year', 'month'])
    
    # 築年数の計算
    for df in [train, test]:
        if 'year_built' in df.columns:
            df['building_age'] = df['year'] - (df['year_built'] // 100)
    
    if 'building_age' not in num_cols:
        num_cols.append('building_age')
    
    # 面積関連の特徴量
    for df in [train, test]:
        # 専有面積あたりの特徴量計算用
        if 'unit_area' in df.columns and 'total_floor_area' in df.columns:
            df['unit_area_ratio'] = df['unit_area'] / (df['total_floor_area'] + 1)
        if 'building_land_area' in df.columns and 'land_area_all' in df.columns:
            df['land_coverage'] = df['building_land_area'] / (df['land_area_all'] + 1)
        if 'house_area' in df.columns and 'snapshot_land_area' in df.columns:
            df['house_land_ratio'] = df['house_area'] / (df['snapshot_land_area'] + 1)
    
    for col in ['unit_area_ratio', 'land_coverage', 'house_land_ratio']:
        if col in train.columns and col not in num_cols:
            num_cols.append(col)
    
    # 使用する特徴量カラムを最終決定
    use_cols = num_cols + cat_cols
    use_cols = [col for col in use_cols if col in train.columns and col in test.columns]
    
    print(f"使用する特徴量数: {len(use_cols)}")
    print(f"  - 数値特徴量: {len([c for c in use_cols if c in num_cols])}")
    print(f"  - カテゴリカル特徴量: {len([c for c in use_cols if c in cat_cols])}")
    
    # データフレーム作成
    X_train = train[use_cols].copy()
    X_test = test[use_cols].copy()
    
    # カテゴリカル変数のエンコーディング
    label_encoders = {}
    for col in cat_cols:
        if col in X_train.columns:
            le = LabelEncoder()
            # trainとtestを結合してfit
            combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            label_encoders[col] = le
    
    # 欠損値処理（数値は-999で埋める）
    X_train = X_train.fillna(-999)
    X_test = X_test.fillna(-999)
    
    # すべてのカラムを数値型に変換
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            le = LabelEncoder()
            combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            if col not in cat_cols:
                cat_cols.append(col)
        else:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(-999)
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(-999)
    
    return X_train, X_test, y_train, use_cols, cat_cols

def train_model(X_train, y_train, cat_cols):
    """LightGBMモデルの学習"""
    print("\nモデル学習中...")
    
    # ハイパーパラメータ
    params = {
        'objective': 'regression',
        'metric': 'mape',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'max_depth': -1,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    # クロスバリデーション
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_pred = np.zeros(len(X_train))
    models = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train[train_idx], y_train[valid_idx]
        
        # カテゴリカルカラムのインデックスを取得
        cat_indices = [X_train.columns.get_loc(col) for col in cat_cols if col in X_train.columns]
        
        train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_indices)
        valid_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_indices)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            feval=mape_lgb,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=True),
                lgb.log_evaluation(period=200)
            ]
        )
        
        models.append(model)
        oof_pred[valid_idx] = model.predict(X_val)
        
        fold_mape = mape(y_val, oof_pred[valid_idx])
        print(f"Fold {fold + 1} MAPE: {fold_mape:.4f}%")
    
    overall_mape = mape(y_train, oof_pred)
    print(f"\n=== Overall OOF MAPE: {overall_mape:.4f}% ===")
    
    return models

def predict_and_submit(models, X_test, test_ids):
    """予測と提出ファイル作成"""
    print("\n予測中...")
    
    # 全モデルの平均で予測
    predictions = np.zeros(len(X_test))
    for model in models:
        predictions += model.predict(X_test)
    predictions /= len(models)
    
    # 負の値を補正（価格は正の値のみ）
    predictions = np.maximum(predictions, 1)
    
    # 整数に丸める
    predictions = np.round(predictions).astype(int)
    
    # 提出ファイル作成
    submission = pd.DataFrame({
        'id': test_ids,
        'money_room': predictions
    })
    
    # idでソート
    submission = submission.sort_values('id')
    
    # 提出ファイル保存
    output_path = f'{OUTPUT_DIR}/submission.csv'
    submission.to_csv(output_path, index=False, header=False)
    print(f"\n提出ファイルを保存しました: {output_path}")
    print(f"予測件数: {len(submission)}")
    print(f"予測価格統計:")
    print(f"  - 平均: {predictions.mean():,.0f}円")
    print(f"  - 中央値: {np.median(predictions):,.0f}円")
    print(f"  - 最小: {predictions.min():,.0f}円")
    print(f"  - 最大: {predictions.max():,.0f}円")
    
    return submission

def main():
    """メイン処理"""
    print("=" * 60)
    print("不動産価格予測モデル")
    print("=" * 60)
    
    # データ読み込み
    train, test = load_data()
    
    # テストIDを保存
    test_ids = test['id'].values
    
    # 特徴量前処理
    X_train, X_test, y_train, use_cols, cat_cols = preprocess_features(train, test)
    
    # モデル学習
    models = train_model(X_train, y_train, cat_cols)
    
    # 予測・提出
    submission = predict_and_submit(models, X_test, test_ids)
    
    print("\n=== 処理完了 ===")
    
    return submission

if __name__ == '__main__':
    main()

