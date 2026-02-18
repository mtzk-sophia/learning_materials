# Signate 物件価格予測
_Exported on 2026/2/19 at 6:14:20 JST from Cursor (2.4.31)_

---

**User**

SIGNATEで以下の予測に取り組みます．予測用のコードを作成して下さい．出力形式はsample_submit.csvを参考にして下さい．
## 学習用データ：train.csvには、2019年〜2022年の1⽉または7⽉にLIFULLポータル上に掲載された物件データが含まれています。このデータは、過去の売買価格(money_room)や物件の特徴を含んでおり、売買価格予測モデルの開発に使⽤できます。
## 評価⽤データ：test.csvには、売買価格(money_room)を除く2023年の1⽉または7⽉にLIFULLポータル上に掲載された物件データが含まれています。idというサンプルIDが新たに付与されています。このデータを基に売買価格（money_room）を予測していただきます。
## 評価方法：評価指標として「MAPE（Mean Absolute Percentage Error）」を使用します。MAPEは正解値の絶対値と実際の値との絶対誤差の比のパーセントの平均値を取ったもので、予測精度が高いほど値が小さくなります。予測値はある年月におけるある物件の売買価格(単位は円)とします。

---

**Cursor**

まず、データの構造を確認させてください。

データの構造が分かりました。次にデータのサイズと特徴量をもう少し確認します。

データ概要を把握しました：
- train.csv: 約36万行、149カラム
- test.csv: 約11万行
- 提出形式: id, money_room (予測価格)

それでは、不動産価格予測モデルのコードを作成します。

```python
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

```

モデル学習は成功しました（OOF MAPE: 15.79%）。予測時にデータ型のエラーがあるので修正します。

```python
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


```

---

**User**

これはどのように実行すれば良いですか？

---

**Cursor**

以下のコマンドで実行できます：

```bash
cd /home/ubuntu/signate/geospatial_data_challenge_2nd
python src/predict.py
```

実行しますか？

```shellscript
cd /home/ubuntu/signate/geospatial_data_challenge_2nd
python src/predict.py
```

実行完了しました！結果をご説明します：

## 実行結果

**モデル性能（5-Fold クロスバリデーション）:**
- Fold 1 MAPE: 15.73%
- Fold 2 MAPE: 15.72%
- Fold 3 MAPE: 15.83%
- Fold 4 MAPE: 15.72%
- Fold 5 MAPE: 15.81%
- **Overall OOF MAPE: 15.76%**

**予測統計:**
- 予測件数: 112,437件
- 平均価格: 約2,837万円
- 中央値: 約2,428万円
- 最小: 約165万円
- 最大: 約1.7億円

**提出ファイル:**

```
/home/ubuntu/signate/geospatial_data_challenge_2nd/data/submit/submission.csv
```

提出ファイルの内容を確認しましょう：

```plaintext
/home/ubuntu/signate/geospatial_data_challenge_2nd/data/submit/submission.csv
```

```plaintext
/home/ubuntu/signate/geospatial_data_challenge_2nd/data/submit/submission.csv
```

提出ファイルが正常に作成されました。

## 実行方法のまとめ

```bash
cd /home/ubuntu/signate/geospatial_data_challenge_2nd
python src/predict.py
```

## 提出ファイル
- **ファイルパス:** `/home/ubuntu/signate/geospatial_data_challenge_2nd/data/submit/submission.csv`
- **形式:** `id,money_room`（ヘッダーなし、sample_submit.csvと同じ形式）

このファイルをSIGNATEにアップロードすれば提出完了です！

```shellscript
cd /home/ubuntu/signate/geospatial_data_challenge_2nd
python src/predict.py
```

```shellscript
cd /home/ubuntu/signate/geospatial_data_challenge_2nd
python src/predict.py
```

---

**User**

ありがとうございます．提出したところ，スコアは17くらいでした．

あなたはWebページのURLを入力に入れれば，そのページの内容を理解し回答してくれますか？

前年のコンペの入賞者の記事を見つけたのですが．．．

---

**Cursor**

スコア17%とのこと、お疲れ様でした！

## URLについて

残念ながら、**URLを直接入力してページ内容を読み込む機能は持っていません**。ただし、以下の方法で対応できます：

1. **Web検索機能**: キーワードで関連情報を検索することは可能です
2. **記事内容のコピー**: 記事の内容をコピーしてこのチャットに貼り付けていただければ、理解して回答できます

## 前年のコンペについて

前年のコンペの入賞者の解法を探してみましょうか？

もう少し具体的に検索してみます：

検索がキャンセルされましたね。

**お勧めの方法：**

既に入賞者の記事を見つけているとのことですので、その記事の**重要な部分（特徴量エンジニアリング、モデル選択、前処理など）をコピーしてこのチャットに貼り付けて**いただければ、その内容を理解して、現在のコードに取り入れる改善点をご提案できます。

特に知りたい情報があれば：
- 使用した特徴量
- モデルのアンサンブル方法
- データの前処理手法
- 外部データの活用方法

などをお知らせください！

---

**User**

記事データをコピペして貼り付けます．

参考にできそうなところがあれば参考にして下さい．



```

国交省コンペ優勝！賃料予測コンペ1位解法

kenkoooo

technology



こんにちは、スタッフエンジニアの @kenkoooo です！

先日開催されたデータサイエンスコンペ「第1回 国土交通省 地理空間情報データチャレンジ ～国土数値情報編～ モデリング部門」で優勝しました！(最終順位表)



この記事では1位解法を解説しています。不動産データ分析の雰囲気が伝われば幸いです。



第1回 国土交通省 地理空間情報データチャレンジ ～国土数値情報編～ モデリング部門とは

国土交通省が主催する、賃貸住宅の賃料を予測するコンペです。株式会社LIFULLが提供する全国の賃貸マンション・アパートの賃料と物件情報を元に、与えられた賃貸物件の賃料を予測します。



上記データの他に、国土交通省が提供する「国土数値情報」というデータを活用できます。また、株式会社ゼンリンが提供する ZENRIN Maps API や、Snowflake合同会社が提供する Snowflake 環境なども利用できます。



1位解法

LightGBM を使う

LightGBM を素のまま使いました。パラメータチューニングも行いませんでしたが、seed を変えたモデルをいくつか用意し、それらの予測値の平均をとりました。



PARAMS = {

    "objective": "regression",

    "metric": "l1",

    "seed": 42,

    "learning_rate": 0.1,

}



lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

model = lgb.train(

    PARAMS,

    lgb_train,

    valid_sets=lgb_eval,

    num_boost_round=100000,

    callbacks=[

        lgb.early_stopping(stopping_rounds=100),

        lgb.log_evaluation(100),

    ],

)

平米単価の誤差を最小化するようにする

コンペの評価指標は賃料の予測値の RMSE となっているので、最初はこれに従い、賃料の RMSE を最小化するように学習させていました。しかし、手元の Cross Validation を見ると Fold ごとに大きくスコアが異なることに気づきました。Fold 内の予測値を見てみると、賃料が50万円を超えるような少数の外れ値に振り回されている雰囲気を感じました。



estie でよく扱うオフィス賃貸の物件では、賃料総額よりも単位面積あたりの賃料（オフィスの場合は坪単価）に注目することが多いです。これは、契約前から間取りや面積が固定されている住宅と異なり、オフィスは何もないフロアに契約後に壁を立てて契約した広さの区画を作るため、契約時まで賃料総額が決まらないからです。これをヒントに、今回のコンペでも賃料総額ではなく、平米単価の誤差を最小化することにしました。



賃料総額が非常に高い募集では、平米単価が高いのもさることながら、専有面積も大きいことが多いです。高い平米単価 × 大きい専有面積 = 非常に高い賃料、となっているわけです。賃料総額ではなく平米単価をターゲットにすることで、外れ値の外れ具合を緩和できた面もあります。



RMSE ではなく MAE を最小化するようにする

機械学習モデルを学習させる際の評価指標として RMSE ではなく MAE を採用しました。RMSE は誤差を二乗して平均を取るため、10万円の誤差1件が1万円の誤差100件と同じ影響を与えることになります。そのため、外れ値に大きく引っ張られやすい指標です。このような性質から、高級賃貸物件に最適化され、大多数の通常物件に対する予測精度が低下する可能性がありました。



一方で、MAEは誤差を絶対値として評価するため、外れ値の影響を抑えることができます。通常の賃貸物件で適切な予測精度を維持しつつ、高級賃貸への依存度を低下させることを目的に、MAEを選択しました。テストデータに高級賃貸物件が多く含まれていると危ない気がしますが、RMSE で学習させたとしても精度高く当てにいくのは難しいと思い、割り切りました。



欠損している面積を埋める

先に述べたように平米単価を予測するようにしたため、専有面積が直接スコアに関わってくる重要な値となりましたが、与えられるデータの一部は専有面積が欠損していました。そこで、まず専有面積予測モデルを作り、これを使って専有面積を埋めてから、平米単価を予測するようにしました。専有面積予測モデルは平米単価予測モデルとほぼ同じ LightGBM のモデルです。



欠損している郵便番号を埋める

後述の通り郵便番号が非常に強い特徴量になっていましたが、与えられるデータの一部は郵便番号が欠損していたので、ゼンリン API を使って郵便番号を埋めました。



同一部屋の賃料から賃料を決め打ちする

test と同じ部屋の別の時期のデータが train に含まれているケースがありました。都心の賃料は上昇傾向にある一方で郊外は下落傾向にあったり、コロナで落ち込んでいる時期があったり、同じ部屋でも時期によって賃料は異なります。募集時期を変数に線形回帰した賃料を使いました。



同一ビルの賃料から賃料を決め打ちする

test と同じ建物のデータが train にも含まれているケースが多数ありました。当初は、同じ物件なら平米単価も同じと考えて train のデータで test を埋めようとしましたが、train の中でも同一物件内で平米単価が大きく異なるケースがいくつか見つかりました。そこで、平米単価をそのまま使うのではなく、専有面積・階数・募集時期で線形回帰した値を使いました。



出力を見てみると極端な値がいくつかあったので、適当な閾値を超える値は使わないようにしました。



特徴量を作る

与えられるデータをそのまま使ったり、集計したりした他、国土数値情報やゼンリン API と組み合わせて427個の特徴量を作りました。RMSE での評価値を改善するため、外れ値である高級賃貸に特化した特徴量も多くあります。



与えられるデータから作る

与えられるデータをそのまま使ったものもありますが、その中でも次の特徴量は特に強かったです。



専有面積

データをいくつか見てみたところ、郊外では専有面積が大きくなると平米単価は下がる傾向がある一方で、都心では専有面積が大きくなると平米単価も上がる傾向がありました。

コンペ終了後に社内の有識者に聞いてみたところ、都心の広い賃貸マンションは富裕層からの需要が高くなるため、広い部屋ほど平米単価が上がる傾向があるとのことでした。

間取りを数値化したもの

1K < 1DK < 1LDK < 2K < … のように間取りに序列をつけて数値化したものです。

築年数

市区町村名

郵便番号

初期から非常に強い特徴量でしたが、なぜかは最後まで分かりませんでした。郵便番号で賃貸を選ぶことはないと思うので、裏に隠れた何らかの地域性が重要だとは思うのですが……

また、与えられるデータから新しく特徴量を作りました。その中でも特に強かった特徴量を紹介します。



同一郵便番号内での平米単価の平均値・分散

建物周辺の募集の平米単価の平均値を「相場」として使えるのではないかと考えました。

郵便番号でグルーピングしたのは、前述の通りなぜか郵便番号単体が非常に強い特徴量だったためです。

周囲 2 km 以内の平米単価の平均値・分散

建物周辺の募集の平米単価の平均値を「相場」として使っています。2 kmという閾値は適当で、地図を見たときに自分が散歩する範囲が自宅の半径 2 km 以内だったためです。

築年数が同一市区町村内の平均の何倍か

築年数は賃貸を選ぶときによく見る指標の一つですが、古い建物が多い地域での築浅の物件は価値が高くなると考え、特徴量にしました。

専有面積が同一市区町村内の平均の何倍か

都心で専有面積が大きい物件は希少価値が高いので、高級賃貸を当てる指標になるのではないかと思い、特徴量にしました。

町名

同じ港区でも住所が新橋か六本木かで賃料が変わりそうだと思ったので入れました。

国土数値情報

特徴量の作成に国土数値情報を活用しました。利用した国土数値情報は以下の10個です。



地価公示

　a. 地価だけでも強い特徴量だけでしたが、直近のものだけでなく過去の年度の値もあるので、変動率のような特徴量も作ることができ、精度改善に繋がりました。



都道府県地価調査

　a. こちらも強い特徴量でした。地価公示とは調査主体が異なるので、地価のアンサンブルのような感じになったのではないでしょうか。



小学校区

　a. 都心では中学受験に備えて有名公立小学校に子どもを入れたいらしい、という話を聞いたので入れました。



駅別乗降客数

　a. 最寄り駅の乗降客数として入れることで、地域の性質を表現できるのではないかと考えました。

　b. 駅自体の情報も入っているので、最寄り駅への距離を出すこともできました。

　c. 乗降客数上位25の駅からの距離を出すことで、「都会度」を表現する特徴量も作りました。



医療機関

　a. 小児科が近い物件はファミリー需要から賃料も高くなるのではないかと考え、最寄りの小児科への距離を算出するのに使いました。



1kmメッシュ別将来推計人口

　a. 会社で事業開発の人が「人口は最も嘘をつかない指標」と言っていたのを聞いて入れました。

　b. 全体的に強く、特に 35〜39 歳の人口、 45〜49 歳の人口、 5〜9 歳の人口の3つが強かったです。



平年値（気候）メッシュ

　a. 過ごしやすい場所ほど賃料が高いのではないかと考えました。

　b. 日照時間や最高気温が特徴量として強かったです。



用途地域

　a. 第一種低層住居専用地域にはコンビニなどの店舗が作れないが、第二種低層住居専用地域には作れる、みたいな情報が効くのではないかと思い、入れました。そこそこ強い特徴量でした。



人口集中地区データ

　a. 地域の性質を表現できるのではないかと考えました。人口密度の値が効いていました。



学校

　a. 最寄りの保育園までの距離を入れるのに使いました。



ハザードマップとして用いられる洪水浸水想定区域データも使えれば良かったのですが、他のデータに比べてサイズが桁違いに大きかったので、後回しにしているうちに終わってしまいました。



ゼンリン API

ゼンリン API を用いて全国のコンビニ一覧を入手し、最寄りのコンビニまでの距離を特徴量として入れました。



1位解法を支える技術

個人的に、アルゴリズムやヒューリスティックの競技プログラミングは好きなのですが、データサイエンスのコンペには苦手意識がありました。過去に参加したコンペでは、notebook やスクリプトが散乱し、データの作成やモデルの学習が再現できなくなり、モチベーションを失っていくことが多々あったためです。



今回は苦手意識を払拭すべく、データや学習の再現性を担保して、継続的にモデルを改善できる開発環境を作ることに重点を置きました。このコンペでは国土数値情報やゼンリンなどの外部データを利用することができるため、モデルの設計を頑張るよりも、データエンジニアリングを頑張って良い特徴量を作ることが重要になると考えたのもあります。

```

---

**Cursor**

素晴らしい記事ですね！非常に参考になるポイントがたくさんあります。この解法を今回の売買価格予測に適用できる改善点をまとめます。

## 主要な改善ポイント

### 1. **平米単価を予測ターゲットにする** 📊
記事の最も重要なポイントです。`money_room`（売買価格）を直接予測するのではなく、`money_room / unit_area`（平米単価）を予測し、最後に専有面積を掛けて戻します。

### 2. **MAE（L1）を使用** 
MAPEの代わりにMAEを学習指標にすることで外れ値の影響を抑制。

### 3. **同一建物の情報を活用**
`building_id`が同じ物件のtrainデータから価格情報を活用。

### 4. **郵便番号の活用**
`post1`, `post2`（郵便番号）は非常に強い特徴量とのこと。

### 5. **重要な特徴量エンジニアリング**
- 同一郵便番号内の平米単価平均・分散
- 同一市区町村内での築年数・面積の相対値
- 周囲の相場情報

---

## 改善版コード

Agentモードに切り替えていただければ、以下の改善を適用できます：

```python
"""
不動産価格予測モデル - 改善版
参考: 第1回地理空間情報データチャレンジ 1位解法
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

def create_target_features(train, test):
    """
    平米単価をターゲットにする & 同一建物の情報を活用
    """
    print("ターゲット特徴量作成中...")
    
    # 専有面積の欠損を埋める（中央値で）
    unit_area_median = train['unit_area'].median()
    train['unit_area'] = train['unit_area'].fillna(unit_area_median)
    test['unit_area'] = test['unit_area'].fillna(unit_area_median)
    
    # 平米単価を計算（ターゲット）
    train['price_per_sqm'] = train['money_room'] / train['unit_area'].replace(0, np.nan)
    train['price_per_sqm'] = train['price_per_sqm'].fillna(train['price_per_sqm'].median())
    
    # 同一建物(building_id)の平米単価統計をtrainから計算
    building_stats = train.groupby('building_id').agg({
        'price_per_sqm': ['mean', 'std', 'min', 'max', 'count'],
        'money_room': ['mean', 'std']
    }).reset_index()
    building_stats.columns = ['building_id', 'bld_price_sqm_mean', 'bld_price_sqm_std', 
                               'bld_price_sqm_min', 'bld_price_sqm_max', 'bld_count',
                               'bld_price_mean', 'bld_price_std']
    
    train = train.merge(building_stats, on='building_id', how='left')
    test = test.merge(building_stats, on='building_id', how='left')
    
    return train, test

def create_location_features(train, test):
    """
    郵便番号・市区町村ごとの相場情報を特徴量として追加
    """
    print("位置情報特徴量作成中...")
    
    # 郵便番号(post1)ごとの平米単価統計
    post_stats = train.groupby('post1').agg({
        'price_per_sqm': ['mean', 'std', 'median'],
        'money_room': ['mean', 'median']
    }).reset_index()
    post_stats.columns = ['post1', 'post_price_sqm_mean', 'post_price_sqm_std', 
                          'post_price_sqm_median', 'post_price_mean', 'post_price_median']
    
    train = train.merge(post_stats, on='post1', how='left')
    test = test.merge(post_stats, on='post1', how='left')
    
    # 市区町村(addr1_1 + addr1_2)ごとの統計
    for df in [train, test]:
        df['city_code'] = df['addr1_1'].astype(str) + '_' + df['addr1_2'].astype(str)
    
    city_stats = train.groupby('city_code').agg({
        'price_per_sqm': ['mean', 'std'],
        'year_built': ['mean'],
        'unit_area': ['mean']
    }).reset_index()
    city_stats.columns = ['city_code', 'city_price_sqm_mean', 'city_price_sqm_std',
                          'city_year_built_mean', 'city_unit_area_mean']
    
    train = train.merge(city_stats, on='city_code', how='left')
    test = test.merge(city_stats, on='city_code', how='left')
    
    # 築年数・面積の地域平均との比
    for df in [train, test]:
        df['year_built_ratio'] = df['year_built'] / df['city_year_built_mean'].replace(0, np.nan)
        df['unit_area_ratio'] = df['unit_area'] / df['city_unit_area_mean'].replace(0, np.nan)
    
    return train, test

def preprocess_features(train, test):
    """特徴量の前処理"""
    print("特徴量前処理中...")
    
    # ターゲット変数（平米単価）
    y_train = train['price_per_sqm'].values
    
    # 不要なカラムを削除
    drop_cols = ['money_room', 'price_per_sqm', 'id']
    
    # 日付関連のカラムを特定
    date_cols = [col for col in train.columns if 'date' in col.lower()]
    
    # テキスト系カラム
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
                 'building_tag_id', 'unit_tag_id', 'statuses', 'city_code']
    
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
    
    # 使用する特徴量カラムを最終決定
    use_cols = num_cols + cat_cols
    use_cols = [col for col in use_cols if col in train.columns and col in test.columns]
    
    print(f"使用する特徴量数: {len(use_cols)}")
    
    # データフレーム作成
    X_train = train[use_cols].copy()
    X_test = test[use_cols].copy()
    
    # カテゴリカル変数のエンコーディング
    for col in cat_cols:
        if col in X_train.columns:
            le = LabelEncoder()
            combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
    
    # 欠損値処理
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
    """LightGBMモデルの学習（MAE使用 + seedアンサンブル）"""
    print("\nモデル学習中...")
    
    n_splits = 5
    seeds = [42, 123, 456]  # 複数seedでアンサンブル
    
    all_models = []
    
    for seed in seeds:
        print(f"\n=== Seed: {seed} ===")
        
        params = {
            'objective': 'regression',
            'metric': 'mae',  # MAEを使用（外れ値に強い）
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
            'seed': seed
        }
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
            print(f"Fold {fold + 1}/{n_splits}")
            
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_tr, y_val = y_train[train_idx], y_train[valid_idx]
            
            cat_indices = [X_train.columns.get_loc(col) for col in cat_cols if col in X_train.columns]
            
            train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_indices)
            valid_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_indices)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=3000,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=500)
                ]
            )
            
            all_models.append(model)
    
    print(f"\n学習完了: {len(all_models)}モデル")
    return all_models

def predict_and_submit(models, X_test, test_unit_area, test_ids):
    """予測と提出ファイル作成"""
    print("\n予測中...")
    
    # 全モデルの平均で平米単価を予測
    price_per_sqm_pred = np.zeros(len(X_test))
    for model in models:
        price_per_sqm_pred += model.predict(X_test)
    price_per_sqm_pred /= len(models)
    
    # 平米単価 × 専有面積 = 売買価格
    predictions = price_per_sqm_pred * test_unit_area
    
    # 負の値を補正
    predictions = np.maximum(predictions, 1)
    
    # 整数に丸める
    predictions = np.round(predictions).astype(int)
    
    # 提出ファイル作成
    submission = pd.DataFrame({
        'id': test_ids,
        'money_room': predictions
    })
    submission = submission.sort_values('id')
    
    output_path = f'{OUTPUT_DIR}/submission_v2.csv'
    submission.to_csv(output_path, index=False, header=False)
    print(f"\n提出ファイルを保存しました: {output_path}")
    print(f"予測価格統計:")
    print(f"  - 平均: {predictions.mean():,.0f}円")
    print(f"  - 中央値: {np.median(predictions):,.0f}円")
    
    return submission

def main():
    print("=" * 60)
    print("不動産価格予測モデル - 改善版")
    print("=" * 60)
    
    # データ読み込み
    train, test = load_data()
    
    # テストIDと専有面積を保存
    test_ids = test['id'].values
    
    # 平米単価ターゲット & 同一建物情報
    train, test = create_target_features(train, test)
    
    # 位置情報特徴量
    train, test = create_location_features(train, test)
    
    # 専有面積を保存（最後に掛け算する）
    test_unit_area = test['unit_area'].values
    
    # 特徴量前処理
    X_train, X_test, y_train, use_cols, cat_cols = preprocess_features(train, test)
    
    # モデル学習
    models = train_model(X_train, y_train, cat_cols)
    
    # 予測・提出
    submission = predict_and_submit(models, X_test, test_unit_area, test_ids)
    
    print("\n=== 処理完了 ===")
    return submission

if __name__ == '__main__':
    main()
```

---

## 主な改善点まとめ

| 改善項目 | 変更内容 |
|---------|---------|
| ターゲット | `money_room` → `price_per_sqm`（平米単価） |
| 損失関数 | MAPE → MAE（外れ値に強い） |
| 同一建物情報 | `building_id`で集計した価格統計を特徴量に追加 |
| 郵便番号活用 | `post1`ごとの価格統計を特徴量に追加 |
| 地域相場 | 市区町村ごとの築年数・面積平均との比を追加 |
| アンサンブル | 複数seed（42, 123, 456）のモデル平均 |

---

適用したい場合は **Agentモードに切り替えて** ください！

```python
"""
不動産価格予測モデル - 改善版
参考: 第1回地理空間情報データチャレンジ 1位解法
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

def create_target_features(train, test):
    """
    平米単価をターゲットにする & 同一建物の情報を活用
    """
    print("ターゲット特徴量作成中...")
    
    # 専有面積の欠損を埋める（中央値で）
    unit_area_median = train['unit_area'].median()
    train['unit_area'] = train['unit_area'].fillna(unit_area_median)
    test['unit_area'] = test['unit_area'].fillna(unit_area_median)
    
    # 平米単価を計算（ターゲット）
    train['price_per_sqm'] = train['money_room'] / train['unit_area'].replace(0, np.nan)
    train['price_per_sqm'] = train['price_per_sqm'].fillna(train['price_per_sqm'].median())
    
    # 同一建物(building_id)の平米単価統計をtrainから計算
    building_stats = train.groupby('building_id').agg({
        'price_per_sqm': ['mean', 'std', 'min', 'max', 'count'],
        'money_room': ['mean', 'std']
    }).reset_index()
    building_stats.columns = ['building_id', 'bld_price_sqm_mean', 'bld_price_sqm_std', 
                               'bld_price_sqm_min', 'bld_price_sqm_max', 'bld_count',
                               'bld_price_mean', 'bld_price_std']
    
    train = train.merge(building_stats, on='building_id', how='left')
    test = test.merge(building_stats, on='building_id', how='left')
    
    return train, test

def create_location_features(train, test):
    """
    郵便番号・市区町村ごとの相場情報を特徴量として追加
    """
    print("位置情報特徴量作成中...")
    
    # 郵便番号(post1)ごとの平米単価統計
    post_stats = train.groupby('post1').agg({
        'price_per_sqm': ['mean', 'std', 'median'],
        'money_room': ['mean', 'median']
    }).reset_index()
    post_stats.columns = ['post1', 'post_price_sqm_mean', 'post_price_sqm_std', 
                          'post_price_sqm_median', 'post_price_mean', 'post_price_median']
    
    train = train.merge(post_stats, on='post1', how='left')
    test = test.merge(post_stats, on='post1', how='left')
    
    # 市区町村(addr1_1 + addr1_2)ごとの統計
    for df in [train, test]:
        df['city_code'] = df['addr1_1'].astype(str) + '_' + df['addr1_2'].astype(str)
    
    city_stats = train.groupby('city_code').agg({
        'price_per_sqm': ['mean', 'std'],
        'year_built': ['mean'],
        'unit_area': ['mean']
    }).reset_index()
    city_stats.columns = ['city_code', 'city_price_sqm_mean', 'city_price_sqm_std',
                          'city_year_built_mean', 'city_unit_area_mean']
    
    train = train.merge(city_stats, on='city_code', how='left')
    test = test.merge(city_stats, on='city_code', how='left')
    
    # 築年数・面積の地域平均との比
    for df in [train, test]:
        df['year_built_ratio'] = df['year_built'] / df['city_year_built_mean'].replace(0, np.nan)
        df['unit_area_ratio'] = df['unit_area'] / df['city_unit_area_mean'].replace(0, np.nan)
    
    return train, test

def preprocess_features(train, test):
    """特徴量の前処理"""
    print("特徴量前処理中...")
    
    # ターゲット変数（平米単価）
    y_train = train['price_per_sqm'].values
    
    # 不要なカラムを削除
    drop_cols = ['money_room', 'price_per_sqm', 'id']
    
    # 日付関連のカラムを特定
    date_cols = [col for col in train.columns if 'date' in col.lower()]
    
    # テキスト系カラム
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
                 'building_tag_id', 'unit_tag_id', 'statuses', 'city_code']
    
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
    
    # 使用する特徴量カラムを最終決定
    use_cols = num_cols + cat_cols
    use_cols = [col for col in use_cols if col in train.columns and col in test.columns]
    
    print(f"使用する特徴量数: {len(use_cols)}")
    
    # データフレーム作成
    X_train = train[use_cols].copy()
    X_test = test[use_cols].copy()
    
    # カテゴリカル変数のエンコーディング
    for col in cat_cols:
        if col in X_train.columns:
            le = LabelEncoder()
            combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
    
    # 欠損値処理
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
    """LightGBMモデルの学習（MAE使用 + seedアンサンブル）"""
    print("\nモデル学習中...")
    
    n_splits = 5
    seeds = [42, 123, 456]  # 複数seedでアンサンブル
    
    all_models = []
    
    for seed in seeds:
        print(f"\n=== Seed: {seed} ===")
        
        params = {
            'objective': 'regression',
            'metric': 'mae',  # MAEを使用（外れ値に強い）
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
            'seed': seed
        }
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
            print(f"Fold {fold + 1}/{n_splits}")
            
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_tr, y_val = y_train[train_idx], y_train[valid_idx]
            
            cat_indices = [X_train.columns.get_loc(col) for col in cat_cols if col in X_train.columns]
            
            train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_indices)
            valid_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_indices)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=3000,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=500)
                ]
            )
            
            all_models.append(model)
    
    print(f"\n学習完了: {len(all_models)}モデル")
    return all_models

def predict_and_submit(models, X_test, test_unit_area, test_ids):
    """予測と提出ファイル作成"""
    print("\n予測中...")
    
    # 全モデルの平均で平米単価を予測
    price_per_sqm_pred = np.zeros(len(X_test))
    for model in models:
        price_per_sqm_pred += model.predict(X_test)
    price_per_sqm_pred /= len(models)
    
    # 平米単価 × 専有面積 = 売買価格
    predictions = price_per_sqm_pred * test_unit_area
    
    # 負の値を補正
    predictions = np.maximum(predictions, 1)
    
    # 整数に丸める
    predictions = np.round(predictions).astype(int)
    
    # 提出ファイル作成
    submission = pd.DataFrame({
        'id': test_ids,
        'money_room': predictions
    })
    submission = submission.sort_values('id')
    
    output_path = f'{OUTPUT_DIR}/submission_v2.csv'
    submission.to_csv(output_path, index=False, header=False)
    print(f"\n提出ファイルを保存しました: {output_path}")
    print(f"予測価格統計:")
    print(f"  - 平均: {predictions.mean():,.0f}円")
    print(f"  - 中央値: {np.median(predictions):,.0f}円")
    
    return submission

def main():
    print("=" * 60)
    print("不動産価格予測モデル - 改善版")
    print("=" * 60)
    
    # データ読み込み
    train, test = load_data()
    
    # テストIDと専有面積を保存
    test_ids = test['id'].values
    
    # 平米単価ターゲット & 同一建物情報
    train, test = create_target_features(train, test)
    
    # 位置情報特徴量
    train, test = create_location_features(train, test)
    
    # 専有面積を保存（最後に掛け算する）
    test_unit_area = test['unit_area'].values
    
    # 特徴量前処理
    X_train, X_test, y_train, use_cols, cat_cols = preprocess_features(train, test)
    
    # モデル学習
    models = train_model(X_train, y_train, cat_cols)
    
    # 予測・提出
    submission = predict_and_submit(models, X_test, test_unit_area, test_ids)
    
    print("\n=== 処理完了 ===")
    return submission

if __name__ == '__main__':
    main()
```

---

**User**

記事の内容を元に改善点を適用して下さい．

---

**Cursor**

記事の内容を元に改善版コードを適用します。

```python
"""
不動産価格予測モデル - 改善版
参考: 第1回地理空間情報データチャレンジ 1位解法

改善点:
1. 平米単価を予測ターゲットにする（外れ値の影響を緩和）
2. MAE（L1）を使用（外れ値に強い）
3. 同一建物(building_id)の情報を活用
4. 郵便番号・市区町村ごとの相場情報を特徴量に追加
5. 複数seedでアンサンブル
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


def create_target_features(train, test):
    """
    平米単価をターゲットにする & 同一建物の情報を活用
    """
    print("ターゲット特徴量作成中...")
    
    # 専有面積の欠損を埋める（中央値で）
    unit_area_median = train['unit_area'].median()
    train['unit_area'] = train['unit_area'].fillna(unit_area_median)
    test['unit_area'] = test['unit_area'].fillna(unit_area_median)
    
    # 専有面積が0の場合も中央値で埋める
    train.loc[train['unit_area'] <= 0, 'unit_area'] = unit_area_median
    test.loc[test['unit_area'] <= 0, 'unit_area'] = unit_area_median
    
    # 平米単価を計算（ターゲット）
    train['price_per_sqm'] = train['money_room'] / train['unit_area']
    
    # 同一建物(building_id)の平米単価統計をtrainから計算
    building_stats = train.groupby('building_id').agg({
        'price_per_sqm': ['mean', 'std', 'min', 'max', 'count'],
        'money_room': ['mean', 'std', 'median']
    }).reset_index()
    building_stats.columns = ['building_id', 'bld_price_sqm_mean', 'bld_price_sqm_std', 
                               'bld_price_sqm_min', 'bld_price_sqm_max', 'bld_count',
                               'bld_price_mean', 'bld_price_std', 'bld_price_median']
    
    train = train.merge(building_stats, on='building_id', how='left')
    test = test.merge(building_stats, on='building_id', how='left')
    
    print(f"  - 同一建物情報: {len(building_stats)}件のbuilding_id")
    
    return train, test


def create_location_features(train, test):
    """
    郵便番号・市区町村ごとの相場情報を特徴量として追加
    """
    print("位置情報特徴量作成中...")
    
    # 郵便番号(post1)ごとの平米単価統計
    post_stats = train.groupby('post1').agg({
        'price_per_sqm': ['mean', 'std', 'median', 'count'],
        'money_room': ['mean', 'median']
    }).reset_index()
    post_stats.columns = ['post1', 'post_price_sqm_mean', 'post_price_sqm_std', 
                          'post_price_sqm_median', 'post_price_sqm_count',
                          'post_price_mean', 'post_price_median']
    
    train = train.merge(post_stats, on='post1', how='left')
    test = test.merge(post_stats, on='post1', how='left')
    
    print(f"  - 郵便番号別統計: {len(post_stats)}件")
    
    # 市区町村(addr1_1 + addr1_2)ごとの統計
    for df in [train, test]:
        df['city_code'] = df['addr1_1'].astype(str) + '_' + df['addr1_2'].astype(str)
    
    city_stats = train.groupby('city_code').agg({
        'price_per_sqm': ['mean', 'std', 'median'],
        'year_built': ['mean'],
        'unit_area': ['mean'],
        'money_room': ['mean', 'median']
    }).reset_index()
    city_stats.columns = ['city_code', 'city_price_sqm_mean', 'city_price_sqm_std',
                          'city_price_sqm_median', 'city_year_built_mean', 
                          'city_unit_area_mean', 'city_price_mean', 'city_price_median']
    
    train = train.merge(city_stats, on='city_code', how='left')
    test = test.merge(city_stats, on='city_code', how='left')
    
    print(f"  - 市区町村別統計: {len(city_stats)}件")
    
    # 築年数・面積の地域平均との比
    for df in [train, test]:
        # 築年数が地域平均の何倍か
        df['year_built_ratio'] = df['year_built'] / df['city_year_built_mean'].replace(0, np.nan)
        # 専有面積が地域平均の何倍か
        df['unit_area_ratio'] = df['unit_area'] / df['city_unit_area_mean'].replace(0, np.nan)
        # 平米単価が地域平均の何倍か（trainのみ意味がある）
        df['price_sqm_city_ratio'] = df.get('price_per_sqm', 0) / df['city_price_sqm_mean'].replace(0, np.nan)
    
    return train, test


def create_additional_features(train, test):
    """
    追加の特徴量エンジニアリング
    """
    print("追加特徴量作成中...")
    
    for df in [train, test]:
        # target_ymから年月を抽出
        df['year'] = df['target_ym'] // 100
        df['month'] = df['target_ym'] % 100
        
        # 築年数の計算
        if 'year_built' in df.columns:
            df['building_age'] = df['year'] - (df['year_built'] // 100)
            df['building_age'] = df['building_age'].clip(lower=0)  # 負の値を0に
        
        # 面積関連の特徴量
        if 'total_floor_area' in df.columns:
            df['unit_total_area_ratio'] = df['unit_area'] / (df['total_floor_area'] + 1)
        
        if 'building_land_area' in df.columns and 'land_area_all' in df.columns:
            df['land_coverage'] = df['building_land_area'] / (df['land_area_all'] + 1)
        
        if 'house_area' in df.columns and 'snapshot_land_area' in df.columns:
            df['house_land_ratio'] = df['house_area'] / (df['snapshot_land_area'] + 1)
        
        # 階数関連
        if 'room_floor' in df.columns and 'floor_count' in df.columns:
            df['floor_ratio'] = df['room_floor'] / (df['floor_count'] + 1)
        
        # 駅距離関連
        if 'walk_distance1' in df.columns:
            df['walk_distance1_log'] = np.log1p(df['walk_distance1'].fillna(0))
        if 'walk_distance2' in df.columns:
            df['walk_distance2_log'] = np.log1p(df['walk_distance2'].fillna(0))
        
        # 管理費・修繕積立金
        if 'money_kyoueki' in df.columns:
            df['kyoueki_per_sqm'] = df['money_kyoueki'] / (df['unit_area'] + 1)
        if 'money_shuuzen' in df.columns:
            df['shuuzen_per_sqm'] = df['money_shuuzen'] / (df['unit_area'] + 1)
    
    return train, test


def preprocess_features(train, test):
    """特徴量の前処理"""
    print("特徴量前処理中...")
    
    # ターゲット変数（平米単価）
    y_train = train['price_per_sqm'].values
    
    # 不要なカラムを削除
    drop_cols = ['money_room', 'price_per_sqm', 'id', 'price_sqm_city_ratio']
    
    # 日付関連のカラムを特定
    date_cols = [col for col in train.columns if 'date' in col.lower()]
    
    # テキスト系カラム
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
                 'building_tag_id', 'unit_tag_id', 'statuses', 'city_code']
    
    # 使用するカラム
    feature_cols = [col for col in train.columns 
                    if col not in drop_cols + date_cols + text_cols]
    
    # 日付からの特徴量抽出
    new_date_cols = []
    for df in [train, test]:
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if df[col].notna().any():
                    year_col = f'{col}_year'
                    month_col = f'{col}_month'
                    df[year_col] = df[col].dt.year
                    df[month_col] = df[col].dt.month
                    if year_col not in new_date_cols:
                        new_date_cols.extend([year_col, month_col])
    
    feature_cols.extend(new_date_cols)
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
            combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            label_encoders[col] = le
    
    # 欠損値処理
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
    """LightGBMモデルの学習（MAE使用 + seedアンサンブル）"""
    print("\nモデル学習中...")
    
    n_splits = 5
    seeds = [42, 123, 456]  # 複数seedでアンサンブル
    
    all_models = []
    oof_pred = np.zeros(len(X_train))
    
    for seed in seeds:
        print(f"\n{'='*40}")
        print(f"Seed: {seed}")
        print(f"{'='*40}")
        
        # MAEを使用（外れ値に強い）
        params = {
            'objective': 'regression',
            'metric': 'mae',
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
            'seed': seed
        }
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
            print(f"\nFold {fold + 1}/{n_splits}")
            
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_tr, y_val = y_train[train_idx], y_train[valid_idx]
            
            cat_indices = [X_train.columns.get_loc(col) for col in cat_cols if col in X_train.columns]
            
            train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_indices)
            valid_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_indices)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=3000,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=500)
                ]
            )
            
            all_models.append(model)
            
            # OOF予測
            if seed == seeds[0]:  # 最初のseedのみOOF計算
                oof_pred[valid_idx] = model.predict(X_val)
    
    # OOF MAE（平米単価）
    oof_mae = np.mean(np.abs(y_train - oof_pred))
    print(f"\n=== OOF MAE (平米単価): {oof_mae:,.2f}円/m² ===")
    
    print(f"\n学習完了: 合計 {len(all_models)} モデル")
    return all_models


def predict_and_submit(models, X_test, test_unit_area, test_ids):
    """予測と提出ファイル作成"""
    print("\n予測中...")
    
    # 全モデルの平均で平米単価を予測
    price_per_sqm_pred = np.zeros(len(X_test))
    for model in models:
        price_per_sqm_pred += model.predict(X_test)
    price_per_sqm_pred /= len(models)
    
    # 平米単価 × 専有面積 = 売買価格
    predictions = price_per_sqm_pred * test_unit_area
    
    # 負の値を補正
    predictions = np.maximum(predictions, 100000)  # 最低10万円
    
    # 整数に丸める
    predictions = np.round(predictions).astype(int)
    
    # 提出ファイル作成
    submission = pd.DataFrame({
        'id': test_ids,
        'money_room': predictions
    })
    submission = submission.sort_values('id')
    
    output_path = f'{OUTPUT_DIR}/submission_v2.csv'
    submission.to_csv(output_path, index=False, header=False)
    
    print(f"\n提出ファイルを保存しました: {output_path}")
    print(f"予測件数: {len(submission)}")
    print(f"\n予測価格統計:")
    print(f"  - 平均: {predictions.mean():,.0f}円")
    print(f"  - 中央値: {np.median(predictions):,.0f}円")
    print(f"  - 最小: {predictions.min():,.0f}円")
    print(f"  - 最大: {predictions.max():,.0f}円")
    print(f"\n平米単価統計:")
    print(f"  - 平均: {price_per_sqm_pred.mean():,.0f}円/m²")
    print(f"  - 中央値: {np.median(price_per_sqm_pred):,.0f}円/m²")
    
    return submission


def main():
    print("=" * 60)
    print("不動産価格予測モデル - 改善版 v2")
    print("参考: 第1回地理空間情報データチャレンジ 1位解法")
    print("=" * 60)
    
    # データ読み込み
    train, test = load_data()
    
    # テストIDを保存
    test_ids = test['id'].values
    
    # 平米単価ターゲット & 同一建物情報
    train, test = create_target_features(train, test)
    
    # 位置情報特徴量
    train, test = create_location_features(train, test)
    
    # 追加特徴量
    train, test = create_additional_features(train, test)
    
    # 専有面積を保存（最後に掛け算する）
    test_unit_area = test['unit_area'].values
    
    # 特徴量前処理
    X_train, X_test, y_train, use_cols, cat_cols = preprocess_features(train, test)
    
    # モデル学習
    models = train_model(X_train, y_train, cat_cols)
    
    # 予測・提出
    submission = predict_and_submit(models, X_test, test_unit_area, test_ids)
    
    print("\n" + "=" * 60)
    print("処理完了!")
    print("=" * 60)
    
    return submission


if __name__ == '__main__':
    main()

```

改善版コードを作成しました。実行します。

---

**User**

目的変数を元の家賃のスケールに戻している部分はどこですか？

---

**Cursor**

`predict_and_submit`関数の中で行っています。該当部分を確認しましょう：

該当部分は **352-353行目** です：

```352:353:/home/ubuntu/signate/geospatial_data_challenge_2nd/src/predict_v2.py
    # 平米単価 × 専有面積 = 売買価格
    predictions = price_per_sqm_pred * test_unit_area
```

## 処理の流れ

1. **学習時**: `price_per_sqm = money_room / unit_area`（平米単価をターゲットに）
2. **予測時**: モデルが平米単価（`price_per_sqm_pred`）を予測
3. **変換時**: `predictions = price_per_sqm_pred * test_unit_area`（専有面積を掛けて元の売買価格に戻す）

`test_unit_area`は`main()`関数内で事前に保存しています：

```python
# 専有面積を保存（最後に掛け算する）
test_unit_area = test['unit_area'].values
```

これにより、平米単価の予測値に各物件の専有面積を掛け算して、最終的な売買価格（円）を算出しています。

```python
    # 平米単価 × 専有面積 = 売買価格
    predictions = price_per_sqm_pred * test_unit_area
```

```python
# 専有面積を保存（最後に掛け算する）
test_unit_area = test['unit_area'].values
```

---

**User**

スコアが42に悪化しました．

---

**Cursor**

スコアが悪化してしまいましたね。原因を分析します。

## 問題点の分析

予測統計を比較すると：

| 指標 | v1（元のモデル） | v2（改善版） |
|------|-----------------|-------------|
| 平均 | 28,365,291円 | 17,544,438円 |
| 中央値 | 24,279,791円 | 9,373,938円 |
| 最大 | 170,559,245円 | **1,776,668,788円** |

**問題**: 
- 平米単価アプローチで極端な外れ値（17億円）が発生
- 全体的に予測値が低くなりすぎている

## 修正方針

平米単価アプローチは賃料予測では有効でしたが、売買価格では異なる特性があるようです。**元のアプローチ（money_roomを直接予測）に戻しつつ**、記事から有効な改善点だけを取り入れます：

1. 同一建物(building_id)の価格統計
2. 郵便番号・市区町村の相場情報
3. 複数seedでのアンサンブル

```python
"""
不動産価格予測モデル - 改善版 v3
元のmoney_room直接予測に戻しつつ、有効な特徴量エンジニアリングを適用

改善点:
1. 同一建物(building_id)の価格統計を特徴量に追加
2. 郵便番号・市区町村ごとの相場情報を特徴量に追加
3. 複数seedでアンサンブル
4. ターゲットはmoney_roomを直接予測（平米単価ではなく）
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


def create_building_features(train, test):
    """
    同一建物(building_id)の価格情報を特徴量として追加
    """
    print("建物情報特徴量作成中...")
    
    # 同一建物(building_id)の価格統計をtrainから計算
    building_stats = train.groupby('building_id').agg({
        'money_room': ['mean', 'std', 'min', 'max', 'median', 'count']
    }).reset_index()
    building_stats.columns = ['building_id', 'bld_price_mean', 'bld_price_std', 
                               'bld_price_min', 'bld_price_max', 'bld_price_median', 'bld_count']
    
    # stdがNaNの場合は0で埋める
    building_stats['bld_price_std'] = building_stats['bld_price_std'].fillna(0)
    
    train = train.merge(building_stats, on='building_id', how='left')
    test = test.merge(building_stats, on='building_id', how='left')
    
    print(f"  - 同一建物情報: {len(building_stats)}件のbuilding_id")
    
    # 専有面積の中央値を計算
    unit_area_median = train['unit_area'].median()
    
    # 平米単価も計算して特徴量に（ターゲットではなく特徴量として）
    for df in [train, test]:
        df['unit_area_filled'] = df['unit_area'].fillna(unit_area_median)
        df['unit_area_filled'] = df['unit_area_filled'].replace(0, unit_area_median)
    
    # 建物ごとの平米単価統計
    train['price_per_sqm_temp'] = train['money_room'] / train['unit_area_filled']
    building_sqm_stats = train.groupby('building_id').agg({
        'price_per_sqm_temp': ['mean', 'std', 'median']
    }).reset_index()
    building_sqm_stats.columns = ['building_id', 'bld_sqm_price_mean', 'bld_sqm_price_std', 'bld_sqm_price_median']
    building_sqm_stats['bld_sqm_price_std'] = building_sqm_stats['bld_sqm_price_std'].fillna(0)
    
    train = train.merge(building_sqm_stats, on='building_id', how='left')
    test = test.merge(building_sqm_stats, on='building_id', how='left')
    
    # 一時カラムを削除
    train = train.drop(columns=['price_per_sqm_temp'])
    
    return train, test


def create_location_features(train, test):
    """
    郵便番号・市区町村ごとの相場情報を特徴量として追加
    """
    print("位置情報特徴量作成中...")
    
    # 郵便番号(post1)ごとの価格統計
    post_stats = train.groupby('post1').agg({
        'money_room': ['mean', 'std', 'median', 'count']
    }).reset_index()
    post_stats.columns = ['post1', 'post_price_mean', 'post_price_std', 
                          'post_price_median', 'post_count']
    post_stats['post_price_std'] = post_stats['post_price_std'].fillna(0)
    
    train = train.merge(post_stats, on='post1', how='left')
    test = test.merge(post_stats, on='post1', how='left')
    
    print(f"  - 郵便番号別統計: {len(post_stats)}件")
    
    # 市区町村(addr1_1 + addr1_2)ごとの統計
    for df in [train, test]:
        df['city_code'] = df['addr1_1'].astype(str) + '_' + df['addr1_2'].astype(str)
    
    city_stats = train.groupby('city_code').agg({
        'money_room': ['mean', 'std', 'median'],
        'year_built': ['mean'],
        'unit_area': ['mean']
    }).reset_index()
    city_stats.columns = ['city_code', 'city_price_mean', 'city_price_std',
                          'city_price_median', 'city_year_built_mean', 
                          'city_unit_area_mean']
    city_stats['city_price_std'] = city_stats['city_price_std'].fillna(0)
    
    train = train.merge(city_stats, on='city_code', how='left')
    test = test.merge(city_stats, on='city_code', how='left')
    
    print(f"  - 市区町村別統計: {len(city_stats)}件")
    
    # 築年数・面積の地域平均との比
    for df in [train, test]:
        df['year_built_ratio'] = df['year_built'] / df['city_year_built_mean'].replace(0, np.nan)
        df['unit_area_ratio'] = df['unit_area'] / df['city_unit_area_mean'].replace(0, np.nan)
    
    return train, test


def create_additional_features(train, test):
    """
    追加の特徴量エンジニアリング
    """
    print("追加特徴量作成中...")
    
    for df in [train, test]:
        # target_ymから年月を抽出
        df['year'] = df['target_ym'] // 100
        df['month'] = df['target_ym'] % 100
        
        # 築年数の計算
        if 'year_built' in df.columns:
            df['building_age'] = df['year'] - (df['year_built'] // 100)
            df['building_age'] = df['building_age'].clip(lower=0)
        
        # 面積関連の特徴量
        if 'total_floor_area' in df.columns:
            df['unit_total_area_ratio'] = df['unit_area'] / (df['total_floor_area'] + 1)
        
        if 'building_land_area' in df.columns and 'land_area_all' in df.columns:
            df['land_coverage'] = df['building_land_area'] / (df['land_area_all'] + 1)
        
        if 'house_area' in df.columns and 'snapshot_land_area' in df.columns:
            df['house_land_ratio'] = df['house_area'] / (df['snapshot_land_area'] + 1)
        
        # 階数関連
        if 'room_floor' in df.columns and 'floor_count' in df.columns:
            df['floor_ratio'] = df['room_floor'] / (df['floor_count'] + 1)
        
        # 駅距離関連
        if 'walk_distance1' in df.columns:
            df['walk_distance1_log'] = np.log1p(df['walk_distance1'].fillna(0))
        if 'walk_distance2' in df.columns:
            df['walk_distance2_log'] = np.log1p(df['walk_distance2'].fillna(0))
        
        # 管理費・修繕積立金（面積あたり）
        if 'money_kyoueki' in df.columns and 'unit_area_filled' in df.columns:
            df['kyoueki_per_sqm'] = df['money_kyoueki'] / (df['unit_area_filled'] + 1)
        if 'money_shuuzen' in df.columns and 'unit_area_filled' in df.columns:
            df['shuuzen_per_sqm'] = df['money_shuuzen'] / (df['unit_area_filled'] + 1)
    
    return train, test


def preprocess_features(train, test):
    """特徴量の前処理"""
    print("特徴量前処理中...")
    
    # ターゲット変数（money_roomを直接予測）
    target = 'money_room'
    y_train = train[target].values
    
    # 不要なカラムを削除
    drop_cols = [target, 'id']
    
    # 日付関連のカラムを特定
    date_cols = [col for col in train.columns if 'date' in col.lower()]
    
    # テキスト系カラム
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
                 'building_tag_id', 'unit_tag_id', 'statuses', 'city_code']
    
    # 使用するカラム
    feature_cols = [col for col in train.columns 
                    if col not in drop_cols + date_cols + text_cols]
    
    # 日付からの特徴量抽出
    new_date_cols = []
    for df in [train, test]:
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if df[col].notna().any():
                    year_col = f'{col}_year'
                    month_col = f'{col}_month'
                    df[year_col] = df[col].dt.year
                    df[month_col] = df[col].dt.month
                    if year_col not in new_date_cols:
                        new_date_cols.extend([year_col, month_col])
    
    feature_cols.extend(new_date_cols)
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
    for col in cat_cols:
        if col in X_train.columns:
            le = LabelEncoder()
            combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
    
    # 欠損値処理
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
    """LightGBMモデルの学習（MAPE最適化 + seedアンサンブル）"""
    print("\nモデル学習中...")
    
    n_splits = 5
    seeds = [42, 123, 456]  # 複数seedでアンサンブル
    
    all_models = []
    oof_pred = np.zeros(len(X_train))
    
    for seed in seeds:
        print(f"\n{'='*40}")
        print(f"Seed: {seed}")
        print(f"{'='*40}")
        
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
            'seed': seed
        }
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
            print(f"\nFold {fold + 1}/{n_splits}")
            
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_tr, y_val = y_train[train_idx], y_train[valid_idx]
            
            cat_indices = [X_train.columns.get_loc(col) for col in cat_cols if col in X_train.columns]
            
            train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_indices)
            valid_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_indices)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=3000,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                feval=mape_lgb,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=500)
                ]
            )
            
            all_models.append(model)
            
            # OOF予測（最初のseedのみ）
            if seed == seeds[0]:
                oof_pred[valid_idx] = model.predict(X_val)
    
    # OOF MAPE
    oof_mape = mape(y_train, oof_pred)
    print(f"\n=== OOF MAPE: {oof_mape:.4f}% ===")
    
    print(f"\n学習完了: 合計 {len(all_models)} モデル")
    return all_models


def predict_and_submit(models, X_test, test_ids):
    """予測と提出ファイル作成"""
    print("\n予測中...")
    
    # 全モデルの平均で予測
    predictions = np.zeros(len(X_test))
    for model in models:
        predictions += model.predict(X_test)
    predictions /= len(models)
    
    # 負の値を補正
    predictions = np.maximum(predictions, 1)
    
    # 整数に丸める
    predictions = np.round(predictions).astype(int)
    
    # 提出ファイル作成
    submission = pd.DataFrame({
        'id': test_ids,
        'money_room': predictions
    })
    submission = submission.sort_values('id')
    
    output_path = f'{OUTPUT_DIR}/submission_v3.csv'
    submission.to_csv(output_path, index=False, header=False)
    
    print(f"\n提出ファイルを保存しました: {output_path}")
    print(f"予測件数: {len(submission)}")
    print(f"\n予測価格統計:")
    print(f"  - 平均: {predictions.mean():,.0f}円")
    print(f"  - 中央値: {np.median(predictions):,.0f}円")
    print(f"  - 最小: {predictions.min():,.0f}円")
    print(f"  - 最大: {predictions.max():,.0f}円")
    
    return submission


def main():
    print("=" * 60)
    print("不動産価格予測モデル - 改善版 v3")
    print("money_roomを直接予測 + 特徴量エンジニアリング強化")
    print("=" * 60)
    
    # データ読み込み
    train, test = load_data()
    
    # テストIDを保存
    test_ids = test['id'].values
    
    # 建物情報特徴量
    train, test = create_building_features(train, test)
    
    # 位置情報特徴量
    train, test = create_location_features(train, test)
    
    # 追加特徴量
    train, test = create_additional_features(train, test)
    
    # 特徴量前処理
    X_train, X_test, y_train, use_cols, cat_cols = preprocess_features(train, test)
    
    # モデル学習
    models = train_model(X_train, y_train, cat_cols)
    
    # 予測・提出
    submission = predict_and_submit(models, X_test, test_ids)
    
    print("\n" + "=" * 60)
    print("処理完了!")
    print("=" * 60)
    
    return submission


if __name__ == '__main__':
    main()

```

---

**User**

スコアが40でした．全然改善しませんね

---

**Cursor**

予測値の統計を比較すると明らかな問題があります：

| バージョン | 平均 | 中央値 | スコア |
|-----------|------|--------|--------|
| v1（元） | 28,365,291円 | 24,279,791円 | **17** |
| v3（現在） | 17,692,455円 | 9,448,123円 | **40** |

**問題**: 追加した特徴量が予測を低く歪めています。建物・郵便番号の統計がtestデータで欠損→-999で埋められ悪影響を与えている可能性が高いです。

元のv1に戻して、最小限の改善だけを試みましょう：

```python
"""
不動産価格予測モデル - v4
v1ベースで最小限の改善のみ適用
- 複数seedアンサンブル
- 外部統計特徴量は使用しない（リーク回避）
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
    """特徴量の前処理（v1ベース）"""
    print("特徴量前処理中...")
    
    # ターゲット変数
    target = 'money_room'
    y_train = train[target].values
    
    # 不要なカラムを削除
    drop_cols = [target, 'id']
    
    # 日付関連のカラムを特定
    date_cols = [col for col in train.columns if 'date' in col.lower()]
    
    # テキスト系カラム
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
            combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            label_encoders[col] = le
    
    # 欠損値処理
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
    """LightGBMモデルの学習（複数seedアンサンブル）"""
    print("\nモデル学習中...")
    
    n_splits = 5
    seeds = [42, 123, 456, 789, 2024]  # 5つのseedでアンサンブル
    
    all_models = []
    oof_pred_sum = np.zeros(len(X_train))
    oof_count = np.zeros(len(X_train))
    
    for seed in seeds:
        print(f"\n{'='*40}")
        print(f"Seed: {seed}")
        print(f"{'='*40}")
        
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
            'seed': seed
        }
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
            print(f"Fold {fold + 1}/{n_splits}", end=" ")
            
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_tr, y_val = y_train[train_idx], y_train[valid_idx]
            
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
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=0)  # ログ非表示
                ]
            )
            
            all_models.append(model)
            
            # OOF予測
            pred = model.predict(X_val)
            oof_pred_sum[valid_idx] += pred
            oof_count[valid_idx] += 1
            
            fold_mape = mape(y_val, pred)
            print(f"MAPE: {fold_mape:.4f}%")
    
    # OOF MAPE（平均）
    oof_pred = oof_pred_sum / oof_count
    oof_mape = mape(y_train, oof_pred)
    print(f"\n=== Overall OOF MAPE: {oof_mape:.4f}% ===")
    
    print(f"\n学習完了: 合計 {len(all_models)} モデル")
    return all_models


def predict_and_submit(models, X_test, test_ids):
    """予測と提出ファイル作成"""
    print("\n予測中...")
    
    # 全モデルの平均で予測
    predictions = np.zeros(len(X_test))
    for model in models:
        predictions += model.predict(X_test)
    predictions /= len(models)
    
    # 負の値を補正
    predictions = np.maximum(predictions, 1)
    
    # 整数に丸める
    predictions = np.round(predictions).astype(int)
    
    # 提出ファイル作成
    submission = pd.DataFrame({
        'id': test_ids,
        'money_room': predictions
    })
    submission = submission.sort_values('id')
    
    output_path = f'{OUTPUT_DIR}/submission_v4.csv'
    submission.to_csv(output_path, index=False, header=False)
    
    print(f"\n提出ファイルを保存しました: {output_path}")
    print(f"予測件数: {len(submission)}")
    print(f"\n予測価格統計:")
    print(f"  - 平均: {predictions.mean():,.0f}円")
    print(f"  - 中央値: {np.median(predictions):,.0f}円")
    print(f"  - 最小: {predictions.min():,.0f}円")
    print(f"  - 最大: {predictions.max():,.0f}円")
    
    return submission


def main():
    print("=" * 60)
    print("不動産価格予測モデル - v4")
    print("v1ベース + 複数seedアンサンブル")
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
    
    print("\n" + "=" * 60)
    print("処理完了!")
    print("=" * 60)
    
    return submission


if __name__ == '__main__':
    main()

```

---

**User**

17.4に上がりました

---

**Cursor**

v4でスコア17.4ですね。v1の17と比較すると少し悪化しています（MAPEは小さいほど良い）。アンサンブルが逆効果だったかもしれません。

次は**対数変換**を試しましょう。不動産価格は対数正規分布に近いことが多く、対数変換することで予測精度が改善されることがあります。

```python
"""
不動産価格予測モデル - v5
v1ベース + 対数変換
- ターゲット(money_room)を対数変換して学習
- 予測後にexp()で元に戻す
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


def preprocess_features(train, test):
    """特徴量の前処理（v1ベース）"""
    print("特徴量前処理中...")
    
    # ターゲット変数
    target = 'money_room'
    y_train = train[target].values
    
    # 不要なカラムを削除
    drop_cols = [target, 'id']
    
    # 日付関連のカラムを特定
    date_cols = [col for col in train.columns if 'date' in col.lower()]
    
    # テキスト系カラム
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
    
    # データフレーム作成
    X_train = train[use_cols].copy()
    X_test = test[use_cols].copy()
    
    # カテゴリカル変数のエンコーディング
    for col in cat_cols:
        if col in X_train.columns:
            le = LabelEncoder()
            combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
    
    # 欠損値処理
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
    """LightGBMモデルの学習（対数変換版）"""
    print("\nモデル学習中...")
    
    # 対数変換
    y_train_log = np.log1p(y_train)
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    models = []
    oof_pred_log = np.zeros(len(X_train))
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',  # 対数空間ではRMSEを使用
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
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train_log[train_idx], y_train_log[valid_idx]
        
        cat_indices = [X_train.columns.get_loc(col) for col in cat_cols if col in X_train.columns]
        
        train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_indices)
        valid_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_indices)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=True),
                lgb.log_evaluation(period=200)
            ]
        )
        
        models.append(model)
        oof_pred_log[valid_idx] = model.predict(X_val)
        
        # 元のスケールでMAPE計算
        y_val_orig = y_train[valid_idx]
        pred_orig = np.expm1(oof_pred_log[valid_idx])
        fold_mape = mape(y_val_orig, pred_orig)
        print(f"Fold {fold + 1} MAPE: {fold_mape:.4f}%")
    
    # OOF MAPE（元のスケール）
    oof_pred = np.expm1(oof_pred_log)
    oof_mape = mape(y_train, oof_pred)
    print(f"\n=== Overall OOF MAPE: {oof_mape:.4f}% ===")
    
    return models


def predict_and_submit(models, X_test, test_ids):
    """予測と提出ファイル作成"""
    print("\n予測中...")
    
    # 全モデルの平均で予測（対数空間）
    predictions_log = np.zeros(len(X_test))
    for model in models:
        predictions_log += model.predict(X_test)
    predictions_log /= len(models)
    
    # 元のスケールに戻す
    predictions = np.expm1(predictions_log)
    
    # 負の値を補正
    predictions = np.maximum(predictions, 1)
    
    # 整数に丸める
    predictions = np.round(predictions).astype(int)
    
    # 提出ファイル作成
    submission = pd.DataFrame({
        'id': test_ids,
        'money_room': predictions
    })
    submission = submission.sort_values('id')
    
    output_path = f'{OUTPUT_DIR}/submission_v5.csv'
    submission.to_csv(output_path, index=False, header=False)
    
    print(f"\n提出ファイルを保存しました: {output_path}")
    print(f"予測件数: {len(submission)}")
    print(f"\n予測価格統計:")
    print(f"  - 平均: {predictions.mean():,.0f}円")
    print(f"  - 中央値: {np.median(predictions):,.0f}円")
    print(f"  - 最小: {predictions.min():,.0f}円")
    print(f"  - 最大: {predictions.max():,.0f}円")
    
    return submission


def main():
    print("=" * 60)
    print("不動産価格予測モデル - v5")
    print("v1ベース + 対数変換")
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
    
    print("\n" + "=" * 60)
    print("処理完了!")
    print("=" * 60)
    
    return submission


if __name__ == '__main__':
    main()

```

素晴らしい結果です！

## 比較

| バージョン | OOF MAPE | 提出スコア |
|-----------|----------|----------|
| v1 | 15.76% | 17 |
| v5（対数変換） | **14.35%** | ??? |

OOF MAPEが **15.76% → 14.35%** に大幅改善しました！

提出ファイル: `/home/ubuntu/signate/geospatial_data_challenge_2nd/data/submit/submission_v5.csv`

予測統計も適切な範囲です：
- 平均: 27,526,140円
- 中央値: 23,516,101円

ぜひ提出してスコアを確認してみてください！

