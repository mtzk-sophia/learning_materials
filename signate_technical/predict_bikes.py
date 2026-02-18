#!/usr/bin/env python3
"""
自転車利用可能台数予測モデル
SIGNATEテクニカルチャレンジ

時系列制約に対応：
- 各予測時点で、その時点より前のデータのみを使用
- 予測対象日ごとに、その日より前のデータから統計量を計算
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# データパス
DATA_DIR = '/home/ubuntu/signate/technical_challenge/data/raw'

def load_data():
    """データの読み込み"""
    status = pd.read_csv(f'{DATA_DIR}/status.csv')
    station = pd.read_csv(f'{DATA_DIR}/station.csv')
    weather = pd.read_csv(f'{DATA_DIR}/weather.csv')
    trip = pd.read_csv(f'{DATA_DIR}/trip.csv')
    
    return status, station, weather, trip

def preprocess_weather(weather):
    """気象データの前処理"""
    # 日付をパース
    weather['date'] = pd.to_datetime(weather['date'], format='%m/%d/%Y')
    weather['year'] = weather['date'].dt.year
    weather['month'] = weather['date'].dt.month
    weather['day'] = weather['date'].dt.day
    
    # precipitationを数値型に変換
    weather['precipitation'] = pd.to_numeric(weather['precipitation'], errors='coerce').fillna(0)
    
    # zip_codeごとの平均を取る（全エリア平均）
    weather_daily = weather.groupby(['year', 'month', 'day']).agg({
        'max_temperature': 'mean',
        'mean_temperature': 'mean',
        'min_temperature': 'mean',
        'max_humidity': 'mean',
        'mean_humidity': 'mean',
        'min_humidity': 'mean',
        'precipitation': 'mean',
        'cloud_cover': 'mean',
        'mean_wind_speed': 'mean',
    }).reset_index()
    
    return weather_daily

def preprocess_trip(trip):
    """トリップデータの前処理 - ステーションごとの統計"""
    trip['start_date'] = pd.to_datetime(trip['start_date'], format='%m/%d/%Y %H:%M')
    trip['end_date'] = pd.to_datetime(trip['end_date'], format='%m/%d/%Y %H:%M')
    trip['start_hour'] = trip['start_date'].dt.hour
    trip['end_hour'] = trip['end_date'].dt.hour
    trip['day_of_week'] = trip['start_date'].dt.dayofweek
    
    # ステーションごとの時間帯別出発・到着数
    trip_start_stats = trip.groupby(['start_station_id', 'start_hour']).size().reset_index(name='avg_departures')
    trip_start_stats.columns = ['station_id', 'hour', 'avg_departures']
    
    trip_end_stats = trip.groupby(['end_station_id', 'end_hour']).size().reset_index(name='avg_arrivals')
    trip_end_stats.columns = ['station_id', 'hour', 'avg_arrivals']
    
    # トリップ数を正規化（日数で割る）
    num_days = (trip['start_date'].max() - trip['start_date'].min()).days + 1
    trip_start_stats['avg_departures'] = trip_start_stats['avg_departures'] / num_days
    trip_end_stats['avg_arrivals'] = trip_end_stats['avg_arrivals'] / num_days
    
    trip_stats = pd.merge(trip_start_stats, trip_end_stats, on=['station_id', 'hour'], how='outer').fillna(0)
    trip_stats['net_flow'] = trip_stats['avg_arrivals'] - trip_stats['avg_departures']
    
    return trip_stats

def create_city_zip_mapping():
    """City to zip_code mapping (approximate based on coordinates)"""
    # サンフランシスコ周辺の都市とzip codeの対応
    # city2 (SF) -> 94107
    # city1 (San Jose) -> 95113
    # city3 (Redwood City) -> 94063
    # city4 (Mountain View) -> 94041
    # city5 (Palo Alto) -> 94301
    return {
        'city1': 95113,
        'city2': 94107,
        'city3': 94063,
        'city4': 94041,
        'city5': 94301
    }

def create_features(status, station, weather_daily, trip_stats):
    """特徴量の作成"""
    print("Creating features...")
    
    # ステーション情報をマージ
    df = status.merge(station[['station_id', 'dock_count', 'city', 'lat', 'long']], 
                     on='station_id', how='left')
    
    # 曜日を追加
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 気象情報をマージ
    df = df.merge(weather_daily, on=['year', 'month', 'day'], how='left')
    
    # トリップ統計をマージ
    df = df.merge(trip_stats, on=['station_id', 'hour'], how='left')
    df['avg_departures'] = df['avg_departures'].fillna(0)
    df['avg_arrivals'] = df['avg_arrivals'].fillna(0)
    df['net_flow'] = df['net_flow'].fillna(0)
    
    # 市のエンコーディング
    le = LabelEncoder()
    df['city_encoded'] = le.fit_transform(df['city'])
    
    # 時間の周期的特徴
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def add_lag_features_for_training(df):
    """訓練データ用のラグ特徴量の追加"""
    print("Adding lag features for training data...")
    
    # 訓練データのみを対象とする
    train_df = df[df['predict'] == 0].copy()
    
    # 各予測対象日の0時のデータを取得
    hour0_data = train_df[train_df['hour'] == 0][
        ['year', 'month', 'day', 'station_id', 'bikes_available']
    ].copy()
    hour0_data = hour0_data.rename(columns={'bikes_available': 'bikes_at_hour0'})
    train_df = train_df.merge(hour0_data, on=['year', 'month', 'day', 'station_id'], how='left')
    
    # 時系列順にソート
    train_df = train_df.sort_values(['station_id', 'year', 'month', 'day', 'hour']).reset_index(drop=True)
    
    # 訓練データ全体から統計量を計算（NaNでないデータのみ）
    valid_train = train_df[train_df['bikes_available'].notna()]
    
    station_hour_avg = valid_train.groupby(['station_id', 'hour'])['bikes_available'].mean().reset_index()
    station_hour_avg = station_hour_avg.rename(columns={'bikes_available': 'station_hour_avg'})
    train_df = train_df.merge(station_hour_avg, on=['station_id', 'hour'], how='left')
    
    station_dow_hour_avg = valid_train.groupby(['station_id', 'day_of_week', 'hour'])['bikes_available'].mean().reset_index()
    station_dow_hour_avg = station_dow_hour_avg.rename(columns={'bikes_available': 'station_dow_hour_avg'})
    train_df = train_df.merge(station_dow_hour_avg, on=['station_id', 'day_of_week', 'hour'], how='left')
    
    station_month_hour_avg = valid_train.groupby(['station_id', 'month', 'hour'])['bikes_available'].mean().reset_index()
    station_month_hour_avg = station_month_hour_avg.rename(columns={'bikes_available': 'station_month_hour_avg'})
    train_df = train_df.merge(station_month_hour_avg, on=['station_id', 'month', 'hour'], how='left')
    
    return train_df


def add_lag_features_for_test(test_df, full_df):
    """
    テストデータ用のラグ特徴量の追加
    各予測対象日に対して、その日より前のデータのみを使用して統計量を計算
    """
    print("Adding lag features for test data (time-series safe)...")
    
    # 予測対象日のリストを取得
    predict_dates = test_df[['year', 'month', 'day']].drop_duplicates().sort_values(['year', 'month', 'day'])
    
    # 結果を格納するリスト
    test_with_features_list = []
    
    # 各予測対象日に対して処理
    for idx, (pred_year, pred_month, pred_day) in predict_dates.iterrows():
        print(f"Processing {pred_year}/{pred_month}/{pred_day}...")
        
        # この日の予測対象データ
        day_test = test_df[
            (test_df['year'] == pred_year) & 
            (test_df['month'] == pred_month) & 
            (test_df['day'] == pred_day)
        ].copy()
        
        # この日の0時のデータを取得（予測対象日の0時は利用可能）
        hour0_data = full_df[
            (full_df['year'] == pred_year) & 
            (full_df['month'] == pred_month) & 
            (full_df['day'] == pred_day) &
            (full_df['hour'] == 0) &
            (full_df['predict'] == 0)
        ][['station_id', 'bikes_available']].copy()
        hour0_data = hour0_data.rename(columns={'bikes_available': 'bikes_at_hour0'})
        day_test = day_test.merge(hour0_data, on='station_id', how='left')
        
        # この日より前のデータのみを使用して統計量を計算
        # 予測対象日の日付をdatetimeに変換
        pred_date = pd.Timestamp(year=pred_year, month=pred_month, day=pred_day)
        
        # この日より前のデータを取得（NaNでないもの）
        historical_data = full_df[
            (full_df['date'] < pred_date) &
            (full_df['bikes_available'].notna())
        ].copy()
        
        # 統計量を計算
        if len(historical_data) > 0:
            # ステーション・時間ごとの平均
            station_hour_avg = historical_data.groupby(['station_id', 'hour'])['bikes_available'].mean().reset_index()
            station_hour_avg = station_hour_avg.rename(columns={'bikes_available': 'station_hour_avg'})
            day_test = day_test.merge(station_hour_avg, on=['station_id', 'hour'], how='left')
            
            # ステーション・曜日・時間ごとの平均
            station_dow_hour_avg = historical_data.groupby(['station_id', 'day_of_week', 'hour'])['bikes_available'].mean().reset_index()
            station_dow_hour_avg = station_dow_hour_avg.rename(columns={'bikes_available': 'station_dow_hour_avg'})
            day_test = day_test.merge(station_dow_hour_avg, on=['station_id', 'day_of_week', 'hour'], how='left')
            
            # ステーション・月・時間ごとの平均
            station_month_hour_avg = historical_data.groupby(['station_id', 'month', 'hour'])['bikes_available'].mean().reset_index()
            station_month_hour_avg = station_month_hour_avg.rename(columns={'bikes_available': 'station_month_hour_avg'})
            day_test = day_test.merge(station_month_hour_avg, on=['station_id', 'month', 'hour'], how='left')
        else:
            # 過去データがない場合は欠損値として扱う
            day_test['station_hour_avg'] = np.nan
            day_test['station_dow_hour_avg'] = np.nan
            day_test['station_month_hour_avg'] = np.nan
        
        test_with_features_list.append(day_test)
    
    # 全ての日のデータを結合
    test_with_features = pd.concat(test_with_features_list, ignore_index=True)
    
    return test_with_features

def prepare_model_data(train_df, test_df):
    """モデル用データの準備"""
    feature_cols = [
        'station_id', 'hour', 'day_of_week', 'month',
        'dock_count', 'city_encoded', 'lat', 'long',
        'is_weekend',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'max_temperature', 'mean_temperature', 'min_temperature',
        'max_humidity', 'mean_humidity', 'min_humidity',
        'precipitation', 'cloud_cover', 'mean_wind_speed',
        'avg_departures', 'avg_arrivals', 'net_flow',
        'bikes_at_hour0', 'station_hour_avg', 'station_dow_hour_avg', 'station_month_hour_avg'
    ]
    
    # 訓練データはbikes_availableがNaNでないもの（予測対象日翌日は除外）
    train_df = train_df[train_df['bikes_available'].notna()].copy()
    
    # 欠損値処理（訓練データとテストデータの両方をチェック）
    for col in feature_cols:
        if col in train_df.columns and col in test_df.columns:
            # 訓練データから中央値を計算
            median_val = train_df[col].median()
            # NaNの場合は0で埋める（統計量が計算できない場合）
            if pd.isna(median_val):
                median_val = 0
            
            # 訓練データとテストデータの両方の欠損値を埋める
            train_df[col] = train_df[col].fillna(median_val)
            test_df[col] = test_df[col].fillna(median_val)
    
    X_train = train_df[feature_cols]
    y_train = train_df['bikes_available']
    X_test = test_df[feature_cols]
    test_ids = test_df['id']
    
    return X_train, y_train, X_test, test_ids, feature_cols

def train_and_predict(X_train, y_train, X_test):
    """モデルの訓練と予測"""
    print("Training model...")
    
    # GradientBoostingRegressorを使用
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    print("Making predictions...")
    predictions = model.predict(X_test)
    
    # 予測値を0以上に制限（負の台数はありえない）
    predictions = np.maximum(predictions, 0)
    
    return predictions, model

def main():
    print("=" * 50)
    print("自転車利用可能台数予測")
    print("=" * 50)
    
    # データ読み込み
    print("\nLoading data...")
    status, station, weather, trip = load_data()
    
    # 前処理
    print("Preprocessing data...")
    weather_daily = preprocess_weather(weather)
    trip_stats = preprocess_trip(trip)
    
    # 特徴量作成
    df = create_features(status, station, weather_daily, trip_stats)
    
    # 訓練データとテストデータを分離
    train_df = df[df['predict'] == 0].copy()
    test_df = df[df['predict'] == 1].copy()
    
    # 訓練データに特徴量を追加
    train_df = add_lag_features_for_training(train_df)
    
    # テストデータに特徴量を追加（時系列制約を守る）
    test_df = add_lag_features_for_test(test_df, df)
    
    # モデル用データ準備
    X_train, y_train, X_test, test_ids, feature_cols = prepare_model_data(train_df, test_df)
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # 訓練と予測
    predictions, model = train_and_predict(X_train, y_train, X_test)
    
    # 特徴量重要度の表示
    print("\nTop 10 Feature Importances:")
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance_df.head(10))
    
    # 提出ファイル作成
    print("\nCreating submission file...")
    submission = pd.DataFrame({
        'id': test_ids.astype(int),
        'bikes_available': predictions.round().astype(int)
    })
    
    # idでソート
    submission = submission.sort_values('id').reset_index(drop=True)
    
    # 保存
    output_path = '/home/ubuntu/signate/technical_challenge/data/processed/submission.csv'
    submission.to_csv(output_path, index=False, header=False)
    
    print(f"\nSubmission file saved to: {output_path}")
    print(f"Total predictions: {len(submission)}")
    print(f"\nPrediction statistics:")
    print(submission['bikes_available'].describe())
    
    # サンプルと比較
    sample_submit = pd.read_csv(f'{DATA_DIR}/sample_submit.csv', header=None, names=['id', 'bikes_available'])
    print(f"\nExpected rows: {len(sample_submit)}")
    print(f"Actual rows: {len(submission)}")
    
    return submission

if __name__ == "__main__":
    main()

