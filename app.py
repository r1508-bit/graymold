import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import japanize_matplotlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import os
import re
from matplotlib.patches import Patch

_CSV_CACHE_MAX = 256
_csv_encoding_cache: Dict[Tuple[str, int, int], str] = {}
_tabular_columns_cache: Dict[Tuple[str, int, int], List[str]] = {}

# 灰色かび病リスクチェック関数
def check_gray_mold_risk(temp_humidity_data: List[Tuple[float, float]], timestamps) -> Tuple[str, int, str]:
    """
    最新の10日間(240時間)のデータからリスクレベルを算出する関数
    """
    # データをタイムスタンプでソート
    data_sorted = sorted(zip(temp_humidity_data, timestamps), key=lambda x: x[1])
    
    if not data_sorted:
        return "データなし", 0, "gray"
    
    # 最新の日時
    latest_timestamp = max(ts for _, ts in data_sorted)
    
    # 10日前の日時
    window_start = latest_timestamp - pd.Timedelta(hours=240)
    
    # 過去10日間のデータを抽出
    recent_window = [(temp, humidity) for (temp, humidity), ts in data_sorted 
                    if window_start <= ts <= latest_timestamp]
    
    # リスク時間を計算
    risk_hours = sum(
        1 for temp, humidity in recent_window
        if 15 <= float(temp) <= 25 and float(humidity) >= 94
    )
    
    # リスクレベルとカラーを決定
    if risk_hours == 0:
        return "極低", risk_hours, "blue"
    elif 0 < risk_hours <= 10:
        return "低", risk_hours, "blue"
    elif 10 < risk_hours <= 20:
        return "中", risk_hours, "green"
    elif 20 < risk_hours <= 39:
        return "高", risk_hours, "orange"
    else:
        return "極高", risk_hours, "red"

# CSVファイルから気温データと相対湿度データを読み込む関数
def read_temperature_and_humidity_data(file_obj, device_type=None, days_to_keep=30):
    """
    様々な形式のセンサーデータファイルから温度と湿度のデータを読み込む関数
    
    Parameters:
    ----------
    file_obj : UploadedFile または str
        Streamlitでアップロードされたファイルオブジェクトまたはファイルパス
    device_type : str, optional
        デバイスタイプ ('HZ', 'PF', 'PF2', 'SB', 'OT', 'HN', None)
    days_to_keep : int, optional
        保持する日数（最新のN日分）。デフォルトは30日。
    """
    
    # メイン処理開始
    temp_path = None
    is_uploaded_file = hasattr(file_obj, 'read')
    try:
        # アップロードされたファイルを一時ファイルに保存
        import tempfile
        if is_uploaded_file:  # UploadedFileオブジェクトの場合
            uploaded_name = getattr(file_obj, 'name', '')
            uploaded_suffix = os.path.splitext(uploaded_name)[1].lower() or '.csv'
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_suffix) as tmp_file:
                tmp_file.write(file_obj.getbuffer())
                temp_path = tmp_file.name
        else:  # 文字列（ファイルパス）の場合
            temp_path = file_obj
        
        # デバイス設定の定義（辞書を活用）
        # 各デバイスタイプに対応する列名とパラメータを集約
        device_configs = {
            'HZ': {
                'temp_cols': ['温度'],
                'humid_cols': ['湿度'],
                'timestamp_cols': ['日付'],
                'encoding': 'shift-jis'
            },
            'PF': {
                'temp_cols': ['気温'],
                'humid_cols': ['相対湿度'],
                'timestamp_cols': ['年月日'],
                'encoding': 'shift-jis'
            },
            'PF2': {
                'temp_cols': ['PF 測定 気温'],
                'humid_cols': ['湿度'],
                'timestamp_cols': ['datetime'],
                'encoding': 'shift-jis',
                'date_time_cols': {'date_col': '日付', 'time_col': '時刻'}
            },
            'SB': {
                # SwitchBot系のすべてのパターンを統合（正常、タイポ、文字化け含む）
                'temp_cols': [
                    'Temperature_Celsius(°C)', 'Temperature_Celsius(℃)', 'Temperature_Celsius(ﾂｰC)',
                    'Temperature_Celsius', 'Temperatre_Celsius(℁E', 'Temperatre_Celsius', 'Temperatre'
                ],
                'humid_cols': ['Relative_Humidity(%)', 'Relativ_Humidity(%)', 'Relativ_Humidity'],
                'timestamp_cols': ['Timestamp', 'Timamp', 'Date'],
                'encoding': 'utf-8'
            },
            'OT': {
                'temp_cols': ['室温'],
                'humid_cols': ['湿度'],
                'timestamp_cols': ['日付'],
                'encoding': 'shift-jis'
            },
            'HN': {
                'temp_cols': ['温度(℃)'],
                'humid_cols': ['相対湿度(％)'],
                'timestamp_cols': ['日時'],
                'encoding': 'utf-8-sig'
            },
            'KN': {
                # 換気ナビ：複数センサの平均値を使用
                'temp_cols': ['温度センサ１(℃)', '温度センサ２(℃)', '温度センサ３(℃)', '温度センサ４(℃)'],
                'humid_cols': ['湿度１(%)', '湿度２(%)'],
                'timestamp_cols': ['日時'],
                'encoding': 'utf-8-sig',
                'use_average': True  # 複数センサの平均値を使用
            }
        }
        
        # デバイスタイプの自動検出を試みる（デバイスタイプが未指定の場合）
        if device_type is None:
            # まずヘッダだけを読み込んで列構造を確認
            cols = read_tabular_columns_robust(temp_path)

            # 特殊パターンの検出（優先順位順）
            # PF2形式の検出（PF 測定 気温 と 日付+時刻が別カラム）
            if any('PF 測定' in col for col in cols) or ('日付' in cols and '時刻' in cols and '湿度' in cols):
                device_type = 'PF2'
            # KN形式の検出（温度センサ１等の全角数字を含む）
            elif any('温度センサ１' in col or '温度センサ２' in col for col in cols):
                device_type = 'KN'
            # SB形式の検出（SwitchBot系、タイポ含む）
            elif any('Timestamp' in col or 'Timamp' in col or 'Temperature' in col or 'Temperatre' in col for col in cols):
                device_type = 'SB'
            else:
                # 各デバイスタイプの特徴と照合
                for dev, config in device_configs.items():
                    # 温度列が存在するか確認
                    for col_name in config['temp_cols']:
                        if col_name in cols:
                            device_type = dev
                            break
                    if device_type:
                        break

            # ファジーマッチングでの検出（最終手段）
            if device_type is None:
                for dev, config in device_configs.items():
                    matched = find_column_fuzzy(cols, config['temp_cols'])
                    if matched:
                        device_type = dev
                        break

            # 自動検出できない場合はデフォルトとしてSBを使用
            if device_type is None:
                device_type = 'SB'
        
        # 設定を取得
        config = device_configs.get(device_type, device_configs['SB'])

        # ファイル読み込み
        df, _ = read_tabular_data_robust(temp_path)
        
        # タイムスタンプ列の特定と処理（デバイスタイプに応じた処理）
        timestamp_found = False
        
        # PF2形式の特別処理（日付と時刻が別々の列）
        if device_type == 'PF2' and '日付' in df.columns and '時刻' in df.columns:
     
            # 時刻列からアスタリスクなどを削除して結合
            date_col = config.get('date_time_cols', {}).get('date_col', '日付')
            time_col = config.get('date_time_cols', {}).get('time_col', '時刻')
            
            df['datetime'] = combine_date_time(df, date_col, time_col)
            
            # NaT値をチェック
            nat_count = df['datetime'].isna().sum()
            if nat_count > 0:
                df = df.dropna(subset=['datetime'])
            
            df = df.set_index('datetime')
            timestamp_found = True
        
        # 一般的なタイムスタンプ列の処理
        if not timestamp_found:
            for ts_col in config['timestamp_cols']:
                if ts_col in df.columns:
                    df['datetime'] = coerce_mixed_datetime_series(df[ts_col])

                    # NaT値をチェック
                    nat_count = df['datetime'].isna().sum()
                    if nat_count > 0:
                        df = df.dropna(subset=['datetime'])
                    df = df.set_index('datetime')
                    timestamp_found = True
                    break
        
        if not timestamp_found:
            # 日付と時刻が別々の列の場合（一般的なケース）
            if '日付' in df.columns and '時刻' in df.columns:
                df['datetime'] = combine_date_time(df, '日付', '時刻')
                
                # NaT値をチェック
                nat_count = df['datetime'].isna().sum()
                if nat_count > 0:
                    df = df.dropna(subset=['datetime'])
                
                df = df.set_index('datetime')
                timestamp_found = True
        
        if not timestamp_found:
            st.error("タイムスタンプ列が見つかりません")
            return None, None, None
        
        if timestamp_found:
            df = df.sort_index()
            if len(df) > 0:  # 空のDataFrameでないことを確認
                # 最新の日付を特定
                latest_date = df.index.max()
                # 期間を計算
                cutoff_date = latest_date - pd.Timedelta(days=days_to_keep)
                # 期間内のデータのみ保持
                df_filtered = df[df.index >= cutoff_date]

                # フィルタ後にデータが残っていることを確認
                if not df_filtered.empty:
                    # 元のデータフレームを更新
                    df = df_filtered
                else:
                    st.warning(f"最新の{days_to_keep}日間にデータがありません。すべてのデータを保持します。")
            
                
        # 温度と湿度の列を特定
        temp_col = None
        humid_col = None

        # KN形式の場合：複数センサの平均値を使用
        if config.get('use_average', False):
            # 温度センサ列を検索
            temp_sensor_cols = []
            for col in config['temp_cols']:
                if col in df.columns:
                    temp_sensor_cols.append(col)
            # 湿度センサ列を検索
            humid_sensor_cols = []
            for col in config['humid_cols']:
                if col in df.columns:
                    humid_sensor_cols.append(col)

            if temp_sensor_cols and humid_sensor_cols:
                # 数値型に変換
                df = convert_to_numeric(df, temp_sensor_cols + humid_sensor_cols)

                # 全てNaNのセンサのみを除外（0値は有効なデータとして扱う）
                valid_temp_cols = [col for col in temp_sensor_cols if df[col].notna().any()]
                valid_humid_cols = [col for col in humid_sensor_cols if df[col].notna().any()]

                # 除外されたセンサがある場合は警告を表示
                excluded_temp = [col for col in temp_sensor_cols if col not in valid_temp_cols]
                excluded_humid = [col for col in humid_sensor_cols if col not in valid_humid_cols]
                if excluded_temp or excluded_humid:
                    excluded_msgs = [f"  - {col}: 全てNaN" for col in excluded_temp + excluded_humid]
                    st.warning("以下のセンサはデータがないため除外されました:\n" + "\n".join(excluded_msgs))

                # 有効なセンサがあるか確認
                if not valid_temp_cols:
                    st.error("有効な温度センサがありません（全てのセンサがNaNです）")
                    return None, None, None
                if not valid_humid_cols:
                    st.error("有効な湿度センサがありません（全てのセンサがNaNです）")
                    return None, None, None

                # 平均値を計算（NaN値はスキップ、0値は有効データとして含む）
                df['_temp_avg'] = df[valid_temp_cols].mean(axis=1, skipna=True)
                df['_humid_avg'] = df[valid_humid_cols].mean(axis=1, skipna=True)
                temp_col = '_temp_avg'
                humid_col = '_humid_avg'
            else:
                st.error(f"センサ列が見つかりません。利用可能な列: {df.columns.tolist()}")
                return None, None, None
        else:
            # 通常のデバイス：完全一致を先に試す
            for col in config['temp_cols']:
                if col in df.columns:
                    temp_col = col
                    break

            for col in config['humid_cols']:
                if col in df.columns:
                    humid_col = col
                    break

            # 見つからない場合はファジーマッチング
            if temp_col is None:
                temp_col = find_column_fuzzy(df.columns.tolist(), config['temp_cols'])

            if humid_col is None:
                humid_col = find_column_fuzzy(df.columns.tolist(), config['humid_cols'])

            if temp_col is None or humid_col is None:
                st.error(f"必要な列が見つかりません。検索した列: {config['temp_cols']}, {config['humid_cols']}")
                st.error(f"利用可能な列: {df.columns.tolist()[:20]}...")  # 最初の20列のみ表示
                return None, None, None

            # 数値型に変換
            df = convert_to_numeric(df, [temp_col, humid_col])
        
        # NaN値を含む行を除外
        df = df.dropna(subset=[temp_col, humid_col])

        # 1時間間隔にリサンプリング
        df = resample_to_hourly(df)

        # データ確認
        if temp_col not in df.columns or humid_col not in df.columns:
            st.error("データ処理中にエラーが発生しました")
            return None, None, None

        result_temp = df[temp_col].tolist()
        result_humid = df[humid_col].tolist()

        # 結果を返す
        return result_temp, result_humid, df.index
    
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return None, None, None
    
    finally:
        # 一時ファイルを確実に削除
        if temp_path and is_uploaded_file:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                st.warning(f"一時ファイル削除エラー: {str(e)}")

def get_file_signature(file_path) -> Optional[Tuple[str, int, int]]:
    """キャッシュキー用のファイル署名を返す"""
    try:
        stat = os.stat(file_path)
        return (os.path.abspath(file_path), int(stat.st_mtime_ns), int(stat.st_size))
    except OSError:
        return None

def _set_cache_value(cache, key, value) -> None:
    """簡易キャッシュへ値を保存する"""
    cache[key] = value
    if len(cache) > _CSV_CACHE_MAX:
        oldest_key = next(iter(cache))
        cache.pop(oldest_key, None)

def _normalize_columns(columns) -> List[str]:
    """列名の前後空白や BOM を除去する"""
    return [str(col).replace('\ufeff', '').strip() for col in columns]

def _get_candidate_encodings(file_path) -> List[str]:
    """読み込み候補の文字コード一覧を返す"""
    default_encodings = ['utf-8-sig', 'utf-8', 'shift-jis', 'cp932']
    signature = get_file_signature(file_path)
    if signature is not None and signature in _csv_encoding_cache:
        cached_encoding = _csv_encoding_cache[signature]
        return [cached_encoding] + [enc for enc in default_encodings if enc != cached_encoding]
    return default_encodings

def _remember_encoding(file_path, encoding) -> None:
    """成功した文字コードをキャッシュへ保存する"""
    signature = get_file_signature(file_path)
    if signature is not None:
        _set_cache_value(_csv_encoding_cache, signature, encoding)

def _read_excel_robust(file_path, **kwargs) -> pd.DataFrame:
    """Excel ファイルを読み込む"""
    try:
        return pd.read_excel(file_path, **kwargs)
    except ImportError as exc:
        raise ValueError("Excel ファイルの読み込みには openpyxl または xlrd が必要です") from exc

def read_tabular_data_robust(file_path, **kwargs):
    """CSV / Excel を判別して堅牢に読み込む"""
    extension = os.path.splitext(str(file_path))[1].lower()
    if extension in ['.xlsx', '.xls']:
        df = _read_excel_robust(file_path, **kwargs)
        df.columns = _normalize_columns(df.columns)
        return df, 'excel'

    last_error = None
    csv_kwargs = dict(kwargs)
    csv_kwargs.setdefault('index_col', False)
    for encoding in _get_candidate_encodings(file_path):
        try:
            df = pd.read_csv(file_path, encoding=encoding, engine='python', **csv_kwargs)
            df.columns = _normalize_columns(df.columns)
            _remember_encoding(file_path, encoding)
            return df, encoding
        except (UnicodeDecodeError, UnicodeError, LookupError) as exc:
            last_error = exc
            continue
        except Exception as exc:
            last_error = exc
            continue

    raise ValueError("ファイルを読み込めませんでした: {} (最後のエラー: {})".format(file_path, last_error))

def read_tabular_columns_robust(file_path) -> List[str]:
    """CSV / Excel の列名だけを堅牢に読み込む"""
    signature = get_file_signature(file_path)
    if signature is not None and signature in _tabular_columns_cache:
        return list(_tabular_columns_cache[signature])

    columns = read_tabular_data_robust(file_path, nrows=0)[0].columns.tolist()
    current_signature = get_file_signature(file_path)
    if current_signature is not None:
        _set_cache_value(_tabular_columns_cache, current_signature, list(columns))
    return columns

def try_multiple_encodings(file_path):
    """互換用ラッパー。CSV / Excel を判別して読み込む"""
    return read_tabular_data_robust(file_path)

def find_column_fuzzy(columns, patterns, threshold=0.6):
    """
    ファジーマッチングで列名を検索する関数

    Parameters:
    ----------
    columns : list
        データフレームの列名リスト
    patterns : list
        検索する列名パターン（文字列または正規表現）
    threshold : float
        類似度の閾値（0-1）

    Returns:
    -------
    str or None
        マッチした列名、見つからない場合はNone
    """
    from difflib import SequenceMatcher

    # まず完全一致を試みる
    for pattern in patterns:
        if pattern in columns:
            return pattern

    # 部分一致を試みる（列名がパターンを含む）
    for pattern in patterns:
        for col in columns:
            if pattern.lower() in col.lower():
                return col

    # 正規表現マッチを試みる
    for pattern in patterns:
        for col in columns:
            try:
                if re.search(pattern, col, re.IGNORECASE):
                    return col
            except re.error:
                continue

    # ファジーマッチング（類似度ベース）
    for pattern in patterns:
        best_match = None
        best_ratio = 0
        for col in columns:
            # 列名を正規化して比較
            ratio = SequenceMatcher(None, pattern.lower(), col.lower()).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = col
        if best_match:
            return best_match

    return None

def combine_date_time(df, date_col='日付', time_col='時刻'):
    """日付列と時刻列を結合してdatetime型の列を作成する関数"""
    try:
        date_series = df[date_col].astype(str).str.strip()
        time_series = df[time_col].astype(str).str.replace('*', '', regex=False).str.strip()
        hour_only_mask = time_series.str.fullmatch(r'\d{1,2}')
        time_series.loc[hour_only_mask] = time_series.loc[hour_only_mask] + ':00'
        return coerce_mixed_datetime_series(date_series + ' ' + time_series)
        
    except Exception as e:
        st.error(f"日付と時刻の結合に失敗: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return pd.Series([pd.NaT] * len(df))

def _normalize_datetime_candidate_text(text: str) -> str:
    """日時候補文字列の軽微な表記ゆれを正規化する"""
    normalized = text.strip().strip('"').strip("'").replace('\u3000', ' ')
    normalized = normalized.replace('*', '')
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.replace('T', ' ')
    normalized = re.sub(r'^(\d{4})\.(\d{1,2})\.(\d{1,2})(.*)$', r'\1/\2/\3\4', normalized)
    normalized = re.sub(r'(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日', r'\1/\2/\3', normalized)
    normalized = re.sub(r'(\d{1,2})時\s*(\d{1,2})分\s*(\d{1,2})秒', r'\1:\2:\3', normalized)
    normalized = re.sub(r'(\d{1,2})時\s*(\d{1,2})分', r'\1:\2', normalized)
    normalized = re.sub(r'(\d{1,2})時', r'\1:00', normalized)
    return normalized.strip()

def _parse_excel_serial_datetime(value) -> pd.Timestamp:
    """Excel シリアル日時を Timestamp に変換する"""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return pd.NaT

    if not (20000 <= numeric_value <= 100000):
        return pd.NaT
    return pd.Timestamp('1899-12-30') + pd.to_timedelta(numeric_value, unit='D')

def _parse_datetime_with_day_rollover(text: str) -> pd.Timestamp:
    """24:00 表記を翌日 00:00 として扱う"""
    match = re.match(r'^(.*?)(?:\s+|T)24:(\d{2})(?::(\d{2}))?(.*)$', text)
    if not match:
        return pd.NaT

    date_part = match.group(1).strip()
    minute_part = match.group(2)
    second_part = match.group(3) or '00'
    suffix_part = match.group(4).strip()
    if minute_part != '00' or second_part != '00':
        return pd.NaT

    base_date = pd.to_datetime(date_part, errors='coerce')
    if pd.isna(base_date):
        return pd.NaT

    rolled = base_date + pd.Timedelta(days=1)
    rollover_text = '{} 00:00:00'.format(rolled.strftime('%Y-%m-%d'))
    if suffix_part:
        rollover_text = '{} {}'.format(rollover_text, suffix_part)
    return pd.to_datetime(rollover_text, errors='coerce')

def _parse_datetime_fallback_value(value) -> pd.Timestamp:
    """単一値の補助日時パーサー"""
    if pd.isna(value):
        return pd.NaT

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _parse_excel_serial_datetime(value)

    text = _normalize_datetime_candidate_text(str(value))
    if not text:
        return pd.NaT

    if re.fullmatch(r'\d+(?:\.\d+)?', text):
        excel_dt = _parse_excel_serial_datetime(text)
        if not pd.isna(excel_dt):
            return excel_dt

    rollover_dt = _parse_datetime_with_day_rollover(text)
    if not pd.isna(rollover_dt):
        return rollover_dt

    return pd.to_datetime(text, errors='coerce')

def coerce_mixed_datetime_series(values) -> pd.Series:
    """混在フォーマットを含む日時列を datetime に寄せて変換する"""
    series = pd.Series(values).copy()
    if series.empty:
        return pd.to_datetime(series, errors='coerce')

    if getattr(series, 'dtype', None) == object:
        series = series.map(
            lambda value: _normalize_datetime_candidate_text(value) if isinstance(value, str) else value
        )

    parsed = pd.to_datetime(series, errors='coerce')
    remaining_mask = parsed.isna() & series.notna()
    if not remaining_mask.any():
        return parsed

    remaining_values = series.loc[remaining_mask]
    try:
        reparsed = pd.to_datetime(remaining_values, errors='coerce', format='mixed')
    except TypeError:
        reparsed = remaining_values.map(lambda value: pd.to_datetime(value, errors='coerce'))

    reparsed_series = pd.Series(reparsed, index=remaining_values.index)
    unresolved_mask = reparsed_series.isna() & remaining_values.notna()
    if unresolved_mask.any():
        reparsed_series.loc[unresolved_mask] = remaining_values.loc[unresolved_mask].map(
            _parse_datetime_fallback_value
        )

    parsed.loc[remaining_mask] = reparsed_series
    return parsed

def convert_to_numeric(df, columns):
    """指定された列を数値型に変換する関数"""
    for col in columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            st.error(f"{col}の数値変換エラー: {str(e)}")
    
    return df

def resample_to_hourly(df):
    """データを1時間間隔にリサンプリングする関数"""
    try:
        # インデックスがdatetimeかチェック
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("日時データの変換に失敗しました")
        # 非数値列を除外
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            return df
        # 数値列のみを使用
        numeric_df = df[numeric_cols].copy()
        # 1時間間隔にリサンプリング
        numeric_df.index = numeric_df.index.floor('h')
        # グループ化とリサンプリング
        df_hourly = numeric_df.groupby(numeric_df.index).mean()
        # 欠損値の補間 (時系列データに適した method='time' を使用)
        df_resampled = df_hourly.resample('1h').interpolate(method='time')
        return df_resampled
    except Exception:
        return df

# 時系列リスク計算関数
def calculate_time_series_risk(temp_humidity_data, timestamps, days_to_show=30):
    """
    過去X日間の各日におけるリスクを計算する関数
    各日付から遡って10日間(240時間)のウィンドウでリスク判定
    """
    # データをタイムスタンプでソート
    data_sorted = sorted(zip(temp_humidity_data, timestamps), key=lambda x: x[1])
    
    # 各日のリスクを計算
    daily_risks = []
    
    if not data_sorted:
        return pd.DataFrame()  # データがない場合は空のDataFrameを返す
    
    # 最新と最古の日付
    latest_date = max(ts.date() for _, ts in data_sorted)
    earliest_date = min(ts.date() for _, ts in data_sorted)
    
    # 表示する日付範囲
    date_range = [latest_date - pd.Timedelta(days=i) for i in range(days_to_show)]
    date_range = [d for d in date_range if d >= earliest_date]
    
    for target_date in sorted(date_range):
        # その日の終わり（次の日の0時）
        end_datetime = datetime.combine(target_date + pd.Timedelta(days=1), datetime.min.time())
        
        # 過去10日間（240時間）のウィンドウ
        window_start = end_datetime - pd.Timedelta(hours=240)
        
        # このウィンドウ内のデータを抽出
        risk_window = [(temp, humidity) for (temp, humidity), ts in data_sorted 
                      if window_start <= ts < end_datetime]
        
        # リスク時間を計算
        risk_hours = sum(
            1 for temp, humidity in risk_window
            if 15 <= float(temp) <= 25 and float(humidity) >= 94
        )
        
        # リスクレベルとカラーを決定
        if risk_hours == 0:
            risk_level, color = "極低", "gray"
        elif 0 < risk_hours <= 10:
            risk_level, color = "低", "blue"
        elif 10 < risk_hours <= 20:
            risk_level, color = "中", "green"
        elif 20 < risk_hours <= 39:
            risk_level, color = "高", "orange"
        else:
            risk_level, color = "極高", "red"
        
        daily_risks.append({
            'date': target_date,
            'risk_hours': risk_hours,
            'risk_level': risk_level,
            'color': color,
            'window_data_count': len(risk_window)
        })
    
    return pd.DataFrame(daily_risks)

# スピードメーターのみを残して他の可視化関数を削除
def plot_speedometer(percentage, color, risk_level, risk_hours):
    """
    車のスピードメーターのような半円型のゲージでリスクを表示
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # 背景の半円弧（薄いグレー）
    theta = np.linspace(180, 0, 100) * np.pi / 180.0
    r = 0.8
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # 背景の弧（グレー）
    ax.plot(x, y, color='lightgray', linewidth=20, solid_capstyle='round')
    
    # リスクに応じた弧の描画
    end_angle = 180 - (percentage * 180)
    theta_risk = np.linspace(180, end_angle, 100) * np.pi / 180.0
    x_risk = r * np.cos(theta_risk)
    y_risk = r * np.sin(theta_risk)
    ax.plot(x_risk, y_risk, color=color, linewidth=20, solid_capstyle='round')
    
    # 針の描画
    needle_angle = (180 - (percentage * 180)) * np.pi / 180.0
    needle_length = 0.9
    ax.plot([0, needle_length * np.cos(needle_angle)], 
            [0, needle_length * np.sin(needle_angle)], 
            color='black', linewidth=2)
    
    # 針の中心点
    circle = plt.Circle((0, 0), 0.05, color='darkgray')
    ax.add_patch(circle)
    
    # リスクレベルを装飾ボックスで表示
    text_box_props = dict(
        boxstyle='round,pad=0.5',
        facecolor=color,
        alpha=0.8,
        edgecolor='none'
    )
    
    # リスクレベルテキスト - 白い文字で目立たせる
    ax.text(0, -0.4, f"リスクレベル: {risk_level}", ha='center', va='center', 
            fontsize=16, fontweight='bold', color='white',
            bbox=text_box_props)
    
    # リスク時間のテキスト
    ax.text(0, -0.6, f"{risk_hours}時間", ha='center', va='center', 
            fontsize=18, color='#303030')
    
    # 目盛り表示（リスクレベルラベル）
    # 各ラベルを対応するリスク範囲の中央に配置
    # 極低=0時間, 低=0-10の中間(5時間), 中=10-20の中間(15時間), 高=30時間, 極高=40+時間
    labels = ["極低", "低", "中", "高", "極高"]
    label_hours = [0, 5, 15, 30, 40]  # 各ラベルの時間位置
    label_angles = [(180 - (h / 40) * 180) * np.pi / 180.0 for h in label_hours]
    for label, angle in zip(labels, label_angles):
        x = 1.1 * np.cos(angle)
        y = 1.1 * np.sin(angle)
        ax.text(x, y, label, ha='center', va='center', fontsize=18)
    
    # スケール表示（時間数）- 0, 10, 20, 40+の位置に表示
    scales = ["0", "10", "20", "", "40+"]
    scale_angles = np.linspace(180, 0, 5) * np.pi / 180.0  # 均等配置
    for scale, angle in zip(scales, scale_angles):
        x = 0.81 * np.cos(angle)
        y = 0.81 * np.sin(angle)
        ax.text(x, y, scale, ha='center', va='center', fontsize=16, color='black')
    
    # グラフの設定
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.8, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig

# 簡略化されたメイン可視化関数（スピードメーターのみ）
def display_risk_visualization(risk_level, risk_hours):
    """
    リスク表示関数（スピードメーターのみ対応）
    """
    # リスクレベルに対応する色を更新
    colors = {
        "極低": "#0000cc",  # 青
        "低": "#0000cc",    # 青
        "中": "#00cc00",    # 緑
        "高": "#ff8000",    # オレンジ
        "極高": "#cc0000",  # 赤
        "データなし": "#808080"  # グレー
    }
    color = colors.get(risk_level, "#808080")
    
    # パーセンテージ計算（40時間を最大値として）
    percentage = min(risk_hours / 40, 1)
    
    # スピードメーターを表示
    return plot_speedometer(percentage, color, risk_level, risk_hours)

def detect_device_type(file_path):
    """
    ファイルのカラム名からデバイスタイプを推測する関数
    ファイル名には依存しない（カラム名のみで判定）
    """
    header = None

    # UTF-8-sig（BOM付き）でまず試す
    try:
        with open(file_path, 'rb') as f:
            raw = f.read(3)
        if raw == b'\xef\xbb\xbf':  # UTF-8 BOM
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                header = f.readline()
            # KN形式：温度センサ１等の全角数字を含む
            if '温度センサ１' in header or '温度センサ２' in header:
                return 'KN'
            # HN形式：温度(℃)と相対湿度(％)がある
            if '温度(℃)' in header and '相対湿度(％)' in header:
                return 'HN'
    except Exception:
        pass

    # UTF-8で試す（SwitchBotシリーズ）
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline()
        # SB形式：SwitchBot系（正常、タイポ含む）
        if any(pattern in header for pattern in ['Timestamp', 'Timamp', 'Temperature', 'Temperatre']):
            return 'SB'
    except UnicodeDecodeError:
        pass

    # Shift-JISで試す（日本語センサー）
    try:
        with open(file_path, 'r', encoding='shift-jis') as f:
            header = f.readline()

        # HZ形式：はかる蔵
        if '日付' in header and '温度' in header and '湿度' in header:
            return 'HZ'
        # PF形式：プロファインダー旧型
        if '年月日' in header and '気温' in header:
            return 'PF'
        # OT形式：おんどとり
        if '日付' in header and '室温' in header:
            return 'OT'
        # PF2形式：プロファインダー新型
        if 'PF 測定' in header or ('日付' in header and '時刻' in header and '湿度' in header):
            return 'PF2'

    except UnicodeDecodeError:
        pass

    # 推測できない場合はSBをデフォルトに
    return 'SB'

# 棒グラフ描画関数
def plot_risk_bar_chart(risk_df):
    """過去の日別リスクを棒グラフで表示する関数（スマホ対応版、日付を上部に配置）"""
    # スマホに適したサイズ比率
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [4, 1]})
    ax = axes[0]  # メインの棒グラフ用
    ax_legend = axes[1]  # 凡例用
    
    # 日付を古い順にソート
    risk_df = risk_df.sort_values('date')
    
    # 日付を文字列フォーマットに変換
    date_labels = [d.strftime('%m/%d') for d in risk_df['date']]
    
    # リスクレベルに対応する色を定義
    risk_colors = {
        "極低": "#0000cc",  # 青
        "低": "#0000cc",    # 青
        "中": "#00cc00",    # 緑
        "高": "#ff8000",    # オレンジ
        "極高": "#cc0000"   # 赤
    }
    
    # データに基づく色のリスト作成
    bar_colors = [risk_colors[level] for level in risk_df['risk_level']]
    
    # インデックスベースのX軸位置を作成
    x_positions = np.arange(len(risk_df))
    
    # 棒グラフをプロット
    bars = ax.bar(
        x_positions,
        risk_df['risk_hours'],
        color=bar_colors,
        width=0.7
    )
    
    # X軸のティックとラベルを非表示にする
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    # Y軸の設定
    ax.set_ylabel('条件を満たす時間数', fontsize=16)
    
    # Y軸の上限を固定して、日付表示用のスペースを確保
    # 基本的に40時間を上限とし、それより大きい値がある場合はそれに少し余裕を持たせる
    max_data = risk_df['risk_hours'].max()

    # 日付ラベルの位置を先に計算（max_data + 8）し、それより上にスペースを確保
    date_label_y = max(43, max_data + 8)
    y_limit = date_label_y + 12  # 日付ラベルより上に余裕を持たせる
    ax.set_ylim(0, y_limit)
    
    # 最初と最後の日付を取得して期間をタイトルに表示
    first_date = date_labels[0]
    last_date = date_labels[-1]
    ax.set_title(f'過去10日間の灰色かび病リスク推移({first_date}～{last_date})', fontsize=18)
    
    # 各棒の上に時間数を表示
    for i, (bar, hours) in enumerate(zip(bars, risk_df['risk_hours'])):
        height = bar.get_height()
        # 時間数を棒の上に表示
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{hours}時間',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # 日付を時間ラベルの上に表示（重なりを避ける）
    # date_label_y は上で計算済み
    for i, (x, date) in enumerate(zip(x_positions, date_labels)):
        # すべての日付を表示するか、偶数番目と最後の日付のみ表示するかを選択可能
        if i % 2 == 0 or i == len(date_labels) - 1:  # 偶数または最後のみ表示
            ax.text(x, date_label_y, date, 
                    ha='center', va='bottom', fontsize=14)
    
    # グリッド線を追加して読みやすくする
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # リスクレベルの凡例をスマホに最適化
    ax_legend.axis('off')  # 軸を非表示
    
    # リスクレベル区分を定義
    risk_levels = ["極低", "低", "中", "高", "極高"]
    risk_colors_list = [risk_colors["極低"], risk_colors["低"], risk_colors["中"], 
                         risk_colors["高"], risk_colors["極高"]]
    
    # 凡例を横に並べて表示（スマホに最適化）
    
    box_width = 0.15
    gap = 0.05
    total_width = (box_width + gap) * len(risk_levels) - gap
    start_x = (1 - total_width) / 2
    
    for i, (level, color) in enumerate(zip(risk_levels, risk_colors_list)):
        x = start_x + i * (box_width + gap)
        # 色付きのボックスを描画
        rect = patches.Rectangle((x, 0.4), box_width, 0.4, facecolor=color)
        ax_legend.add_patch(rect)
        # テキストラベルを追加
        ax_legend.text(x + box_width/2, 0.15, level, ha='center', va='center', fontsize=16)
    
    # 凡例のタイトル
    ax_legend.text(0.5, 1.0, "リスクレベル区分", ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)  # サブプロット間の隙間を調整
    return fig

def plot_risk_heatmap(risk_df):
    """リスクをカレンダー形式のヒートマップで表示する関数（スマホ対応版）"""
    # スマホ向けにサイズとレイアウトを調整
    fig, axes = plt.subplots(2, 1, figsize=(8, 1.8), gridspec_kw={'height_ratios': [1, 0.5]})
    ax = axes[0]  # メインのヒートマップ用
    ax_legend = axes[1]  # 凡例用
    
    # 日付を古い順にソートし、月-日形式に変換
    risk_df = risk_df.sort_values('date')
    date_labels = [d.strftime('%m/%d') for d in risk_df['date']]
    
    # カラーマップの設定（リスク区分に対応した離散的な色）
    from matplotlib.colors import ListedColormap, BoundaryNorm
    # リスクレベルに対応する色
    colors = [
        '#0000cc',  # 極低（0時間）: 青
        '#0000cc',  # 低（1-10時間）: 青
        '#00cc00',  # 中（11-20時間）: 緑
        '#ff8000',  # 高（21-39時間）: オレンジ
        '#cc0000',  # 極高（40+時間）: 赤
    ]
    risk_cmap = ListedColormap(colors)
    # リスク区分の境界値
    boundaries = [0, 1, 11, 21, 40, 100]
    norm = BoundaryNorm(boundaries, risk_cmap.N)
    
    # リスク時間数を2D配列に変換
    risk_matrix = risk_df['risk_hours'].values.reshape(1, -1)
    
    # ヒートマップの描画
    ax.pcolormesh(risk_matrix, cmap=risk_cmap, norm=norm, edgecolors='white', linewidth=1)
    
    # Y軸ラベルの設定（空にする）
    ax.set_yticks([])
    
    # X軸の設定
    total_days = len(date_labels)
    xtick_positions = np.arange(total_days) + 0.5
    
    # 最初と最後の日付のみラベル表示
    ax.set_xticks([xtick_positions[0], xtick_positions[-1]])
    ax.set_xticklabels([date_labels[0], date_labels[-1]], fontsize=12, fontweight='bold')
    
    # 日付の区切り線（薄く）
    for x in xtick_positions:
        ax.axvline(x, color='lightgray', linestyle='-', linewidth=0.3, alpha=0.2)
    
    
    # タイトル
    ax.set_title(f"過去30日間の灰色かび病リスク推移 ({date_labels[0]}～{date_labels[-1]})", fontsize=18)
    
    # 凡例を描画するサブプロットの設定
    ax_legend.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    return fig

def main():
    st.set_page_config(page_title="🍓 イチゴ灰色かび病リスク判断ツール", layout="wide")
    st.title("🍓 イチゴ灰色かび病リスク判断ツール")
    st.header("👇CSVファイルをアップロードしてください\n※アップロードされたファイルは終了時自動的に削除されます。")

    # ファイルアップローダー（センサータイプは自動検出のみ）
    uploaded_file = st.file_uploader("CSVファイルを選択してください。", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        # センサータイプは自動検出（Noneを渡す）
        selected_device = None
        
        # プログレスバーで処理状況を表示
        with st.spinner('ファイルを読み込んでいます...'):
            # 拡張版関数を使用してデータを読み込む（30日分のみ）
            temperature, relative_humidity, timestamps = read_temperature_and_humidity_data(
                uploaded_file, device_type=selected_device, days_to_keep=30
            )

        if temperature is not None and relative_humidity is not None:
            # データのプレビューを表示
            
            # 温度・湿度データをリスト化
            temp_humidity_data = list(zip(temperature, relative_humidity))

            # 現在のリスク計算（最新の10日間）
            current_risk_level, current_risk_hours, current_color = check_gray_mold_risk(temp_humidity_data, timestamps)
            
            # 時系列リスク計算
            time_series_risk_df = calculate_time_series_risk(temp_humidity_data, timestamps, days_to_show=28)

            # 1. メインの現在リスク表示（スピードメーター）
            # リスクレベル表示
            st.markdown(f"<h2 style='color: {current_color};'>灰色かび病の発病リスク: {current_risk_level}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:18px; font-weight:bold;'>過去10日間で条件を満たす時間数: {current_risk_hours}時間</p>", unsafe_allow_html=True)

            # 推奨対策表示
            recommendations = {
                "極低": "本病の発生リスクは極めて低いので、他作業（収穫等）に集中してください。",
                "低": "本病の発生リスクは低いですが、耕種的防除を実施するとなお良いでしょう。",
                "中": "耕種的防除及び薬剤防除（予防剤）の実施がお勧めです。",
                "高": "耕種的防除及び薬剤防除（治療剤）の実施が必要です。",
                "極高": "今すぐに耕種的防除及び薬剤防除（治療剤）の実施が必要です。",
                "データなし": "データが不足しています。CSVファイルを確認してください。"
            }
            st.markdown(f"<p style='font-size:16px;'><span style='font-weight:bold;'>推奨対策:</span> {recommendations[current_risk_level]}</p>", unsafe_allow_html=True)
            # スピードメーターを表示
            risk_viz_fig = display_risk_visualization(current_risk_level, current_risk_hours)
            st.pyplot(risk_viz_fig)
            
            # 区切り線を表示
            st.markdown("---")
            
            # 2. 時系列リスクの棒グラフ表示
            if not time_series_risk_df.empty:
                
                # 棒グラフ用に直近10日間のデータを抽出
                recent_10_days = time_series_risk_df.sort_values('date', ascending=False).head(10).sort_values('date')
                bar_fig = plot_risk_bar_chart(recent_10_days)
                st.pyplot(bar_fig)
                
                # 区切り線を表示
                st.markdown("---")
                
                # 3. 時系列リスクのヒートマップ表示（30日間）
                heat_fig = plot_risk_heatmap(time_series_risk_df)
                st.pyplot(heat_fig)
                
                # 詳細データテーブルを表示
                with st.expander("詳細データを表示"):
                    display_df = time_series_risk_df[['date', 'risk_hours', 'risk_level']].copy()
                    
                    # .dt アクセサーを使う前に型チェックを追加
                    if pd.api.types.is_datetime64_any_dtype(display_df['date']):
                        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                    else:
                        display_df['date'] = display_df['date'].astype(str)
                    
                    display_df.columns = ['日付', 'リスク時間数', 'リスクレベル']
                    st.dataframe(display_df)
            else:
                st.warning("表示可能な時系列データがありません。より長期間のデータを含むCSVファイルをアップロードしてください。")

    # 免責事項・コピーライト表示
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 11px;'>"
        "本ツールの判断結果は参考情報であり、実際の防除判断は現場の状況を確認の上、ご自身の責任で行ってください。"
        "</p>"
        "<p style='text-align: center; color: gray; font-size: 12px;'>"
        "&copy; 大分県農林水産研究指導センター農業研究部"
        "</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
