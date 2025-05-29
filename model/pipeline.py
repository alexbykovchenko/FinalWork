import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import dill
from typing import Tuple, List

# Константы для лучшей читаемости
TARGET_ACTIONS = [
    'sub_car_claim_click', 'sub_car_claim_submit_click',
    'sub_open_dialog_click', 'sub_custom_question_submit_click',
    'sub_call_number_click', 'sub_callback_submit_click',
    'sub_submit_success', 'sub_car_request_submit_click'
]

TIME_OF_DAY_BINS = [
    (0, 6, 0),  # ночь
    (6, 12, 1),  # утро
    (12, 18, 2),  # день
    (18, 24, 3)  # вечер
]


def load_data(sessions_path: str, hits_path: str) -> pd.DataFrame:
    import pandas as pd
    """Загружает и объединяет данные сессий и хитов, создает целевую переменную."""
    dtype_sessions = {
        'session_id': 'str',
        'client_id': 'str',
        'device_brand': 'category',
        'device_os': 'category',
        'device_model': 'str'  # Оставляем как строку для последующей обработки
    }

    dtype_hits = {
        'session_id': 'str',
        'event_action': 'category'
    }

    df_sessions = pd.read_csv(sessions_path, low_memory=False, dtype=dtype_sessions)
    df_hits = pd.read_csv(hits_path, low_memory=False, dtype=dtype_hits)

    df_hits['event_target'] = df_hits['event_action'].isin(TARGET_ACTIONS).astype(np.int8)
    df_hits_agg = df_hits.groupby('session_id')['event_target'].max().astype(np.int8)

    merged = df_sessions.merge(df_hits_agg, on='session_id', how='left')
    merged['event_target'] = merged['event_target'].fillna(0).astype(np.int8)
    return merged


def clip_outliers(X: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    from typing import Tuple
    """Обрабатывает выбросы в числовых признаках с помощью метода IQR."""
    X = X.copy()

    def get_iqr_bounds(data: pd.Series) -> Tuple[float, float]:
        """Вложенная функция для расчета границ IQR."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if lower_bound < 1:
            lower_bound = 1

        return float(lower_bound), float(upper_bound)

    # Обрабатываем screen_pixels если он есть
    if 'screen_pixels' in X.columns:
        lower, upper = get_iqr_bounds(X['screen_pixels'])
        X['screen_pixels'] = X['screen_pixels'].clip(lower=lower, upper=upper)

    # Обрабатываем visit_number если он есть
    if 'visit_number' in X.columns:
        lower, upper = get_iqr_bounds(X['visit_number'])
        X['visit_number'] = X['visit_number'].clip(lower=lower, upper=upper)

    return X


def create_temporal_features(X: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    """Создает временные признаки из даты и времени визита."""
    X = X.copy()

    if not all(col in X.columns for col in ['visit_date', 'visit_time']):
        return X

    dt = pd.to_datetime(X['visit_date'])
    X['month'] = dt.dt.month.astype(np.int8)
    X['day_of_week'] = dt.dt.dayofweek.astype(np.int8)
    X['quarter'] = dt.dt.quarter.astype(np.int8)
    X['week_in_month'] = ((dt.dt.day - 1) // 7 + 1).astype(np.int8)

    hour = pd.to_datetime(X['visit_time'], format='%H:%M:%S').dt.hour.astype(np.int8)
    X['hour'] = hour
    time_of_day = np.zeros_like(hour, dtype=np.int8)
    for start, end, val in TIME_OF_DAY_BINS:
        mask = (hour >= start) & (hour < end)
        time_of_day[mask] = val
        X['time_of_day'] = time_of_day

    return X


def process_screen_resolution(X: pd.DataFrame) -> pd.DataFrame:
    """Обрабатывает разрешение экрана, преобразуя его в количество пикселей."""
    X = X.copy()
    if 'device_screen_resolution' not in X.columns:
        return X

    X['device_screen_resolution'] = X['device_screen_resolution'].replace('(not set)', np.nan)
    X['screen_pixels'] = X['device_screen_resolution'].apply(
        lambda x: int(x.split('x')[0]) * int(x.split('x')[1]) if isinstance(x, str) and 'x' in x else np.nan
    ).astype(np.float32)

    return X


def process_device_brand(X: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    """Обрабатывает device_brand, заменя (not set) на самый популярный бренд для данной категории."""
    X = X.copy()

    if not all(col in X.columns for col in ['device_brand', 'device_category']):
        return X

    # Заменяем (not set) на NA
    X['device_brand'] = X['device_brand'].replace('(not set)', pd.NA)

    # Находим самые популярные бренды для каждой категории
    filtered_df = X[['device_category', 'device_brand']].copy()
    popular_brands = filtered_df.groupby(['device_category', 'device_brand']).size().reset_index(name='count')
    popular_brands = popular_brands.loc[popular_brands.groupby('device_category')['count'].idxmax()]

    # Создаем словарь для замены
    brand_mapping = popular_brands.set_index('device_category')['device_brand'].to_dict()

    # Заполняем пропущенные значения
    X['device_brand'] = X.apply(
        lambda row: brand_mapping.get(row['device_category'], 'unknown')
        if pd.isna(row['device_brand'])
        else row['device_brand'],
        axis=1
    )

    return X


def process_device_os(X: pd.DataFrame) -> pd.DataFrame:
    """Заполняет пропуски в device_os на основе самого частого ОС для данного бренда."""
    X = X.copy()

    if not all(col in X.columns for col in ['device_brand', 'device_os']):
        return X

    # Создаем mapping между брендом и самой частой ОС
    filtered_df = X[['device_brand', 'device_os']].dropna(subset=['device_brand'])
    brand_os_mapping = filtered_df.groupby('device_brand')['device_os'].first().to_dict()

    # Заполняем пропущенные значения в device_os
    X['device_os'] = X.apply(
        lambda row: brand_os_mapping.get(row['device_brand'], row['device_os']),
        axis=1
    )

    return X


def remove_unused_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Удаляет ненужные колонки после извлечения из них полезной информации."""
    X = X.copy()
    cols_to_drop = []

    if all(col in X.columns for col in ['visit_date', 'visit_time']):
        cols_to_drop.extend(['visit_date', 'visit_time'])

    if 'device_screen_resolution' in X.columns:
        cols_to_drop.append('device_screen_resolution')

    if 'session_id' in X.columns:
        cols_to_drop.append('session_id')

    if 'client_id' in X.columns:
        cols_to_drop.append('client_id')

    if 'device_model' in X.columns:
        cols_to_drop.append('device_model')

    return X.drop(cols_to_drop, axis=1, errors='ignore')


def build_feature_pipeline() -> Pipeline:
    """Создает пайплайн для обработки признаков."""
    return Pipeline([
        ('temporal_features', FunctionTransformer(create_temporal_features)),
        ('screen_resolution', FunctionTransformer(process_screen_resolution)),
        ('process_device_brand', FunctionTransformer(process_device_brand)),
        ('process_device_os', FunctionTransformer(process_device_os)),
        ('clip_outliers', FunctionTransformer(clip_outliers)),
        ('remove_columns', FunctionTransformer(remove_unused_columns)),
    ])


def get_actual_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Определяет, какие признаки будут после преобразований."""
    feature_pipeline = build_feature_pipeline()
    sample_features = feature_pipeline.fit_transform(df.head())

    expected_numerical = ['month', 'day_of_week', 'hour', 'time_of_day',
                          'quarter', 'week_in_month', 'screen_pixels', 'visit_number']
    numerical = [col for col in expected_numerical if col in sample_features.columns]

    categorical = [col for col in sample_features.columns
                   if col not in numerical and col != 'event_target']

    return numerical, categorical


def build_model_pipeline(numerical_features: List[str],
                         categorical_features: List[str]) -> Pipeline:
    """Создает полный пайплайн модели."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(
            min_frequency=5,
            max_categories=7,
            handle_unknown='infrequent_if_exist',
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    return Pipeline(steps=[
        ('feature_engineering', build_feature_pipeline()),
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            class_weight='balanced',
            max_depth=10,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42
        ))
    ])


def train_and_save_model() -> Pipeline:
    """Обучает модель и сохраняет ее в файл."""
    df = load_data('../data/ga_sessions.csv', '../data/ga_hits.csv')

    X = df.drop('event_target', axis=1)
    y = df['event_target']

    numerical_features, categorical_features = get_actual_features(X)

    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)

    full_pipeline = build_model_pipeline(numerical_features, categorical_features)
    full_pipeline.fit(X, y)

    metadata = {
        "name": "Final work",
        "author": "Alexey Bykovchenko",
        "version": 1,
        "date": datetime.now().isoformat(),
        "type": "RF",
        "accuracy": 0.69,
        "features": {
            "numerical": numerical_features,
            "categorical": categorical_features
        }
    }

    with open('best_model_pipeline.pkl', 'wb') as f:
        dill.dump((full_pipeline, metadata), f)

    return full_pipeline


if __name__ == "__main__":
    pipeline = train_and_save_model()
    print("Модель успешно обучена и сохранена!")