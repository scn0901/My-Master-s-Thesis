# %% [markdown]
# # RQ3

# %% [markdown]
# ## import modules

# %%
import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    make_scorer
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
warnings.filterwarnings('ignore')

# %% [markdown]
# ## configuration

# %%
# --- Data ---
CSV_PATH_LIST = [
    '../../dataset/id_if_cib.csv',
    '../../dataset/id_create_time.csv',
    '../../dataset/id_basic_feature_engineering.csv',
    '../../dataset/id_time_feature_engineering.csv',
    '../../dataset/id_user_feature_engineering_cib.csv',
    '../../dataset/id_traces.csv',
    '../../dataset/id_virality_index_cib.csv'
]
INDEX_COL = 'id'
DV_COL = 'virality_index'                   # <-- change to your dependent variable column
IV_COLS = [
    # Basic features (15)
    'hashtag_num',
    'hashtag_rate_outside_top_500',
    'has_hashtag_dem',
    'has_hashtag_rep',
    'has_hashtag_traffic',
    'domain_num',
    'domain_rate_outside_top_100',
    'has_domain_platform',
    'has_domain_politics_civic',
    'has_domain_news_media',
    'has_domain_commerce',
    'has_domain_fundraising',
    'is_duet',
    'is_stitch',
    'is_reply',

    # Time features (3)
    'publish_hour_edt',
    'publish_weekday_edt',
    'days_since_start',

    # User features (11)
    'query_failed',
    'if_username_autogen',
    'is_verified',
    'if_has_bio',
    'log1p_video_count',
    'log1p_follower_count',
    'log1p_following_count',
    'log1p_follower_following_rate',
    'if_username_has_political_keyword',
    'if_display_name_has_political_keyword',
    'if_bio_has_political_keyword',

    # Trace features (5)
    'co_hashtagseq',
    'co_domain',
    'text_similarity',
    'video_similarity',
    'time_synchronization',
]                      # <-- list of feature columns, or None to use all columns except DV_COL and MODEL_EXCLUDE_COLS

# --- Time split settings ---
TIME_COL = 'create_time'            # Unix timestamp (seconds)
SPLIT_MODE = 'random'               # 'random' or 'time'
TEST_SIZE = 0.2                     # used only when SPLIT_MODE == 'random'
TIME_CUTOFF_TS = 1727740800         # used only when SPLIT_MODE == 'time' (example: 2024-10-01 00:00:00 UTC)

# --- Reproducibility ---
RANDOM_STATE = 42
N_JOBS = -1

# --- Preprocessing ---
USE_NUMERIC_SCALING = True          # numeric: median impute + optional scaling
DROP_ROWS_WITH_MISSING_DV = True

# --- Cross-validation / Grid search ---
CV_N_SPLITS = 5
GRID_REFIT_METRIC = 'rmse'          # one of: 'mae', 'rmse', 'r2'
GRID_VERBOSE = 1

# --- Models to run ---
# You can run one or both: ['random_forest', 'ridge']
MODELS_TO_RUN = ['random_forest', 'ridge']

# --- Hyperparameter grids (common / practical defaults) ---
RF_PARAM_GRID = {
    'model__n_estimators': [300],          # fixed
    'model__max_depth': [None, 12, 20],    # 3
    'model__min_samples_leaf': [1, 3],     # 2
    'model__max_features': ['sqrt', 0.5]   # 2
}

RIDGE_PARAM_GRID = {
    'model__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
}

# --- Reporting ---
TOP_N_FEATURES_TO_SHOW = None

# --- CIB filtering ---
CIB_COL = 'if_cib'
CIB_FILTER_MODE = 'cib_only'   # 'all' | 'cib_only' | 'noncib_only'

# --- Columns NOT used as model features ---
MODEL_EXCLUDE_COLS = [TIME_COL, CIB_COL]   # e.g., exclude raw timestamp from training features

# --- Feature type overrides ---
# Columns listed here will be forced into the corresponding type group (if they exist in X).
FORCE_NUMERIC_COLS = [
    # Basic features (4)
    'hashtag_num',
    'hashtag_rate_outside_top_500',
    'domain_num',
    'domain_rate_outside_top_100',

    # Time features (1)
    'days_since_start',

    # User features (4)
    'log1p_video_count',
    'log1p_follower_count',
    'log1p_following_count',
    'log1p_follower_following_rate',
]
FORCE_CATEGORICAL_COLS = [
    # Basic features (11)
    'has_hashtag_dem',
    'has_hashtag_rep',
    'has_hashtag_traffic',
    'has_domain_platform',
    'has_domain_politics_civic',
    'has_domain_news_media',
    'has_domain_commerce',
    'has_domain_fundraising',
    'is_duet',
    'is_stitch',
    'is_reply',

    # Time features (2)
    'publish_hour_edt',
    'publish_weekday_edt',

    # User features (7)
    'query_failed',
    'if_username_autogen',
    'is_verified',
    'if_has_bio',
    'if_username_has_political_keyword',
    'if_display_name_has_political_keyword',
    'if_bio_has_political_keyword',

    # Trace features (5)
    'co_hashtagseq',
    'co_domain',
    'text_similarity',
    'video_similarity',
    'time_synchronization',
]

# %% [markdown]
# ## helper dataclass

# %%
@dataclass
class TrainResult:
    model_name: str
    best_estimator: Pipeline
    best_params: dict
    best_cv_scores: dict
    train_cv_scores: dict
    test_scores: dict
    feature_effects: pd.DataFrame

# %% [markdown]
# ## utility functions

# %%
def load_data(csv_path_list, index_col: str) -> pd.DataFrame:
    '''
    Load one or multiple CSV files and merge them by index_col.
    Files are aligned by index (id) and concatenated column-wise.
    '''
    if isinstance(csv_path_list, str):
        csv_path_list = [csv_path_list]

    if not csv_path_list:
        raise ValueError('CSV_PATH_LIST is empty.')

    data_list = []
    for path in csv_path_list:
        df = pd.read_csv(path)
        if index_col not in df.columns:
            raise ValueError(f'Index column "{index_col}" not found in file: {path}')
        df = df.set_index(index_col)
        data_list.append(df)

    # Inner join on index to keep only ids shared by all files
    data = pd.concat(data_list, axis=1, join='inner')

    # Optional: check duplicated column names after merge
    duplicated_cols = data.columns[data.columns.duplicated()].tolist()
    if duplicated_cols:
        raise ValueError(
            f'Duplicate column names found after merging: {duplicated_cols[:10]}'
            + (' ...' if len(duplicated_cols) > 10 else '')
        )

    return data

# %%
def apply_cib_filter(data: pd.DataFrame, cib_col: str = 'if_cib', mode: str = 'all') -> pd.DataFrame:
    '''
    Filter rows by CIB label.
    mode:
    - 'all': keep all rows
    - 'cib_only': keep rows where if_cib == 1
    - 'noncib_only': keep rows where if_cib == 0
    '''
    if mode == 'all':
        return data.copy()

    if cib_col not in data.columns:
        raise ValueError(f'CIB column "{cib_col}" not found in data.')

    if mode == 'cib_only':
        return data.loc[data[cib_col] == 1].copy()

    if mode == 'noncib_only':
        return data.loc[data[cib_col] == 0].copy()

    raise ValueError('CIB_FILTER_MODE must be one of: \'all\', \'cib_only\', \'noncib_only\'')

# %%
def resolve_iv_cols_for_prepare(iv_cols, dv_col, split_mode, time_col):
    '''
    Build the feature column list used in prepare_xy().
    - Keep IV_COLS as true model features
    - If time-based split is used, temporarily append TIME_COL for splitting
    '''
    if iv_cols is None:
        # None means "use all columns except DV" in prepare_xy
        # In this case, no special handling is needed here.
        return None

    cols = list(iv_cols)

    if split_mode == 'time' and time_col not in cols and time_col != dv_col:
        cols.append(time_col)

    return cols

# %%
def prepare_xy(
    data: pd.DataFrame,
    dv_col: str,
    iv_cols=None,
    drop_rows_with_missing_dv: bool = True
):
    '''
    Split data into X and y. If iv_cols is None, use all columns except dv_col.
    '''
    if dv_col not in data.columns:
        raise ValueError(f'Dependent variable column "{dv_col}" not found in data.')

    working_data = data.copy()

    if drop_rows_with_missing_dv:
        working_data = working_data.dropna(subset=[dv_col])

    if iv_cols is None:
        iv_cols = [col for col in working_data.columns if col != dv_col]
    else:
        missing_cols = [col for col in iv_cols if col not in working_data.columns]
        if missing_cols:
            raise ValueError(f'Some IV columns are missing in data: {missing_cols}')

    X = working_data[iv_cols].copy()
    y = working_data[dv_col].copy()

    return X, y

# %%
def infer_feature_types(
    X: pd.DataFrame,
    force_numeric_cols=None,
    force_categorical_cols=None
):
    '''
    Infer numeric and categorical feature columns with manual overrides.

    Priority:
    1) FORCE_NUMERIC_COLS / FORCE_CATEGORICAL_COLS (manual override)
    2) dtype-based inference for remaining columns

    Notes:
    - Only columns present in X will be considered.
    - If a column is listed in both force lists, raise an error.
    '''
    force_numeric_cols = force_numeric_cols or []
    force_categorical_cols = force_categorical_cols or []

    # Keep only columns that actually exist in X
    force_numeric_cols = [c for c in force_numeric_cols if c in X.columns]
    force_categorical_cols = [c for c in force_categorical_cols if c in X.columns]

    # Check conflict
    overlap = sorted(set(force_numeric_cols) & set(force_categorical_cols))
    if overlap:
        raise ValueError(
            f'Columns cannot be in both FORCE_NUMERIC_COLS and FORCE_CATEGORICAL_COLS: {overlap}'
        )

    # 1) Start from dtype-based inference
    inferred_numeric_cols = X.select_dtypes(include=['number', 'bool']).columns.tolist()
    inferred_categorical_cols = [c for c in X.columns if c not in inferred_numeric_cols]

    # 2) Remove all forced cols from inferred groups
    forced_all = set(force_numeric_cols) | set(force_categorical_cols)
    inferred_numeric_cols = [c for c in inferred_numeric_cols if c not in forced_all]
    inferred_categorical_cols = [c for c in inferred_categorical_cols if c not in forced_all]

    # 3) Rebuild final lists with overrides taking priority
    numeric_cols = force_numeric_cols + inferred_numeric_cols
    categorical_cols = force_categorical_cols + inferred_categorical_cols

    # 4) Final sanity check (preserve original column order)
    numeric_set = set(numeric_cols)
    categorical_set = set(categorical_cols)

    numeric_cols = [c for c in X.columns if c in numeric_set]
    categorical_cols = [c for c in X.columns if c in categorical_set]

    if set(numeric_cols) & set(categorical_cols):
        raise RuntimeError('Internal error: numeric/categorical overlap detected.')

    if set(numeric_cols + categorical_cols) != set(X.columns):
        missing = sorted(set(X.columns) - set(numeric_cols) - set(categorical_cols))
        raise RuntimeError(f'Internal error: some columns were not assigned a type: {missing}')

    return numeric_cols, categorical_cols

# %%
def make_one_hot_encoder():
    '''
    Create OneHotEncoder with backward compatibility across sklearn versions.
    '''
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    except TypeError:
        # older sklearn
        return OneHotEncoder(handle_unknown='ignore', sparse=True)

# %%
def build_preprocessor(numeric_cols, categorical_cols, use_numeric_scaling=True):
    '''
    Build ColumnTransformer:
    - Numeric: median impute + optional scaling
    - Categorical: most_frequent impute + one-hot encoding
    '''
    numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
    if use_numeric_scaling:
        numeric_steps.append(('scaler', StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', make_one_hot_encoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ],
        remainder='drop'
    )

    return preprocessor

# %%
def get_model_and_param_grid(model_name: str, random_state: int = 42):
    '''
    Return a model instance and its grid-search parameter grid.
    '''
    if model_name == 'random_forest':
        model = RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1
        )
        param_grid = RF_PARAM_GRID

    elif model_name == 'ridge':
        model = Ridge(random_state=random_state)
        param_grid = RIDGE_PARAM_GRID

    else:
        raise ValueError(f'Unsupported model_name: {model_name}')

    return model, param_grid

# %%
def build_pipeline(preprocessor, model):
    '''
    Build a unified preprocessing + model pipeline.
    '''
    return Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', model)
    ])

# %%
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    split_mode: str = 'random',
    test_size: float = 0.2,
    random_state: int = 42,
    time_col: str = 'create_time',
    time_cutoff_ts: int = None
):
    '''
    Split data into train/test either randomly or by time cutoff.
    - Random split: train_test_split
    - Time split: rows with time < cutoff -> train, >= cutoff -> test
    '''
    if split_mode == 'random':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )
        split_info = {
            'split_mode': 'random',
            'test_size': test_size
        }
        return X_train, X_test, y_train, y_test, split_info

    if split_mode == 'time':
        if time_col not in X.columns:
            raise ValueError(f'Time column "{time_col}" not found in X for time-based split.')

        if time_cutoff_ts is None:
            raise ValueError('TIME_CUTOFF_TS must be provided for time-based split.')

        # Sort by time for readability / reproducibility in inspection
        order = np.argsort(X[time_col].values)
        X_sorted = X.iloc[order].copy()
        y_sorted = y.iloc[order].copy()

        train_mask = X_sorted[time_col] < time_cutoff_ts
        test_mask = X_sorted[time_col] >= time_cutoff_ts

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            raise ValueError(
                'Time split produced an empty train or test set. '
                'Please adjust TIME_CUTOFF_TS.'
            )

        X_train = X_sorted.loc[train_mask].copy()
        X_test = X_sorted.loc[test_mask].copy()
        y_train = y_sorted.loc[train_mask].copy()
        y_test = y_sorted.loc[test_mask].copy()

        split_info = {
            'split_mode': 'time',
            'time_col': time_col,
            'time_cutoff_ts': int(time_cutoff_ts),
            'train_time_min': int(X_train[time_col].min()),
            'train_time_max': int(X_train[time_col].max()),
            'test_time_min': int(X_test[time_col].min()),
            'test_time_max': int(X_test[time_col].max())
        }
        return X_train, X_test, y_train, y_test, split_info

    raise ValueError("split_mode must be either 'random' or 'time'")

# %%
def drop_model_exclude_cols(X_train: pd.DataFrame, X_test: pd.DataFrame, exclude_cols):
    '''
    Drop columns from both train/test feature sets before model training.
    These columns can still be used earlier for splitting or bookkeeping.
    '''
    exclude_cols = [col for col in exclude_cols if col in X_train.columns or col in X_test.columns]

    X_train_out = X_train.drop(columns=[c for c in exclude_cols if c in X_train.columns]).copy()
    X_test_out = X_test.drop(columns=[c for c in exclude_cols if c in X_test.columns]).copy()

    return X_train_out, X_test_out

# %%
def evaluate_regression(y_true, y_pred):
    '''
    Return regression metrics as a dict:
    - MAE
    - RMSE
    - R^2
    '''
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

# %%
def make_scoring_dict():
    '''
    Create scoring dict for cross-validation and grid search.
    Notes:
    - MAE / RMSE are set as "negative" in sklearn scoring convention for optimization.
    - R2 is positive (higher is better).
    '''
    rmse_scorer = make_scorer(
        root_mean_squared_error,
        greater_is_better=False
    )

    scoring = {
        'mae': 'neg_mean_absolute_error',
        'rmse': rmse_scorer,
        'r2': 'r2'
    }
    return scoring

# %%
def summarize_grid_best_scores(grid_search: GridSearchCV):
    '''
    Extract the best CV scores from GridSearchCV and convert signs for readability.
    '''
    idx = grid_search.best_index_
    cv_results = grid_search.cv_results_

    best_scores = {
        'cv_mae_mean': -cv_results['mean_test_mae'][idx],
        'cv_mae_std': cv_results['std_test_mae'][idx],
        'cv_rmse_mean': -cv_results['mean_test_rmse'][idx],
        'cv_rmse_std': cv_results['std_test_rmse'][idx],
        'cv_r2_mean': cv_results['mean_test_r2'][idx],
        'cv_r2_std': cv_results['std_test_r2'][idx]
    }
    return best_scores

# %%
def cross_validate_on_train(best_pipeline, X_train, y_train, cv, scoring, n_jobs=-1):
    '''
    Run an additional cross-validation on the training set using the best estimator.
    This is optional but useful for reporting.
    '''
    cv_res = cross_validate(
        estimator=best_pipeline,
        X=X_train,
        y=y_train,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=False
    )

    summary = {
        'cv_mae_mean': -np.mean(cv_res['test_mae']),
        'cv_mae_std': np.std(cv_res['test_mae']),
        'cv_rmse_mean': -np.mean(cv_res['test_rmse']),
        'cv_rmse_std': np.std(cv_res['test_rmse']),
        'cv_r2_mean': np.mean(cv_res['test_r2']),
        'cv_r2_std': np.std(cv_res['test_r2'])
    }
    return summary

# %%
def get_feature_names_from_pipeline(fitted_pipeline: Pipeline):
    '''
    Get transformed feature names from the fitted ColumnTransformer.
    '''
    preprocessor = fitted_pipeline.named_steps['preprocess']
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        # Fallback if sklearn version / custom transformers cause issues
        feature_names = np.array([f'feature_{i}' for i in range(
            fitted_pipeline.named_steps['preprocess'].transform(
                pd.DataFrame(columns=[])
            ).shape[1]
        )])
    return feature_names

# %%
def extract_feature_effects(fitted_pipeline: Pipeline, top_n: int = 20):
    '''
    Extract feature importances or coefficients (if supported by the model).
    Returns a DataFrame sorted by absolute effect size.
    '''
    model = fitted_pipeline.named_steps['model']
    preprocessor = fitted_pipeline.named_steps['preprocess']

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        # If feature names cannot be extracted, return empty DataFrame
        return pd.DataFrame(columns=['feature', 'value', 'abs_value', 'type'])

    if hasattr(model, 'feature_importances_'):
        values = model.feature_importances_
        effect_type = 'feature_importance'
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        if np.ndim(coef) > 1:
            # Multi-output regression (not typical in this template)
            coef = np.ravel(coef)
        values = coef
        effect_type = 'coefficient'
    else:
        return pd.DataFrame(columns=['feature', 'value', 'abs_value', 'type'])

    effect_df = pd.DataFrame({
        'feature': feature_names,
        'value': values
    })
    effect_df['abs_value'] = effect_df['value'].abs()
    effect_df['type'] = effect_type
    if top_n:
        effect_df = effect_df.sort_values('abs_value', ascending=False).head(top_n).reset_index(drop=True)
    else:
        effect_df = effect_df.sort_values('abs_value', ascending=False).reset_index(drop=True)

    return effect_df

# %%
def fit_with_grid_search(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols,
    categorical_cols,
    use_numeric_scaling: bool = True,
    random_state: int = 42,
    cv_n_splits: int = 5,
    grid_refit_metric: str = 'rmse',
    n_jobs: int = -1,
    verbose: int = 1
) -> TrainResult:
    '''
    Train one model with preprocessing + GridSearchCV (CV only on training set).
    '''
    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        use_numeric_scaling=use_numeric_scaling
    )

    model, param_grid = get_model_and_param_grid(model_name, random_state=random_state)
    pipeline = build_pipeline(preprocessor, model)

    cv = KFold(n_splits=cv_n_splits, shuffle=True, random_state=random_state)
    scoring = make_scoring_dict()

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit=grid_refit_metric,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=False
    )

    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_scores = summarize_grid_best_scores(grid_search)

    # Optional extra CV report on the best model (still train-set only)
    train_cv_scores = cross_validate_on_train(
        best_pipeline=best_pipeline,
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs
    )

    feature_effects = extract_feature_effects(best_pipeline, top_n=TOP_N_FEATURES_TO_SHOW)

    # Placeholder test scores; filled later
    result = TrainResult(
        model_name=model_name,
        best_estimator=best_pipeline,
        best_params=best_params,
        best_cv_scores=best_cv_scores,
        train_cv_scores=train_cv_scores,
        test_scores={},
        feature_effects=feature_effects
    )
    return result

# %%
def pretty_print_scores(title: str, scores: dict):
    '''
    Print metrics in a compact readable format.
    '''
    print(f'\n{title}')
    for k, v in scores.items():
        if isinstance(v, (int, float, np.floating)):
            print(f'  {k}: {v:.6f}')
        else:
            print(f'  {k}: {v}')

# %%
def run_experiment():
    '''
    Main workflow:
    1) Load data
    2) Prepare X/y (TIME_COL can remain in X if needed for time split)
    3) Split train/test (random split or time-based split)
    4) Remove columns not used as model features (e.g., raw timestamp TIME_COL)
    5) Infer numeric/categorical feature types from training features
    6) Train models with preprocessing + CV + grid search (training set only)
    7) Evaluate the selected model on the test set
    8) Extract feature importance / coefficients (if supported)
    '''
    # 1) Load
    data = load_data(CSV_PATH_LIST, INDEX_COL)
    print(f'Data shape (before CIB filter): {data.shape}')

    # 1.5) Filter by CIB mode
    data = apply_cib_filter(data, cib_col=CIB_COL, mode=CIB_FILTER_MODE)
    print(f'Data shape (after CIB filter: {CIB_FILTER_MODE}): {data.shape}')

    # 2) Prepare X / y
    iv_cols_for_prepare = resolve_iv_cols_for_prepare(
        iv_cols=IV_COLS,
        dv_col=DV_COL,
        split_mode=SPLIT_MODE,
        time_col=TIME_COL
    )

    X, y = prepare_xy(
        data=data,
        dv_col=DV_COL,
        iv_cols=iv_cols_for_prepare,
        drop_rows_with_missing_dv=DROP_ROWS_WITH_MISSING_DV
    )
    print(f'\nX shape (note: may include TIME_COL): {X.shape}')
    print(f'y shape: {y.shape}')

    # 3) Split data (TIME_COL may still be present here for time-based split)
    X_train, X_test, y_train, y_test, split_info = split_data(
        X=X,
        y=y,
        split_mode=SPLIT_MODE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        time_col=TIME_COL,
        time_cutoff_ts=TIME_CUTOFF_TS
    )

    print('\nSplit info:')
    for k, v in split_info.items():
        print(f'  {k}: {v}')
    print(f'  train size: {len(X_train)}')
    print(f'  test size: {len(X_test)}')

    # 4) Drop columns that should NOT be used as model features
    X_train, X_test = drop_model_exclude_cols(
        X_train=X_train,
        X_test=X_test,
        exclude_cols=MODEL_EXCLUDE_COLS
    )

    print(f'\nX_train shape after excluding non-feature cols: {X_train.shape}')
    print(f'X_test shape after excluding non-feature cols: {X_test.shape}')

    # 5) Infer feature types on the actual model inputs
    numeric_cols, categorical_cols = infer_feature_types(
        X_train,
        force_numeric_cols=FORCE_NUMERIC_COLS,
        force_categorical_cols=FORCE_CATEGORICAL_COLS
    )
    print(f'\nNumeric features (used by model): {numeric_cols}')
    print(f'Categorical features (used by model): {categorical_cols}')

    # 6) Train + 7) Evaluate + 8) Show feature effects
    all_results = []
    summary_rows = []

    for model_name in MODELS_TO_RUN:
        print('\n' + '=' * 80)
        print(f'Training model: {model_name}')
        print('=' * 80)

        result = fit_with_grid_search(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            use_numeric_scaling=USE_NUMERIC_SCALING,
            random_state=RANDOM_STATE,
            cv_n_splits=CV_N_SPLITS,
            grid_refit_metric=GRID_REFIT_METRIC,
            n_jobs=N_JOBS,
            verbose=GRID_VERBOSE
        )

        # Evaluate on test set
        y_pred_test = result.best_estimator.predict(X_test)
        test_scores = evaluate_regression(y_test, y_pred_test)
        result.test_scores = test_scores

        # Print results
        print('\nBest params:')
        for k, v in result.best_params.items():
            print(f'  {k}: {v}')

        pretty_print_scores('Best CV scores from grid search (training set only):', result.best_cv_scores)
        pretty_print_scores('Additional CV scores on best estimator (training set only):', result.train_cv_scores)
        pretty_print_scores('Test set scores:', result.test_scores)

        # Feature effects
        if not result.feature_effects.empty:
            if TOP_N_FEATURES_TO_SHOW:
                print(f'\nTop {TOP_N_FEATURES_TO_SHOW} feature effects ({result.feature_effects["type"].iloc[0]}):')
            else:
                print(f'\nAll feature effects ({result.feature_effects["type"].iloc[0]}):')
            print(result.feature_effects.to_string(index=False))
        else:
            print('\nThis model does not provide feature_importances_ or coef_.')

        if model_name == 'ridge':
            print('\nNote: For linear models with one-hot encoded features, coefficient interpretation requires caution.')
            print('      Coefficients depend on encoding scheme, scaling, and reference categories.')

        all_results.append(result)

        summary_rows.append({
            # --- Model identity ---
            'model': model_name,

            # --- Data / split info ---
            'split_mode': SPLIT_MODE,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_iv_raw': X.shape[1],

            # --- Best CV scores from GridSearchCV (training set only) ---
            'grid_cv_mae_mean': result.best_cv_scores.get('cv_mae_mean'),
            'grid_cv_mae_std': result.best_cv_scores.get('cv_mae_std'),
            'grid_cv_rmse_mean': result.best_cv_scores.get('cv_rmse_mean'),
            'grid_cv_rmse_std': result.best_cv_scores.get('cv_rmse_std'),
            'grid_cv_r2_mean': result.best_cv_scores.get('cv_r2_mean'),
            'grid_cv_r2_std': result.best_cv_scores.get('cv_r2_std'),

            # --- Re-run CV scores on best estimator (training set only) ---
            'train_cv_mae_mean': result.train_cv_scores.get('cv_mae_mean'),
            'train_cv_mae_std': result.train_cv_scores.get('cv_mae_std'),
            'train_cv_rmse_mean': result.train_cv_scores.get('cv_rmse_mean'),
            'train_cv_rmse_std': result.train_cv_scores.get('cv_rmse_std'),
            'train_cv_r2_mean': result.train_cv_scores.get('cv_r2_mean'),
            'train_cv_r2_std': result.train_cv_scores.get('cv_r2_std'),

            # --- Test set scores ---
            'test_mae': result.test_scores.get('MAE'),
            'test_rmse': result.test_scores.get('RMSE'),
            'test_r2': result.test_scores.get('R2'),

            # --- Generalization gap (test - train CV) ---
            # Positive gap for error metrics means worse on test
            'gap_mae_test_minus_traincv': (
                result.test_scores.get('MAE') - result.train_cv_scores.get('cv_mae_mean')
                if result.test_scores and result.train_cv_scores else np.nan
            ),
            'gap_rmse_test_minus_traincv': (
                result.test_scores.get('RMSE') - result.train_cv_scores.get('cv_rmse_mean')
                if result.test_scores and result.train_cv_scores else np.nan
            ),
            # For R2, negative gap means worse on test (since higher is better)
            'gap_r2_test_minus_traincv': (
                result.test_scores.get('R2') - result.train_cv_scores.get('cv_r2_mean')
                if result.test_scores and result.train_cv_scores else np.nan
            ),

            # --- Metadata ---
            'grid_refit_metric': GRID_REFIT_METRIC,
            'cv_n_splits': CV_N_SPLITS,
            'random_state': RANDOM_STATE,

            # --- (Optional) Keep best params as a compact string for logging ---
            'best_params': str(result.best_params)
        })

    # Final summary table
    summary_df = pd.DataFrame(summary_rows).sort_values('test_rmse', ascending=True).reset_index(drop=True)
    print('\n' + '=' * 80)
    print('Final model comparison (sorted by test_rmse)')
    print('=' * 80)
    print(summary_df.to_string(index=False))

    return {
        'data': data,
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'results': all_results,
        'summary_df': summary_df
    }

# %% [markdown]
# ## run

# %%
outputs = run_experiment()

# %%


# %%


# %%



