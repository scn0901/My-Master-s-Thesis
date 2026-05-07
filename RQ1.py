# %%
# %%
# === 0. Imports and global settings ===

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

RESULT_DIR = os.path.abspath(os.path.join('..', '..', 'result', 'RQ1'))
os.makedirs(RESULT_DIR, exist_ok=True)

FONT_PATH = os.path.expanduser('~/fonts/noto/NotoSansCJKsc-Regular.otf')
if not os.path.exists(FONT_PATH):
    raise FileNotFoundError(f'Font file not found: {FONT_PATH}')

font_manager.fontManager.addfont(FONT_PATH)
FONT_NAME = font_manager.FontProperties(fname=FONT_PATH).get_name()

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams.update({
    'font.family': FONT_NAME,
    'axes.unicode_minus': False,
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 10.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

BASE_PALETTE = sns.color_palette('colorblind')
GROUP_PALETTE = {
    0: BASE_PALETTE[0],
    1: BASE_PALETTE[1]
}
TRACE_PALETTE = sns.color_palette('crest', n_colors=5)

SAVEFIG_KW = {
    'dpi': 300,
    'bbox_inches': 'tight',
    'facecolor': 'white'
}

START_DATE = pd.Timestamp('2024-08-01')


# %%
# === 1. Read data and validate columns ===


id_if_cib = pd.read_csv('../../dataset/id_if_cib.csv', index_col='id')
id_basic_feature_engineering = pd.read_csv('../../dataset/id_basic_feature_engineering.csv', index_col='id')
id_time_feature_engineering = pd.read_csv('../../dataset/id_time_feature_engineering.csv', index_col='id')
id_user_feature_engineering_sample = pd.read_csv('../../dataset/id_user_feature_engineering_sample.csv', index_col='id')
id_traces = pd.read_csv('../../dataset/id_traces.csv', index_col='id')
data = pd.concat([id_if_cib, id_basic_feature_engineering, id_time_feature_engineering, id_traces], axis=1, join='inner').copy()
data_cib = data[data['if_cib']==1].copy()
data_sample = pd.concat([data, id_user_feature_engineering_sample], axis=1, join='inner').copy()


common_required_cols = [
    'if_cib',
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
    'publish_hour_edt',
    'publish_weekday_edt',
    'days_since_start',
    'co_hashtagseq',
    'co_domain',
    'text_similarity',
    'video_similarity',
    'time_synchronization'
]

sample_extra_required_cols = [
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
    'if_bio_has_political_keyword'
]

for df_name, df_obj in [('data', data), ('data_cib', data_cib)]:
    missing_cols = [col for col in common_required_cols if col not in df_obj.columns]
    if missing_cols:
        raise KeyError(f'Missing required columns in {df_name}: {missing_cols}')

missing_cols = [col for col in common_required_cols + sample_extra_required_cols if col not in data_sample.columns]
if missing_cols:
    raise KeyError(f'Missing required columns in data_sample: {missing_cols}')

binary_cols_common = [
    'if_cib',
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
    'co_hashtagseq',
    'co_domain',
    'text_similarity',
    'video_similarity',
    'time_synchronization'
]

binary_cols_sample_extra = [
    'query_failed',
    'if_username_autogen',
    'is_verified',
    'if_has_bio',
    'if_username_has_political_keyword',
    'if_display_name_has_political_keyword',
    'if_bio_has_political_keyword'
]

numeric_cols_common = [
    'hashtag_num',
    'hashtag_rate_outside_top_500',
    'domain_num',
    'domain_rate_outside_top_100',
    'publish_hour_edt',
    'publish_weekday_edt',
    'days_since_start'
]

numeric_cols_sample_extra = [
    'log1p_video_count',
    'log1p_follower_count',
    'log1p_following_count',
    'log1p_follower_following_rate'
]

for col in numeric_cols_common:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data_cib[col] = pd.to_numeric(data_cib[col], errors='coerce')
    data_sample[col] = pd.to_numeric(data_sample[col], errors='coerce')

for col in numeric_cols_sample_extra:
    data_sample[col] = pd.to_numeric(data_sample[col], errors='coerce')

for col in binary_cols_common:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)
    data_cib[col] = pd.to_numeric(data_cib[col], errors='coerce').fillna(0).astype(int)
    data_sample[col] = pd.to_numeric(data_sample[col], errors='coerce').fillna(0).astype(int)

for col in binary_cols_sample_extra:
    data_sample[col] = pd.to_numeric(data_sample[col], errors='coerce').fillna(0).astype(int)

content_numeric_cols = [
    'hashtag_num',
    'hashtag_rate_outside_top_500',
    'domain_num',
    'domain_rate_outside_top_100'
]

content_binary_cols = [
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
    'is_reply'
]

creator_numeric_cols = [
    'log1p_video_count',
    'log1p_follower_count',
    'log1p_following_count',
    'log1p_follower_following_rate'
]

creator_binary_cols = [
    'if_username_autogen',
    'is_verified',
    'if_has_bio',
    'if_username_has_political_keyword',
    'if_display_name_has_political_keyword',
    'if_bio_has_political_keyword'
]

trace_cols = [
    'co_hashtagseq',
    'co_domain',
    'text_similarity',
    'video_similarity',
    'time_synchronization'
]

TRACE_LABELS_CN = {
    'co_hashtagseq': '话题标签序列一致',
    'co_domain': '外链域名共现',
    'text_similarity': '文本内容相似',
    'video_similarity': '视频内容相似',
    'time_synchronization': '发布时间同步'
}

VAR_LABELS_CN = {
    'if_cib': '是否CIB',
    'hashtag_num': '话题标签数量',
    'hashtag_rate_outside_top_500': 'Top500外话题标签占比',
    'has_hashtag_dem': '含民主党相关标签',
    'has_hashtag_rep': '含共和党相关标签',
    'has_hashtag_traffic': '含引流型标签',
    'domain_num': '外链域名数量',
    'domain_rate_outside_top_100': 'Top100外域名占比',
    'has_domain_platform': '含平台型域名',
    'has_domain_politics_civic': '含政治/公共事务域名',
    'has_domain_news_media': '含新闻媒体域名',
    'has_domain_commerce': '含商业域名',
    'has_domain_fundraising': '含筹款域名',
    'is_duet': 'Duet视频',
    'is_stitch': 'Stitch视频',
    'is_reply': 'Reply视频',
    'publish_hour_edt': '发布时间（EDT小时）',
    'publish_weekday_edt': '发布时间（EDT星期）',
    'days_since_start': '距2024-08-01天数',
    'query_failed': '创作者信息查询失败',
    'if_username_autogen': '用户名疑似自动生成',
    'is_verified': '创作者账号已认证',
    'if_has_bio': '创作者有简介',
    'log1p_video_count': 'ln(1+历史发帖量)',
    'log1p_follower_count': 'ln(1+粉丝数)',
    'log1p_following_count': 'ln(1+关注数)',
    'log1p_follower_following_rate': 'ln(1+粉丝/关注比)',
    'if_username_has_political_keyword': '用户名含政治关键词',
    'if_display_name_has_political_keyword': '显示名含政治关键词',
    'if_bio_has_political_keyword': '简介含政治关键词'
}

GROUP_LABELS_CN = {
    0: '非CIB',
    1: 'CIB'
}

WEEKDAY_LABELS_CN = {
    0: '周一',
    1: '周二',
    2: '周三',
    3: '周四',
    4: '周五',
    5: '周六',
    6: '周日'
}

print(f'data shape: {data.shape}')
print(f'data_cib shape: {data_cib.shape}')
print(f'data_sample shape: {data_sample.shape}')
print(f'Result directory: {RESULT_DIR}')

# %%
# %%
# === 2. Helper functions ===

def save_csv(df, filename, index=False):
    path = os.path.join(RESULT_DIR, filename)
    df.to_csv(path, index=index, encoding='utf-8-sig')


def save_figure(fig, filename):
    path = os.path.join(RESULT_DIR, filename)
    fig.savefig(path, **SAVEFIG_KW)
    plt.close(fig)


def format_p_value(p):
    if pd.isna(p):
        return 'NA'
    if p < 1e-4:
        return f'{p:.2e}'
    return f'{p:.4f}'


def significance_star(p):
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def apply_ratio_formatter(ax, axis='x'):
    formatter = FuncFormatter(lambda x, pos: f'{x:.0%}')
    if axis in ['x', 'both']:
        ax.xaxis.set_major_formatter(formatter)
    if axis in ['y', 'both']:
        ax.yaxis.set_major_formatter(formatter)


def summarize_series(series):
    s = pd.to_numeric(series, errors='coerce')
    s_valid = s.dropna()

    out = {
        'n': int(s_valid.shape[0]),
        'missing_n': int(s.isna().sum()),
        'mean': np.nan,
        'std': np.nan,
        'min': np.nan,
        'q1': np.nan,
        'median': np.nan,
        'q3': np.nan,
        'max': np.nan
    }

    if s_valid.empty:
        return out

    out.update({
        'mean': s_valid.mean(),
        'std': s_valid.std(ddof=1),
        'min': s_valid.min(),
        'q1': s_valid.quantile(0.25),
        'median': s_valid.median(),
        'q3': s_valid.quantile(0.75),
        'max': s_valid.max()
    })
    return out


def p_adjust_bh(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan, dtype=float)

    valid_mask = ~np.isnan(p_values)
    p = p_values[valid_mask]

    if p.size == 0:
        return adjusted

    order = np.argsort(p)
    ranked_p = p[order]
    n = len(ranked_p)

    bh = ranked_p * n / np.arange(1, n + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0, 1)

    adjusted_valid = np.empty_like(bh)
    adjusted_valid[order] = bh
    adjusted[valid_mask] = adjusted_valid
    return adjusted


def balanced_sample(df, group_col=None, max_n=50000, random_state=RANDOM_SEED):
    if group_col is None:
        if len(df) <= max_n:
            return df.copy()
        return df.sample(n=max_n, random_state=random_state).copy()

    sampled_parts = []
    for _, part in df.groupby(group_col):
        n_take = min(len(part), max_n)
        if len(part) > n_take:
            sampled_parts.append(part.sample(n=n_take, random_state=random_state))
        else:
            sampled_parts.append(part.copy())
    return pd.concat(sampled_parts, axis=0).copy()


def mannwhitney_summary(x, y):
    x = pd.to_numeric(pd.Series(x), errors='coerce').dropna().to_numpy()
    y = pd.to_numeric(pd.Series(y), errors='coerce').dropna().to_numpy()

    out = {
        'u_stat': np.nan,
        'p_value': np.nan,
        'rank_biserial': np.nan
    }

    if len(x) == 0 or len(y) == 0:
        return out

    try:
        u_stat, p_value = stats.mannwhitneyu(x, y, alternative='two-sided', method='auto')
    except TypeError:
        u_stat, p_value = stats.mannwhitneyu(x, y, alternative='two-sided')

    rank_biserial = 2 * u_stat / (len(x) * len(y)) - 1

    out.update({
        'u_stat': u_stat,
        'p_value': p_value,
        'rank_biserial': rank_biserial
    })
    return out


def binary_group_test(df, group_col, value_col):
    sub = df[[group_col, value_col]].dropna().copy()
    table = pd.crosstab(sub[group_col], sub[value_col]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    n_total = table.to_numpy().sum()
    if n_total == 0:
        return {
            'method': 'NA',
            'chi2_stat': np.nan,
            'p_value': np.nan,
            'cramers_v': np.nan,
            'odds_ratio': np.nan
        }

    chi2_stat, chi2_p, _, expected = stats.chi2_contingency(table.to_numpy(), correction=False)
    cramers_v = np.sqrt(chi2_stat / n_total)

    odds_ratio = np.nan
    a = table.loc[1, 1]
    b = table.loc[1, 0]
    c = table.loc[0, 1]
    d = table.loc[0, 0]
    if b * c > 0:
        odds_ratio = (a * d) / (b * c)

    method = 'chi2'
    p_value = chi2_p

    if (expected < 5).any():
        odds_ratio_fisher, fisher_p = stats.fisher_exact(table.to_numpy())
        method = 'fisher'
        p_value = fisher_p
        odds_ratio = odds_ratio_fisher

    return {
        'method': method,
        'chi2_stat': chi2_stat,
        'p_value': p_value,
        'cramers_v': cramers_v,
        'odds_ratio': odds_ratio
    }


def numeric_group_summary_and_test(df, group_col, vars_list):
    summary_rows = []
    test_rows = []

    for var in vars_list:
        for group_value in [0, 1]:
            summary = summarize_series(df.loc[df[group_col] == group_value, var])
            summary_rows.append({
                'variable': var,
                'variable_label_cn': VAR_LABELS_CN[var],
                'group_value': group_value,
                'group_label_cn': GROUP_LABELS_CN[group_value],
                **summary
            })

        s0 = df.loc[df[group_col] == 0, var]
        s1 = df.loc[df[group_col] == 1, var]

        sum0 = summarize_series(s0)
        sum1 = summarize_series(s1)
        test_out = mannwhitney_summary(s1, s0)

        test_rows.append({
            'variable': var,
            'variable_label_cn': VAR_LABELS_CN[var],
            'n_non_cib': sum0['n'],
            'n_cib': sum1['n'],
            'mean_non_cib': sum0['mean'],
            'mean_cib': sum1['mean'],
            'median_non_cib': sum0['median'],
            'median_cib': sum1['median'],
            'mean_diff_cib_minus_non_cib': sum1['mean'] - sum0['mean'],
            'median_diff_cib_minus_non_cib': sum1['median'] - sum0['median'],
            **test_out
        })

    summary_df = pd.DataFrame(summary_rows)
    test_df = pd.DataFrame(test_rows)
    test_df['p_value_bh'] = p_adjust_bh(test_df['p_value'].to_numpy())
    test_df['significance'] = test_df['p_value_bh'].apply(significance_star)
    return summary_df.round(6), test_df.round(6)


def binary_group_summary_and_test(df, group_col, vars_list):
    summary_rows = []
    test_rows = []

    for var in vars_list:
        for group_value in [0, 1]:
            n_group = int((df[group_col] == group_value).sum())
            n_positive = int(df.loc[df[group_col] == group_value, var].sum())
            summary_rows.append({
                'variable': var,
                'variable_label_cn': VAR_LABELS_CN[var],
                'group_value': group_value,
                'group_label_cn': GROUP_LABELS_CN[group_value],
                'n_group': n_group,
                'n_positive': n_positive,
                'share_positive': n_positive / n_group if n_group > 0 else np.nan
            })

        share_non = df.loc[df[group_col] == 0, var].mean()
        share_cib = df.loc[df[group_col] == 1, var].mean()
        test_out = binary_group_test(df, group_col, var)

        test_rows.append({
            'variable': var,
            'variable_label_cn': VAR_LABELS_CN[var],
            'share_non_cib': share_non,
            'share_cib': share_cib,
            'share_diff_cib_minus_non_cib': share_cib - share_non,
            **test_out
        })

    summary_df = pd.DataFrame(summary_rows)
    test_df = pd.DataFrame(test_rows)
    test_df['p_value_bh'] = p_adjust_bh(test_df['p_value'].to_numpy())
    test_df['significance'] = test_df['p_value_bh'].apply(significance_star)
    return summary_df.round(6), test_df.round(6)


def trace_feature_test(df_cib, feature_cols):
    summary_rows = []
    test_rows = []

    for trace in trace_cols:
        mask_pos = df_cib[trace] == 1
        mask_neg = df_cib[trace] == 0

        for var in feature_cols:
            if df_cib[var].dropna().isin([0, 1]).all():
                share_pos = df_cib.loc[mask_pos, var].mean()
                share_neg = df_cib.loc[mask_neg, var].mean()
                summary_rows.append({
                    'trace': trace,
                    'trace_label_cn': TRACE_LABELS_CN[trace],
                    'variable': var,
                    'variable_label_cn': VAR_LABELS_CN[var],
                    'trace_positive_mean_or_share': share_pos,
                    'trace_negative_mean_or_share': share_neg,
                    'difference_positive_minus_negative': share_pos - share_neg,
                    'stat_type': 'binary'
                })

                test_df_temp = df_cib[[trace, var]].rename(columns={trace: 'trace_flag'})
                test_out = binary_group_test(test_df_temp, 'trace_flag', var)

                test_rows.append({
                    'trace': trace,
                    'trace_label_cn': TRACE_LABELS_CN[trace],
                    'variable': var,
                    'variable_label_cn': VAR_LABELS_CN[var],
                    'stat_type': 'binary',
                    'trace_positive_mean_or_share': share_pos,
                    'trace_negative_mean_or_share': share_neg,
                    'difference_positive_minus_negative': share_pos - share_neg,
                    **test_out
                })
            else:
                sum_pos = summarize_series(df_cib.loc[mask_pos, var])
                sum_neg = summarize_series(df_cib.loc[mask_neg, var])
                summary_rows.append({
                    'trace': trace,
                    'trace_label_cn': TRACE_LABELS_CN[trace],
                    'variable': var,
                    'variable_label_cn': VAR_LABELS_CN[var],
                    'trace_positive_mean_or_share': sum_pos['mean'],
                    'trace_negative_mean_or_share': sum_neg['mean'],
                    'difference_positive_minus_negative': sum_pos['mean'] - sum_neg['mean'],
                    'stat_type': 'numeric'
                })

                test_out = mannwhitney_summary(df_cib.loc[mask_pos, var], df_cib.loc[mask_neg, var])

                test_rows.append({
                    'trace': trace,
                    'trace_label_cn': TRACE_LABELS_CN[trace],
                    'variable': var,
                    'variable_label_cn': VAR_LABELS_CN[var],
                    'stat_type': 'numeric',
                    'trace_positive_mean_or_share': sum_pos['mean'],
                    'trace_negative_mean_or_share': sum_neg['mean'],
                    'difference_positive_minus_negative': sum_pos['mean'] - sum_neg['mean'],
                    'method': 'mannwhitney',
                    'chi2_stat': np.nan,
                    'odds_ratio': np.nan,
                    'cramers_v': np.nan,
                    **test_out
                })

    summary_df = pd.DataFrame(summary_rows).round(6)
    test_df = pd.DataFrame(test_rows)
    test_df['p_value_bh'] = p_adjust_bh(test_df['p_value'].to_numpy())
    test_df['significance'] = test_df['p_value_bh'].apply(significance_star)
    return summary_df, test_df.round(6)

# %%
# %%
# === 3. 4.1 Sample overview ===

table_41_sample_overview = pd.DataFrame([
    {
        'dataset': 'data',
        'n_total': len(data),
        'n_cib': int(data['if_cib'].sum()),
        'n_non_cib': int((data['if_cib'] == 0).sum()),
        'share_cib': data['if_cib'].mean()
    },
    {
        'dataset': 'data_cib',
        'n_total': len(data_cib),
        'n_cib': int(data_cib['if_cib'].sum()) if 'if_cib' in data_cib.columns else len(data_cib),
        'n_non_cib': int((data_cib['if_cib'] == 0).sum()) if 'if_cib' in data_cib.columns else 0,
        'share_cib': data_cib['if_cib'].mean() if 'if_cib' in data_cib.columns else 1.0
    },
    {
        'dataset': 'data_sample',
        'n_total': len(data_sample),
        'n_cib': int(data_sample['if_cib'].sum()),
        'n_non_cib': int((data_sample['if_cib'] == 0).sum()),
        'share_cib': data_sample['if_cib'].mean()
    }
]).round(6)

save_csv(table_41_sample_overview, 'table_41_sample_overview.csv', index=False)

table_41_sample_query = (
    data_sample
    .groupby(['if_cib', 'query_failed'])
    .size()
    .rename('n')
    .reset_index()
)

table_41_sample_query['group_label_cn'] = table_41_sample_query['if_cib'].map(GROUP_LABELS_CN)
table_41_sample_query['query_failed_label_cn'] = table_41_sample_query['query_failed'].map({0: '查询成功', 1: '查询失败'})

group_n = data_sample.groupby('if_cib').size().rename('group_n').reset_index()
table_41_sample_query = table_41_sample_query.merge(group_n, on='if_cib', how='left')
table_41_sample_query['share_within_group'] = table_41_sample_query['n'] / table_41_sample_query['group_n']
table_41_sample_query = table_41_sample_query.round(6)

save_csv(table_41_sample_query, 'table_41_sample_query_status.csv', index=False)


# %%
# === 4. 4.2 Content feature differences: tables ===

table_42_content_numeric_summary, table_42_content_numeric_tests = numeric_group_summary_and_test(
    data,
    'if_cib',
    content_numeric_cols
)
save_csv(table_42_content_numeric_summary, 'table_42_content_numeric_summary.csv', index=False)
save_csv(table_42_content_numeric_tests, 'table_42_content_numeric_tests.csv', index=False)

table_42_content_binary_summary, table_42_content_binary_tests = binary_group_summary_and_test(
    data,
    'if_cib',
    content_binary_cols
)
save_csv(table_42_content_binary_summary, 'table_42_content_binary_summary.csv', index=False)
save_csv(table_42_content_binary_tests, 'table_42_content_binary_tests.csv', index=False)

table_42_content_numeric_tests[['variable', 'mean_diff_cib_minus_non_cib', 'rank_biserial', 'p_value_bh']]

# %%
# %%
# === 5. 4.2 Content feature differences: plots ===

# Numeric features
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

sparse_numeric_vars = {'domain_num', 'domain_rate_outside_top_100'}

for ax, var in zip(axes.flat, content_numeric_cols):
    plot_df = data[['if_cib', var]].dropna().copy()

    max_n = 5000 if var in sparse_numeric_vars else 50000
    plot_df = balanced_sample(plot_df, group_col='if_cib', max_n=max_n, random_state=RANDOM_SEED)
    plot_df['group_label_cn'] = plot_df['if_cib'].map(GROUP_LABELS_CN)

    sns.boxplot(
        data=plot_df,
        x='group_label_cn',
        y=var,
        order=['非CIB', 'CIB'],
        palette=[GROUP_PALETTE[0], GROUP_PALETTE[1]],
        width=0.55,
        showfliers=(var in sparse_numeric_vars),
        fliersize=1.2 if var in sparse_numeric_vars else 0,
        ax=ax
    )

    row = table_42_content_numeric_tests.loc[table_42_content_numeric_tests['variable'] == var].iloc[0]
    text = (
        f'RBC={row["rank_biserial"]:.3f}\n'
        f'校正p={format_p_value(row["p_value_bh"])}{row["significance"]}'
    )
    ax.text(
        0.03, 0.97, text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.8')
    )

    ax.set_title(f'{VAR_LABELS_CN[var]}：CIB vs 非CIB')
    ax.set_xlabel('')
    ax.set_ylabel(VAR_LABELS_CN[var])

    if 'rate' in var:
        apply_ratio_formatter(ax, axis='y')

    sns.despine(ax=ax)

fig.suptitle('内容数值特征的组间差异', y=1.02, fontsize=15)
fig.tight_layout()
save_figure(fig, 'fig_42_content_numeric_boxplots.png')


# Binary features
fig, axes = plt.subplots(1, 3, figsize=(19, 8), gridspec_kw={'width_ratios': [1.0, 1.0, 0.85]})

hashtag_vars = [
    'has_hashtag_dem',
    'has_hashtag_rep',
    'has_hashtag_traffic'
]

domain_vars = [
    'has_domain_platform',
    'has_domain_politics_civic',
    'has_domain_news_media',
    'has_domain_commerce',
    'has_domain_fundraising'
]

format_vars = [
    'is_duet',
    'is_stitch',
    'is_reply'
]

plot_rows = []
for var in hashtag_vars + domain_vars + format_vars:
    row = table_42_content_binary_summary.loc[
        table_42_content_binary_summary['variable'] == var,
        ['group_label_cn', 'share_positive']
    ].copy()
    row['variable'] = var
    row['variable_label_cn'] = VAR_LABELS_CN[var]
    plot_rows.append(row)

plot_df = pd.concat(plot_rows, axis=0, ignore_index=True)

# Hashtag-related panel
sns.barplot(
    data=plot_df.loc[plot_df['variable'].isin(hashtag_vars)],
    x='share_positive',
    y='variable_label_cn',
    hue='group_label_cn',
    hue_order=['非CIB', 'CIB'],
    palette=[GROUP_PALETTE[0], GROUP_PALETTE[1]],
    orient='h',
    ax=axes[0]
)
axes[0].set_title('标签相关二元特征占比')
axes[0].set_xlabel('占比')
axes[0].set_ylabel('')
apply_ratio_formatter(axes[0], axis='x')
axes[0].legend(title='', frameon=False, loc='lower right')
sns.despine(ax=axes[0])

# Domain-related panel
sns.barplot(
    data=plot_df.loc[plot_df['variable'].isin(domain_vars)],
    x='share_positive',
    y='variable_label_cn',
    hue='group_label_cn',
    hue_order=['非CIB', 'CIB'],
    palette=[GROUP_PALETTE[0], GROUP_PALETTE[1]],
    orient='h',
    ax=axes[1]
)
axes[1].set_title('外链域名相关二元特征占比')
axes[1].set_xlabel('占比')
axes[1].set_ylabel('')
apply_ratio_formatter(axes[1], axis='x')
axes[1].legend(title='', frameon=False, loc='lower right')
sns.despine(ax=axes[1])

# Format-related panel
sns.barplot(
    data=plot_df.loc[plot_df['variable'].isin(format_vars)],
    x='share_positive',
    y='variable_label_cn',
    hue='group_label_cn',
    hue_order=['非CIB', 'CIB'],
    palette=[GROUP_PALETTE[0], GROUP_PALETTE[1]],
    orient='h',
    ax=axes[2]
)
axes[2].set_title('互动形式相关二元特征占比')
axes[2].set_xlabel('占比')
axes[2].set_ylabel('')
apply_ratio_formatter(axes[2], axis='x')
axes[2].legend(title='', frameon=False, loc='lower right')
sns.despine(ax=axes[2])

fig.suptitle('内容二元特征的组间差异', y=1.02, fontsize=15)
fig.tight_layout()
save_figure(fig, 'fig_42_content_binary_prevalence.png')

# %%
# %%
# === 6. 4.3 Publish-time differences: tables ===

table_43_days_summary, table_43_days_tests = numeric_group_summary_and_test(
    data,
    'if_cib',
    ['days_since_start']
)
save_csv(table_43_days_summary, 'table_43_publish_days_summary.csv', index=False)
save_csv(table_43_days_tests, 'table_43_publish_days_tests.csv', index=False)

hour_dist = (
    data.groupby(['if_cib', 'publish_hour_edt'])
    .size()
    .rename('n')
    .reset_index()
)

hour_total = data.groupby('if_cib').size().rename('group_n').reset_index()
hour_dist = hour_dist.merge(hour_total, on='if_cib', how='left')
hour_dist['share_within_group'] = hour_dist['n'] / hour_dist['group_n']
hour_dist['group_label_cn'] = hour_dist['if_cib'].map(GROUP_LABELS_CN)
hour_dist = hour_dist.sort_values(['if_cib', 'publish_hour_edt']).round(6)
save_csv(hour_dist, 'table_43_publish_hour_distribution.csv', index=False)

weekday_dist = (
    data.groupby(['if_cib', 'publish_weekday_edt'])
    .size()
    .rename('n')
    .reset_index()
)

weekday_total = data.groupby('if_cib').size().rename('group_n').reset_index()
weekday_dist = weekday_dist.merge(weekday_total, on='if_cib', how='left')
weekday_dist['share_within_group'] = weekday_dist['n'] / weekday_dist['group_n']
weekday_dist['group_label_cn'] = weekday_dist['if_cib'].map(GROUP_LABELS_CN)
weekday_dist['weekday_label_cn'] = weekday_dist['publish_weekday_edt'].map(WEEKDAY_LABELS_CN)
weekday_dist = weekday_dist.sort_values(['if_cib', 'publish_weekday_edt']).round(6)
save_csv(weekday_dist, 'table_43_publish_weekday_distribution.csv', index=False)

hour_table = pd.crosstab(data['if_cib'], data['publish_hour_edt']).reindex(index=[0, 1], fill_value=0)
weekday_table = pd.crosstab(data['if_cib'], data['publish_weekday_edt']).reindex(index=[0, 1], fill_value=0)

hour_chi2, hour_p, _, _ = stats.chi2_contingency(hour_table.to_numpy())
weekday_chi2, weekday_p, _, _ = stats.chi2_contingency(weekday_table.to_numpy())

table_43_publish_tests = pd.DataFrame([
    {
        'dimension': 'publish_hour_edt',
        'dimension_label_cn': '发布时间（EDT小时）',
        'method': 'chi2',
        'statistic': hour_chi2,
        'p_value': hour_p,
        'cramers_v': np.sqrt(hour_chi2 / hour_table.to_numpy().sum())
    },
    {
        'dimension': 'publish_weekday_edt',
        'dimension_label_cn': '发布时间（EDT星期）',
        'method': 'chi2',
        'statistic': weekday_chi2,
        'p_value': weekday_p,
        'cramers_v': np.sqrt(weekday_chi2 / weekday_table.to_numpy().sum())
    },
    {
        'dimension': 'days_since_start',
        'dimension_label_cn': '距2024-08-01天数',
        'method': 'mannwhitney',
        'statistic': table_43_days_tests.loc[0, 'u_stat'],
        'p_value': table_43_days_tests.loc[0, 'p_value'],
        'cramers_v': np.nan
    }
])

table_43_publish_tests['p_value_bh'] = p_adjust_bh(table_43_publish_tests['p_value'].to_numpy())
table_43_publish_tests['significance'] = table_43_publish_tests['p_value_bh'].apply(significance_star)
table_43_publish_tests = table_43_publish_tests.round(6)
save_csv(table_43_publish_tests, 'table_43_publish_time_tests.csv', index=False)


# %%
# === 7. 4.3 Publish-time differences: plots ===

fig, ax = plt.subplots(figsize=(12, 5.2))

plot_hour = hour_dist.copy()
for group_value in [0, 1]:
    full_hours = pd.DataFrame({'publish_hour_edt': np.arange(24)})
    part = plot_hour.loc[plot_hour['if_cib'] == group_value, ['publish_hour_edt', 'share_within_group']]
    part = full_hours.merge(part, on='publish_hour_edt', how='left').fillna(0)
    ax.plot(
        part['publish_hour_edt'],
        part['share_within_group'],
        marker='o',
        linewidth=2,
        label=GROUP_LABELS_CN[group_value],
        color=GROUP_PALETTE[group_value]
    )

ax.set_title('CIB 与非CIB的视频发布时间小时分布')
ax.set_xlabel('EDT小时')
ax.set_ylabel('组内占比')
ax.set_xticks(np.arange(24))
apply_ratio_formatter(ax, axis='y')
ax.legend(title='', frameon=False)
sns.despine(ax=ax)

fig.tight_layout()
save_figure(fig, 'fig_43_publish_hour_distribution.png')


fig, ax = plt.subplots(figsize=(10, 5.2))

plot_weekday = weekday_dist.copy()
plot_weekday['weekday_label_cn'] = pd.Categorical(
    plot_weekday['weekday_label_cn'],
    categories=[WEEKDAY_LABELS_CN[i] for i in range(7)],
    ordered=True
)

sns.barplot(
    data=plot_weekday,
    x='weekday_label_cn',
    y='share_within_group',
    hue='group_label_cn',
    hue_order=['非CIB', 'CIB'],
    palette=[GROUP_PALETTE[0], GROUP_PALETTE[1]],
    ax=ax
)

ax.set_title('CIB 与非CIB的视频发布时间星期分布')
ax.set_xlabel('')
ax.set_ylabel('组内占比')
apply_ratio_formatter(ax, axis='y')
ax.legend(title='', frameon=False)
sns.despine(ax=ax)

fig.tight_layout()
save_figure(fig, 'fig_43_publish_weekday_distribution.png')


timeline_df = data[['if_cib', 'days_since_start']].copy()
timeline_df['week_index'] = np.floor(timeline_df['days_since_start'] / 7).astype(int)
timeline_df['week_start'] = START_DATE + pd.to_timedelta(timeline_df['week_index'] * 7, unit='D')

timeline_plot = (
    timeline_df.groupby(['if_cib', 'week_start'])
    .size()
    .rename('n')
    .reset_index()
)

group_total = timeline_plot.groupby('if_cib')['n'].sum().rename('group_n').reset_index()
timeline_plot = timeline_plot.merge(group_total, on='if_cib', how='left')
timeline_plot['share_within_group'] = timeline_plot['n'] / timeline_plot['group_n']
timeline_plot['group_label_cn'] = timeline_plot['if_cib'].map(GROUP_LABELS_CN)

fig, axes = plt.subplots(1, 2, figsize=(16, 5.2))

sns.lineplot(
    data=timeline_plot,
    x='week_start',
    y='share_within_group',
    hue='group_label_cn',
    hue_order=['非CIB', 'CIB'],
    palette=[GROUP_PALETTE[0], GROUP_PALETTE[1]],
    linewidth=2,
    marker='o',
    ax=axes[0]
)

axes[0].set_title('按周聚合的发布时间推进趋势')
axes[0].set_xlabel('')
axes[0].set_ylabel('组内占比')
apply_ratio_formatter(axes[0], axis='y')
axes[0].legend(title='', frameon=False)
sns.despine(ax=axes[0])

plot_df = balanced_sample(
    data[['if_cib', 'days_since_start']].dropna(),
    group_col='if_cib',
    max_n=50000,
    random_state=RANDOM_SEED
).copy()
plot_df['group_label_cn'] = plot_df['if_cib'].map(GROUP_LABELS_CN)

sns.ecdfplot(
    data=plot_df,
    x='days_since_start',
    hue='group_label_cn',
    hue_order=['非CIB', 'CIB'],
    palette=[GROUP_PALETTE[0], GROUP_PALETTE[1]],
    linewidth=2,
    ax=axes[1]
)

axes[1].set_title('发布时间距起点天数的累计分布')
axes[1].set_xlabel('距2024-08-01天数')
axes[1].set_ylabel('累计占比')
axes[1].legend(title='', frameon=False)
sns.despine(ax=axes[1])

fig.tight_layout()
save_figure(fig, 'fig_43_publish_timeline_and_ecdf.png')

# %%
# %%
# === 8. 4.4 Creator feature differences based on sampled data ===

table_44_query_failed_summary, table_44_query_failed_tests = binary_group_summary_and_test(
    data_sample,
    'if_cib',
    ['query_failed']
)
save_csv(table_44_query_failed_summary, 'table_44_query_failed_summary.csv', index=False)
save_csv(table_44_query_failed_tests, 'table_44_query_failed_tests.csv', index=False)

data_sample_valid = data_sample.loc[data_sample['query_failed'] == 0].copy()

table_44_creator_numeric_summary, table_44_creator_numeric_tests = numeric_group_summary_and_test(
    data_sample_valid,
    'if_cib',
    creator_numeric_cols
)
save_csv(table_44_creator_numeric_summary, 'table_44_creator_numeric_summary.csv', index=False)
save_csv(table_44_creator_numeric_tests, 'table_44_creator_numeric_tests.csv', index=False)

table_44_creator_binary_summary, table_44_creator_binary_tests = binary_group_summary_and_test(
    data_sample_valid,
    'if_cib',
    creator_binary_cols
)
save_csv(table_44_creator_binary_summary, 'table_44_creator_binary_summary.csv', index=False)
save_csv(table_44_creator_binary_tests, 'table_44_creator_binary_tests.csv', index=False)

table_44_valid_sample_overview = pd.DataFrame([
    {
        'subset': 'all_sample',
        'n_total': len(data_sample),
        'n_cib': int(data_sample['if_cib'].sum()),
        'n_non_cib': int((data_sample['if_cib'] == 0).sum())
    },
    {
        'subset': 'query_success_only',
        'n_total': len(data_sample_valid),
        'n_cib': int(data_sample_valid['if_cib'].sum()),
        'n_non_cib': int((data_sample_valid['if_cib'] == 0).sum())
    }
])
save_csv(table_44_valid_sample_overview, 'table_44_valid_sample_overview.csv', index=False)


# %%
# === 9. 4.4 Creator feature differences: plots ===

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={'width_ratios': [1, 1.8, 1.4]})

query_plot = table_44_query_failed_summary.copy()
sns.barplot(
    data=query_plot,
    x='group_label_cn',
    y='share_positive',
    order=['非CIB', 'CIB'],
    palette=[GROUP_PALETTE[0], GROUP_PALETTE[1]],
    ax=axes[0]
)
axes[0].set_title('创作者信息查询失败比例')
axes[0].set_xlabel('')
axes[0].set_ylabel('占比')
apply_ratio_formatter(axes[0], axis='y')
sns.despine(ax=axes[0])

plot_df = balanced_sample(
    data_sample_valid[['if_cib'] + creator_numeric_cols].dropna(),
    group_col='if_cib',
    max_n=5000,
    random_state=RANDOM_SEED
).copy()
plot_df['group_label_cn'] = plot_df['if_cib'].map(GROUP_LABELS_CN)
long_df = plot_df.melt(
    id_vars=['if_cib', 'group_label_cn'],
    value_vars=creator_numeric_cols,
    var_name='variable',
    value_name='value'
)
long_df['variable_label_cn'] = long_df['variable'].map(VAR_LABELS_CN)

sns.boxplot(
    data=long_df,
    x='variable_label_cn',
    y='value',
    hue='group_label_cn',
    hue_order=['非CIB', 'CIB'],
    palette=[GROUP_PALETTE[0], GROUP_PALETTE[1]],
    showfliers=False,
    ax=axes[1]
)
axes[1].set_title('创作者数值特征比较（仅query_failed=0）')
axes[1].set_xlabel('')
axes[1].set_ylabel('数值')
axes[1].tick_params(axis='x', rotation=18)
axes[1].legend(title='', frameon=False)
sns.despine(ax=axes[1])

binary_plot = table_44_creator_binary_summary.copy()
binary_plot['variable_label_cn'] = pd.Categorical(
    binary_plot['variable_label_cn'],
    categories=[VAR_LABELS_CN[v] for v in creator_binary_cols],
    ordered=True
)

sns.barplot(
    data=binary_plot,
    x='share_positive',
    y='variable_label_cn',
    hue='group_label_cn',
    hue_order=['非CIB', 'CIB'],
    palette=[GROUP_PALETTE[0], GROUP_PALETTE[1]],
    orient='h',
    ax=axes[2]
)
axes[2].set_title('创作者二元特征比较（仅query_failed=0）')
axes[2].set_xlabel('占比')
axes[2].set_ylabel('')
apply_ratio_formatter(axes[2], axis='x')
axes[2].legend(title='', frameon=False, loc='lower right')
sns.despine(ax=axes[2])

fig.tight_layout()
save_figure(fig, 'fig_44_creator_feature_comparison.png')

# %%
# %%
# === 10. 4.5 Trace profile within CIB: prevalence and overlap ===

if 'if_cib' in data_cib.columns:
    cib_full = data_cib.loc[data_cib['if_cib'] == 1].copy()
else:
    cib_full = data_cib.copy()

rows = []
for trace in trace_cols:
    positive_n = int((cib_full[trace] == 1).sum())
    rows.append({
        'trace': trace,
        'trace_label_cn': TRACE_LABELS_CN[trace],
        'n_positive': positive_n,
        'share_in_cib': positive_n / len(cib_full)
    })

table_45_trace_prevalence = pd.DataFrame(rows).round(6)
save_csv(table_45_trace_prevalence, 'table_45_trace_prevalence.csv', index=False)

combo_series = cib_full[trace_cols].apply(
    lambda row: ' + '.join([TRACE_LABELS_CN[col] for col in trace_cols if row[col] == 1]) if row.sum() > 0 else '无轨迹',
    axis=1
)

table_45_trace_combinations = (
    combo_series.value_counts(dropna=False)
    .rename_axis('trace_combination_cn')
    .reset_index(name='n')
)
table_45_trace_combinations['share_in_cib'] = table_45_trace_combinations['n'] / len(cib_full)
table_45_trace_combinations = table_45_trace_combinations.round(6)
save_csv(table_45_trace_combinations, 'table_45_trace_combinations.csv', index=False)

overlap_count = pd.DataFrame(index=trace_cols, columns=trace_cols, dtype=float)
overlap_jaccard = pd.DataFrame(index=trace_cols, columns=trace_cols, dtype=float)

for trace_i in trace_cols:
    mask_i = cib_full[trace_i] == 1
    for trace_j in trace_cols:
        mask_j = cib_full[trace_j] == 1
        intersection = int((mask_i & mask_j).sum())
        union = int((mask_i | mask_j).sum())
        overlap_count.loc[trace_i, trace_j] = intersection
        overlap_jaccard.loc[trace_i, trace_j] = intersection / union if union > 0 else np.nan

overlap_count.index = [TRACE_LABELS_CN[c] for c in overlap_count.index]
overlap_count.columns = [TRACE_LABELS_CN[c] for c in overlap_count.columns]
overlap_jaccard.index = [TRACE_LABELS_CN[c] for c in overlap_jaccard.index]
overlap_jaccard.columns = [TRACE_LABELS_CN[c] for c in overlap_jaccard.columns]

save_csv(overlap_count.round(6), 'table_45_trace_overlap_count.csv', index=True)
save_csv(overlap_jaccard.round(6), 'table_45_trace_overlap_jaccard.csv', index=True)

trace_profile_vars = [
    'hashtag_num',
    'hashtag_rate_outside_top_500',
    'domain_num',
    'domain_rate_outside_top_100',
    'has_hashtag_traffic',
    'has_domain_news_media',
    'has_domain_platform',
    'is_reply',
    'publish_hour_edt',
    'days_since_start'
]

# Remove variables with no variation within CIB
trace_profile_vars = [
    var for var in trace_profile_vars
    if cib_full[var].nunique(dropna=False) > 1
]

table_45_trace_profile_summary, table_45_trace_profile_tests = trace_feature_test(cib_full, trace_profile_vars)
save_csv(table_45_trace_profile_summary, 'table_45_trace_profile_summary.csv', index=False)
save_csv(table_45_trace_profile_tests, 'table_45_trace_profile_tests.csv', index=False)


# %%
# === 11. 4.5 Trace profile within CIB: plots ===

fig, ax = plt.subplots(figsize=(7.2, 6.2))

sns.heatmap(
    overlap_jaccard,
    annot=True,
    fmt='.2f',
    cmap='YlGnBu',
    vmin=0,
    vmax=1,
    linewidths=0.5,
    square=True,
    cbar_kws={'label': 'Jaccard重叠系数'},
    ax=ax
)

ax.set_title('CIB内部各协调轨迹的重叠程度')
ax.set_xlabel('协调轨迹')
ax.set_ylabel('协调轨迹')

fig.tight_layout()
save_figure(fig, 'fig_45_trace_overlap_heatmap.png')


heatmap_df = (
    table_45_trace_profile_summary
    .pivot(index='trace_label_cn', columns='variable_label_cn', values='trace_positive_mean_or_share')
)

trace_order_cn = [TRACE_LABELS_CN[col] for col in trace_cols]
feature_order_cn = [VAR_LABELS_CN[col] for col in trace_profile_vars]

heatmap_df = heatmap_df.reindex(index=trace_order_cn, columns=feature_order_cn)

annot_df = heatmap_df.copy().astype(object)
for i in range(heatmap_df.shape[0]):
    for j in range(heatmap_df.shape[1]):
        value = heatmap_df.iloc[i, j]
        if pd.isna(value):
            annot_df.iloc[i, j] = ''
        else:
            annot_df.iloc[i, j] = f'{value:.2f}'

fig, ax = plt.subplots(figsize=(12, 5.8))

sns.heatmap(
    heatmap_df,
    annot=annot_df,
    fmt='',
    cmap='Blues',
    linewidths=0.5,
    cbar_kws={'label': '均值 / 占比'},
    ax=ax
)

ax.set_title('不同协调轨迹阳性子组的特征画像（仅CIB内部）')
ax.set_xlabel('特征')
ax.set_ylabel('协调轨迹')
ax.tick_params(axis='x', rotation=25)

fig.tight_layout()
save_figure(fig, 'fig_45_trace_profile_heatmap.png')


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

trace_plot_vars = [var for var in ['hashtag_num', 'domain_num', 'days_since_start'] if var in trace_profile_vars]

for ax, var in zip(axes, trace_plot_vars):
    parts = []
    for trace in trace_cols:
        part = cib_full.loc[cib_full[trace] == 1, [var]].dropna().copy()
        part['trace_label_cn'] = TRACE_LABELS_CN[trace]
        parts.append(part)
    plot_df = pd.concat(parts, axis=0, ignore_index=True)

    sns.boxplot(
        data=plot_df,
        x='trace_label_cn',
        y=var,
        order=trace_order_cn,
        palette=TRACE_PALETTE,
        showfliers=False,
        width=0.6,
        ax=ax
    )

    ax.set_title(f'{VAR_LABELS_CN[var]}在不同轨迹阳性子组中的分布')
    ax.set_xlabel('')
    ax.set_ylabel(VAR_LABELS_CN[var])
    ax.tick_params(axis='x', rotation=18)
    sns.despine(ax=ax)

# Hide unused axes if needed
for ax in axes[len(trace_plot_vars):]:
    ax.axis('off')

fig.suptitle('不同协调轨迹对应的关键特征差异（轨迹子组可重叠）', y=1.03, fontsize=15)
fig.text(
    0.5, -0.03,
    '注：每个箱线图对应“该轨迹=1”的CIB子样本；同一视频可同时进入多个轨迹子组，因此此图为描述性比较而非互斥分组比较。',
    ha='center',
    fontsize=11
)
fig.tight_layout()
save_figure(fig, 'fig_45_trace_key_feature_boxplots.png')

# %%
# %%
# === 12. Build result manifest ===

manifest = pd.DataFrame([
    ['table_41_sample_overview.csv', '4.1 数据样本规模概览'],
    ['table_41_sample_query_status.csv', '4.1 抽样样本中的创作者信息查询状态'],
    ['table_42_content_numeric_summary.csv', '4.2 内容数值特征分组描述统计'],
    ['table_42_content_numeric_tests.csv', '4.2 内容数值特征组间检验'],
    ['table_42_content_binary_summary.csv', '4.2 内容二元特征分组描述统计'],
    ['table_42_content_binary_tests.csv', '4.2 内容二元特征组间检验'],
    ['fig_42_content_numeric_boxplots.png', '4.2 内容数值特征箱线图（稀疏变量显示离群点）'],
    ['fig_42_content_binary_prevalence.png', '4.2 内容二元特征占比图（标签、域名与互动形式分面展示）'],
    ['table_43_publish_days_summary.csv', '4.3 发布时间距起点天数描述统计'],
    ['table_43_publish_days_tests.csv', '4.3 发布时间距起点天数组间检验'],
    ['table_43_publish_hour_distribution.csv', '4.3 发布时间小时分布'],
    ['table_43_publish_weekday_distribution.csv', '4.3 发布时间星期分布'],
    ['table_43_publish_time_tests.csv', '4.3 发布时间特征整体检验'],
    ['fig_43_publish_hour_distribution.png', '4.3 发布时间小时分布图'],
    ['fig_43_publish_weekday_distribution.png', '4.3 发布时间星期分布图'],
    ['fig_43_publish_timeline_and_ecdf.png', '4.3 发布时间时间推进趋势与累计分布图'],
    ['table_44_query_failed_summary.csv', '4.4 创作者信息查询失败比例描述统计'],
    ['table_44_query_failed_tests.csv', '4.4 创作者信息查询失败比例组间检验'],
    ['table_44_valid_sample_overview.csv', '4.4 仅保留query_failed=0后的样本规模'],
    ['table_44_creator_numeric_summary.csv', '4.4 创作者数值特征描述统计'],
    ['table_44_creator_numeric_tests.csv', '4.4 创作者数值特征组间检验'],
    ['table_44_creator_binary_summary.csv', '4.4 创作者二元特征描述统计'],
    ['table_44_creator_binary_tests.csv', '4.4 创作者二元特征组间检验'],
    ['fig_44_creator_feature_comparison.png', '4.4 创作者特征比较图'],
    ['table_45_trace_prevalence.csv', '4.5 各协调轨迹在CIB中的出现频率'],
    ['table_45_trace_combinations.csv', '4.5 CIB内部轨迹组合分布'],
    ['table_45_trace_overlap_count.csv', '4.5 CIB内部轨迹重叠计数矩阵'],
    ['table_45_trace_overlap_jaccard.csv', '4.5 CIB内部轨迹Jaccard重叠矩阵'],
    ['table_45_trace_profile_summary.csv', '4.5 不同轨迹阳性子组的特征画像描述统计'],
    ['table_45_trace_profile_tests.csv', '4.5 不同轨迹阳性子组的特征画像检验'],
    ['fig_45_trace_overlap_heatmap.png', '4.5 CIB内部轨迹重叠热力图'],
    ['fig_45_trace_profile_heatmap.png', '4.5 不同轨迹阳性子组的特征画像热力图'],
    ['fig_45_trace_key_feature_boxplots.png', '4.5 不同轨迹阳性子组的关键特征箱线图']
], columns=['filename', 'description'])

save_csv(manifest, 'result_manifest_rq1.csv', index=False)

print('All RQ1 outputs have been saved successfully.')
print(f'Output directory: {RESULT_DIR}')

# %%


# %%


# %%



