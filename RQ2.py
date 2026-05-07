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

RESULT_DIR = os.path.abspath(os.path.join('..', '..', 'result', 'RQ2'))
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

# %%
# === 1. Read data and validate columns ===

id_virality = pd.read_csv('../../dataset/id_virality.csv', index_col='id')
id_if_cib = pd.read_csv('../../dataset/id_if_cib.csv', index_col='id')
id_traces = pd.read_csv('../../dataset/id_traces.csv', index_col='id')

data = pd.concat([id_virality, id_if_cib, id_traces], axis=1, join='inner')

required_cols = [
    'view_count',
    'like_count',
    'comment_count',
    'share_count',
    'interaction_count',
    'log1p_view_count',
    'log1p_like_count',
    'log1p_comment_count',
    'log1p_share_count',
    'log1p_interaction_count',
    'interaction_rate',
    'share_rate',
    'share_share',
    'if_cib',
    'co_hashtagseq',
    'co_domain',
    'text_similarity',
    'video_similarity',
    'time_synchronization'
]

missing_cols = [col for col in required_cols if col not in data.columns]
if missing_cols:
    raise KeyError(f'Missing required columns: {missing_cols}')

impact_cols = [
    'view_count',
    'like_count',
    'comment_count',
    'share_count',
    'interaction_count',
    'log1p_view_count',
    'log1p_like_count',
    'log1p_comment_count',
    'log1p_share_count',
    'log1p_interaction_count',
    'interaction_rate',
    'share_rate',
    'share_share'
]

key_metrics = [
    'log1p_view_count',
    'log1p_interaction_count',
    'share_rate'
]

ratio_cols = [
    'interaction_rate',
    'share_rate',
    'share_share'
]

trace_cols = [
    'co_hashtagseq',
    'co_domain',
    'text_similarity',
    'video_similarity',
    'time_synchronization'
]

for col in impact_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

for col in ['if_cib'] + trace_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)

METRIC_LABELS_CN = {
    'view_count': '播放量',
    'like_count': '点赞量',
    'comment_count': '评论量',
    'share_count': '分享量',
    'interaction_count': '互动总量',
    'log1p_view_count': 'ln(1+播放量)',
    'log1p_like_count': 'ln(1+点赞量)',
    'log1p_comment_count': 'ln(1+评论量)',
    'log1p_share_count': 'ln(1+分享量)',
    'log1p_interaction_count': 'ln(1+互动总量)',
    'interaction_rate': '互动率',
    'share_rate': '分享率',
    'share_share': '分享占互动比'
}

TRACE_LABELS_CN = {
    'co_hashtagseq': '话题标签序列一致',
    'co_domain': '外链域名共现',
    'text_similarity': '语音内容相似',
    'video_similarity': '视频内容相似',
    'time_synchronization': '发帖时间同步'
}

GROUP_LABELS_CN = {
    0: '非CIB',
    1: 'CIB'
}

print(f'Data shape: {data.shape}')
print(f'Result directory: {RESULT_DIR}')

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
    formatter = FuncFormatter(lambda x, pos: f'{x:.1%}')
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


def mannwhitney_summary(x, y):
    x = pd.to_numeric(pd.Series(x), errors='coerce').dropna().to_numpy()
    y = pd.to_numeric(pd.Series(y), errors='coerce').dropna().to_numpy()

    out = {
        'u_stat': np.nan,
        'p_value': np.nan,
        'rank_biserial': np.nan,
        'ks_stat': np.nan,
        'ks_p_value': np.nan
    }

    if len(x) == 0 or len(y) == 0:
        return out

    try:
        u_stat, p_value = stats.mannwhitneyu(x, y, alternative='two-sided', method='auto')
    except TypeError:
        u_stat, p_value = stats.mannwhitneyu(x, y, alternative='two-sided')

    try:
        ks_stat, ks_p_value = stats.ks_2samp(x, y, alternative='two-sided', method='auto')
    except TypeError:
        ks_stat, ks_p_value = stats.ks_2samp(x, y, alternative='two-sided')

    rank_biserial = 2 * u_stat / (len(x) * len(y)) - 1

    out.update({
        'u_stat': u_stat,
        'p_value': p_value,
        'rank_biserial': rank_biserial,
        'ks_stat': ks_stat,
        'ks_p_value': ks_p_value
    })
    return out


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

# %%
# === 3. 5.1 Overall descriptive statistics of impact metrics ===

overall_rows = []
for metric in impact_cols:
    summary = summarize_series(data[metric])
    overall_rows.append({
        'metric': metric,
        'metric_label_cn': METRIC_LABELS_CN[metric],
        **summary
    })

table_51_overall_summary = pd.DataFrame(overall_rows).round(6)
save_csv(table_51_overall_summary, 'table_51_impact_overall_summary.csv', index=False)

table_51_overall_summary.head()

# %%
# === 4. 5.1 Overall distribution plots for key metrics ===

fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

for ax, metric in zip(axes, key_metrics):
    plot_df = balanced_sample(
        data[[metric]].dropna(),
        group_col=None,
        max_n=200000,
        random_state=RANDOM_SEED
    )

    s = plot_df[metric]

    sns.histplot(
        s,
        bins=50,
        kde=True,
        stat='density',
        color=BASE_PALETTE[0],
        alpha=0.75,
        edgecolor=None,
        ax=ax
    )

    ax.axvline(s.mean(), color=BASE_PALETTE[1], linestyle='--', linewidth=1.6, label='均值')
    ax.axvline(s.median(), color='black', linestyle='-', linewidth=1.6, label='中位数')

    ax.set_title(f'{METRIC_LABELS_CN[metric]}的总体分布')
    ax.set_xlabel(METRIC_LABELS_CN[metric])
    ax.set_ylabel('密度')

    if metric in ratio_cols:
        apply_ratio_formatter(ax, axis='x')

    ax.legend(frameon=False)
    sns.despine(ax=ax)

fig.suptitle('传播效果关键指标的总体分布', y=1.03, fontsize=15)
fig.tight_layout()

save_figure(fig, 'fig_51_overall_distribution_key_metrics.png')

# %%
# === 5. 5.2 Group summary by CIB vs non-CIB ===

rows = []
for metric in impact_cols:
    for group_value in [0, 1]:
        summary = summarize_series(data.loc[data['if_cib'] == group_value, metric])
        rows.append({
            'metric': metric,
            'metric_label_cn': METRIC_LABELS_CN[metric],
            'group_value': group_value,
            'group_label_cn': GROUP_LABELS_CN[group_value],
            **summary
        })

table_52_group_summary = pd.DataFrame(rows).round(6)
save_csv(table_52_group_summary, 'table_52_impact_by_cib_summary.csv', index=False)

table_52_group_summary.head()

# %%
# === 6. 5.2 Statistical tests: CIB vs non-CIB ===

rows = []

for metric in impact_cols:
    s_cib = data.loc[data['if_cib'] == 1, metric]
    s_non = data.loc[data['if_cib'] == 0, metric]

    sum_cib = summarize_series(s_cib)
    sum_non = summarize_series(s_non)
    test_out = mannwhitney_summary(s_cib, s_non)

    mean_diff = np.nan
    median_diff = np.nan
    if not pd.isna(sum_cib['mean']) and not pd.isna(sum_non['mean']):
        mean_diff = sum_cib['mean'] - sum_non['mean']
    if not pd.isna(sum_cib['median']) and not pd.isna(sum_non['median']):
        median_diff = sum_cib['median'] - sum_non['median']

    direction = 'no difference'
    if not pd.isna(test_out['rank_biserial']):
        if test_out['rank_biserial'] > 0:
            direction = 'CIB higher'
        elif test_out['rank_biserial'] < 0:
            direction = 'Non-CIB higher'

    rows.append({
        'metric': metric,
        'metric_label_cn': METRIC_LABELS_CN[metric],
        'n_cib': sum_cib['n'],
        'n_non_cib': sum_non['n'],
        'mean_cib': sum_cib['mean'],
        'mean_non_cib': sum_non['mean'],
        'median_cib': sum_cib['median'],
        'median_non_cib': sum_non['median'],
        'mean_diff_cib_minus_non_cib': mean_diff,
        'median_diff_cib_minus_non_cib': median_diff,
        'u_stat': test_out['u_stat'],
        'p_value': test_out['p_value'],
        'rank_biserial': test_out['rank_biserial'],
        'ks_stat': test_out['ks_stat'],
        'ks_p_value': test_out['ks_p_value'],
        'direction': direction
    })

table_52_tests = pd.DataFrame(rows)
table_52_tests['p_value_bh'] = p_adjust_bh(table_52_tests['p_value'].to_numpy())
table_52_tests['ks_p_value_bh'] = p_adjust_bh(table_52_tests['ks_p_value'].to_numpy())
table_52_tests['significance'] = table_52_tests['p_value_bh'].apply(significance_star)
table_52_tests = table_52_tests.round(6)

save_csv(table_52_tests, 'table_52_cib_vs_noncib_tests.csv', index=False)

table_52_tests[['metric', 'mean_diff_cib_minus_non_cib', 'median_diff_cib_minus_non_cib', 'rank_biserial', 'p_value_bh', 'direction']]

# %%
# === 7. 5.2 Key metric boxplots: CIB vs non-CIB ===

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
group_order = ['非CIB', 'CIB']

for ax, metric in zip(axes, key_metrics):
    plot_df = data[['if_cib', metric]].dropna().copy()
    plot_df = balanced_sample(
        plot_df,
        group_col='if_cib',
        max_n=50000,
        random_state=RANDOM_SEED
    )
    plot_df['group_label_cn'] = plot_df['if_cib'].map(GROUP_LABELS_CN)

    sns.boxplot(
        data=plot_df,
        x='group_label_cn',
        y=metric,
        order=group_order,
        showfliers=False,
        width=0.55,
        palette=[GROUP_PALETTE[0], GROUP_PALETTE[1]],
        ax=ax
    )

    row = table_52_tests.loc[table_52_tests['metric'] == metric].iloc[0]

    text = (
        f'n(非CIB)={int(row["n_non_cib"]):,}\n'
        f'n(CIB)={int(row["n_cib"]):,}\n'
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

    ax.set_title(f'{METRIC_LABELS_CN[metric]}：CIB vs 非CIB')
    ax.set_xlabel('')
    ax.set_ylabel(METRIC_LABELS_CN[metric])

    if metric in ratio_cols:
        apply_ratio_formatter(ax, axis='y')

    sns.despine(ax=ax)

fig.suptitle('CIB 与非 CIB 在关键传播指标上的差异', y=1.03, fontsize=15)
fig.tight_layout()

save_figure(fig, 'fig_52_cib_vs_noncib_boxplot_key_metrics.png')

# %%
# === 8. 5.2 Key metric ECDF plots: CIB vs non-CIB ===

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
group_order = ['非CIB', 'CIB']

for i, (ax, metric) in enumerate(zip(axes, key_metrics)):
    plot_df = data[['if_cib', metric]].dropna().copy()
    plot_df = balanced_sample(
        plot_df,
        group_col='if_cib',
        max_n=50000,
        random_state=RANDOM_SEED
    )
    plot_df['group_label_cn'] = plot_df['if_cib'].map(GROUP_LABELS_CN)

    sns.ecdfplot(
        data=plot_df,
        x=metric,
        hue='group_label_cn',
        hue_order=group_order,
        palette=[GROUP_PALETTE[0], GROUP_PALETTE[1]],
        linewidth=2,
        ax=ax
    )

    ax.set_title(f'{METRIC_LABELS_CN[metric]}的累计分布')
    ax.set_xlabel(METRIC_LABELS_CN[metric])
    ax.set_ylabel('累计占比')

    if metric in ratio_cols:
        apply_ratio_formatter(ax, axis='x')

    if i == 0:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title('')
            legend.set_frame_on(False)
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    sns.despine(ax=ax)

fig.suptitle('CIB 与非 CIB 的关键传播指标分布差异', y=1.03, fontsize=15)
fig.tight_layout()

save_figure(fig, 'fig_52_cib_vs_noncib_ecdf_key_metrics.png')

# %%
# === 9. 5.3 Restrict to CIB data ===

cib_data = data.loc[data['if_cib'] == 1].copy()

if cib_data.empty:
    raise ValueError('No CIB observations found in data.')

print(f'CIB-only data shape: {cib_data.shape}')

# %%
# === 10. 5.3 Trace prevalence and strategy combination tables ===

rows = []
for trace in trace_cols:
    positive_n = int((cib_data[trace] == 1).sum())
    rows.append({
        'trace': trace,
        'trace_label_cn': TRACE_LABELS_CN[trace],
        'n_positive': positive_n,
        'share_in_cib': positive_n / len(cib_data)
    })

table_53_trace_prevalence = pd.DataFrame(rows).round(6)
save_csv(table_53_trace_prevalence, 'table_53_cib_trace_prevalence.csv', index=False)

combo_series = cib_data[trace_cols].apply(
    lambda row: ' + '.join([TRACE_LABELS_CN[col] for col in trace_cols if row[col] == 1]) if row.sum() > 0 else 'None',
    axis=1
)

table_53_strategy_combinations = (
    combo_series
    .value_counts(dropna=False)
    .rename_axis('strategy_combination_cn')
    .reset_index(name='n')
)

table_53_strategy_combinations['share_in_cib'] = (
    table_53_strategy_combinations['n'] / len(cib_data)
)

table_53_strategy_combinations = table_53_strategy_combinations.round(6)
save_csv(table_53_strategy_combinations, 'table_53_cib_strategy_combinations.csv', index=False)

table_53_trace_prevalence

# %%
# === 11. 5.3 Trace overlap matrices ===

count_matrix = pd.DataFrame(index=trace_cols, columns=trace_cols, dtype=float)
jaccard_matrix = pd.DataFrame(index=trace_cols, columns=trace_cols, dtype=float)

for trace_i in trace_cols:
    mask_i = cib_data[trace_i] == 1
    for trace_j in trace_cols:
        mask_j = cib_data[trace_j] == 1
        intersection = int((mask_i & mask_j).sum())
        union = int((mask_i | mask_j).sum())

        count_matrix.loc[trace_i, trace_j] = intersection
        jaccard_matrix.loc[trace_i, trace_j] = intersection / union if union > 0 else np.nan

count_matrix.index = [TRACE_LABELS_CN[col] for col in count_matrix.index]
count_matrix.columns = [TRACE_LABELS_CN[col] for col in count_matrix.columns]

jaccard_matrix.index = [TRACE_LABELS_CN[col] for col in jaccard_matrix.index]
jaccard_matrix.columns = [TRACE_LABELS_CN[col] for col in jaccard_matrix.columns]

save_csv(count_matrix.round(6), 'table_53_cib_trace_overlap_count.csv', index=True)
save_csv(jaccard_matrix.round(6), 'table_53_cib_trace_overlap_jaccard.csv', index=True)

count_matrix

# %%
# === 12. 5.3 Trace overlap heatmap ===

fig, ax = plt.subplots(figsize=(7.2, 6.2))

sns.heatmap(
    jaccard_matrix,
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

ax.set_title('CIB样本中各协调轨迹的重叠程度')
ax.set_xlabel('协调轨迹')
ax.set_ylabel('协调轨迹')

fig.tight_layout()
save_figure(fig, 'fig_53_cib_trace_overlap_heatmap.png')

# %%
# === 13. 5.3 Statistical tests within CIB: trace-positive vs trace-negative ===

rows = []

for trace in trace_cols:
    mask_pos = cib_data[trace] == 1
    mask_neg = cib_data[trace] == 0

    for metric in impact_cols:
        s_pos = cib_data.loc[mask_pos, metric]
        s_neg = cib_data.loc[mask_neg, metric]

        sum_pos = summarize_series(s_pos)
        sum_neg = summarize_series(s_neg)
        test_out = mannwhitney_summary(s_pos, s_neg)

        mean_diff = np.nan
        median_diff = np.nan
        if not pd.isna(sum_pos['mean']) and not pd.isna(sum_neg['mean']):
            mean_diff = sum_pos['mean'] - sum_neg['mean']
        if not pd.isna(sum_pos['median']) and not pd.isna(sum_neg['median']):
            median_diff = sum_pos['median'] - sum_neg['median']

        direction = 'no difference'
        if not pd.isna(test_out['rank_biserial']):
            if test_out['rank_biserial'] > 0:
                direction = 'Trace-positive higher'
            elif test_out['rank_biserial'] < 0:
                direction = 'Trace-negative higher'

        rows.append({
            'trace': trace,
            'trace_label_cn': TRACE_LABELS_CN[trace],
            'metric': metric,
            'metric_label_cn': METRIC_LABELS_CN[metric],
            'n_trace_positive': sum_pos['n'],
            'n_trace_negative': sum_neg['n'],
            'mean_trace_positive': sum_pos['mean'],
            'mean_trace_negative': sum_neg['mean'],
            'median_trace_positive': sum_pos['median'],
            'median_trace_negative': sum_neg['median'],
            'mean_diff_positive_minus_negative': mean_diff,
            'median_diff_positive_minus_negative': median_diff,
            'u_stat': test_out['u_stat'],
            'p_value': test_out['p_value'],
            'rank_biserial': test_out['rank_biserial'],
            'ks_stat': test_out['ks_stat'],
            'ks_p_value': test_out['ks_p_value'],
            'direction': direction
        })

table_53_trace_tests = pd.DataFrame(rows)
table_53_trace_tests['p_value_bh'] = p_adjust_bh(table_53_trace_tests['p_value'].to_numpy())
table_53_trace_tests['ks_p_value_bh'] = p_adjust_bh(table_53_trace_tests['ks_p_value'].to_numpy())
table_53_trace_tests['significance'] = table_53_trace_tests['p_value_bh'].apply(significance_star)
table_53_trace_tests = table_53_trace_tests.round(6)

save_csv(table_53_trace_tests, 'table_53_cib_trace_tests.csv', index=False)

table_53_trace_tests[['trace', 'metric', 'median_diff_positive_minus_negative', 'rank_biserial', 'p_value_bh', 'direction']].head(12)

# %%
# === 14. 5.3 Key metric summary for trace-positive subgroups ===

rows = []

for trace in trace_cols:
    for metric in key_metrics:
        summary = summarize_series(cib_data.loc[cib_data[trace] == 1, metric])
        rows.append({
            'trace': trace,
            'trace_label_cn': TRACE_LABELS_CN[trace],
            'metric': metric,
            'metric_label_cn': METRIC_LABELS_CN[metric],
            **summary
        })

table_53_trace_key_metric_summary = pd.DataFrame(rows).round(6)
save_csv(table_53_trace_key_metric_summary, 'table_53_cib_trace_key_metric_summary.csv', index=False)

table_53_trace_key_metric_summary

# %%
# === 15. 5.3 Effect-size heatmap for key metrics within CIB ===

effect_heatmap = (
    table_53_trace_tests
    .loc[table_53_trace_tests['metric'].isin(key_metrics), ['trace_label_cn', 'metric_label_cn', 'rank_biserial']]
    .pivot(index='trace_label_cn', columns='metric_label_cn', values='rank_biserial')
)

p_heatmap = (
    table_53_trace_tests
    .loc[table_53_trace_tests['metric'].isin(key_metrics), ['trace_label_cn', 'metric_label_cn', 'p_value_bh']]
    .pivot(index='trace_label_cn', columns='metric_label_cn', values='p_value_bh')
)

trace_order_cn = [TRACE_LABELS_CN[col] for col in trace_cols]
metric_order_cn = [METRIC_LABELS_CN[col] for col in key_metrics]

effect_heatmap = effect_heatmap.reindex(index=trace_order_cn, columns=metric_order_cn)
p_heatmap = p_heatmap.reindex(index=trace_order_cn, columns=metric_order_cn)

annot = effect_heatmap.copy().astype(object)
for i in range(effect_heatmap.shape[0]):
    for j in range(effect_heatmap.shape[1]):
        value = effect_heatmap.iloc[i, j]
        pval = p_heatmap.iloc[i, j]
        if pd.isna(value):
            annot.iloc[i, j] = ''
        else:
            annot.iloc[i, j] = f'{value:.2f}{significance_star(pval)}'

fig, ax = plt.subplots(figsize=(8.2, 5.6))

sns.heatmap(
    effect_heatmap,
    annot=annot,
    fmt='',
    cmap='coolwarm',
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    cbar_kws={'label': '秩二分相关'},
    ax=ax
)

ax.set_title('不同协调轨迹对应的关键传播指标差异（仅在CIB内部比较）')
ax.set_xlabel('传播效果指标')
ax.set_ylabel('协调轨迹')

fig.tight_layout()
save_figure(fig, 'fig_53_cib_trace_effect_heatmap_key_metrics.png')

# %%
# === 16. 5.3 Descriptive boxplots for overlapping trace-positive subgroups ===

fig, axes = plt.subplots(1, 3, figsize=(19, 5))
trace_order_cn = [TRACE_LABELS_CN[col] for col in trace_cols]

for ax, metric in zip(axes, key_metrics):
    parts = []

    for trace in trace_cols:
        part = cib_data.loc[cib_data[trace] == 1, [metric]].dropna().copy()
        part['trace_label_cn'] = TRACE_LABELS_CN[trace]
        parts.append(part)

    plot_df = pd.concat(parts, axis=0, ignore_index=True)

    sns.boxplot(
        data=plot_df,
        x='trace_label_cn',
        y=metric,
        order=trace_order_cn,
        palette=TRACE_PALETTE,
        showfliers=False,
        width=0.6,
        ax=ax
    )

    ax.set_title(f'{METRIC_LABELS_CN[metric]}在不同协调轨迹阳性子组中的分布')
    ax.set_xlabel('')
    ax.set_ylabel(METRIC_LABELS_CN[metric])
    ax.tick_params(axis='x', rotation=20)

    if metric in ratio_cols:
        apply_ratio_formatter(ax, axis='y')

    sns.despine(ax=ax)

fig.suptitle('不同协调轨迹的关键传播指标比较（轨迹子组可重叠）', y=1.04, fontsize=15)
fig.text(
    0.5, -0.02,
    '注：每个箱线图对应“该轨迹=1”的 CIB 子样本；同一视频可同时进入多个子组，因此此图为描述性比较而非互斥分组比较。',
    ha='center',
    fontsize=11
)

fig.tight_layout()
save_figure(fig, 'fig_53_cib_trace_boxplot_key_metrics.png')

# %%
# === 17. Optional: build a simple manifest of saved outputs ===

manifest = pd.DataFrame([
    ['table_51_impact_overall_summary.csv', '5.1 全部传播指标的总体描述统计'],
    ['fig_51_overall_distribution_key_metrics.png', '5.1 关键指标总体分布图'],
    ['table_52_impact_by_cib_summary.csv', '5.2 CIB / 非CIB 分组描述统计'],
    ['table_52_cib_vs_noncib_tests.csv', '5.2 CIB / 非CIB 组间检验结果'],
    ['fig_52_cib_vs_noncib_boxplot_key_metrics.png', '5.2 关键指标箱线图比较'],
    ['fig_52_cib_vs_noncib_ecdf_key_metrics.png', '5.2 关键指标累计分布比较'],
    ['table_53_cib_trace_prevalence.csv', '5.3 CIB 内部各协调轨迹的出现频率'],
    ['table_53_cib_strategy_combinations.csv', '5.3 CIB 内部策略组合分布'],
    ['table_53_cib_trace_overlap_count.csv', '5.3 CIB 内部轨迹重叠计数矩阵'],
    ['table_53_cib_trace_overlap_jaccard.csv', '5.3 CIB 内部轨迹 Jaccard 重叠矩阵'],
    ['fig_53_cib_trace_overlap_heatmap.png', '5.3 CIB 内部轨迹重叠热力图'],
    ['table_53_cib_trace_tests.csv', '5.3 各轨迹阳性 vs 阴性的统计检验结果'],
    ['table_53_cib_trace_key_metric_summary.csv', '5.3 各轨迹阳性子组的关键指标描述统计'],
    ['fig_53_cib_trace_effect_heatmap_key_metrics.png', '5.3 各轨迹对应关键指标效应量热力图'],
    ['fig_53_cib_trace_boxplot_key_metrics.png', '5.3 各轨迹阳性子组的关键指标箱线图']
], columns=['filename', 'description'])

save_csv(manifest, 'result_manifest_rq2.csv', index=False)

print('All RQ2 outputs have been saved successfully.')
print(f'Output directory: {RESULT_DIR}')

# %%


# %%


# %%



