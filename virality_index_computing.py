# %% [markdown]
# # virality_index_computing

# %% [markdown]
# ## import modules

# %%
import numpy as np
import pandas as pd

# %% [markdown]
# ## load dataset

# %%
id_virality = pd.read_csv('../../dataset/id_virality.csv', index_col='id')
id_virality

# %%
id_if_cib = pd.read_csv('../../dataset/id_if_cib.csv', index_col='id')
id_if_cib

# %% [markdown]
# ## merge

# %%
id_virality_if_cib = pd.concat([id_virality, id_if_cib], axis=1, join='inner')
id_virality_if_cib

# %% [markdown]
# ## filter out cib

# %%
id_virality_cib = id_virality_if_cib[id_virality_if_cib['if_cib']==1]
id_virality_cib.drop(['if_cib'], axis=1, inplace=True)
id_virality_cib

# %% [markdown]
# ## count np.nan

# %%
id_virality_cib.isna().sum(axis=0)

# %% [markdown]
# ## get columns

# %%
print(id_virality_cib.columns.to_list())

# %% [markdown]
# ## def virality_index

# %%
def _to_1d_array(x):
    """将单个值或列表统一转为 1 维 numpy 数组"""
    if np.isscalar(x):
        return np.array([x], dtype=float)
    return np.asarray(x, dtype=float).reshape(-1)


def fit_virality_stats(share_count, view_count, interaction_count, q=0.05):
    """
    用一批样本拟合 virality_index 所需的参考参数：
    1. 平滑常数 tau_v, tau_e
    2. 三个组成部分的均值和标准差（用于 z-score）

    建议在训练集或全样本上先拟合一次，再用于后续计算。
    """
    s = _to_1d_array(share_count)
    v = _to_1d_array(view_count)
    it = _to_1d_array(interaction_count)

    if not (len(s) == len(v) == len(it)):
        raise ValueError("share_count, view_count, interaction_count 长度必须一致。")

    # 仅从正的分母中估计平滑常数
    pos_v = v[v > 0]
    pos_it = it[it > 0]

    tau_v = float(np.quantile(pos_v, q)) if pos_v.size > 0 else 1.0
    tau_e = float(np.quantile(pos_it, q)) if pos_it.size > 0 else 1.0

    # 构造平滑后的有效比率
    adj_share_rate = np.where(v > 0, s / (v + tau_v), 0.0)
    adj_share_share = np.where(it > 0, s / (it + tau_e), 0.0)

    # 对 share_count 做 log 变换
    log_share = np.log1p(np.clip(s, a_min=0, a_max=None))

    def safe_mean_std(x):
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x, ddof=0))
        if sd == 0 or np.isnan(sd):
            sd = 1.0
        return mu, sd

    mu_log_share, sd_log_share = safe_mean_std(log_share)
    mu_rate, sd_rate = safe_mean_std(adj_share_rate)
    mu_share, sd_share = safe_mean_std(adj_share_share)

    return {
        "tau_v": tau_v,
        "tau_e": tau_e,
        "mu_log_share": mu_log_share,
        "sd_log_share": sd_log_share,
        "mu_rate": mu_rate,
        "sd_rate": sd_rate,
        "mu_share": mu_share,
        "sd_share": sd_share,
    }


def virality_index(share_count, view_count, interaction_count, stats=None, q=0.05):
    """
    计算 virality_index。
    
    支持：
    - 单个值输入
    - 多个值组成的列表输入
    
    参数：
    - share_count, view_count, interaction_count: 标量或列表
    - stats: 参考参数字典；若为 None 且输入为列表，则自动根据当前列表拟合；
             若输入为单个值，则必须提供 stats
    - q: 平滑常数使用的分位数，默认 0.05

    返回：
    - 若输入是单个值，返回 float
    - 若输入是列表，返回 list[float]
    """
    scalar_input = (
        np.isscalar(share_count)
        and np.isscalar(view_count)
        and np.isscalar(interaction_count)
    )

    s = _to_1d_array(share_count)
    v = _to_1d_array(view_count)
    it = _to_1d_array(interaction_count)

    if not (len(s) == len(v) == len(it)):
        raise ValueError("share_count, view_count, interaction_count 长度必须一致。")

    # 单个值无法自己生成 z-score 的参考分布，因此需要外部传入 stats
    if stats is None:
        if len(s) == 1:
            raise ValueError("单个值计算 virality_index 时，必须提供 stats。")
        stats = fit_virality_stats(s, v, it, q=q)

    tau_v = stats["tau_v"]
    tau_e = stats["tau_e"]

    adj_share_rate = np.where(v > 0, s / (v + tau_v), 0.0)
    adj_share_share = np.where(it > 0, s / (it + tau_e), 0.0)
    log_share = np.log1p(np.clip(s, a_min=0, a_max=None))

    z_log_share = (log_share - stats["mu_log_share"]) / stats["sd_log_share"]
    z_rate = (adj_share_rate - stats["mu_rate"]) / stats["sd_rate"]
    z_share = (adj_share_share - stats["mu_share"]) / stats["sd_share"]

    vi = (z_log_share + z_rate + z_share) / 3.0

    if scalar_input:
        return float(vi[0])
    return vi.tolist()

# %% [markdown]
# ## select q

# %%
rows = []
for q in [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]:
    stats = fit_virality_stats(id_virality_cib['share_count'], id_virality_cib['view_count'], id_virality_cib['interaction_count'], q)
    rows.append({'q': q, **stats})
    id_virality_cib['virality_index'] = virality_index(id_virality_cib['share_count'], id_virality_cib['view_count'], id_virality_cib['interaction_count'], stats, q)
    id_virality_cib.nlargest(20, 'virality_index')[['share_count', 'view_count', 'interaction_count', 'virality_index']].to_csv(f'../../dataset/q/virality_index_top20_{q}.csv')
pd.DataFrame(rows).set_index('q').to_csv('../../dataset/q/stats.csv')

# %% [markdown]
# ## compute virality_index

# %%
q = 0.5
stats = fit_virality_stats(id_virality_cib['share_count'], id_virality_cib['view_count'], id_virality_cib['interaction_count'], q)
id_virality_cib['virality_index'] = virality_index(id_virality_cib['share_count'], id_virality_cib['view_count'], id_virality_cib['interaction_count'], stats, q)

# %% [markdown]
# ## save dataset

# %%
id_virality_index_cib = id_virality_cib[['virality_index']]
id_virality_index_cib.to_csv('../../dataset/id_virality_index_cib.csv')

# %%


# %%


# %%



