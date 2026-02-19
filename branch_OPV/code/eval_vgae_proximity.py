import numpy as np
from pathlib import Path

def mahalanobis2(Z, mean, cov_inv):
    D = Z - mean
    return np.einsum('ni,ij,nj->n', D, cov_inv, D)

def whiten(Z, mean, cov):
    # Cholesky (cov must be PD; for shrinkage cov先用 Ledoit-Wolf)
    L = np.linalg.cholesky(cov)
    return np.linalg.solve(L, (Z - mean).T).T, L

def fid_gaussian(m1, c1, m2, c2):
    # simplified FID (no sqrtm numerical guard here for brevity)
    from scipy.linalg import sqrtm
    diff = m1 - m2
    covmean = sqrtm(c1.dot(c2))
    if np.iscomplexobj(covmean): covmean = covmean.real
    return diff.dot(diff) + np.trace(c1 + c2 - 2*covmean)

# === paths ===
root = Path('.')
m = np.load('z_mean.npy')          # [z_dim]
C = np.load('z_cov.npy')           # [z_dim, z_dim]
C_inv = np.linalg.inv(C)

Z_train = np.load('z_bank.npy')    # [N_train, z_dim]
Z_gen   = np.load('z_gen.npy')     # [N_gen, z_dim]

# --- 1) 硬门控统计 ---
# 例：q=0.90 的门；若高斯假设成立，阈值可用 chi2.ppf(0.9, z_dim)
from scipy.stats import chi2
zdim = m.shape[0]
tau = chi2.ppf(0.90, df=zdim)

d2_gen = mahalanobis2(Z_gen, m, C_inv)
pass_mask = d2_gen <= tau
pass_rate = pass_mask.mean()
d2_pass = d2_gen[pass_mask]

print(f"[HARD GATE] tau(chi2,90%)={tau:.2f}, pass_rate={pass_rate:.2%}, "
      f"median_d2_pass={np.median(d2_pass):.2f}, p90_d2_pass={np.percentile(d2_pass,90):.2f}")

# --- 2) 软约束统计（中心化密度/距离）---
d2_train = mahalanobis2(Z_train, m, C_inv)
mu_d2, sd_d2 = d2_train.mean(), d2_train.std()
d2_gen_z = (d2_gen - mu_d2) / (sd_d2 + 1e-8)

print(f"[SOFT] d2_gen z-score: mean={d2_gen_z.mean():.3f}, "
      f"median={np.median(d2_gen_z):.3f}, p90={np.percentile(d2_gen_z,90):.3f}")

# 把 -0.5*d2 做成“中心化密度分”，均值应 ~0（更便于和属性奖励同量纲）
s_train = -0.5*d2_train
s_gen   = -0.5*d2_gen
s0 = (s_gen - s_train.mean()) / (s_train.std() + 1e-8)
print(f"[SOFT] centered log-density z-score: mean={s0.mean():.3f}")

# --- 3) 分布相似度（FID/KL）---
m_gen = Z_gen.mean(0)
C_gen = np.cov(Z_gen.T)
fid = fid_gaussian(m_gen, C_gen, m, C)
print(f"[DIST] FID(gaussian)={fid:.3f}")

# KL(train || gen), KL(gen || train) for Gaussians
from numpy.linalg import slogdet
def kl_gauss(m0,C0,m1,C1):
    k = m0.shape[0]
    sign0, logdet0 = slogdet(C0)
    sign1, logdet1 = slogdet(C1)
    C1_inv = np.linalg.inv(C1)
    diff = (m1 - m0)[:,None]
    term = np.trace(C1_inv.dot(C0)) + diff.T.dot(C1_inv).dot(diff) - k + (logdet1 - logdet0)
    return 0.5*float(term)

print(f"[DIST] KL(train||gen)={kl_gauss(m, C, m_gen, C_gen):.3f}  "
      f"KL(gen||train)={kl_gauss(m_gen, C_gen, m, C):.3f}")

# --- 4) 白化检验（贴近N(0,I)吗）---
Zt_w, L = whiten(Z_train, m, C)
Zg_w = np.linalg.solve(L, (Z_gen - m).T).T
mw = Zg_w.mean(0)
Cw = np.cov(Zg_w.T)
err_mean = np.linalg.norm(mw)       # 应小
err_cov  = np.linalg.norm(Cw - np.eye(zdim), 'fro')  # 应小
d2w = np.sum(Zg_w**2,1)             # 应近似 ~ chi2(z_dim)
print(f"[WHITE] mean_norm={err_mean:.3f}, cov_Fro_err={err_cov:.3f}, "
      f"median(||z_w||^2)={np.median(d2w):.2f}, p90={np.percentile(d2w,90):.2f}, "
      f"chi2_med~{chi2.ppf(0.5,zdim):.2f}")
