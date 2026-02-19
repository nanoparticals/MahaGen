
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.linalg import eigh
from math import sqrt, pi

def sqrtm_psd(M):
    vals, vecs = eigh(M)
    vals[vals < 0] = 0.0
    return (vecs * np.sqrt(vals)) @ vecs.T

def fid_gaussian(m1, C1, m2, C2):
    diff = m1 - m2
    covmean = sqrtm_psd(C1 @ C2)
    return float(diff @ diff + np.trace(C1 + C2 - 2*covmean))

def w2_gaussian(m1, C1, m2, C2):
    diff = m1 - m2
    term_mean = diff @ diff
    C1_sqrt = sqrtm_psd(C1)
    mid = C1_sqrt @ C2 @ C1_sqrt
    term_cov = np.trace(C1 + C2 - 2*sqrtm_psd(mid))
    return float(term_mean + term_cov)

def bhattacharyya_gaussian(m1, C1, m2, C2):
    # Bhattacharyya coefficient between Gaussians (multivariate)
    C = 0.5 * (C1 + C2)
    diff = (m2 - m1)
    term1 = 0.125 * diff @ np.linalg.inv(C) @ diff
    detC  = np.linalg.det(C)
    det1  = np.linalg.det(C1)
    det2  = np.linalg.det(C2)
    # guard small determinants
    detC  = max(detC, 1e-12)
    det1  = max(det1, 1e-12)
    det2  = max(det2, 1e-12)
    term2 = 0.5 * np.log(detC / np.sqrt(det1 * det2))
    # Bhattacharyya distance
    DB = term1 + term2
    # coefficient = exp(-DB), in [0,1]
    return float(np.exp(-DB)), float(DB)

def pca_components_on_train(Z_train, k=2):
    X = Z_train - Z_train.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    P = Vt[:k].T  # [dim,k]
    mean = Z_train.mean(0)
    return P, mean

def project(Z, P, mean_ref):
    return (Z - mean_ref) @ P

def ellipse_params(C2):
    vals, vecs = eigh(C2)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:,order]
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    return vals, angle

def gaussian_pdf_2d(X, Y, mean, cov):
    # X,Y are meshgrid; mean is (2,), cov is (2,2)
    pos = np.stack([X, Y], axis=-1)  # [..., 2]
    invC = np.linalg.inv(cov)
    diff = pos - mean
    # Mahalanobis
    M = diff[...,0]* (invC[0,0]*diff[...,0] + invC[0,1]*diff[...,1]) + \
        diff[...,1]* (invC[1,0]*diff[...,0] + invC[1,1]*diff[...,1])
    detC = max(np.linalg.det(cov), 1e-12)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(detC))
    return norm * np.exp(-0.5*M)

def plot_latent_overlap(z_train_path, z_gen_path, out_png):
    Zt = np.load(z_train_path)
    Zg = np.load(z_gen_path)

    # stats in full space
    mt = Zt.mean(0); Ct = np.cov(Zt.T)
    mg = Zg.mean(0); Cg = np.cov(Zg.T)

    # metrics (Gaussian)
    FID = fid_gaussian(mt, Ct, mg, Cg)
    W2  = w2_gaussian(mt, Ct, mg, Cg)
    BC, DB = bhattacharyya_gaussian(mt, Ct, mg, Cg)

    # PCA to 2D on training set
    P, mref = pca_components_on_train(Zt, k=2)
    Zt2 = project(Zt, P, mref)
    Zg2 = project(Zg, P, mref)
    mt2 = project(mt[None,:], P, mref)[0]
    mg2 = project(mg[None,:], P, mref)[0]
    Ct2 = P.T @ Ct @ P
    Cg2 = P.T @ Cg @ P

    # grid for overlap (numeric integral of min of the two 2D Gaussians)
    all2 = np.vstack([Zt2, Zg2])
    xmin, ymin = all2.min(0) - 1.0
    xmax, ymax = all2.max(0) + 1.0
    nx = ny = 300
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)
    Pt = gaussian_pdf_2d(X, Y, mt2, Ct2)
    Pg = gaussian_pdf_2d(X, Y, mg2, Cg2)
    Pmin = np.minimum(Pt, Pg)
    # integral via Riemann sum
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    overlap_integral = float((Pmin.sum() * dx * dy))

    # scatter (downsample)
    def downsample(Z2, n=4000):
        if Z2.shape[0] > n:
            idx = np.random.default_rng(0).choice(Z2.shape[0], size=n, replace=False)
            return Z2[idx]
        return Z2

    Zt2s = downsample(Zt2, 4000)
    Zg2s = downsample(Zg2, 4000)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(Zt2s[:,0], Zt2s[:,1], s=6, alpha=0.3, label="Train (PCA2)")
    ax.scatter(Zg2s[:,0], Zg2s[:,1], s=6, alpha=0.3, label="Generated/Test (PCA2)")

    # 1σ/2σ ellipses
    from matplotlib.patches import Ellipse
    def add_ellipse(ax, mean2, C2, label_prefix):
        vals, vecs = eigh(C2)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:,order]
        angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
        for nsig in [1.0, 2.0]:
            width = 2*nsig*np.sqrt(vals[0])
            height = 2*nsig*np.sqrt(vals[1])
            e = Ellipse(xy=mean2, width=width, height=height, angle=angle,
                        fill=False, lw=2, alpha=0.9)
            ax.add_patch(e)
        ax.annotate(f"{label_prefix} 2σ", xy=mean2)

    add_ellipse(ax, mt2, Ct2, "Train")
    add_ellipse(ax, mg2, Cg2, "Gen/Test")

    # Filled contour for min(Pt, Pg): shows "overlap" visually
    cs = ax.contourf(X, Y, np.minimum(Pt, Pg), levels=8, alpha=0.4)

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"Latent overlap (Gaussian approx)\n"
                 f"FID={FID:.3f} | W2={W2:.3f} | W={sqrt(W2):.3f} | "
                 f"Bhattacharyya coef={BC:.3f} | Overlap integral≈{overlap_integral:.3f}")
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    return {
        "FID_Gaussian": FID,
        "W2_Gaussian": W2,
        "W_Gaussian": sqrt(W2),
        "Bhattacharyya_coefficient": BC,
        "Bhattacharyya_distance": DB,
        "OverlapIntegral_minPDF_2D": overlap_integral
    }

if __name__ == "__main__":
    root = Path("/mnt/data")
    zt = root / "z_bank.npy"
    zg = root / "z_gen.npy"
    if not zt.exists() or not zg.exists():
        # quick demo with mock data if files missing
        rng = np.random.default_rng(0)
        dim = 16
        A = rng.normal(size=(dim, dim)); C = A @ A.T / dim + 0.2*np.eye(dim)
        m = rng.normal(size=(dim,))
        Zt = rng.multivariate_normal(m, C, size=5000)
        m2 = m + 0.3 * rng.normal(size=(dim,))
        C2 = C.copy(); C2[0,0] *= 1.6
        Zg = rng.multivariate_normal(m2, C2, size=5000)
        np.save(root / "z_bank.npy", Zt)
        np.save(root / "z_gen.npy", Zg)
        zt = root / "z_bank.npy"
        zg = root / "z_gen.npy"
    metrics = plot_latent_overlap(zt, zg, root / "latent_overlap.png")
    print(metrics)
