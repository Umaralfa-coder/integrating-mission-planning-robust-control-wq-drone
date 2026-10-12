import pickle, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

with open('./parsed.pkl','rb') as f:
    D = pickle.load(f)

csv_epoch = D['csv_epoch']; csv_x = D['csv_x']; csv_y = D['csv_y']
ct = D['ct']
mission_start_epoch = D['mission_start_epoch']
video_start_epoch = D['video_start_epoch']

order = np.argsort(csv_epoch)
csv_epoch_s = csv_epoch[order]; csv_x_s = csv_x[order]; csv_y_s = csv_y[order]

def pos_at(epoch):
    e = np.clip(epoch, csv_epoch_s[0], csv_epoch_s[-1])
    x = np.interp(e, csv_epoch_s, csv_x_s)
    y = np.interp(e, csv_epoch_s, csv_y_s)
    return x, y

# build training arrays, filtering valid DO
epochs = ct['epoch']
do = ct['do']
valid = np.array([v is not None for v in do])
epochs_v = epochs[valid]
do_v = np.array([float(v) for v in do[valid]])
xs_v, ys_v = pos_at(epochs_v)

order2 = np.argsort(epochs_v)
epochs_v = epochs_v[order2]; do_v = do_v[order2]; xs_v = xs_v[order2]; ys_v = ys_v[order2]

print("training points total:", len(epochs_v), "DO range", do_v.min(), do_v.max(), "std", do_v.std())

# spatial grid bounds from full mission CSV path, padded
pad = 3.0
xmin, xmax = csv_x.min()-pad, csv_x.max()+pad
ymin, ymax = csv_y.min()-pad, csv_y.max()+pad
ngrid = 90
gx = np.linspace(xmin, xmax, ngrid)
gy = np.linspace(ymin, ymax, ngrid)
GX, GY = np.meshgrid(gx, gy)
grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=1)

ELL_SPACE = 9.0     # m
ELL_TIME  = 75.0    # s
NOISE     = 0.05 * do_v.std() if do_v.std() > 0 else 1e-3
SIGMA_F2  = max(do_v.var(), 1e-6)

def gp_predict(train_x, train_y, train_t, train_v, cur_t, qx, qy, qt):
    n = len(train_v)
    if n == 0:
        return np.full(len(qx), np.nan)
    mu = train_v.mean()
    dv = train_v - mu
    # training kernel (space+time)
    dX = train_x[:,None]-train_x[None,:]
    dY = train_y[:,None]-train_y[None,:]
    dT = train_t[:,None]-train_t[None,:]
    K = SIGMA_F2*np.exp(-(dX**2+dY**2)/(2*ELL_SPACE**2) - (dT**2)/(2*ELL_TIME**2))
    K += (NOISE**2)*np.eye(n)
    # query-train kernel
    dXq = qx[:,None]-train_x[None,:]
    dYq = qy[:,None]-train_y[None,:]
    dTq = qt - train_t[None,:]
    Kq = SIGMA_F2*np.exp(-(dXq**2+dYq**2)/(2*ELL_SPACE**2) - (dTq**2)/(2*ELL_TIME**2))
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, dv))
    except np.linalg.LinAlgError:
        alpha = np.linalg.solve(K, dv)
    pred = mu + Kq @ alpha
    return pred

outdir = './gp_frames'
os.makedirs(outdir, exist_ok=True)

vmin, vmax = do_v.min(), do_v.max()
vpad = (vmax-vmin)*0.15 if vmax>vmin else 0.05
vmin -= vpad; vmax += vpad

meta = []
N = len(epochs_v)
for i in range(N):
    cur_epoch = epochs_v[i]
    tr_mask = epochs_v <= cur_epoch
    tx = xs_v[tr_mask]; ty = ys_v[tr_mask]; tt = epochs_v[tr_mask]-mission_start_epoch; tv = do_v[tr_mask]
    qt = cur_epoch - mission_start_epoch
    pred = gp_predict(tx, ty, tt, tv, qt, grid_pts[:,0], grid_pts[:,1], qt)
    Z = pred.reshape(GX.shape)

    cur_x, cur_y = pos_at(cur_epoch)
    video_t = cur_epoch - video_start_epoch

    fig, ax = plt.subplots(figsize=(5.6, 3.55), dpi=100)
    Zc = np.clip(Z, vmin, vmax)
    levels = np.linspace(vmin, vmax, 21)
    cf = ax.contourf(GX, GY, Zc, levels=levels, cmap='viridis', vmin=vmin, vmax=vmax, extend='neither')
    ax.plot(tx, ty, '.', color='white', ms=2.5, alpha=0.55, mec='none')
    ax.plot(cur_x, cur_y, 'o', color='#F2A93B', ms=7, mec='black', mew=0.8, zorder=5)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Local X (m)', fontsize=9)
    ax.set_ylabel('Local Y (m)', fontsize=9)
    ax.set_title('GP spatiotemporal estimate — Dissolved Oxygen (live)', fontsize=9.5)
    ax.tick_params(labelsize=8)
    cbar = fig.colorbar(cf, ax=ax, pad=0.03, fraction=0.06)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('DO (mg/L)', fontsize=8, labelpad=6)
    fig.subplots_adjust(left=0.10, right=0.88, top=0.90, bottom=0.13)
    fname = os.path.join(outdir, f'gp_{i:04d}.png')
    fig.savefig(fname)
    plt.close(fig)
    meta.append((video_t, fname))
    if i % 50 == 0:
        print("rendered", i, "/", N, "video_t=", video_t)

import json
with open('./gp_frame_index.json','w') as f:
    json.dump(meta, f)
print("done, frames:", len(meta))
