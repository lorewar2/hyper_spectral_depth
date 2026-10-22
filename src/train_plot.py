import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
import matplotlib.pyplot as plt
import h5py

from model import SpectralModel

def main():
    #plot_depth_csv('depth_per_pixel.csv', drange=[-3, -2.5])
    #plot_depth_csv('depth_per_pixel.csv', drange=[-2.5, -2.0])
    plot_depth_csv('depth_per_pixel.csv', drange=[-2, -1.5])
    plot_depth_csv('depth_per_pixel.csv', drange=[-1.5, -1])
    plot_depth_csv('depth_per_pixel.csv', drange=[-1, -0.5])
    plot_depth_csv('depth_per_pixel.csv', drange=[-0.5, 0])
    plot_depth_csv('depth_per_pixel.csv', drange=[0, 0.3])
    plot_depth_csv('depth_per_pixel.csv', drange=[0.3, 0.6])
    #plot_depth_csv('depth_per_pixel.csv', drange=[0.6, 0.9])
    #plot_depth_csv('depth_per_pixel.csv', drange=[1, 1.5])
    #plot_depth_csv('depth_per_pixel.csv', drange=[1.5, 2.0])
    #plot_depth_csv('depth_per_pixel.csv', drange=[2.0, 2.5])
    return

def plot_depth_csv(csv_path, drange=None, invert=False,
                   vmin=None, vmax=None, cmap='viridis',
                   save_path=None, title=None):
    """
    Load a per-pixel depth CSV (columns: row, col, depth) and plot the depth map.

    drange : (lo, hi) or None
        If given, only pixels whose depth is within [lo, hi] are shown in color;
        all other pixels are greyed out (but still drawn). Set invert=True to
        flip it: grey out the [lo, hi] band and show everything else.
    vmin, vmax : color-scale limits (default: autoscale on the shown pixels)
    cmap       : matplotlib colormap name
    save_path  : if given, save the figure there
    """
    # --- load ---
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)   # row, col, depth
    rows = data[:, 0].astype(int)
    cols = data[:, 1].astype(int)
    depth = data[:, 2]

    # --- rebuild the image grid (order-independent) ---
    nR, nC = rows.max() + 1, cols.max() + 1
    grid = np.full((nR, nC), np.nan)
    grid[rows, cols] = depth

    # --- apply the depth-range filter ---
    disp = grid.copy()
    if drange is not None:
        lo, hi = drange
        in_range = (grid >= lo) & (grid <= hi)
        keep = ~in_range if invert else in_range
        disp = np.where(keep, grid, np.nan)       # dropped pixels -> nan (greyed)

    # --- plot ---
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad('lightgrey')                 # masked / missing pixels
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(disp, cmap=cmap_obj, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label='latent depth  z = f_S(x, y)')
    if title is None:
        title = ('Depth map' if drange is None else
                 f'Depth {"outside" if invert else "within"} [{drange[0]}, {drange[1]}]')
    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
    return disp

def test():
    # data loading
    set_seeds(0)
    file_path       = "./data/Plant1_L1_13.dspkd"
    SUBTRACT_DARK   = False      # subtract stored dark model (set False if preproc already did it)
    APPLY_CORRECTION = False     # apply per-channel spectral-response correction
    with h5py.File(file_path, "r") as f:
        data   = np.asarray(f["data"], dtype=np.float64)             # (pixels, channels)
        wl     = np.asarray(f["header/channelWavelengths"]).ravel()  # (channels,) nm
        corr   = np.asarray(f["header/correctionFactor"]).ravel()    # (channels,)
        dark_S = np.asarray(f["header/darkPureSpectra"])             # (2, channels)
        dark_C = np.asarray(f["header/darkConcentrations"])          # (pixels, 2)
        nRows  = int(round(sca(f["header/nRows"])))
        nCols  = int(round(sca(f["header/nCols"])))
        nWav   = int(round(sca(f["header/nWavelengths"])))
        nFr    = int(round(sca(f["header/nFrames"])))
        maxC   = int(round(sca(f["header/maxComponents"])))
        preproc = mstr(f["header/preprocessType"])
        fname   = mstr(f["header/Filename"])

    print(f"file           : {fname}")
    print(f"preprocessType : {preproc}")
    print(f"data           : {data.shape}  (pixels x channels)")
    print(f"grid           : {nRows} x {nCols} = {nRows*nCols}   nFrames={nFr}  nWav={nWav}")
    print(f"maxComponents  : {maxC}")
    print(f"wavelengths    : {wl.min():.1f}-{wl.max():.1f} nm across {wl.size} channels")

    assert data.shape[0] == nRows * nCols, (
        "pixel count != nRows*nCols -- likely multiple frames; reshape as "
        "(nRows, nCols, nFr, nWav) and pick a frame before unmixing.")
    assert data.shape[1] == wl.size, "channel count mismatch"

    # preprocessing
    D = data.copy()
    if SUBTRACT_DARK:
        D = D - dark_C @ dark_S
    if APPLY_CORRECTION:
        D = D * corr[None, :]
    D = np.clip(D, 0.0, None)

    cube = D.reshape(nRows, nCols, nWav, order="F")

    A_raw = cube.reshape(nRows * nCols, nWav).astype('float32')          # (N, C) spectra
    rr, cc = np.meshgrid(np.arange(nRows), np.arange(nCols), indexing='ij')
    S_raw = np.stack([cc.ravel(), rr.ravel()], axis=1).astype('float32')  # (N, 2) = (x=col, y=row)

    # scale coords + spectra (fit on the full image)
    s_scaler = preprocessing.StandardScaler().fit(S_raw)
    a_scaler = preprocessing.StandardScaler().fit(A_raw)
    S = torch.tensor(s_scaler.transform(S_raw), dtype=torch.float32)
    A = torch.tensor(a_scaler.transform(A_raw), dtype=torch.float32)

    # train
    model, loss_list = train(
        S, A,
        S_hidden_list=[2560, 2560, 2560, 2560], A_hidden_list=[2560, 2560, 2560, 2560], latent_dim=1,
        epochs=50_000, batch_size=None, optim='adam', lr=1e-4, seed=0,
    )

   # depth
    depth = get_depth(model, S)[:, 0]            # (N,)
    depth_img = depth.reshape(nRows, nCols)      # C-order, matches how A_raw/S_raw were built

    # ---- save depth for each pixel to CSV ----
    # S_raw[:, 0] = col (x), S_raw[:, 1] = row (y); same pixel order as `depth`
    out = np.column_stack([S_raw[:, 1], S_raw[:, 0], depth])   # row, col, depth
    np.savetxt('depth_per_pixel.csv', out, delimiter=',',
               header='row,col,depth', comments='', fmt=['%d', '%d', '%.8f'])
    print(f'wrote depth_per_pixel.csv  ({out.shape[0]} pixels)')

    # plot
    plt.figure(figsize=(5, 5))
    im = plt.imshow(depth_img, cmap='viridis')
    plt.colorbar(im, label='latent depth  z = f_S(x, y)')
    plt.title('Spatial embedding (depth) map')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('depth_map.png', dpi=150)
    plt.show()
    return

def mstr(ds):   # decode a MATLAB char array (uint16) -> str
    a = np.asarray(ds).ravel()
    return "".join(chr(int(v)) for v in a if v != 0)

def sca(ds):    # read a 1x1 scalar
    return float(np.asarray(ds).ravel()[0])

@torch.no_grad()
def get_depth(model, S, device='cpu'):
    """The latent_dim=1 'depth' z = f_S(x, y) for every coordinate -> (N, latent_dim)."""
    model = model.to(device).eval()
    return model.spatial_embedding(S.to(device)).cpu().numpy()

def set_seeds(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def get_depth(model, S, device='cpu'):
    """The 1-D spatial latent ('depth') z = f_S(x, y) for every coordinate -> (N, latent_dim)."""
    model = model.to(device).eval()
    return model.spatial_embedding(S.to(device)).cpu().numpy()

def load_rescale_input_data(S, A):
    """Standard-scale coords S (N x 2) and spectra A (N x C).
    Returns tensors plus the fitted scalers (needed to invert at eval time)."""
    assert S.shape[0] == A.shape[0], 'ERROR: S and A have different N!'
    s_scaler = preprocessing.StandardScaler().fit(S)
    a_scaler = preprocessing.StandardScaler().fit(A)
    S_torch = torch.tensor(s_scaler.transform(S), dtype=torch.float32)
    A_torch = torch.tensor(a_scaler.transform(A), dtype=torch.float32)
    return S_torch, A_torch, s_scaler, a_scaler


def train(S, A,
          model=None, S_hidden_list=None, A_hidden_list=None,
          activation_fn=nn.ReLU(), latent_dim=1,
          epochs=1000, batch_size=None, checkpoint=100, save_dir=None,
          loss_reduction='mean', optim='adam', lr=1e-3, weight_decay=0,
          momentum=0, seed=0, save_final=False, device=None):
    """
    Train coordinate -> spectrum model.
    S : torch.Tensor (N x 2)  pixel coordinates
    A : torch.Tensor (N x C)  spectra (one row per pixel)
    """
    set_seeds(seed)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')

    N, C = A.shape
    if model is None:
        model = SpectralModel(C, S_hidden_list, A_hidden_list,
                              activation_fn=activation_fn, latent_dim=latent_dim)
    model = model.to(device)
    S, A = S.to(device), A.to(device)

    if optim == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim == 'adagrad':
        opt = torch.optim.Adagrad(model.parameters(), weight_decay=weight_decay)
    else:
        raise ValueError(f'unknown optimizer: {optim}')

    loss_function = nn.MSELoss(reduction=loss_reduction)
    loss_list = np.zeros(epochs)

    for epoch in range(epochs):
        if epoch % checkpoint == 0:
            print(f'epoch: {epoch} loss: {loss_list[epoch]}')
            if save_dir is not None:
                torch.save(model, f'{save_dir}/model_epoch_{epoch}.pt')

        if batch_size is not None:
            perm = torch.randperm(N, device=device)
            for i in range(0, N, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = loss_function(model(S[idx]), A[idx])
                loss.backward()
                opt.step()
                loss_list[epoch] += loss.item()
                print(loss)
        else:
            opt.zero_grad()
            loss = loss_function(model(S), A)
            loss.backward()
            opt.step()
            loss_list[epoch] += loss.item()
            print(loss)

    if save_final and save_dir is not None:
        torch.save(model, f'{save_dir}/final_model.pt')
        np.savetxt(f'{save_dir}/loss_list.txt', loss_list)

    return model, loss_list

if __name__ == '__main__':
    main()
    