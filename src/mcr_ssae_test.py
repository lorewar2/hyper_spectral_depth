import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
import matplotlib.pyplot as plt
import h5py

def main():
    # load data
    A_raw, S_raw = load_data()
    # train code
    model = train(A_raw, S_raw)
    # plot code
    draw_abundence_map(model, A_raw, S_raw)
    return

def train(A_raw, S_raw):
    model = None
    # to do
    return model

def draw_abundence_map(model, A_raw, S_raw):
    # to do 
    return

def load_data():
    # data loading
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
    return A_raw, S_raw

def mstr(ds):   # decode a MATLAB char array (uint16) -> str
    a = np.asarray(ds).ravel()
    return "".join(chr(int(v)) for v in a if v != 0)

def sca(ds):    # read a 1x1 scalar
    return float(np.asarray(ds).ravel()[0])

if __name__ == '__main__':
    main()