import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import coherence
from datetime import datetime, timedelta

from helper_funcs import *
from weather import *


# -----------------------------
# 0) Parameters
# -----------------------------
FS_CPD = 48.0          # samples per day (30-min cadence -> 48 samples/day)
NPERSEG = 1008           # Welch segment length = 21 days
F_MARKS = [2.0, 4.0]   # semi-diurnal and quarter-diurnal markers in cycles/day
OUT_PNG = "./figures/coherence_das_psd_vs_tides.png"

# -----------------------------
# 1) Load data
# -----------------------------
file_path = '/data/localraid/saved_specs/600s_window/2025-04-01 00:00:25__2025-05-01 00:00:25_f0.01:49.9_1475_spec.txt'
spec = np.loadtxt(file_path, delimiter=',')
f_size, t_size = spec.shape

t_start = datetime.strptime(file_path.split('/')[-1].split('_')[0], '%Y-%m-%d %H:%M:%S')
t_end   = datetime.strptime(file_path.split('/')[-1].split('_')[2], '%Y-%m-%d %H:%M:%S')
t_bin_length = (t_end - t_start).total_seconds() / t_size
time_axis = [t_start + timedelta(seconds=t_bin_length * j) for j in range(t_size)]

target_f = 1.1
f_bins = np.linspace(0.0, 50.016666666666666, f_size)
idx = (np.abs(f_bins - target_f)).argmin()
psd_series = pd.Series(spec[idx,:], index=pd.to_datetime(time_axis))
psd_series = psd_series[f'{t_start.date()} 00:00:00':f'{t_end.date()} 23:59:59']

lons = [1.402138]; lats = [52.94651]
times = pd.date_range(f'{t_start.date()} 00:00:00', f'{t_end.date()} 23:00:00', freq=timedelta(minutes=30))
if os.path.isfile(f'./checkpoints/tidal_heights_{t_start}_{t_end}.csv'):
    h = pd.read_csv(f'./checkpoints/tidal_heights_{t_start}_{t_end}.csv', index_col=False, header=0, delimiter=',', names=["H"])
    uv = pd.read_csv(f'./checkpoints/tidal_currents_{t_start}_{t_end}.csv', index_col=False, header=0, delimiter=',')
else:
    h, uv = get_tidal_data(lons=lons, lats=lats, times=times)
    h = pd.DataFrame(h, columns=['H'])
    uv = pd.DataFrame(uv, columns=['u','v'])
    h.to_csv(f'./checkpoints/tidal_heights_{t_start}_{t_end}.csv', index=False)
    uv.to_csv(f'./checkpoints/tidal_currents_{t_start}_{t_end}.csv', index=False)

window_seconds = 30 * 60
half_window = window_seconds / 2.0
targets = pd.DatetimeIndex(times)
vals = psd_series.values
idx = psd_series.index

cosine_30min = []
for t in targets:
    offs = (idx - t).total_seconds()
    sel = np.abs(offs) <= half_window
    if not np.any(sel):
        cosine_30min.append(np.nan)
        continue
    offs_sel = offs[sel]
    samples = vals[sel].astype(float)
    w = 0.5 * (1.0 + np.cos(2.0 * np.pi * offs_sel / window_seconds))
    good = np.isfinite(samples) & (w > 0)
    if not np.any(good):
        cosine_30min.append(np.nan)
        continue
    w = w[good]
    s = samples[good]
    cosine_30min.append(np.sum(w * s) / np.sum(w))

psd_series = psd_series.interpolate()
psd_30min_cosine = pd.Series(cosine_30min, index=targets)


# -----------------------------
# 2) Align lengths and timestamps
# -----------------------------
n = min(len(psd_30min_cosine), len(h), len(uv))
psd = psd_30min_cosine.iloc[:n]
h   = h.iloc[:n]
uv  = uv.iloc[:n]

h.index  = psd.index
uv.index = psd.index

# -----------------------------
# 3) Build predictors |H| and |U|
# -----------------------------
H = h["H"].values
H_abs   = h["H"].abs().values
U_speed = np.sqrt(uv["u"].values**2 + uv["v"].values**2)/100
PSD     = psd_30min_cosine.values

# -----------------------------
# 4) Coherence spectra
# -----------------------------
fH, Cxy_H = coherence(PSD, H, fs=FS_CPD, nperseg=NPERSEG)
fH_abs, Cxy_H_abs = coherence(PSD, H_abs, fs=FS_CPD, nperseg=NPERSEG)
fU, Cxy_U = coherence(PSD, U_speed, fs=FS_CPD, nperseg=NPERSEG)

# -----------------------------
# 5) Plot
# -----------------------------
plt.figure(figsize=(9, 4.8))
plt.plot(fH, Cxy_H, label="PSD @ 1 Hz vs H", lw=2)
plt.plot(fH_abs, Cxy_H_abs, label="PSD @ 1 Hz vs |H|", lw=2)
plt.plot(fU, Cxy_U, label="PSD @ 1 Hz vs |U|", lw=2)

for fm in F_MARKS:
    plt.axvline(fm, color="crimson", ls="--", lw=1)

plt.xlim(0, 6)
plt.ylim(0, 1)
plt.xlabel("Modulation frequency (cycles/day)")
plt.ylabel("Magnitude-squared coherence")
plt.title(f"Coherence of DAS PSD @ {target_f} Hz with tidal predictors")
plt.legend(frameon=False)
plt.tight_layout()
# plt.savefig(OUT_PNG, dpi=200)
# plt.show()

# -----------------------------
# 6) Read off the values at 2 and 4 cpd (optional)
# -----------------------------
def pick(freqs, Cxy, f0):
    idx = np.argmin(np.abs(freqs - f0))
    return float(freqs[idx]), float(Cxy[idx])

for name, (freqs, Cxy) in {"H": (fH, Cxy_H), "|H|": (fH_abs, Cxy_H_abs), "|U|": (fU, Cxy_U)}.items():
    f2, c2 = pick(freqs, Cxy, 2.0)
    f4, c4 = pick(freqs, Cxy, 4.0)
    print(f"{name}: coherence @ {f2:.2f} cpd = {c2:.3f}; @ {f4:.2f} cpd = {c4:.3f}")



print("\n\n-----------------\n\n")

fs = np.array([1/50, 1/14, 0.1, 1.0, 1.5, 5, 10])
# fs = np.linspace(0.0, 50.016666666666666, f_size)
# f_idx = (np.abs(fs - 5.0)).argmin()
# fs = fs[:f_idx]
# vars = ['|U|']
# constituents = [1.96, 2.0, 3.86, 3.94, 4.0]

# df = pd.DataFrame(index=pd.MultiIndex.from_product([vars, constituents], names=('vars', 'constituents')), columns=fs)
# df = pd.DataFrame(index=fs, columns=constituents)
for cha in [750, 1475]:
    df = pd.DataFrame(columns=fs)
    t0 = datetime(year=2024, month=4, day=1); t1 = datetime(year=2025, month=7, day=31)
    # t0 = datetime(year=2024, month=10, day=1); t1 = datetime(year=2025, month=2, day=28)

    lons = [1.402138]; lats = [52.94651]
    times = pd.date_range(f'{t0.date()} 00:00:00', f'{t1.date()} 23:30:00', freq=timedelta(minutes=30))
    if os.path.isfile(f'./checkpoints/tidal_heights_{t0}_{t1}.csv'):
        h = pd.read_csv(f'./checkpoints/tidal_heights_{t0}_{t1}.csv', index_col=False, header=0, delimiter=',', names=["H"])
        uv = pd.read_csv(f'./checkpoints/tidal_currents_{t0}_{t1}.csv', index_col=False, header=0, delimiter=',')
    else:
        h, uv = get_tidal_data(lons=lons, lats=lats, times=times)
        h = pd.DataFrame(h, columns=['H'])
        uv = pd.DataFrame(uv, columns=['u','v'])
        h.to_csv(f'./checkpoints/tidal_heights_{t0}_{t1}.csv', index=False)
        uv.to_csv(f'./checkpoints/tidal_currents_{t0}_{t1}.csv', index=False)

    files = get_spec_files(f'/data/localraid/saved_specs/600s_window/', t0, t1+timedelta(days=1), cha)
    for i, freq in enumerate(fs):
        cosine_30min = []
        for file_path, t_start in files: 
            spec = np.loadtxt(file_path, delimiter=',')
            f_size, t_size = spec.shape
            f_bins = np.linspace(0.0, 50.016666666666666, f_size)
            f_idx = (np.abs(f_bins - freq)).argmin()
            
            t_start = datetime.strptime(file_path.split('/')[-1].split('_')[0], '%Y-%m-%d %H:%M:%S')
            t_end   = datetime.strptime(file_path.split('/')[-1].split('_')[2], '%Y-%m-%d %H:%M:%S')
            t_bin_length = (t_end - t_start).total_seconds() / t_size
            time_axis = [t_start + timedelta(seconds=t_bin_length * j) for j in range(t_size)]
            print(t_start, t_end)
            
            psd_series = pd.Series(spec[f_idx,:], index=pd.to_datetime(time_axis))
            psd_series = psd_series[f'{t_start.date()} 00:00:00':f'{t_end.date()} 23:59:59']
            vals = psd_series.values
            idx = psd_series.index
            
            targets = pd.DatetimeIndex(pd.date_range(f'{t_start.date()} 00:00:00', f'{(t_end-timedelta(days=1)).date()} 23:30:00', freq=timedelta(minutes=30)))
            for t in targets:
                offs = (idx - t).total_seconds()
                sel = np.abs(offs) <= half_window
                if not np.any(sel):
                    cosine_30min.append(np.nan)
                    continue
                offs_sel = offs[sel]
                samples = vals[sel].astype(float)
                w = 0.5 * (1.0 + np.cos(2.0 * np.pi * offs_sel / window_seconds))
                good = np.isfinite(samples) & (w > 0)
                if not np.any(good):
                    cosine_30min.append(np.nan)
                    continue
                w = w[good]
                s = samples[good]
                cosine_30min.append(np.sum(w * s) / np.sum(w))
        
        # print(cosine_30min)
        targets = pd.DatetimeIndex(pd.date_range(f'{t0.date()} 00:00:00', f'{(t_end-timedelta(days=1)).date()} 23:30:00', freq=timedelta(minutes=30)))
        psd_30min_cosine = pd.Series(cosine_30min, index=targets)
        
        # n = min(len(psd_30min_cosine), len(h), len(uv))
        # uv  = uv.iloc[:n]
        U_speed = np.sqrt(uv["u"].values**2 + uv["v"].values**2)/100
        print(psd_30min_cosine.shape)
        print(U_speed.shape)
        mask = ~np.isnan(psd_30min_cosine)
        psd_30min_cosine = psd_30min_cosine[mask]
        U_speed = U_speed[mask]
        print(psd_30min_cosine.shape)
        print(U_speed.shape)
        
        PSD = psd_30min_cosine.values

        # fH, Cxy_H = coherence(PSD, H, fs=FS_CPD, nperseg=NPERSEG)
        # fH_abs, Cxy_H_abs = coherence(PSD, H_abs, fs=FS_CPD, nperseg=NPERSEG)
        fU, Cxy_U = coherence(PSD, U_speed, fs=FS_CPD, nperseg=NPERSEG, noverlap=int(NPERSEG/2))
        
        # print(f"[{freq:.2f} Hz]", end='')
        # for name, (freqs, Cxy) in {"H": (fH, Cxy_H), "|H|": (fH_abs, Cxy_H_abs), "|U|": (fU, Cxy_U)}.items():
        #     f2, c2 = pick(freqs, Cxy, 2.0)
        #     f4, c4 = pick(freqs, Cxy, 4.0)
        #     df.loc[(name, 2.0), freq] = c2
        #     df.loc[(name, 4.0), freq] = c4
            # print(f"{name}: {c2:.3f}; {c4:.3f}", end='')
        # print('||')
        # for constituent in constituents:
        #     f, c = pick(fU, Cxy_U, constituent)
        #     df.loc[freq, constituent] = c
        
        df[freq] = Cxy_U
    df.index = fU

    df.to_csv(f'./checkpoints/das_coherence_{cha}.csv', sep=',')





#### PLOT CODE: 
import pandas as pd
import matplotlib.pyplot as plt
def plot(cha):
    df = pd.read_csv(f'../Nodes_Analysis/checkpoints/das_coherence_{cha}.csv', index_col=0)
    df = df.drop(columns=['0.02'])
    df = df.rename(columns={'0.07142857142857142':'0.07 Hz', '0.1': '0.1 Hz', '1.0': '1.0 Hz', '1.5': '1.5 Hz', '5.0': '5.0 Hz', '10.0':'10.0 Hz'})
    ax = df.plot(cmap='viridis')
    ax.set_xlim(0,6)
    ax.set_ylim(0,0.8)
    ax.axvline(2.0, c='k', ls='--', alpha=0.5)
    ax.text(2.0, 0.99, 'S2', ha='left', va='top', rotation=90, transform=ax.get_xaxis_transform())
    ax.axvline(1.93, c='k', ls='--', alpha=0.5)
    ax.text(1.93, 0.99, 'M2', ha='right', va='top', rotation=90, transform=ax.get_xaxis_transform())
    ax.axvline(3.86, c='k', ls='--', alpha=0.5)
    ax.text(3.86, 0.99, 'M4', ha='right', va='top', rotation=90, transform=ax.get_xaxis_transform())
    ax.set_xlabel('Cycles/day')
    ax.set_ylabel('Magnitude-squared coherence')
    plt.tight_layout()
    plt.show()
