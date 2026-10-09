import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import coherence
from obspy import Stream, read, read_inventory
from obspy.signal import PPSD

from helper_funcs import extract_psd_time_series

# -----------------------------
# 0) Parameters
# -----------------------------
FS_CPD = 48.0          # samples per day (30-min cadence -> 48 samples/day)
NPERSEG = 96           # Welch segment length = 2 days -> good compromise for 9–10 day record
F_MARKS = [2.0, 4.0]   # semi-diurnal and quarter-diurnal markers in cycles/day
OUT_PNG = "./figures/coherence_psd_vs_tides.png"

# -----------------------------
# 1) Load data
# -----------------------------
station = '03479'
comp = 'Z'
st = Stream()
for d in range(1,10):
    st += read(f'/data/localraid/UEA_nodes_mseed_mV/rotated/5J.{station}..DP{comp}..2025-04-{d:02d}T00:00:00.000000Z.MSEED')
st.merge(method=1, fill_value='interpolate')
ppsd = PPSD(st[0].stats, period_limits=(0.02,100), metadata=read_inventory(f'/data/localraid/UEA_nodes_mseed_mV/stations/5J.{station}..DP{comp}.76_6'))
ppsd.add(st)

target_f = 1.0
psd, times = extract_psd_time_series(ppsd, target_f, True, smooth=False)
psd = pd.DataFrame(psd, index=times, columns=["psd_db"]).sort_index()

h = pd.read_csv("./checkpoints/tidal_heights.csv", header=None, names=["H"])
uv = pd.read_csv("./checkpoints/tidal_currents.csv")

# -----------------------------
# 2) Align lengths and timestamps
# -----------------------------
n = min(len(psd), len(h), len(uv))
psd = psd.iloc[:n]
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
PSD     = psd["psd_db"].values

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
plt.title(f"Coherence of PSD @ {target_f} Hz with tidal predictors")
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
fs = np.array([1/14, 0.1, 1.0, 1.5, 5.0, 10.0])
# fs = 1/np.asarray(ppsd.period_bin_centers)[::-1]
# f_idx = (np.abs(fs - 5.0)).argmin()
# fs = fs[:f_idx]
# vars = ['H', '|H|', '|U|']
# constituents = [1.96, 2.0, 3.86, 3.94, 4.0]

# df = pd.DataFrame(index=pd.MultiIndex.from_product([vars, constituents], names=('vars', 'constituents')), columns=fs)
# df = pd.DataFrame(index=fs, columns=constituents)
df = pd.DataFrame(columns=fs)

for freq in fs:
    psd, times = extract_psd_time_series(ppsd, freq, True, smooth=False)
    psd = pd.DataFrame(psd, index=times, columns=["psd_db"]).sort_index()
    PSD = psd["psd_db"].values

    # fH, Cxy_H = coherence(PSD, H, fs=FS_CPD, nperseg=NPERSEG)
    # fH_abs, Cxy_H_abs = coherence(PSD, H_abs, fs=FS_CPD, nperseg=NPERSEG)
    fU, Cxy_U = coherence(PSD, U_speed, fs=FS_CPD, nperseg=NPERSEG)
    
    # print(f"[{freq:.2f} Hz]", end='')
    # for name, (freqs, Cxy) in {"H": (fH, Cxy_H), "|H|": (fH_abs, Cxy_H_abs), "|U|": (fU, Cxy_U)}.items():
    #     f2, c2 = pick(freqs, Cxy, 2.0)
    #     f4, c4 = pick(freqs, Cxy, 4.0)
    #     df.loc[(name, 2.0), freq] = c2
    #     df.loc[(name, 4.0), freq] = c4
    #     print(f"{name}: {c2:.3f}; {c4:.3f}", end='')
    # print('||')
    # for constituent in constituents:
    #     f, c = pick(fU, Cxy_U, constituent)
    #     df.loc[freq, constituent] = c
    
    df[freq] = Cxy_U
df.index = fU

df.to_csv('./checkpoints/node_coherence.csv', sep=',')
