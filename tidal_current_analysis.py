# fetch_and_compare_tide_currents.py
import numpy as np, pandas as pd, xarray as xr, requests, io
from datetime import datetime, timedelta
from scipy.signal import butter, filtfilt, coherence, detrend, hilbert
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import Stream, read, read_inventory
from obspy.signal import PPSD
import pyTMD
from pyTMD.io import FES
from helper_funcs import *

lat, lon = 52.902596, 1.378968
t0 = '2025-04-01'
t1 = '2025-04-10'
dt_minutes = 30
psd_cadence = f'{dt_minutes}min'

station = '03479'
comp = 'Z'
st = Stream()
for d in range(1,10):
    st += read(f'/data/localraid/UEA_nodes_mseed_mV/rotated/5J.{station}..DP{comp}..2025-04-{d:02d}T00:00:00.000000Z.MSEED')
st.merge(method=1, fill_value='interpolate', interpolation_samples=10)
# st.detrend('linear')
# st.taper(max_percentage=0.05, type='cosine')
# st.filter('bandpass', freqmin=0.2, freqmax=5.0, corners=4, zerophase=True)
ppsd = PPSD(st[0].stats, period_limits=(0.02,100), metadata=read_inventory(f'/data/localraid/UEA_nodes_mseed_mV/stations/5J.{station}..DP{comp}.76_6'))
ppsd.add(st)
psd_series, times = extract_psd_time_series(ppsd, 1, True, smooth=False)
# ppsd.plot_spectrogram(clim=[-180, -100], cmap='jet', yaxis_frequency=True, show=True)

tidal_df = pd.read_csv('/local/gfs19eku/Documents/NoisePy4DAS/results/checkpoints/CRO_final.csv', parse_dates=True, index_col=0)
dt = tidal_df.index.to_series().diff().dt.total_seconds()
times = pd.to_datetime(times)
match_idx = tidal_df.index.isin(times)
tidal_df = tidal_df.loc[match_idx, :]
tidal_series = tidal_df['ASLVBG02'].reindex(psd_series.index)

lons = [1.402138]
lats = [52.94651]

# tidal_h, tidal_uv = get_tidal_data(lons=lons, lats=lats, times=times)
# pd.DataFrame(tidal_h).to_csv('./tidal_heights.csv', index=False)
# pd.DataFrame(tidal_uv).to_csv('./tidal_currents.csv', index=False)
tidal_h = pd.read_csv('./checkpoints/tidal_heights.csv', index_col=False, header=0, delimiter=',')['0'].to_numpy()
tidal_uv = pd.read_csv('./checkpoints/tidal_currents.csv', index_col=False, header=0, delimiter=',')
curr_speed = np.sqrt(tidal_uv['u']**2 + tidal_uv['v']**2)/100
if type(curr_speed) == pd.Series:   curr = curr_speed; curr.index = times
else:                               curr = pd.Series(curr_speed, index=times)

# fig, ax = plt.subplots(1,1,figsize=(12,4))
# ax.plot(times, curr)
# ax.set_ylabel('FES Current speed (cm/s)', color='C0')
# ax1 = ax.twinx()
# ax1.plot(times, tidal_h, color='C1')
# ax1.set_ylabel('FES Tidal Height', color='C1')
# ax.grid(axis='x')

# fig, (ax,ax1) = plt.subplots(2,1,figsize=(12,4))
# ax.plot(times, tidal_uv['u'], label='u')
# ax.plot(times, tidal_uv['v'], color='C1', label='v')
# ax.set_ylabel('Tidal speed')
# ax.legend(loc='upper left')
# ax1.plot(times, curr, label='speed')
# ax1.set_ylabel('Tidal speed')
# ax.grid(); ax1.grid()

tide = tidal_series.reindex(times).interpolate()   # align and fill

# detrend and z-score
tide_z = zscore(detrend(tidal_h))
curr_z = zscore(detrend(curr.values))


from scipy.signal.windows import hann
def bandpass(x, low_period, high_period, fs):
    # convert to Hz
    low_f = 1/(low_period*3600.0); high_f = 1/(high_period*3600.0)
    b,a = butter(4, [low_f, high_f], btype='bandpass', fs=fs)
    return filtfilt(b,a,x)


# fig, ax = plt.subplots(1,1,figsize=(12,4))
# ax.plot(times, curr, color='C0')
# ax.set_ylabel('Current speed (cm/s)', color='C0')
# ax2 = ax.twinx()
# ax2.plot(times, psd_series, color='C1')
# ax2.set_ylabel('PSD @ 1 Hz (dB)', color='C1')
# # ax.set_ylim([-135, -115])
# ax.grid(axis='x')
# plt.show()

tidal_h = pd.Series(tidal_h, index=times)
tide_z = pd.Series(tide_z, index=times)
curr_z = pd.Series(curr_z, index=times)
fig, ax, corr_matrix, lags, starts = rolling_xcorr_plot(psd_series, tidal_h, 
                                                        24, overlap_frac=0.75, show=False, var_label='Tidal height (m)', normalise_window=False)
fig, ax, corr_matrix, lags, starts = rolling_xcorr_plot(psd_series, tide_z, 
                                                        24, overlap_frac=0.75, show=False, var_label='Tidal height', normalise_window=False)
# fig, ax, corr_matrix, lags, starts = rolling_xcorr_plot(psd_series, curr_z, 
#                                                         12, overlap_frac=0.75, show=False, var_label='Tidal current', normalise_window=False)
# fig, ax, corr_matrix, lags, starts = rolling_xcorr_plot(psd_series, curr_z, 
#                                                         6, overlap_frac=0.75, show=False, var_label='Tidal current', normalise_window=False)

# fig, ax = full_series_xcorr(psd_series, curr_z, normalise=False, max_lag_hours=48, label='tidal_current_48max', save=False)

plt.show()
