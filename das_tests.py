from obspy import read, read_inventory, UTCDateTime, Stream
from obspy.signal import PPSD
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from helper_funcs import *
from weather import *


# station = '04915'   # 03479
# comp = 'E'
# # for i in range(1,10):
# st = Stream()
# for d in range(1,10):
#     st += read(f'/data/localraid/UEA_nodes_mseed_mV/rotated/5J.{station}..DP{comp}..2025-04-{d:02d}T00:00:00.000000Z.MSEED')
# st.merge(method=1, fill_value='interpolate')
# # st.detrend('linear')
# st.taper(max_percentage=0.05, type='cosine')
# # st.filter('bandpass', freqmin=0.2, freqmax=5.0, corners=4, zerophase=True)
# ppsd = PPSD(st[0].stats, period_limits=(0.02,100), metadata=read_inventory(f'/data/localraid/UEA_nodes_mseed_mV/stations/5J.{station}..DP{comp}.76_6'))
# ppsd.add(st)
# node_psd_series, times = extract_psd_time_series(ppsd, 1.0, True, smooth=False)

file_path = '/data/localraid/saved_specs/600s_window/2024-12-01 00:00:06__2025-01-01 00:00:06_f0.01:49.9_1475_spec.txt'
# file_path = '/data/localraid/saved_specs/non_months/600s_window/2025-04-01 00:00:25__2025-04-10 00:00:25_f0.01:49.9_1475_spec.txt'
spec = np.loadtxt(file_path, delimiter=',')
f_size, t_size = spec.shape

t_start = datetime.strptime(file_path.split('/')[-1].split('_')[0], '%Y-%m-%d %H:%M:%S')
t_end   = datetime.strptime(file_path.split('/')[-1].split('_')[2], '%Y-%m-%d %H:%M:%S')
t_bin_length = (t_end - t_start).total_seconds() / t_size
time_axis = [t_start + timedelta(seconds=t_bin_length * j) for j in range(t_size)]

target_f = 1.6
f_bins = np.linspace(0.0, 50.016666666666666, f_size)
idx = (np.abs(f_bins - target_f)).argmin()
psd_series = pd.Series(spec[idx,:], index=pd.to_datetime(time_axis))
psd_series = psd_series[f'{t_start.date()} 00:00:00':f'{t_end.date()} 23:59:59']

lons = [1.402138]; lats = [52.94651]
times = pd.date_range(f'{t_start.date()} 00:00:00', f'{(t_end-timedelta(days=1)).date()} 23:00:00', freq=timedelta(minutes=30))
# tidal_h, tidal_uv = get_tidal_data(lons=lons, lats=lats, times=times)
# curr = np.sqrt(tidal_uv['u']**2 + tidal_uv['v']**2)
# curr = pd.Series(np.asarray(curr), index=times).interpolate()
# current_df = pd.read_csv('/local/gfs19eku/Documents/Nodes_Analysis/checkpoints/tidal_currents.csv')
# curr_speed = np.sqrt(current_df['u'].to_numpy()**2 + current_df['v'].to_numpy()**2)/100
# curr = pd.Series(curr_speed, index=times)

df_wave = plot_waverider_csv('../NoisePy4DAS/results/checkpoints/hpg_wave.csv', False, get_df=True)
df_met = plot_met_csv('../NoisePy4DAS/results/checkpoints/hpg_met.csv', False, get_df=True)
df_era5 = plot_era5_csv('../NoisePy4DAS/results/checkpoints/combined_weather.csv', False, get_df=True)
weather_df = pd.concat([df_wave, df_met, df_era5], axis=1)[t_start.date():t_end.date()]
weather_df = weather_df.reindex(times)

tidal_df = pd.read_csv('/local/gfs19eku/Documents/NoisePy4DAS/results/checkpoints/CRO_final.csv', parse_dates=True, index_col=0)
match_idx = tidal_df.index.isin(times)
tidal_df = tidal_df.loc[match_idx, :]
tidal_series = tidal_df['ASLVBG02'].reindex(times)

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
weather_df['Wind(m/s)'] = weather_df['Wind(m/s)'].interpolate()
weather_df['Hs(Hm0)(m)'] = weather_df['Hs(Hm0)(m)'].interpolate()
tide = tidal_series.reindex(times).interpolate()


# fig, (ax,ax0,ax1,ax2) = plt.subplots(4,1, figsize=(12,6), sharex=True)
# ax.plot(targets, psd_30min_cosine, label='30min cosine-window')
# ax.plot(psd_series.index, psd_series.values, alpha=0.25, label='raw samples')
# ax.set_ylabel('Amplitude (dB rel. $(nm/m/s)^2/Hz$)')
# ax.annotate('(a)', xy=(0,1), xycoords='axes fraction', xytext=(0.5,-0.5), textcoords='offset fontsize', verticalalignment='top')
# ax0.plot(times, node_psd_series)
# ax0.set_ylabel(r'Amplitude (dB rel. $m^2/s^4/Hz$)')
# ax0.set_ylim(-130, -110)
# ax0.annotate('(b)', xy=(0,1), xycoords='axes fraction', xytext=(0.5,-0.5), textcoords='offset fontsize', verticalalignment='top')
# ax1.plot(times, curr)
# ax1.set_ylabel('Tidal current (m/s)')
# ax1.annotate('(c)', xy=(0,1), xycoords='axes fraction', xytext=(0.5,-0.5), textcoords='offset fontsize', verticalalignment='top')
# ax2.plot(times, tide)
# ax2.set_ylabel('Tidal height (m)')
# ax2.annotate('(d)', xy=(0,1), xycoords='axes fraction', xytext=(0.5,-0.5), textcoords='offset fontsize', verticalalignment='top')
# ax.grid(axis='x'); ax0.grid(axis='x'); ax1.grid(axis='x'); ax2.grid(axis='x')
# ax.set_xlim(times[0], times[-1])
# ax.legend(loc='upper right')
# plt.subplots_adjust(hspace=0)


# fig, ax = plt.subplots(1,1,figsize=(12,4))
# ax.plot(times, curr, color='C0')
# ax.set_ylabel('Current speed (cm/s)', color='C0')
# ax2 = ax.twinx()
# ax2.plot(times, psd_30min_cosine, color='C1')
# ax2.set_ylabel(f'Strainrate PSD @ {target_f} Hz (dB)', color='C1')
# ax.grid(axis='x')


# fig, (ax,ax1,ax2) = plt.subplots(3,1, figsize=(12,6), sharex=True)
# ax.plot(targets, psd_30min_cosine, label='30min cosine-window')
# ax.plot(psd_series.index, psd_series.values, alpha=0.25, label='raw samples')
# ax.set_ylabel('Strainrate derived PSD (dB)')
# ax1.plot(times, weather_df['Wind(m/s)'])
# ax1.set_ylabel('Wind speed (m/s)')
# ax2.plot(times, weather_df['Hs(Hm0)(m)'])
# ax2.set_ylabel('Wave height (m)')
# ax.grid(axis='x'); ax1.grid(axis='x'); ax2.grid(axis='x')
# ax.legend(loc='upper right')
# ax.xlim(times[0], times[-1])
# plt.subplots_adjust(hspace=0)

fig, ax, corr_matrix, lags, starts = rolling_xcorr_plot(psd_30min_cosine, weather_df['Wind(m/s)'], window_hours=24, overlap_frac=0.75, show=True, var_label='Wind speed (m/s)', normalise_window=False)


# print(f'\nWind')
# pearson = psd_30min_cosine.corr(weather_df['Wind(m/s)'], method='pearson')
# spearman = psd_30min_cosine.corr(weather_df['Wind(m/s)'], method='spearman')
# print(f"Pearson r = {pearson:.3f}")
# print(f"Spearman rho = {spearman:.3f}")

# print(f'\nWh')
# pearson = psd_30min_cosine.corr(weather_df['Hs(Hm0)(m)'], method='pearson')
# spearman = psd_30min_cosine.corr(weather_df['Hs(Hm0)(m)'], method='spearman')
# print(f"Pearson r = {pearson:.3f}")
# print(f"Spearman rho = {spearman:.3f}")

# print(f'\nCurrent')
# pearson = psd_30min_cosine.corr(curr, method='pearson')
# spearman = psd_30min_cosine.corr(curr, method='spearman')
# print(f"Pearson r = {pearson:.3f}")
# print(f"Spearman rho = {spearman:.3f}")

# print(f'\nTidal height')
# pearson = psd_30min_cosine.corr(tide, method='pearson')
# spearman = psd_30min_cosine.corr(tide, method='spearman')
# print(f"Pearson r = {pearson:.3f}")
# print(f"Spearman rho = {spearman:.3f}")

plt.show()