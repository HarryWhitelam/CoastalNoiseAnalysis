from obspy import read, read_inventory, UTCDateTime, Stream
from obspy.signal import PPSD
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import coherence
from datetime import timedelta
from helper_funcs import *
from weather import *


### xcorr of tidal data and spectra
station = '04915'   # 03479
comp = 'E'
# for i in range(1,10):
st = Stream()
for d in range(1,10):
    st += read(f'/data/localraid/UEA_nodes_mseed_mV/rotated/5J.{station}..DP{comp}..2025-04-{d:02d}T00:00:00.000000Z.MSEED')
st.merge(method=1, fill_value='interpolate')
# st.detrend('linear')
# st.taper(max_percentage=0.05, type='cosine')
# st.filter('bandpass', freqmin=0.2, freqmax=5.0, corners=4, zerophase=True)
ppsd = PPSD(st[0].stats, period_limits=(0.02,100), metadata=read_inventory(f'/data/localraid/UEA_nodes_mseed_mV/stations/5J.{station}..DP{comp}.76_6'))
ppsd.add(st)
psd_series, times = extract_psd_time_series(ppsd, 1.0, True, smooth=False)
# psd_series.to_csv('/checkpoints/psd_1hz.csv')

tidal_df = pd.read_csv('/local/gfs19eku/Documents/NoisePy4DAS/results/checkpoints/CRO_final.csv', parse_dates=True, index_col=0)
dt = tidal_df.index.to_series().diff().dt.total_seconds()
tidal_df['derivative'] = tidal_df['ASLVBG02'].diff() / dt
tidal_df['derivative_2'] = tidal_df['derivative'].diff() / dt
times = pd.to_datetime(times)
match_idx = tidal_df.index.isin(times)
tidal_df = tidal_df.loc[match_idx, :]

df_wave = plot_waverider_csv('../NoisePy4DAS/results/checkpoints/hpg_wave.csv', False, get_df=True)
df_met = plot_met_csv('../NoisePy4DAS/results/checkpoints/hpg_met.csv', False, get_df=True)
df_era5 = plot_era5_csv('../NoisePy4DAS/results/checkpoints/combined_weather.csv', False, get_df=True)
weather_df = pd.concat([df_wave, df_met, df_era5], axis=1)[times[0].date():times[-1].date()]
weather_df = weather_df.reindex(times)
weather_df['Hs(Hm0)(m)'] = weather_df['Hs(Hm0)(m)'].interpolate()

current_df = pd.read_csv('/local/gfs19eku/Documents/Nodes_Analysis/checkpoints/tidal_currents.csv')
curr_speed = np.sqrt(current_df['u'].to_numpy()**2 + current_df['v'].to_numpy()**2)
curr = pd.Series(curr_speed, index=times) / 100

delta = times[-1] - times[0]
ts_delta = delta.total_seconds() / (psd_series.shape[0]-1)
tidal_delta = delta.total_seconds() / (tidal_df.shape[0]-1)
if ts_delta != tidal_delta: print(f'deltas do not match! ts: {ts_delta} || tidal: {tidal_delta}')

# tidal_series = tidal_df['ASLVBG02'].reindex(psd_series.index)
# fig, ax = plt.subplots(1,1, figsize=(12,4))
# ax.plot(psd_series.index, weather_df['Hs(Hm0)(m)'])
# ax.set_ylabel('Tidal height (m)', color='C0')
# ax1 = ax.twinx()
# ax1.plot(psd_series.index, psd_series, color='C1')
# ax1.set_ylabel('PSD @ 1 Hz (dB)', color='C1')
# ax.grid(axis='x')

tidal_series = zscore(tidal_df['ASLVBG02'].reindex(psd_series.index))
tidal_d_series = zscore(tidal_df['derivative'].reindex(psd_series.index))
tidal_series_norm = tidal_series.abs()
tidal_d_series_norm = tidal_d_series.abs()

# fig, ax, corr_matrix, lags, starts = rolling_xcorr_plot(psd_series, weather_df['Hs(Hm0)(m)'], window_hours=24, overlap_frac=0.75, show=False, var_label='Sig. Wave Height (m)', normalise_window=False)
# plt.show()

for window_hours in [24]:
    fig, ax, corr_matrix, lags, starts = rolling_xcorr_plot(psd_series, curr, 
                                                            window_hours, overlap_frac=0.75, show=True, var_label='Tidal current (m/s)', normalise_window=False)
    # plt.savefig(f'./figures/correlations/{window_hours}hr/xcorr_tidal_height.png', bbox_inches='tight')
    # zs_height = zero_lag_series(corr_matrix, lags, starts, psd_series.index)
    # avg_height = avg_lag_series(corr_matrix, lags, starts, psd_series.index)
    fig, ax, corr_matrix, lags, starts = rolling_xcorr_plot(psd_series, tidal_series_norm, 
                                                            window_hours, overlap_frac=0.75, show=True, var_label='Tidal height (m)', normalise_window=False)
    # plt.savefig(f'./figures/correlations/{window_hours}hr/xcorr_tidal_height_norm.png', bbox_inches='tight')
    # zs_height_norm = zero_lag_series(corr_matrix, lags, starts, psd_series.index)
    # avg_height_norm = avg_lag_series(corr_matrix, lags, starts, psd_series.index)

    # fig, ax, corr_matrix, lags, starts = rolling_xcorr_plot(psd_series, tidal_d_series, 
    #                                                         window_hours, overlap_frac=0.75, show=False, var_label='Tidal speed (m/s)', normalise_window=False)
    # plt.savefig(f'./figures/correlations/{window_hours}hr/xcorr_tidal_speed.png', bbox_inches='tight')
    # zs_speed = zero_lag_series(corr_matrix, lags, starts, psd_series.index)
    # avg_speed = avg_lag_series(corr_matrix, lags, starts, psd_series.index)
    # fig, ax, corr_matrix, lags, starts = rolling_xcorr_plot(psd_series, tidal_d_series_norm, 
    #                                                         window_hours, overlap_frac=0.75, show=False, var_label='Tidal speed (m/s)', normalise_window=False)
    # plt.savefig(f'./figures/correlations/{window_hours}hr/xcorr_tidal_speed_norm.png', bbox_inches='tight')
    # zs_speed_norm = zero_lag_series(corr_matrix, lags, starts, psd_series.index)
    # avg_speed_norm = avg_lag_series(corr_matrix, lags, starts, psd_series.index)
    # plt.close('all')

#     # arr = [
#     #     ['height', 'Tidal height (m)', zs_height, tidal_series],
#     #     ['height_norm', 'Tidal height (m)', zs_height_norm, tidal_series_norm],
#     #     ['speed', 'Tidal speed (m/s)', zs_speed, tidal_d_series],
#     #     ['speed_norm', 'Tidal speed (m/s)', zs_speed_norm, tidal_d_series_norm],
#     # ]
#     # for (f_label, ax_label, zs_series, val_series) in arr:
#     #     fig, (ax0,ax1,ax2) = plt.subplots(3,1, figsize=(12,8), sharex=True)
#     #     fig.subplots_adjust(hspace=0)
#     #     ax0.plot(zs_series.index, zs_series.values)
#     #     ax0.set_ylabel('Corr @ 0 s lag')
#     #     ax1.plot(tidal_df.index, val_series)
#     #     ax1.set_ylabel(ax_label)
#     #     ax2.plot(tidal_df.index, psd_series)
#     #     ax2.set_ylabel('PSD at 1 Hz (dB)')
#     #     ax0.grid(); ax1.grid(); ax2.grid()
#     #     plt.savefig(f'./figures/correlations/{window_hours}hr/zero_lag_{f_label}.png', bbox_inches='tight')
#     #     plt.close('all')
    
#     arr = [
#         ['height', 'Tidal height (m)', avg_height, tidal_series],
#         ['height_norm', 'Tidal height (m)', avg_height_norm, tidal_series_norm],
#         ['speed', 'Tidal speed (m/s)', avg_speed, tidal_d_series],
#         ['speed_norm', 'Tidal speed (m/s)', avg_speed_norm, tidal_d_series_norm],
#     ]
#     for (f_label, ax_label, avg_series, val_series) in arr:
#         fig, (ax0,ax1,ax2) = plt.subplots(3,1, figsize=(12,8), sharex=True)
#         fig.subplots_adjust(hspace=0)
#         ax0.plot(avg_series.index, avg_series.values)
#         ax0.set_ylabel(f'Corr @ {avg_series.name} s lag')
#         ax1.plot(tidal_df.index, val_series)
#         ax1.set_ylabel(ax_label)
#         ax2.plot(tidal_df.index, psd_series)
#         ax2.set_ylabel('PSD at 1 Hz (dB)')
#         ax0.grid(); ax1.grid(); ax2.grid()
#         plt.savefig(f'./figures/correlations/{window_hours}hr/avg_lag_{f_label}.png', bbox_inches='tight')
#         plt.close('all')


# full_series_xcorr(psd_series, tidal_series, normalise=False, max_lag_hours=48, label='tidal_height_48max')
# full_series_xcorr(psd_series, tidal_series_norm, normalise=False, max_lag_hours=48, label='norm_tidal_height_48max')


# ### tide values plot
# fig, (ax, ax0, ax1) = plt.subplots(3,1, figsize=(12,8), sharex=True)
# ax.plot(tidal_df.index, psd_series)
# ax.set_ylabel('PSD at 1 Hz (dB)')

# ax0.plot(tidal_df.index, tidal_series)
# ax0.set_ylabel('Tidal height (m)')

# ax1.plot(tidal_df.index, curr)
# ax1.set_ylabel('Current speed (FES) (cm/s)')

# ax.grid('both')
# ax0.grid('both')
# ax1.grid('both')
# fig.subplots_adjust(hspace=0)
# plt.show()
# plt.savefig('./figures/correlations/tidal_values.png', bbox_inches='tight')



### diags
# plt.figure(figsize=(12,4))
# ax = psd_series.plot(label='PSD (dB)')
# ax2 = ax.twinx()
# tidal_series.plot(ax=ax2, color='C1', label='Tide')
# ax.set_ylabel('PSD (dB)')
# ax2.set_ylabel('Tidal height (m)')
# ax.legend(loc='upper left'); ax2.legend(loc='upper right')
# plt.show()

# print("PSD span:", psd_series.index[0], "->", psd_series.index[-1], "count:", len(psd_series))
# dt = pd.Series(psd_series.index).diff().dropna().dt.total_seconds()
# print("PSD median dt (s):", dt.median(), "min/max:", dt.min(), dt.max())
# print("PSD gaps > 2*median:", (dt > 2*dt.median()).sum())

# print("\nTide span:", tidal_df.index[0], "->", tidal_df.index[-1], "count:", len(tidal_df))
# tdt = tidal_df.index.to_series().diff().dt.total_seconds().dropna()
# print("Tide median dt (s):", tdt.median(), "min/max:", tdt.min(), tdt.max())
# print("Tide gaps > 2*median:", (tdt > 2*tdt.median()).sum())

# window_s = int(12*3600 / (pd.Series(psd_series.index).diff().median().total_seconds()))
# psd_rms = psd_series.rolling(window=window_s, center=True, min_periods=1).std()
# tidal_rms = tidal_series.rolling(window=window_s, center=True, min_periods=1).std()
# plt.figure(figsize=(10,4)); psd_rms.plot(label='PSD std (rolling)'); plt.legend(); plt.grid(); plt.show()
# plt.figure(figsize=(10,4)); tidal_rms.plot(label='Tide std (rolling)'); plt.legend(); plt.grid(); plt.show()

# fig, ax, corr_norm, lags, starts = rolling_xcorr_plot(psd_series, tidal_series, window_hours=6, normalise_window=True, show=False)
# zs_norm = zero_lag_series(corr_norm, lags, starts, psd_series.index)
# fig, ax, corr_raw, lags2, starts2 = rolling_xcorr_plot(psd_series, tidal_series, window_hours=6, normalise_window=False, show=False)
# zs_raw = zero_lag_series(corr_raw, lags2, starts2, psd_series.index)
# plt.figure(); zs_norm.plot(label='per-window norm'); zs_raw.plot(label='raw'); plt.legend(); plt.grid(); plt.show()

# psd_series.groupby(psd_series.index.date).apply(lambda s: s.mean()).plot(kind='bar'); plt.title('PSD mean per day'); plt.show()
# tidal_series.groupby(tidal_series.index.date).apply(lambda s: s.mean()).plot(kind='bar'); plt.title('Tide mean per day'); plt.show()

# s_good = starts[0]
# s_bad = starts[len(starts)//2]
# for s, label in [(s_good,'good'), (s_bad,'bad')]:
#     x = psd_series.to_numpy()[s:s+window_s]
#     y = tidal_series.to_numpy()[s:s+window_s]
#     fs = 1.0 / pd.Series(psd_series.index).diff().median().total_seconds()
#     f, Cxy = coherence(x, y, fs=fs, nperseg=max(16, len(x)//4))
#     plt.semilogy(f, Cxy); plt.title(label); plt.show()





# fig, axs = plt.subplots(3,1, sharex=True)
# for ax, corr, label in zip(axs.ravel(), [asl_xcorr, d1_xcorr, d2_xcorr], ['tidal height', 'tidal height derivative', 'tidal height second derivative']):
#     imax = np.argmax(corr)
#     imin = np.argmin(corr)
#     max_seconds = lags[imax] * ts_delta
#     min_seconds = lags[imin] * ts_delta
#     ax.plot(lags * ts_delta, corr)
#     ax.axvline(max_seconds, color='C1', linestyle='--')
#     ax.axvline(min_seconds, color='C2', linestyle='--')
#     ax.set_ylabel('normalised cross-correlation')
#     if corr[imax] > abs(corr[imin]):
#         ax.set_title(f'{label} (max {corr[imax]:.3f} at {max_seconds:.1f}s)')
#     else:
#         ax.set_title(f'{label} (min {corr[imin]:.3f} at {min_seconds:.1f}s)')
#     ax.grid(True)
# plt.xticks(np.arange(-(60*60*24), 60*60*24, 60*60*4), [str(n).zfill(2) + ':00' for n in np.arange(-24, 24, 4)])
# ax.set_xlabel('Lag (s)')
# fig.suptitle(f'Cross-corr between PSD@{1.0}Hz and specified tidal values between {raw_times[0]} and {raw_times[-1]}.')
# # plt.savefig(f'/figures/correlations/xcorr.png')
# plt.show()



# # nearest to road: 05100
# station = '05100'
# comps = ['N', 'E', 'Z']
# d0 = 1; d1 = 2

# st = Stream()
# for comp in comps:
#     for d in range(d0, d1+1):
#         st += read(f'/data/localraid/UEA_nodes_mseed_mV/rotated/5J.{station}..DP{comp}..2025-04-{d:02d}T00:00:00.000000Z.MSEED')


# st_filt = st.copy()
# st_filt.filter("highpass", freq=30.0)

# for tr in st_filt: 
#     tr.stats.channel += '_filt'
#     st += tr
# st.plot()

# fig, axs = plt.subplots(nrows=3)
# for ax, comp in zip(axs, ['N', 'E', 'Z']):
#     st1 = st.select(component=comp)
#     st2 = st_filt.select(component=comp)
    
#     for tr in st1:
#         ax.plot(tr.times('matplotlib'), tr.data,
#                 label="Raw ({})".format(tr.stats.channel))
#     for tr in st2:
#         ax.plot(tr.times(), tr.data,
#                 label="Highpass 30Hz ({})".format(tr.stats.channel))
#     ax.legend()
#     ax.xaxis_date()
# fig.autofmt_xdate()
# plt.show()
