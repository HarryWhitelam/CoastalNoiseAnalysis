import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ast
from scipy.signal import savgol_filter

from helper_funcs import get_weather_data

# def slice_pearson(x): return x[0]
# def slice_spearman(x): return x[1]

# fname = 'curr_corr'
# file_path = f'/data/localraid/weather_corrs/{fname}.csv'

# df = pd.read_csv(file_path, index_col=(0,1), converters={"2024-04-01": ast.literal_eval, "2024-05-01": ast.literal_eval, "2024-06-01": ast.literal_eval, "2024-07-01": ast.literal_eval, "2024-08-01": ast.literal_eval, "2024-09-01": ast.literal_eval, "2024-10-01": ast.literal_eval, "2024-11-01": ast.literal_eval, "2024-12-01": ast.literal_eval, "2025-01-01": ast.literal_eval, "2025-02-01": ast.literal_eval, "2025-03-01": ast.literal_eval, "2025-04-01": ast.literal_eval, "2025-05-01": ast.literal_eval, "2025-06-01": ast.literal_eval, "2025-07-01": ast.literal_eval})

# # df = df.drop(['2025-06-01'], axis=1)

# df_pearson = df.applymap(slice_pearson)
# df_spearman = df.applymap(slice_spearman)
# df_pearson = df_pearson.astype(np.float64)
# df_spearman = df_pearson.astype(np.float64)
# print(df_spearman)

# df_pearson.to_csv(f'/data/localraid/weather_corrs/{fname}_pearson.csv')
# df_spearman.to_csv(f'/data/localraid/weather_corrs/{fname}_spearman.csv')



# ['wind', 'wh', 'tidalh', 'curr']
fname = 'wh'
# label = 'tidal current'
# 0.130

# df_pearson = pd.read_csv(f'/data/localraid/weather_corrs/{fname}_corr_pearson.csv', index_col=(0,1))
# df_pearson_cha = df_pearson.loc[cha].T
# ax = df_pearson_cha.plot(marker='D')
# ax.set_title(f'DAS spectral correlation with {label}')
# ax.set_ylabel('Pearson coefficient')
# plt.xticks(rotation=90)

#### 1 line per frequency
# for cha in [750,788, 875, 1475]:
#     df_spearman = pd.read_csv(f'/data/localraid/weather_corrs/{fname}_corr_spearman.csv', index_col=(0,1))
#     df_spearman_cha = df_spearman.loc[cha].T
#     ax = df_spearman_cha.plot(marker='D')
#     ax.set_title(f'Channel {cha} spectral correlation with {label}')
#     ax.set_ylabel('Spearman coefficient')
#     plt.xticks(rotation=90)
# plt.show()


#### 1 line per channel
# df_spearman = pd.read_csv(f'/data/localraid/weather_corrs/{fname}_corr_spearman.csv', index_col=(1,0))
# fig, axs = plt.subplots(2, 6, figsize=(12,8), sharex=True, sharey=True)
# # for ax, f in zip(axs.flat, [0.1, 0.3, 0.5, 1.0, 1.6, 2.0, 5.0, 10.0]):
# for ax, f in zip(axs.flat, [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]):
#     df_spearman_f = df_spearman.loc[f].T
#     df_spearman_f.plot(ax=ax, marker='D')
#     ax.set_title(f'{f} Hz')
#     ax.set_ylabel('Spearman coefficient')
#     ax.tick_params(labelrotation=90)
#     ax.xaxis.set_minor_locator(AutoMinorLocator())
#     ax.grid(axis='y')
#     ax.legend().set_visible(False)
#     ax.set_ylim(-0.15, 0.82)
# # fig.suptitle(f'Spectral correlation with {label}')
# axs.flat[-1].legend().set_visible(True)
# plt.tight_layout()
# plt.subplots_adjust(wspace=0, hspace=0.130)

### 1 line per freq
df_spearman = pd.read_csv(f'/data/localraid/weather_corrs/{fname}_corr_spearman.csv', index_col=(0,1))
df_std = pd.read_csv(f'/data/localraid/weather_corrs/{fname}_std_corr_spearman.csv', index_col=(0,1))
df_spearman = df_spearman.rename(columns={'2024-04-01':'2024-04', '2024-05-01':'2024-05', '2024-06-01':'2024-06', '2024-07-01':'2024-07', '2024-08-01':'2024-08',
       '2024-09-01':'2024-09', '2024-10-01':'2024-10', '2024-11-01':'2024-11', '2024-12-01':'2024-12', '2025-01-01':'2025-01',
       '2025-02-01':'2025-02', '2025-03-01':'2025-03', '2025-04-01':'2025-04', '2025-05-01':'2025-05', '2025-06-01':'2025-06',
       '2025-07-01':'2025-07'})
df_std = df_std.rename(columns={'2024-04-01':'2024-04', '2024-05-01':'2024-05', '2024-06-01':'2024-06', '2024-07-01':'2024-07', '2024-08-01':'2024-08',
       '2024-09-01':'2024-09', '2024-10-01':'2024-10', '2024-11-01':'2024-11', '2024-12-01':'2024-12', '2025-01-01':'2025-01',
       '2025-02-01':'2025-02', '2025-03-01':'2025-03', '2025-04-01':'2025-04', '2025-05-01':'2025-05', '2025-06-01':'2025-06',
       '2025-07-01':'2025-07'})
fig, axs = plt.subplots(2, 2, figsize=(12,8), sharex=True, sharey=True)
fs = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]
# for ax, f in zip(axs.flat, [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]):
for ax, cha in zip(axs.flat, [750, 788, 875, 1475]):
    df_spearman_cha = df_spearman.loc[cha].T
    df_spearman_cha = df_spearman_cha[fs]
    df_std_cha = df_std.loc[cha].T
    df_std_cha = df_std_cha[fs]
    df_spearman_cha.plot(ax=ax, marker='D', cmap='viridis', yerr=df_std_cha)
    ax.set_title(f'Channel {cha}', fontsize=14)
    ax.set_ylabel('Spearman coefficient', fontsize=14)
    ax.tick_params(labelrotation=90)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(axis='y')
    ax.legend().set_visible(False)
    ax.set_ylim(-0.19, 0.91)
    ax.axvline(6, color=(1, 0, 0, 0.4))
    # ax.axvline(9, color=(1, 0, 0, 0.4))
    # ax.axvline(10, color=(1, 0, 0, 0.4))
    ax.axvline(14, color=(1, 0, 0, 0.4))
    # ax.xaxis.set_ticklabels(['2024-04', '2024-06', '2024-08', '2024-10', '2024-12', '2025-02', '2025-04', '2025-06'])
    ax.xaxis.set_ticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], minor=True)
axs.flat[-1].legend(loc='center left', bbox_to_anchor=(1,1), title='f (Hz)').set_visible(True)
fig.suptitle('DAS correlation with significant wave height')
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0.130)


# fig, axs = plt.subplots(4,1, figsize=(12,8), sharex=True)
# t0 = datetime(year=2024, month=4, day=1); t1 = datetime(year=2025, month=8, day=1)
# times = pd.date_range(f'{t0.date()} 00:00:00', f'{t1.date()} 00:00:00', freq=timedelta(minutes=30))
# weather_df = get_weather_data(t0, t1, times, ['Wind(m/s)', 'Hs(Hm0)(m)', 'ASLVBG02', 'curr'])
# for ax, col, ylabel in zip(axs.flat, weather_df.columns, ['Wind speed (m/s)', 'Wave height (m)', 'Tidal height (m)', 'Tidal current (cm/s)']):
#     ax.plot(weather_df[col])
#     lobf = savgol_filter(weather_df[col], 48, 2)
#     ax.plot(weather_df.index, lobf)
#     ax.tick_params(labelrotation=90)
#     ax.xaxis.set_minor_locator(AutoMinorLocator())
#     ax.grid(axis='both')
#     ax.set_ylabel(f'{ylabel}')
# plt.tight_layout()
# plt.subplots_adjust(hspace=0)


plt.show()
