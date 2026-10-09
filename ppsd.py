import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from obspy import read, read_inventory, UTCDateTime, Stream
from obspy.signal import PPSD
from obspy.imaging.util import _set_xaxis_obspy_dates
import matplotlib.pyplot as plt
from datetime import datetime

from helper_funcs import extract_psd_time_series

# tidal_df = pd.read_csv('/local/gfs19eku/Documents/NoisePy4DAS/results/checkpoints/CRO_final.csv', parse_dates=True, index_col=0)
# tidal_df['derivative'] = tidal_df.diff()['ASLVBG02'] / (tidal_df.index[1] - tidal_df.index[0]).total_seconds()
# tidal_df['derivative_2'] = tidal_df.diff()['derivative'] / (tidal_df.index[1] - tidal_df.index[0]).total_seconds()

# stations = ['04915', '05014', '03479', '03713', '05330', '05100']       # [0-2] trim-sidestrand, [3-5] cliff-road
stations = ['04915', '05100']
comps = ['E']
n_days = 10
with tqdm(total=len(stations) * len(comps)) as pbar:
    for station in stations:
        for comp in comps:
            st = Stream()
            for d in range(1, n_days+1):
                st += read(f'/data/localraid/UEA_nodes_mseed_mV/rotated/5J.{station}..DP{comp}..2025-04-{d:02d}T00:00:00.000000Z.MSEED')
            
            ppsd = PPSD(st[0].stats, period_limits=(0.02,100), metadata=read_inventory(f'/data/localraid/UEA_nodes_mseed_mV/stations/5J.{station}..DP{comp}.76_6'))
            ppsd.add(st)

            # ppsd.plot(filename=f'figures/{station}/ppsd_{st[0].id}_{st[0].stats.starttime}_{st[-1].stats.endtime}.png', xaxis_frequency=True, show_coverage=False, period_lim=(0.02, 100))
            fig = ppsd.plot(xaxis_frequency=True, show_coverage=False, period_lim=(0.02, 50), show=False)
            fig.get_axes()[0].set_ylim(-180, -90)
            fig.get_axes()[0].set_title(f'Geophone {station}')
            plt.tight_layout()
            plt.savefig(f'./figures/{station}/ppsd_{st[0].id}_{st[0].stats.starttime}_{st[-1].stats.endtime}.eps')

            # ppsd.plot_spectrogram(filename=f'figures/{station}/spec_{st[0].id}_{st[0].stats.starttime}_{st[-1].stats.endtime}.png', cmap='jet', yaxis_frequency=True)
            # filename=f'figures/{station}/spec_{st[0].id}_{st[0].stats.starttime}_{st[-1].stats.endtime}_limited.png', 
            fig = ppsd.plot_spectrogram(clim=[-180, -100], cmap='jet', yaxis_frequency=True, show=False)
            fig.get_axes()[0].set_title(f'Geophone {station}')
            plt.tight_layout()
            plt.savefig(f'./figures/{station}/spec_{st[0].id}_{st[0].stats.starttime}_{st[-1].stats.endtime}.eps')
            
            pbar.update(1)


# DATA FOR 05014 ON 2025/09/04 CORRUPTED
