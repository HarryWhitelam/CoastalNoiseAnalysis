from obspy import UTCDateTime, read, Stream, read_inventory
from obspy.clients.fdsn import Client
from obspy.signal import PPSD
import matplotlib.pyplot as plt
from datetime import timedelta


client = Client('http://eida.bgs.ac.uk')
# Download station list

# Save and plot data
station = "BEDF"
comps = ['E']#
t_start = UTCDateTime('2024-12-11T00:00:00.00')
t_end = UTCDateTime('2024-12-14T00:00:00.00')
# t_end = UTCDateTime('2025-04-01T:12:29.59')
st_bgs = client.get_waveforms("GB", station, "00", "HH?", t_start, t_end)
st_uoe = Stream()

# st_bgs.write(f"{station}_2025-04-01T:00:00.00.mseed", format="MSEED")

for comp in comps:
    st_ppsd = st_bgs.select(component=comp)
    inventory = read_inventory(f'{station.lower()}_station.xml')
    
    ppsd = PPSD(st_ppsd[0].stats, metadata=inventory, ppsd_length=900)   # period_limits=(0.02,100)
    ppsd.add(st_ppsd)

    # ppsd.plot(filename=f'figures/{station}/ppsd_{st_ppsd[0].id}_{st_ppsd[0].stats.starttime}_{st_ppsd[-1].stats.endtime}.png', xaxis_frequency=True, show_coverage=False)
    # ppsd.plot_spectrogram(filename=f'figures/{station}/spec_{st_ppsd[0].id}_{st_ppsd[0].stats.starttime}_{st_ppsd[-1].stats.endtime}.png', cmap='jet', yaxis_frequency=True)
    # ppsd.plot_spectrogram(filename=f'figures/{station}/spec_{st_ppsd[0].id}_{st_ppsd[0].stats.starttime}_{st_ppsd[-1].stats.endtime}_limited.png', clim=[-180, -100], cmap='jet', yaxis_frequency=True)

    # fig = ppsd.plot(xaxis_frequency=True, show_coverage=False)
    # fig = ppsd.plot_spectrogram(yaxis_frequency=True)
    fig = ppsd.plot(xaxis_frequency=True, show_coverage=False, period_lim=(0.02, 50), show=False)
    fig.get_axes()[0].set_ylim(-180, -90)
    fig.get_axes()[0].set_title(station)
    plt.tight_layout()
    plt.savefig(f'./figures/{station}/ppsd_{st_ppsd[0].id}_{st_ppsd[0].stats.starttime}_{st_ppsd[-1].stats.endtime}.eps')
    
    # fig = ppsd.plot_spectrogram(clim=[-180, -100], cmap='jet', yaxis_frequency=True, show=False)
    # fig.get_axes()[0].set_title(station)
    # plt.tight_layout()
    # plt.savefig(f'./figures/{station}/spec_{st_ppsd[0].id}_{st_ppsd[0].stats.starttime}_{st_ppsd[-1].stats.endtime}.eps')

# colours = ['black', 'red']
# fig, axes = plt.subplots(nrows=len(comps))

# for comp, ax in zip(comps, axes):
#     for tr in st_bed.select(component=comp):
#         ax.plot(tr.times('matplotlib'), tr.data, color=colours[0])
#     for tr in st_uoe.select(component=comp):
#         ax.plot(tr.times('matplotlib'), tr.data, color=colours[1])
#     ax.xaxis_date()

# fig.autofmt_xdate()
# plt.show()
