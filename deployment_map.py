from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib_map_utils.core.scale_bar import scale_bar
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from shapely.geometry import box as shapely_box
import pandas as pd
import geopandas as gpd


def deployment_map():
    crs = ccrs.PlateCarree()
    # os_request = cimgt.OSM()
    os_request = cimgt.StadiaMapsTiles(apikey='c3e0a719-1e2c-4797-9276-16a7292f91a8', style='stamen_terrain')
    
    fig_w = 10.0
    fig_h = 8.0
    rel_im_size = 0.4
    
    fig, ax = plt.subplots(figsize=(fig_w,fig_h), subplot_kw={'projection': os_request.crs})
    
    # min_lon, max_lon, min_lat, max_lat
    ext = [1.365, 1.390, 52.895, 52.910]
    ax.set_extent(ext)
    ax.add_image(os_request, 15)
    # ax.set_rasterized(True)

    gridlines = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.2, color='black', alpha=0.5, linestyle='--')
    gridlines.xformatter = LONGITUDE_FORMATTER
    gridlines.yformatter = LATITUDE_FORMATTER
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {'size':10}
    gridlines.ylabel_style = {'size':10}
    
    
    fig_w = ext[1] - ext[0]
    fig_h = ext[3] - ext[2]
    
    # offset_x = (fig_h-fig_w)/fig_h * rel_im_size/2 if fig_h > fig_w else 0
    # offset_y = (fig_w-fig_h)/fig_w * rel_im_size/2 if fig_w > fig_h else 0
    
    # axins = inset_axes(ax, width=f'{rel_im_size*100}%', height=f'{rel_im_size*100}%', borderpad=0, bbox_to_anchor=(0 + offset_x, 0 + offset_y, 1, 1), bbox_transform=ax.transAxes, axes_class=cartopy.mpl.geoaxes.GeoAxes, axes_kwargs=dict(projection=crs))
    
    fig.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.06)
    bbox = ax.get_position()
    inset_size = min(bbox.width, bbox.height) * rel_im_size
    x0 = bbox.x1 - inset_size
    y0 = bbox.y1 - inset_size
    axins = fig.add_axes([x0, y0, inset_size, inset_size], projection=crs)
    
    axins.add_feature(cartopy.feature.LAND, facecolor='lightgrey', zorder=0)
    axins.coastlines(edgecolor='black', lw=0.7, zorder=3)
    # axins.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.65, lw=0.1, zorder=-1)
    inset_extent = [-10, 2, 50, 59]
    axins.set_extent(inset_extent)
    axins.tick_params(top=False, right=False, left=False, bottom=False)
    axins.tick_params(labelleft=False, labelbottom=False, labelright=False, labeltop=False)
    
    bbox = shapely_box(ext[0]-0.1, ext[2]-0.1, ext[1]+0.1, ext[3]+0.1)
    axins.add_geometries([bbox], ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2, zorder=5)

    das_gps = pd.read_csv('./results/checkpoints/final_gps_track_1m.csv', index_col=0)
    geometry = gpd.points_from_xy(das_gps.lon, das_gps.lat)
    gdf = gpd.GeoDataFrame(das_gps, geometry=geometry, crs='EPSG:4326')
    ax.plot(gdf.geometry.x.values, gdf.geometry.y.values, transform=ccrs.PlateCarree(), color='blue', linewidth=1.5, zorder=6)
    for cha in [402, 2043]:
        row = das_gps.loc[cha]
        ax.scatter(row['lon'], row['lat'], transform=ccrs.PlateCarree(), color='blue', s=12, zorder=7)
        if cha == 750:
            ax.text(row['lon'], row['lat'], str(cha), transform=ccrs.PlateCarree(),
                fontsize=14, ha='left', va='center', color='blue', zorder=9)
        else:
            ax.text(row['lon'], row['lat'], str(cha), transform=ccrs.PlateCarree(),
                fontsize=14, ha='left', va='bottom', color='blue', zorder=9)
    
    # nodes_gps = pd.read_csv('./results/checkpoints/nodes.csv', index_col=0, header=0)
    # ax.scatter(nodes_gps['Lon'].values, nodes_gps['Lat'].values, transform=ccrs.PlateCarree(), color='black', s=12, zorder=7)
    # for i, row in nodes_gps.iterrows():
    #     ax.text(row['Lon'], row['Lat'], str(row.name)[-5:], transform=ccrs.PlateCarree(),
    #             fontsize=14, ha='right', va='top', color='black', zorder=9)
    
    # wacr_lat, wacr_lon = 52.725, 0.627
    # axins.scatter(wacr_lon, wacr_lat, c='k', transform=ccrs.PlateCarree(), s=15, zorder=10, marker='o')
    # axins.text(wacr_lon, wacr_lat, 'WACR', transform=ccrs.PlateCarree(), fontsize=8, ha='right', va='top', color='black', zorder=9)
    
    # bedf_lat, bedf_lon = 52.25, 1.259
    # axins.scatter(bedf_lon, bedf_lat, c='k', transform=ccrs.PlateCarree(), s=15, zorder=10, marker='o')
    # axins.text(bedf_lon, bedf_lat, 'BEDF', transform=ccrs.PlateCarree(), fontsize=8, ha='right', va='top', color='black', zorder=9)
    
    # scale_bar(ax, style='boxes', location='upper left', bar={"projection": 'EPSG:4326'})
    ax.add_artist(ScaleBar(dx=1, location='upper left'))
    
    ### trim and sstrand arrows
    # ax.annotate("Trimingham", xytext=(1.3857, 52.901), xy=(1.3897, 52.9002), arrowprops=dict(arrowstyle="->", facecolor='black'), fontsize=12, transform=ccrs.PlateCarree())
    # ax.annotate("Sidestrand", xytext=(1.3671, 52.9076), xy=(1.3659, 52.90857), arrowprops=dict(arrowstyle="->", facecolor='black'), fontsize=12, transform=ccrs.PlateCarree())
    # ax.annotate("Forest", xytext=(1.382872, 52.898934), xy=(1.383683, 52.9), arrowprops=dict(arrowstyle="->", facecolor='black'), fontsize=12, transform=ccrs.PlateCarree())
    
    plt.show()

deployment_map()
