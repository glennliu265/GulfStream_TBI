#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 12:21:03 2023

@author: gliu
"""


import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

#from sklearn.metrics.pairwise import haversine_distances
import haversine as hs   
from haversine import Unit

#%% Set Paths

path = "/Users/gliu/Dropbox (MIT)/Glenn Liu’s files/Home/Work_Portable/ICTP/Project/Data/"
ncname  = "era5_sst_1940_2022_1deg.nc"


figpath = "/Users/gliu/Dropbox (MIT)/Glenn Liu’s files/Home/Work_Portable/ICTP/Project/Figures/"

#%% Functions


def get_gs_coords_alltime(da,n_roll,return_grad=True):
    
    '''
    INPUTS:
    da: a dataarray containing sst on a regular grid (dimensions nx * ny)
    m       : desired month for calculation
    n_roll  : grid boxes to roll by
    
    OUTPUTS:
    
    gs_coords: a dataarray containing lat/lon of maximum sst meridional gradient (dimensions nx * 2)
    
    '''
    
    # Compute Gradient
    da      = da.sst
    da_grad = get_total_gradient(da,n_roll)
    
    # Subset region, find max latitude for each longitude
    da_grad  = da_grad.sel(lat = slice(20,50), lon = slice(360-75,360-50))
    lats_max = da_grad.argmax(dim='lat')
    
    # Retrieve max gradient at each longitude
    ntime,nlon = lats_max.values.shape
    max_gradient = np.zeros((ntime,nlon)) * np.nan
    for t in range(ntime):
        lat_indices_t = lats_max.isel(time=t)
        grad_along_gs = da_grad.isel(time=t,lat=lat_indices_t)
        max_gradient[t,:] = grad_along_gs
    if return_grad:
        return lats_max,max_gradient,da_grad
    return lats_max,max_gradient



    
    lats_max = da_grad.isel(lat = lats_max)
    
    if return_grad:
        return lats_max,da_grad
    return lats_max


def get_total_gradient(da, n_roll):
    
    # Get Longitude
    lats = da.lat
    lats0 = np.squeeze(np.dstack((lats,np.zeros(len(lats)))))
    lats1 = np.squeeze(np.dstack((lats,np.ones(len(lats)))))
    
    # Distances
    xdist = hs.haversine_vector(lats0,lats1,Unit.KILOMETERS)
    ydist = 111*(lats[1]-lats[0]) # take distance between latitudes as 111 km
    
    # Compute Meridional
    y1    = da.roll({'lat':n_roll},roll_coords=False)
    y2    = da.roll({'lat':-1*n_roll},roll_coords=False)
    ygrad = (y1 - y2)/(2 * n_roll * ydist)
    #ygrad1 = (da.roll({'lat':n_roll}) - da.roll({'lat':-1*n_roll}))/(2 * n_roll * ydist)
    
    # Compute Zonal
    x1  = da.roll({'lon':n_roll},roll_coords=False)
    x2  = da.roll({'lon':-1*n_roll},roll_coords=False)
    xgrad = (x1 - x2)/(2 * n_roll * xdist[None,:,None])
    #xgrad1 = (da.roll({'lon':n_roll}) - da.roll({'lon':-1*n_roll}))/(2 * n_roll * xdist[None,:,None])
    
    # Get total gradient
    grad_tot = (xgrad**2 + ygrad**2)**0.5
    
    # Get maximum gradient
    
    
    
    return grad_tot

#%%  Load the data
n_roll  = 1
sst     = xr.open_dataset(path+ncname)

# Compute gradient
gradsst = get_total_gradient(sst.sst,n_roll)

#%% Get the latitude locations and the gulf stream

lat_max,max_gradient,sst_grads = get_gs_coords_alltime(sst,n_roll,return_grad=True)


#%% Visualize Gulf Stream Index (Mean) by Month

ntime,nlon = max_gradient.shape
nlat       = sst_grads.shape[1]
yrs        = 1940 + np.arange(0,ntime/12)

fig,ax = plt.subplots(1,1,figsize=(8,4))

for im in range(12):
    plot_index = np.nanmean(max_gradient.reshape(int(ntime/12),12,nlon)[:,im,:],1) * 100
    ax.plot(yrs,plot_index,label=im+1)

ax.legend(ncol = 4,fontsize=8)
ax.set_xlim([1940,2025])
ax.grid(True,ls="dotted")
ax.set_xlabel("Year")
ax.set_ylabel("Mean Gradient Across the Gulf Stream (K km$^{-1})$")
ax.set_title("Mean Gulf Stream Index in ERSST by Month")
figname = "%sMean_SST_ERSST.png" % figpath
plt.savefig(figname,dpi=150)

#%% Plot actual gradients of the gulf stream 

im   = 0
vlms = [0,5]
fig,axs = plt.subplots(3,4,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(18,9),
                       constrained_layout=True)
for im in range(12):
    ax = axs.flatten()[im]
    
    ax.coastlines()
    ax.set_extent((-80,-50,32,50))
    
    plotsst = sst_grads.values.reshape(int(ntime/12),12,nlat,nlon)[:,im,:,:].mean(0) * 100
    pcm = ax.pcolormesh(sst_grads.lon,sst_grads.lat,plotsst,vmin=vlms[0],vmax=vlms[1])
    
    for y in range(len(yrs)):
        plot_index = sst_grads.lat.values[lat_max.values.reshape(int(ntime/12),12,nlon)[y,im,:]]
        
        ax.scatter(lat_max.lon,plot_index,alpha=0.3,transform=ccrs.PlateCarree(),color="k")
        #plot_index = np.nanmean(max_gradient.reshape(int(ntime/12),12,nlon)[:,im,:],1)
        #ax.plot(yrs,plot_index,label=im+1)
    ax.set_title("Month %i" % (im+1))

cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.026,pad=0.01)
cb.set_label("Mean Meridional Gradient of SST (K km$^{-1})$")

figname = "%sMean_SST_ERSST_Gulf_StreamPath.png" % figpath
plt.savefig(figname,dpi=150)

#%% Scrap ----------------------------------------------------------------------

#gradsst.sst.isel(time=0).plot()


fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
vgrad_mean = gradsst.mean('time')
im = ax.pcolormesh(vgrad_mean.lon,vgrad_mean.lat,vgrad_mean*100)
#im2 = ax.plot(vgrad_mean.lon,vgrad_mean.lat.isel(lat = lats_max),transform=ccrs.PlateCarree())
#im2 = ax.contour(sst.lon,sst.lat,sst_dec.mean('time'),levels=20,c='r')
ax.set_extent((-80,-40,20,50))
ax.coastlines()
plt.colorbar(im,ax=ax)


#%%


klon = 22
ktime = 223


sst_array = da.values

fig,ax = plt.subplots(1,1)

ax.plot(da.roll({'lat':1}).isel(time=ktime,lon=klon),label="roll 1")
ax.plot(da.roll({'lat':-1}).isel(time=ktime,lon=klon),label="roll -1")

#ax.plot(da.roll(1,dim='lat').isel(lat=klat,lon=klon),label="roll 1")
#ax.plot(da.roll(-1,dim='lat').isel(lat=klat,lon=klon),label="roll -1")


ygrad = np.roll(sst_array,1,1) - np.roll(sst_array,1,-1)

#ax.plot(np.roll(sst_array,1,1)[ktime,:,klon],label="roll 1")
#ax.plot(np.roll(sst_array,-1,1)[ktime,:,klon],label="roll -1")
#
ax.set_xlabel("Latitude")

ax.legend()

#ax.set_xlim([80,100])

#%% Scrap

da = sst.sst
n_roll = 1


lats  = da.lat
lats0 = np.squeeze(np.dstack((lats,np.zeros(len(lats)))))
lats1 = np.squeeze(np.dstack((lats,np.ones(len(lats)))))

#xdist = haversine_distances(lats0,lats1) * 6371

xdist = hs.haversine_vector(lats0,lats1,Unit.KILOMETERS)
ydist = 111*(lats[1]-lats[0]) # take distance between latitudes as 111 km

# Compute Meridional
y1    = da.roll({'lat':n_roll},roll_coords=False)
y2    = da.roll({'lat':-1*n_roll},roll_coords=False)
ygrad = (y1 - y2)/(2 * n_roll * ydist)
#ygrad1 = (da.roll({'lat':n_roll}) - da.roll({'lat':-1*n_roll}))/(2 * n_roll * ydist)


# Compute Zonal
x1  = da.roll({'lon':n_roll},roll_coords=False)
x2  = da.roll({'lon':-1*n_roll},roll_coords=False)
xgrad = (x1 - x2)/(2 * n_roll * xdist[None,:,None])
#xgrad1 = (da.roll({'lon':n_roll}) - da.roll({'lon':-1*n_roll}))/(2 * n_roll * xdist[None,:,None])

grad_tot = (xgrad**2 + ygrad**2)**0.5

#%%


#ygrad      = da.diff('lat',n=n_roll)
#ygrad_roll = np.roll(sst_array,axis=1,shift=1) - np.roll(sst_array,axis=1,shift=-1)


x1 = da.roll({'lat':n_roll},roll_coords=False)
x2 = da.roll({'lat':-1*n_roll},roll_coords=False)



fig,ax = plt.subplots(1,1)




ax.plot(x1.lat,x1.isel(lon=klon,time=ktime),label="roll 1")
ax.plot(x2.lat,x2.isel(lon=klon,time=ktime),label="roll -1",ls='dashed')
#ax.plot(x1.lat,(x1-x2).isel(lon=klon,time=ktime),label="diff")
ax.legend()

#ax.plot(da.roll({'lat':-1}).isel(time=ktime,lon=klon),label="roll -1")
