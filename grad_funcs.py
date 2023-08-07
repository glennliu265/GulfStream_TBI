#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:37:52 2023

"""

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import haversine as hs   
from haversine import Unit

#%% Functions


def get_gs_coords_alltime(da,n_roll,return_grad=True,latname='lat',lonname='lon',
                          latslice=[20,50],lonslice=[-75,-50]):
    '''
    INPUTS:
    da: a dataarray containing sst on a regular grid (dimensions nx * ny)
    m       : desired month for calculation
    n_roll  : grid boxes to roll by
    
    OUTPUTS:
    
    gs_coords: a dataarray containing lat/lon of maximum sst meridional gradient (dimensions nx * 2)
    
    '''
    # Flip Latitude to go from -90 to 90
    if (da[latname][1] - da[latname][0]) < 0:
        print("Flipping Latitude to go from South to North")
        da = da.isel(**{latname:slice(None,None,-1)})
        
    # Flip longitude to go from -180 to 180
    if np.any(da[lonname]>180):
        print("Flipping Longitude to go from -180 to 180")
        newcoord = {lonname : ((da[lonname] + 180) % 360) - 180}
        da       = da.assign_coords(newcoord).sortby(lonname)
    
    # Compute Gradient
    da_grad = get_total_gradient(da,n_roll,latname=latname,lonname=lonname)
    
    # Subset region, find max latitude for each longitude
    da_grad  = da_grad.sel(**{latname : slice(latslice[0],latslice[1]),lonname : slice(lonslice[0],lonslice[1])})
    
    # Remove edge values (a dumb fix :(...)
    mask     = np.ones(da_grad.shape) 
    mask[:,[0,-1],:] = 0
    mask[:,:,[0,-1]] = 0
    da_grad = da_grad * mask
    
    # Get maximum latitudes
    lats_max = da_grad.argmax(dim=latname,skipna=True)
    
    # Retrieve max gradient at each longitude
    ntime,nlon  = lats_max.values.shape
    max_gradient = np.zeros((ntime,nlon)) * np.nan
    for t in range(ntime):
        lat_indices_t = lats_max.isel(time=t)
        grad_along_gs = da_grad.isel(**{'time':t,latname:lat_indices_t})
        max_gradient[t,:] = grad_along_gs
    if return_grad:
        return lats_max,max_gradient,da_grad
    return lats_max,max_gradient


def get_total_gradient(da, n_roll,latname='lat',lonname="lon"):
    
    # Get Longitude
    lats  = da[latname]
    lats0 = np.squeeze(np.dstack((lats,np.zeros(len(lats)))))
    lats1 = np.squeeze(np.dstack((lats,np.ones(len(lats)))))
    
    # Distances
    xdist = hs.haversine_vector(lats0,lats1,Unit.KILOMETERS)
    ydist = 111*(lats[1]-lats[0]) # take distance between latitudes as 111 km
    
    # Compute Meridional
    y1    = da.roll({latname:n_roll},roll_coords=False)
    y2    = da.roll({latname:-1*n_roll},roll_coords=False)
    ygrad = (y1 - y2)/(2 * n_roll * ydist)
    #ygrad1 = (da.roll({'lat':n_roll}) - da.roll({'lat':-1*n_roll}))/(2 * n_roll * ydist)
    
    # Compute Zonal
    x1  = da.roll({lonname:n_roll},roll_coords=False)
    x2  = da.roll({lonname:-1*n_roll},roll_coords=False)
    xgrad = (x1 - x2)/(2 * n_roll * xdist[None,:,None])
    #xgrad1 = (da.roll({'lon':n_roll}) - da.roll({'lon':-1*n_roll}))/(2 * n_roll * xdist[None,:,None])
    
    # Get total gradient
    grad_tot = (xgrad**2 + ygrad**2)**0.5
    
    return grad_tot

def lon360to180_xr(ds,lonname='lon'):
    # Based on https://stackoverflow.com/questions/53345442/about-changing-longitude-array-from-0-360-to-180-to-180-with-python-xarray
    ds.coords[lonname] = (ds.coords[lonname] + 180) % 360 - 180
    ds = ds.sortby(ds[lonname])
    return ds


def detrend(da, dim='time', deg=1):
    # detrend the dataset da along the time dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def find_nino(ts_nino34):
    
    '''
    The function looks for 6 consecutive months in which the index has been above 0.4 (Nino) or below 
    -0.4 (Nina). Once the events have been individuated, it creates a list containing the dates 
    corresponding to the events. 
    
    It returns:
    1) A dataarray containing monthly time series of Nino 3.4 index
    2) The list of dates corresponding to those months which have faced a El Nino event
    3) The list of dates corresponding to those months which have faced a La Nina event

    '''
    
    detect_nino = []
    dates_nino = []
    detect_nina = []
    dates_nina = []
    
    for i in range(0,len(ts_nino34)):
        if ts_nino34[i] > 0.4:
            # Put the date in which an event > 0.4 was detected into a list
            detect_nino.append(ts_nino34[i].time.values)
        else:
            if len(detect_nino) > 5:
                # If the list contains more than 5 months, 
                dates_nino.append(detect_nino)
            detect_nino = []

        if ts_nino34[i] < -0.4:
            detect_nina.append(ts_nino34[i].time.values)
        else:
            if len(detect_nina) > 5:
                dates_nina.append(detect_nina)
            detect_nina = []
            
    flat_nino = [dat for innerList in dates_nino for dat in innerList]
    flat_nina = [dat for innerList in dates_nina for dat in innerList]
    
    return flat_nino, flat_nina

def find_enso(sst,plotfig=False):
    
    '''
    
    sst is a dataarray containing the time series of global sst 
    
    '''
    # First I select the region used to define Nino3.4
    sst34 = sst.sel(lat=slice(-5,5), lon= slice(190,240))
    
    # Detrend, deseasonalize data and make field mean, make rolling mean on the time-series
    sst34 = detrend(sst34)
    mon_deseason = (sst34.groupby('time.month') - sst34.groupby('time.month').mean()).mean(dim=['lon','lat'])
    anomalies = mon_deseason - mon_deseason.mean(dim=['time'])
    nino34 = anomalies.rolling(time=6,center=True).mean()
    
    # Get the timeseries of Nino3.4 index and retrieves the dates corresponding to El Nino/La Nina events
    nino_date, nina_date = find_nino(nino34)
    
    # For a graphical visualization of the results
    if plotfig:
        fig,ax = plt.subplots(figsize=(11,7))
        ax.plot(nino34.time,nino34)
        plt.plot(nino34.sel(time=nino_date).time,nino34.sel(time=nino_date),'o',markersize=5)
        plt.plot(nino34.sel(time=nina_date).time,nino34.sel(time=nina_date),'o',markersize=5)
        plt.axhline(y=0.4,color='k')
        plt.axhline(y=-0.4,color='k')
        plt.xlim(sst.isel(time=0).time, sst.isel(time=-1).time)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Nino3.4 index', fontsize=12)
    
    
    return nino34, nino_date, nina_date
