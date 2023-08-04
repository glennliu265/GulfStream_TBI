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


def get_gs_coords_alltime(da,n_roll,varname,return_grad=True,latname='lat',lonname='lon',
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
    da      = da[varname]
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