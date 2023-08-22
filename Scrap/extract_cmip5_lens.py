#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Mon Aug  7 18:35:36 2023

@author: gliu
"""

# jan feb dec
# 1940 - 2023
# -76 to -45 
# 30 to 50


import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import haversine as hs   
from haversine import Unit
import sys
import glob

#%% Calculation choices
# ---------------------

# Processing Options
n_roll       = 1 # Model grids to roll the data by when computing gradients.

# ex: n_roll=1 shifts the latitudes by 1 for the gradient computation
#latslice     = [20,50]
#lonslice     = [-75,-50] # Use negative degrees for deg West
bbox         = [-76,-45,30,50]
ystart       = "1940-01-01"
yend         = "2023-12-31"

out_lonname  = "lon"
out_latname  = "lat"
varname      = "sshf" # sst or sshf
outpath      = "/stormtrack/data3/glliu/01_Data/ICTP/Data/"

#%%  Import Custom Modules
# ------------------------

sys.path.append("..")
import dataset_params as dparams
import grad_funcs as gf

#%% Part 1. Compute the Gulf Stream Index Values/Gradients
# --------------------------------------------------------
modelnames = dparams.indicts_keys
modeldicts = dparams.datasets_dict

# Loop by model
for m in range(len(modelnames)):
    modelname = modelnames[m]
    mdict     = modeldicts[modelname]
    latname   = mdict['latname']
    lonname   = mdict['lonname']
    
    # 1 Slice and merge the files --------------------
    if varname == "ts":
        mpath  = mdict['sst_path']
    elif varname == "sshf":
        mpath  = mdict['sshf_path']
        
    nclist = glob.glob(mpath + "/" + "*.nc")
    nclist.sort()
    nens   = len(nclist)
    print("Found %i files!" % (nens))
    
    sst_allens = []
    # Loop by ens -------------------------
    for e in range(nens):
        ds = xr.open_dataset(nclist[e])
        da = ds[mdict[varname]]
        
        
        # Flip longitudes
        if np.any(da.lon > 180):
            #print("Flipping Longitude")
            da = gf.lon360to180_xr(da,lonname=lonname)
        
        # Flip latitudes
        if (da[latname][1] - da[latname][0]) < 0:
            #print("Flipping Latitude to go from South to North")
            da = da.isel(**{latname:slice(None,None,-1)})
        
        # Select desired region
        da_reg = da
        #da_reg  = da.sel(**{latname : slice(bbox[2],bbox[3]),lonname : slice(bbox[0],bbox[1])})
        
        # Select desired time
        #da_reg  = da_reg.sel(time=slice(ystart,yend)).load()
        
        sst_allens.append(da_reg)
        # <End Ensemble Loop>
        
    # Combine dimensions
    sst_allens = xr.concat(sst_allens,dim='ens')
    savenetcdf = "%s%s_Crop_%s_global.nc" % (outpath,varname,modelname)
    sst_allens.to_netcdf(savenetcdf,
             encoding={mdict[varname]: {'zlib': True}})





