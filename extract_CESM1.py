#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Get CESM1 LENs data, historical and RCP85
Created on Mon Aug  7 23:36:27 2023

@author: gliu
"""

#%% 
import sys
import xarray as xr
import numpy as np

#%%

sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
import amv.loaders as dl
from amv import proc

sys.path.append("/stormtrack/data3/glliu/01_Data/ICTP/GulfStream_TBI")
import grad_funcs as gf

#%% Crop Parameters and Paths

varnames        = ["TS","SHFLX"]
out_varnames    = ["sst","sshf"]
datpath         = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/atm/proc/tseries/monthly/"
outpath         =  "/stormtrack/data3/glliu/01_Data/ICTP/Data/"

bbox            = [-80,-45,30,50]
ystart          = "1920-01-01"
yend            = "2024-01-31"



mnum_rcp    = np.concatenate([np.arange(1,36),np.arange(101,106)])
ntime_rcp   = 1140

mnum_htr    = np.concatenate([np.arange(1,36),np.arange(101,108)])
ntime_htr   = 1032


latname = 'lat'
lonname = 'lon'

crop_global   = True
calc_enso     = True
#%% Load 1 ensemble member


for e in range(42):
    for v in range(2):
        
        # Crop a region and save
        vname  =  varnames[v]
        
        # First, load Historical
        ds_htr = dl.load_htr(vname,mnum_htr[e],)
        ds_htr = ds_htr.sel(time=slice(ystart,None))
        
        # Second, load RCP85
        ds_rcp = dl.load_rcp85(vname,mnum_rcp[e],)
        ds_rcp = ds_rcp.sel(time=slice(None,yend))
        
        # Merge the two scenarios
        ds_merge = xr.concat([ds_htr,ds_rcp],dim='time')
        ds_merge = ds_merge.load()
        
        # Flip the longitude
        if np.any(ds_merge.lon > 180):
            #print("Flipping Loengitude")
            ds_merge = proc.lon360to180_xr(ds_merge,lonname='lon')
            
        # Select region
        if crop_global:
            ds_reg = ds_merge
        else:
            ds_reg  = ds_merge.sel(**{latname : slice(bbox[2],bbox[3]),lonname : slice(bbox[0],bbox[1])})
        
        # Rename
        ds_reg = ds_reg.rename(out_varnames[v])#({vname:out_varnames[v]})
        
        # Save it
        if crop_global:
            savenetcdf = "%sCESM1_Crop_%s_global_ens%02i.nc" % (outpath,out_varnames[v],e+1)
        else:
            savenetcdf = "%sCESM1_Crop_%s_ens%02i.nc" % (outpath,out_varnames[v],e+1)
            
        if calc_enso: # Don't save the global file
            ds_reg.to_netcdf(savenetcdf,
                     encoding={out_varnames[v]: {'zlib': True}})
            
        
        # Compute ENSO Indices
        if calc_enso and out_varnames[v] == "sst":
            
            # Get the enso indices
            nino34,nino_date,nina_date = gf.find_enso(ds_reg)
            
            # Save the outputs
            nino34.to_netcdf(outpath + "nino34_CESM1_ens%02i.nc" % (e+1)) #netCDF format
            np.save("%snino_date_CESM1_ens%02i.npy" % (outpath,e+1),nino_date)
            np.save("%snina_date_CESM1_ens%02i.npy" % (outpath,e+1),nina_date)
            
            print("Completed Ens %02i for %s" % (e+1,out_varnames[v]))
            
            
        
