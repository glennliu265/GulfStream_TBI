#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:37:52 2023

@author: gliu
"""

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import haversine as hs   
from haversine import Unit
import sys

#%% Dataset Information and Processing Options

# Name of NetCDF, latitude, longitude
ncname      = "era5_sst_shf_dec_jan_feb1940_2023_025deg.nc" #"ERA5_sst_test.nc"
latname     = "latitude"
lonname     = "longitude"
varname     = "sst"

# Output NetCDF will use these names...
out_latname = "lat"
out_lonname = "lon"

# Processing Options
n_roll      = 1 # Model grids to roll the data by when computing gradients.
# ex: n_roll=1 shifts the latitudes by 1 for the gradient computation
latslice    = [20,50]
lonslice    = [-75,-50] # Use negative degrees for deg West

#%% Specify the user

user = "Glenn" # 

if user == "Glenn":
    
    path    = r'/Users/gliu/ICTP_temp/'#"/Users/gliu/Dropbox (MIT)/Glenn Liu’s files/Home/Work_Portable/ICTP/Project/Data/" # Path to Data
    outpath = path#"/Users/gliu/Dropbox (MIT)/Glenn Liu’s files/Home/Work_Portable/ICTP/Project/Data/Processed/" # Path for output of processed data
    figpath = r'/Users/gliu/ICTP_temp/figs/'#"/Users/gliu/Dropbox (MIT)/Glenn Liu’s files/Home/Work_Portable/ICTP/Project/Figures/" # Path to output figures
    
# Copy this section and add your own local paths, if wanted -------------------
elif user == "YourName": 
    
    path    = "Path To Data"
    outpath = "Path for output of processed data"
    figpath = "Path to output figures"

# -----------------------------------------------------------------------------
elif user == None:
    path    = "../../Data/"
    outpath = "../../Data/Processed"
    figpath = "../../Figures/"
    
#%% Functions

sys.path.append("..")
from grad_funcs import get_gs_coords_alltime,get_total_gradient
import grad_funcs as gf

#%%  Load the data
sst     = xr.open_dataset(path+ncname).load()
sst     = sst[varname] # Turn into data array

#%% Get the latitude locations and the gulf stream

# Compute Variables
lat_max,max_gradient,sst_grads = get_gs_coords_alltime(sst,n_roll,return_grad=True,
                                                       lonname=lonname,latname=latname,lonslice=lonslice,latslice=latslice)

# Compute to kilometers
max_gradient   = max_gradient * 100
sst_grads      = sst_grads    * 100

# Retrieve latitudes
ntime,nlon     = lat_max.shape
lat_max_values = np.zeros((ntime,nlon)) * np.nan
for t in range(ntime):
    lat_indices = lat_max.isel(time=t).values
    lat_max_values[t,:] = sst_grads[latname].values[lat_indices]

#%% Make DataArray Outputs and save

# Make Data Array for max_gradient --------------------------------------------
dims = {'time':sst_grads.time.values,
      out_lonname:sst_grads[lonname].values}

attrs = {"long_name" :"max_gradient_along_gulfstream",
         "units"     :"Kelvin per kilometer",
         "varname"   :varname,
         "n_rolll"   :n_roll,
         "lonslice"  :lonslice,
         "latslice"  :latslice,
         }
da_maxgradient = xr.DataArray(max_gradient,
                              dims=dims,
                              coords = dims,
                              attrs=attrs,
                              name="max_gradient")

savename = "%sERA5_25deg_GSvariable_sst_maxgradient.nc" % (outpath)
da_maxgradient.to_netcdf(savename)

                              

# Make Data Array for latitude ------------------------------------------------
attrs = {"long_name" :"latitude_of_max_gradient",
         "units"     :"degrees North",
         "varname"   :varname,
         "n_rolll"   :n_roll,
         "lonslice"  :lonslice,
         "latslice"  :latslice,
         }
da_lat_max = xr.DataArray(lat_max,
                              dims=dims,
                              coords = dims,
                              attrs=attrs,
                              name="lat_max")
savename = "%sERA5_25deg_GSvariable_sst_latmax.nc" % (outpath)
da_lat_max.to_netcdf(savename)


# Make Data Array for latitude indices ------------------------------------------------
attrs = {"long_name" :"latitude_index_of_max_gradient",
         "units"     :"degrees North",
         "varname"   :"sst",
         "n_rolll"   :n_roll,
         "lonslice"  :lonslice,
         "latslice"  :latslice,
         }

da_lat_max = xr.DataArray(lat_max,
                              dims=dims,
                              coords = dims,
                              attrs=attrs,
                              name="lat_max_id")
savename = "%sERA5_25deg_GSvariable_sst_latmaxid.nc" % (outpath)
da_lat_max.to_netcdf(savename)



