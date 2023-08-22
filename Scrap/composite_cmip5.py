#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 21:57:01 2023

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import xarray as xr

from grad_funcs import get_gs_coords_alltime,get_total_gradient
from global_land_mask import globe

from matplotlib import colors
from cartopy import feature
from scipy import stats

#%% 

modelname       = "canesm2_lens"#"csiro_mk36_lens"

## months to perform analysis, for full year set analysis_months = range(1,13)

analysis_months = [2]
#figtitle        = 'SHF_ERA5_Feb.png'
#dslink         = '/Users/noahrosenberg/Downloads/era5_sst_1940_2022_1deg.nc'
#dslink          = 'SST_Crop_csiro_mk36_lens.nc'

datpath         = '/Users/gliu/ICTP_temp/'
nc_name         = datpath+"SST_Crop_canesm2_lens.nc"
nino_name       = datpath+"canesm2_lens_ENSO_nino_date.npy"
nina_name       = datpath+"canesm2_lens_ENSO_nina_date.npy"


unit_conv   = 1/1000 #unit scaling for the gradient, set to 1/1000 for m**-1 or 100 for (100 km)**-1
#SHF in J/m2 (1/1000)
#SST in per 100 km
nroll       = 1

latname = 'lat' #also rename these for grad calculation function
lonname = 'lon'
varname = 'ts'




#%% Load variable, subset months

#ds = xr.load_dataset(f'{dslink}',engine='netcdf4')
ds = xr.load_dataset(nc_name,engine='netcdf4')
ds = ds.where(ds.time.dt.month.isin(analysis_months),drop=True).load()

sst_all  = ds[varname] # set to data variable you want to look at
lons = ds[lonname] #rename based on latitude/longitude variables
lats = ds[latname]

dlon = lons[1].values - lons[0].values
dlat = lats[1].values - lats[0].values

#%% Make land mask (cant do this because I dont have the globe package)

x   = lons.to_numpy()
x   = np.where(x>180,x-360,x)
x,y = np.meshgrid(x,lats)
sst = sst_all.isel(ens=0).where(globe.is_ocean(y,x)) ## mask out land values 

#%% Get the Nino/Nina Dates
nino_dates = np.load(nino_name,allow_pickle=True) #read in files generated from Nino3.4.ipynb containing ENSO dates
nina_dates = np.load(nina_name,allow_pickle=True)


#%% get dates from Nino script and dataarray into same format ( just try one ensemble member)


# This is the scrap
# sst   = sst_all.isel(ens=e)

# nens  = len(nino_dates)
# #ntime = len(nino_dates[0])

# nino_months = [] #np.empty((nens,ntime),dtype='datetime64[M]') #* np.nan
# # for e in range(nens):
# #     nno_months = []

# # Retrieve months of each nino event
# ntime      = len(nino_dates[e])

# for t in range(ntime):
#     nino_months.append(nino_dates[e][t].astype('datetime64[M]'))
#     #nno_months.append()
# nino_months = np.array(nino_months,dtype='datetime64[M]')

# # Get first year of SST
# year0        = sst.time.dt.year.min()
# year0        = int(year0)

# #
# nino_indices      = nino_months.astype('int')+(1970-year0)*12

# sst_month_indices = (sst.time.dt.year-year0)*12 + sst.time.dt.month

# sst_nino     = sst.where(sst_month_indices.isin(nino_indices),drop=True)
# sst_nino     = sst_nino # []

e     = 0
sst   = sst_all.sel(ens=e)

def get_enso_dates(ndates,sst):
    
    
    # convert cftime to datetime64
    ntime  = len(ndates)
    enso_months = []
    for t in range(ntime):
        enso_months.append(ndates[t].astype('datetime64[M]'))
        #nno_months.append()
    enso_months = np.array(enso_months,dtype='datetime64[M]')
    
    # Get first year of SST
    year0        = sst.time.dt.year.min()
    year0        = int(year0)
    
    # Convert to indices
    enso_indices      = enso_months.astype('int')+(1970-year0)*12
    sst_month_indices = (sst.time.dt.year-year0)*12 + sst.time.dt.month

    # Subset ssts
    sst_enso     = sst.where(sst_month_indices.isin(enso_indices),drop=True)
    sst_enso     = sst_enso # []
    return enso_months,sst_enso

nino_indices,sst_nino =get_enso_dates(nino_dates[e],sst)
nina_indices,sst_nina =get_enso_dates(nina_dates[e],sst)

#%%

fig,ax = plt.subplots(figsize=(10,10),subplot_kw={'projection':ccrs.PlateCarree()})
im = ax.pcolormesh(lons,lats,sst_nino.mean('time').where(globe.is_ocean(y,x)),transform=ccrs.PlateCarree())
#im2 = ax.contour(sst.lon,sst.lat,tot_grad.mean('time'),levels=(1,288),c='r')

ax.set_extent((-80,-40,20,75))
plt.colorbar(im,ax=ax)
ax.coastlines()

#%% plot difference in gradients between 2 composites
nina_grad = get_total_gradient(sst_nina,nroll,latname=latname,lonname=lonname)*unit_conv
nino_grad = get_total_gradient(sst_nino,nroll,latname=latname,lonname=lonname)*unit_conv

nino_grad_mean = sst_nino.mean('time')#nino_grad.mean('time')
nina_grad_mean = sst_nina.mean('time')#nina_grad.mean('time')
nina_grad_std = sst_nina.std('time')#nina_grad.std('time')

pvals = np.zeros((len(lons),len(lats)))

for i in range(len(lons)):
    for j in range(len(lats)):
        pvals[i,j] = stats.kstest(nino_grad[:,j,i].to_numpy(), nina_grad[:,j,i].to_numpy()).pvalue


tot_grad = get_total_gradient(sst,nroll,latname=latname,lonname=lonname)*unit_conv

tot_grad_mean = tot_grad.mean('time')
tot_grad_std = tot_grad.std('time')

#nino_diff_zscore = abs(nino_grad_mean-nina_grad_mean)/nina_grad_std


#pvals = norm.sf(abs(nino_diff_zscore)) < 0.05

#%% Compute significance via bootstrapping
nino_grad_annual = nino_grad.groupby('time.year').mean('time')
nina_grad_annual = nina_grad.groupby('time.year').mean('time')

nino_years = nino_grad_annual.year.data
nina_years = nina_grad_annual.year.data

import random

sig = nino_grad_mean-nina_grad_mean


for i in range(5):
    rs_nino = np.random.choice(nino_years,10)
    rs_nina = np.random.choice(nina_years,10)
    comp = nino_grad_annual.where(nino_grad_annual.year.isin(rs_nino)).mean('year') - nina_grad_annual.where(nina_grad_annual.year.isin(rs_nina)).mean('year')
    sig = sig.where(np.sign(comp) == np.sign(sig))
    
    

#%%

fig,ax = plt.subplots(figsize=(6,5),subplot_kw={'projection':ccrs.PlateCarree()})
im = ax.pcolormesh(lons,lats,
                   nino_grad_mean-nina_grad_mean,
                   transform=ccrs.PlateCarree(),
                   #vmin=0,vmax=0.03,
                   cmap="RdBu_r",norm=colors.CenteredNorm(0,1))#(0,0.0005))#,cmap='RdBu_r')

stip = ax.contourf(lons,lats,(pvals < 0.01).T,transform=ccrs.PlateCarree(),
                    hatches = ['','.....'],levels = [0,0.5,5],alpha=0,edgecolor='gray')

ax.coastlines()
ax.add_feature(feature.LAND,facecolor='k')#,transform=ccrs.PlateCarree())

#ax.set_extent((-76,-40,31,48))
plt.colorbar(im,ax=ax,fraction=0.02, pad=0.04,extend='both',
             #label = r'$|\nabla$SHF| (J m$^{-3}$)')#' (100 km)$^{-1}$)')
             label = r'$|\nabla$SST| ($\degree$ C (100 km)$^{-1}$)')
ax.set_title('Difference in SHF gradient (February composite, Nino-Nina)',fontsize=10)
plt.tight_layout()
#plt.savefig(figtitle)

plt.show()
#%%

fig,ax = plt.subplots(figsize=(6,5),subplot_kw={'projection':ccrs.PlateCarree()})
im = ax.pcolormesh(lons,lats,
                   nino_grad_mean-nina_grad_mean,
                   transform=ccrs.PlateCarree(),
                   #vmin=0,vmax=0.03,
                   cmap="RdBu_r",norm=colors.CenteredNorm(0,0.8))#(0,0.0005))#,cmap='RdBu_r')

stip = ax.contourf(lons,lats,sig,transform=ccrs.PlateCarree(),
                    hatches = ['..'],alpha=0,edgecolor='gray')

ax.coastlines()
ax.add_feature(feature.LAND,facecolor='k')#,transform=ccrs.PlateCarree())


ax.set_extent((-76,-50,31,48))
plt.colorbar(im,ax=ax,fraction=0.02, pad=0.04,extend='both',
             label = r'$|\nabla$SHF| (J m$^{-3}$)')#' (100 km)$^{-1}$)')
             #label = r'$|\nabla$SST| ($\degree$ C (100 km)$^{-1}$)')
ax.set_title('Difference in SHF gradient (February composite, Nino-Nina)',fontsize=10)
plt.tight_layout()
#plt.savefig(figtitle)

plt.show()