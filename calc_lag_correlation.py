#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute the lag correlation between ENSO and GSI

Created on Tue Aug  8 05:39:31 2023

@author: gliu
"""

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import haversine as hs   
from haversine import Unit
import sys
import glob
import pickle

sys.path.append("..")
import dataset_params as dparams
import grad_funcs as gf
from global_land_mask import globe

from matplotlib import colors
from cartopy import feature
from scipy import stats

#%%

e           = 0
vname       = "sshf"
datpath     = '/Users/gliu/ICTP_temp/'
figpath     = '/Users/gliu/ICTP_temp/figs/'
analysis_months = [3]
nc_name_sst     = "%sCESM1_Crop_sst_global_ens%02i.nc" % (datpath,e+1)
nc_name_shf     = "%sCESM1_Crop_sshf_global_ens%02i.nc" % (datpath,e+1)

#limask      = 

# Processing Options
n_roll      = 1 # Model grids to roll the data by when computing gradients.
# ex: n_roll=1 shifts the latitudes by 1 for the gradient computation
latslice    = [20,50]
lonslice    = [-75,-50] # Use negative degrees for deg West

latname  = "lat"
lonname  = "lon"

if vname == "sst":
    unit_conv = 1/100 # Per KM
    cbar_lbl  = r'$|\nabla$SST| ($\degree$ C (100 km)$^{-1}$)'
elif vname == "sshf":
    unit_conv = 1/1000 # J/m2
    cbar_lbl  = r'$|\nabla$SHF| (J m$^{-3}$)'#' (100 km)$^{-1}$)'


#%% A dumb function

def calc_leadlag(maxlag,gsi,nino34):
    lags = np.arange(maxlag+1)
    ntime = len(gsi)
    
    # Compute cases where ENSO leads
    ninoleads = []
    for l in range(len(lags)):
        gsi_in  = gsi[:(ntime-l)]
        nino_in = nino34[l:] 
        
        nino_in[np.isnan(nino_in)] = 0
        gsi_in[np.isnan(gsi_in)] = 0
        
        corr = np.corrcoef(nino_in,gsi_in)[0,1]
        ninoleads.append(corr)
    
    # Compute case where ENSO lags
    ninolags = []
    for l in range(len(lags)):
        gsi_in  = gsi[l:]
        nino_in = nino34[:(ntime-l)]
        
        
        nino_in[np.isnan(nino_in)] = 0
        gsi_in[np.isnan(gsi_in)] = 0
        
        corr = np.corrcoef(nino_in,gsi_in)[0,1]
        ninolags.append(corr)
        
        
    # Put it all together
    nlags        = len(lags)*2
    lags_all     = np.hstack([-1*lags[::-1],lags[1:]])
    leadlag_corr = np.hstack([np.flip(np.array(ninoleads)),np.array(ninolags)[1:]])
    
    return leadlag_corr,lags_all


def regress2ts(var,ts,normalizeall=0,nanwarn=1,verbose=True):
    # var = [lon x lat x time], ts = [time]
    
    # Anomalize and normalize the data (time series is assumed to have been normalized)
    if normalizeall == 1:
        varmean = np.nanmean(var,2)
        varstd  = np.nanstd(var,2)
        var = (var - varmean[:,:,None]) /varstd[:,:,None]
    
    # Get variable shapes
    if len(var.shape) > 2:
        reshapeflag = True
        if verbose:
            print("Lon and lat are uncombined!")
        londim = var.shape[0]
        latdim = var.shape[1]
    else:
        reshapeflag=False
    
    # Combine the spatial dimensions 
    if len(var.shape)>2:
        var = np.reshape(var,(londim*latdim,var.shape[2]))
    
    var_reg,_ = regress_2d(ts,var,nanwarn=nanwarn)
    
    
    # Reshape to match lon x lat dim
    if reshapeflag:
        var_reg = np.reshape(var_reg,(londim,latdim))
    

    return var_reg

def regress_2d(A,B,nanwarn=1,verbose=True):
    """
    Regresses A (independent variable) onto B (dependent variable), where
    either A or B can be a timeseries [N-dimensions] or a space x time matrix 
    [N x M]. Script automatically detects this and permutes to allow for matrix
    multiplication.
    
    Returns the slope (beta) for each point, array of size [M]
    
    
    """
    # Determine if A or B is 2D and find anomalies
    bothND = False # By default, assume both A and B are not 2-D.
    # Note: need to rewrite function such that this wont be a concern...
    
    # Compute using nan functions (slower)
    if np.any(np.isnan(A)) or np.any(np.isnan(B)):
        if nanwarn == 1:
            print("NaN Values Detected...")
    
        # 2D Matrix is in A [MxN]
        if len(A.shape) > len(B.shape):
            
            # Tranpose A so that A = [MxN]
            if A.shape[1] != B.shape[0]:
                A = A.T
            
            # Set axis for summing/averaging
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis
            Aanom = A - np.nanmean(A,axis=a_axis)[:,None]
            Banom = B - np.nanmean(B,axis=b_axis)
            
        # 2D matrix is B [N x M]
        elif len(A.shape) < len(B.shape):
            
            # Tranpose B so that it is [N x M]
            if B.shape[0] != A.shape[0]:
                B = B.T
            
            # Set axis for summing/averaging
            a_axis = 0
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.nanmean(A,axis=a_axis)
            Banom = B - np.nanmean(B,axis=b_axis)[None,:]
        
        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom = np.nansum(Aanom2,axis=a_axis)    
        
        # Calculate Beta
        beta = Aanom @ Banom / denom
            
        b = (np.nansum(B,axis=b_axis) - beta * np.nansum(A,axis=a_axis))/A.shape[a_axis]
    else:
        # 2D Matrix is in A [MxN]
        if len(A.shape) > len(B.shape):
            
            # Tranpose A so that A = [MxN]
            if A.shape[1] != B.shape[0]:
                A = A.T
                
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis
            Aanom = A - np.mean(A,axis=a_axis)[:,None]
            Banom = B - np.mean(B,axis=b_axis)
            
        # 2D matrix is B [N x M]
        elif len(A.shape) < len(B.shape):
            
            # Tranpose B so that it is [N x M]
            if B.shape[0] != A.shape[0]:
                B = B.T
            
            # Set axis for summing/averaging
            a_axis = 0
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.mean(A,axis=a_axis)
            Banom = B - np.mean(B,axis=b_axis)[None,:]
            
        # A is [P x N], B is [N x M]
        elif len(A.shape) == len(B.shape):
            if verbose:
                print("Note, both A and B are 2-D...")
            bothND = True
            if A.shape[1] != B.shape[0]:
                print("WARNING, Dimensions not matching...")
                print("A is %s, B is %s" % (str(A.shape),str(B.shape)))
                print("Detecting common dimension")
                # Get intersecting indices 
                intersect, ind_a, ind_b = np.intersect1d(A.shape,B.shape, return_indices=True)
                if ind_a[0] == 0: # A is [N x P]
                    A = A.T # Transpose to [P x N]
                if ind_b[0] == 1: # B is [M x N]
                    B = B.T # Transpose to [N x M]
                print("New dims: A is %s, B is %s" % (str(A.shape),str(B.shape)))
            
            # Set axis for summing/averaging
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.mean(A,axis=a_axis)[:,None]
            Banom = B - np.mean(B,axis=b_axis)[None,:]

        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom = np.sum(Aanom2,axis=a_axis)
        if bothND:
            denom = denom[:,None] # Broadcast
            
        # Calculate Beta
        beta = Aanom @ Banom / denom
            
        if bothND:
            b = (np.sum(B,axis=b_axis)[None,:] - beta * np.sum(A,axis=a_axis)[:,None])/A.shape[a_axis]
        else:
            b = (np.sum(B,axis=b_axis) - beta * np.sum(A,axis=a_axis))/A.shape[a_axis]
    
    return beta,b
#%%  Load and compute ERA 

# Load GSI, take mean gradient, and resample from daily --> monthly
era_gsi     = xr.open_dataset(datpath+"ERA5_25deg_GSvariable_sst_maxgradient.nc").load()
era_gsi     = era_gsi.mean('lon').max_gradient
era_gsi     = era_gsi.resample(time="M").mean('time').sel(time=slice('1940-01-01','2022-12-31')) # 998

# Load ENSO Index
era_nino   = xr.open_dataarray(datpath+"nino34_ERA5_1x1.nc")


#%% Compute the lead/lag correlation

era_ll,lags = calc_leadlag(36,era_gsi.values,era_nino.values)

# # Compute Significance Value
p       = 0.05
tails   = 2
dof     = len(era_gsi)
rhocrit = gf.ttest_rho(p,tails,dof)
#%% Plot it
xtks = np.arange(-36,37,6)

# Make the Plot
fig,ax = plt.subplots(1,1,constrained_layout=True)
#ax.plot(-1*lags[::-1],np.array(ninoleads)[::-1],label="ENSO Leads",lw=2.5,color="cornflowerblue")
#ax.plot(lags,np.array(ninolags),label="GSI Leads",lw=2.5,color="goldenrod")

ax.plot(lags,era_ll,label="ERA5 Interim")
ax.set_xlabel("<--          ENSO Leads || GSI Leads (months) -->")
ax.set_ylabel("Correlation")
ax.axhline([0],ls='solid',color="k")
ax.axhline([rhocrit],ls='dashed',color="gray",label=r"$\rho$ = %.2f" % (rhocrit))
ax.axhline([-rhocrit],ls='dashed',color="gray")
ax.axvline([0],ls='solid',color="k")
ax.set_xticks(xtks)
ax.set_xlim([xtks[0],xtks[-1]])
ax.legend()
ax.set_title("ENSO-GSI Lead-Lag, CESM1 Ens%02i" % (e+1))

figname = "%sERA5_ENSO-GSI_LeadLag_%s_ens%02i.png" % (figpath,vname,e+1)
plt.savefig(figname,bbox_tight='inches',dpi=150,transparent=True)


#%% Do the same with CESM

gsi_cesm = []

for e in range(40):
    ncname = "%sCESM1_GSvariable_sst_maxgradient_ens%02i.nc" % (datpath,e+1)
    ds     = xr.open_dataset(ncname).max_gradient
    ds     = ds.sel(time=slice('1940-02-01',"2023-01-01")).load().mean('lon')
    gsi_cesm.append(ds.values)
    
gsi_cesm = np.array(gsi_cesm)



#%% Load the SSTs

# Name of NetCDF, latitude, longitude
ncname      = "era5_sst_shf_dec_jan_feb1940_2023_025deg.nc" #"ERA5_sst_test.nc"
latname     = "latitude"
lonname     = "longitude"
varname     = "sst"

# Get the SSTs
sst         = xr.open_dataset(datpath+ncname)
sst         = sst[varname] # Turn into data array
sst         = sst.resample(time="M").mean('time').sel(time=slice('1940-01-01','2022-12-31'))
lons        = sst.longitude
lats        = sst.latitude
sst         = sst.values

#%% Regress to the ENSO Index

enso_index_in = era_nino / np.nanstd(era_nino)
var_reg       = regress2ts(sst.transpose(2,1,0),enso_index_in.values,normalizeall=0)

plt.pcolormesh(lons,lats,var_reg.T)




