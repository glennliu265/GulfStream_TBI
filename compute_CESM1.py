#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute ENSO and Gulf Stream indices for CESM1

Created on Tue Aug  8 00:00:38 2023

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

import cmocean
#%%

e           = 0
vname       = "sst"
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
    unit_conv = 100 # Per KM
    cbar_lbl  = r'$|\nabla$SST| ($\degree$ C (100 km)$^{-1}$)'
elif vname == "sshf":
    unit_conv = 1/1000 # J/m2
    cbar_lbl  = r'$|\nabla$SHF| (J m$^{-3}$)'#' (100 km)$^{-1}$)'

#%% Compute ENSO

# Load the SST
ds     = xr.open_dataset(nc_name_sst)
da_sst = ds['sst'].load()
lons   = ds[lonname]
lats   = ds[latname]

# Load the gradoent var
ds1   = xr.open_dataset(nc_name_shf)
da_shf = ds1['sshf'].load()

if vname == "sst":
    da = da_sst
elif vname == "sshf":
    da = da_shf

#Compute ENSO
nino34,nino_date,nina_date = gf.find_enso(da_sst)

# Select an analysis month
da_month = da.where(da.time.dt.month.isin(analysis_months),drop=True)

# Convert to indices and grab composites
nino_index,sst_nino,knino = gf.get_enso_dates(nino_date,da_month,return_id=True)
nina_index,sst_nina,knina = gf.get_enso_dates(nina_date,da_month,return_id=True)

#%% Create a mask

x   = lons.to_numpy()
x   = np.where(x>180,x-360,x)
x,y = np.meshgrid(x,lats)
mask = globe.is_ocean(y,x)
#mask = np.ones(x.shape)
#mask[globe.is_ocean(y,x)] = 1
#mask[~mask] = np.nan
#da  = da.where(globe.is_ocean(y,x)) ## mask out land values 

#%% Make a plot to check

sst        = da_sst
times      = np.arange(0,len(sst.time.values))
times_plot = times[::120]
times_lbl  = (times_plot/12 + 1920).astype(int)

# Make the Plot
fig,ax = plt.subplots(figsize=(12,4))
ax.plot(times,nino34)
ax.scatter(times[knino],nino34[knino],color="r",marker='o',s=15,alpha=0.8)
ax.scatter(times[knina],nino34[knina],color="b",marker='d',s=15,alpha=0.8)
ax.set_xticks(times_plot,labels=times_lbl)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Nino 3.4 index', fontsize=12)
plt.axhline(y=0.4,color='k')
plt.axhline(y=-0.4,color='k')
ax.set_title("Nino 3.4 Index in CESM1, Ensemble Member 01")

# Sorry I suck at date time in xarrays, getting cftime errors :( ....
# plt.plot(nino34.sel(time=nino_date).time,nino34.sel(time=nino_date),'o',markersize=5)
# plt.plot(nino34.sel(time=nina_date).time,nino34.sel(time=nina_date),'o',markersize=5)
# plt.xlim(sst.isel(time=0).time, sst.isel(time=-1).time)
#%% Check the global plots

fig,axs = plt.subplots(2,1,subplot_kw={'projection':ccrs.PlateCarree()})
plot_ssts = [sst_nino,sst_nina]
for ii in range(2):
    ax = axs[ii]
    pcm = ax.pcolormesh(lons,lats,plot_ssts[ii].mean('time'))
    cb = fig.colorbar(pcm,ax=ax)




#%% Compute the gradients

nina_grad = gf.get_total_gradient(sst_nina,n_roll,latname=latname,lonname=lonname)*unit_conv
nino_grad = gf.get_total_gradient(sst_nino,n_roll,latname=latname,lonname=lonname)*unit_conv

#%% Composite Things

nino_grad_mean = nino_grad.mean('time')#nino_grad.mean('time')
nina_grad_mean = nina_grad.mean('time')#nina_grad.mean('time')
nina_grad_std  = nina_grad.std('time')#nina_grad.std('time')

#%% Compute Significance

nino_grad_annual = nino_grad.groupby('time.year').mean('time')
nina_grad_annual = nina_grad.groupby('time.year').mean('time')

nino_years = nino_grad_annual.year.data
nina_years = nina_grad_annual.year.data
sig        = nino_grad_mean-nina_grad_mean

for i in range(10):
    rs_nino = np.random.choice(nino_years,10)
    rs_nina = np.random.choice(nina_years,10)
    comp = nino_grad_annual.where(nino_grad_annual.year.isin(rs_nino)).mean('year') - nina_grad_annual.where(nina_grad_annual.year.isin(rs_nina)).mean('year')
    sig = sig.where(np.sign(comp) == np.sign(sig))

#%% Plot Nino-nina

if vname == "sst":
    vlms = [-5e-1,5e-1]
elif vname == "sshf":
    vlms = [-5e-5,5e-5]
    

fig,ax = plt.subplots(figsize=(6,5),subplot_kw={'projection':ccrs.PlateCarree()})
im = ax.pcolormesh(lons,lats,
                   (nino_grad_mean-nina_grad_mean),
                   transform=ccrs.PlateCarree(),
                   vmin=vlms[0],vmax=vlms[-1],
                   cmap="RdBu_r")#norm=colors.CenteredNorm(0,0.8))#(0,0.0005))#,cmap='RdBu_r')

stip = ax.contourf(lons,lats,sig*mask,transform=ccrs.PlateCarree(),
                    hatches = ['..'],alpha=0,edgecolor='gray')

ax.coastlines()
ax.add_feature(feature.LAND,facecolor='k',zorder=1)#,transform=ccrs.PlateCarree())

ax.set_extent((-76,-50,31,48))
plt.colorbar(im,ax=ax,fraction=0.02, pad=0.04,extend='both',
             label = cbar_lbl)
ax.set_title('Difference in %s gradient (February, Nino-Nina), CESM1 Ens%02i' % (vname.upper(),e+1),fontsize=10)
plt.tight_layout()

figname = "%sCESM1_ENSOdiff_%s_ens%02i.png" % (figpath,vname,e+1)
plt.savefig(figname,bbox_tight='inches',dpi=150,transparent=True)

plt.show()


#%% Plot Contour of SST composites for each state

clvls = np.arange(275,296,1)
fig,axs = plt.subplots(1,2,figsize=(12,6),subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)

for ii in range(2):
    ax = axs[ii]
    
    if ii == 0:
        plotsst  = sst_nino.mean('time')
        plotgrad = nino_grad_mean
        title    = "Nino Composite"
    else:
        plotsst  = sst_nina.mean('time')
        plotgrad = nina_grad_mean
        title    = "Nina Composite"
    
    ax.set_extent((-76,-50,31,48))
    
    im = ax.pcolormesh(lons,lats,
                        plotgrad*mask,
                        transform=ccrs.PlateCarree(),
                        vmin=-5e-4,vmax=5e-4,
                        cmap="RdBu_r",)#norm=colors.CenteredNorm(0,0.8))#(0,0.0005))#,cmap='RdBu_r')
    plt.colorbar(im,ax=ax,fraction=0.045, pad=0.04,extend='both',orientation='horizontal',
                  label = cbar_lbl)
    
    cl = ax.contour(lons,lats,plotsst*mask,colors='k',levels=clvls,transform=ccrs.PlateCarree())
    #ax.clabel(cl,clvls[::2])
    
    # Add Gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.75, color='black', alpha=0.5, linestyle=':', draw_labels=True)
    gl.top_labels=False
    gl.right_labels=False
    if ii == 1:
        gl.left_labels=False
    
    ax.set_title(title,fontsize=20)
    ax.add_feature(feature.LAND,facecolor='gray',zorder=1)#,transform=ccrs.PlateCarree())

figname = "%sCESM1_ENSOComposites_%s_ens%02i.png" % (figpath,vname,e+1)
plt.savefig(figname,bbox_tight='inches',dpi=150,transparent=True)


#%% Compute the Gulf Str for CESM1

# Compute Variables
lat_max,max_gradient,sst_grads = gf.get_gs_coords_alltime(da_sst,n_roll,return_grad=True,
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
    
    
# # Retrieve max gradients for sshf
# _,_,shf_grads = gf.get_gs_coords_alltime(da_shf,n_roll,return_grad=True,
#                                                        lonname=lonname,latname=latname,lonslice=lonslice,latslice=latslice)

# max_gradient_shf = np.zeros((ntime,nlon)) * np.nan
# for t in range(ntime):
    
#     lat_indices = lat_max.isel(time=t).values
#     check       = shf_grads.lat.values[lat_indices]
#     max_gradient_shf[t,:] = shf_grads.isel(**{'time':t,'lat':lat_indices})#.values[t,lat_indices,:]
    


#%%

# Make Data Array for max_gradient --------------------------------------------
dims = {'time'     :sst_grads.time.values,
        'lon'      :sst_grads[lonname].values}
attrs = {"long_name" :"max_gradient_along_gulfstream",
         "units"     :"Kelvin per kilometer",
         "varname"   :"sst",
         "n_rolll"   :n_roll,
         "lonslice"  :lonslice,
         "latslice"  :latslice,
         }

da_maxgradient = xr.DataArray(max_gradient,
                              dims=dims,
                              coords = dims,
                              attrs=attrs,
                              name="max_gradient")

savename = "%sCESM1_GSvariable_sst_maxgradient_ens%02i.nc" % (datpath,e+1)
da_maxgradient.to_netcdf(savename)


# Make Data Array for latitude ------------------------------------------------
attrs = {"long_name" :"latitude_of_max_gradient",
         "units"     :"degrees North",
         "varname"   :"sst",
         "n_rolll"   :n_roll,
         "lonslice"  :lonslice,
         "latslice"  :latslice,
         }
da_lat_max = xr.DataArray(lat_max_values,
                              dims=dims,
                              coords = dims,
                              attrs=attrs,
                              name="lat_max")
savename = "%sCESM1_GSvariable_sst_latmax_ens%02i.nc" % (datpath,e+1)
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
savename = "%sCESM1_GSvariable_sst_latmaxid_ens%02i.nc" % (datpath,e+1)
da_lat_max.to_netcdf(savename)

#%% Do Lag Correlation beteween ENSO Index and Gulf Stream Index


# Compute Gulf Stream Index (Temperature based)
gsi = max_gradient.mean(1)

fig,ax = plt.subplots(1,1)
ax.plot(times[2::12],gsi[2::12],label="Gulf Stream Index")
ax.plot(times[2::12],nino34[2::12],label="Nino3.4")
ax.legend()

#%%

lags = np.arange(36)
nlags = len(lags)*2
leadlag_corr = np.zeros(nlags)

# Fill nan points with zeros


# Compute Significance Value
p     = 0.05
tails = 2
dof   = len(gsi)
rhocrit = gf.ttest_rho(p,tails,dof)

# Compute cases where ENSO leads
ninoleads = []
for l in range(len(lags)):
    gsi_in  = gsi[:(ntime-l)]
    nino_in = nino34[l:] 
    
    nino_in[np.isnan(nino_in)] = 0
    
    corr = np.corrcoef(nino_in,gsi_in)[0,1]
    ninoleads.append(corr)
    
# Compute case where ENSO lags
ninolags = []
for l in range(len(lags)):
    gsi_in  = gsi[l:]
    nino_in = nino34[:(ntime-l)]
    nino_in[np.isnan(nino_in)] = 0
    corr = np.corrcoef(nino_in,gsi_in)[0,1]
    ninolags.append(corr)
    
#%% Plot Lead Lag

xtks = np.arange(-36,37,6)

# Make the Plot
fig,ax = plt.subplots(1,1,constrained_layout=True)
ax.plot(-1*lags[::-1],np.array(ninoleads)[::-1],label="ENSO Leads",lw=2.5,color="cornflowerblue")
ax.plot(lags,np.array(ninolags),label="GSI Leads",lw=2.5,color="goldenrod")
ax.set_xlabel("GSI Leads (months)")
ax.set_ylabel("Correlation")
ax.axhline([0],ls='solid',color="k")
ax.axhline([rhocrit],ls='dashed',color="gray",label=r"$\rho$ = %.2f" % (rhocrit))
ax.axhline([-rhocrit],ls='dashed',color="gray")
ax.axvline([0],ls='solid',color="k")
ax.set_xticks(xtks)
ax.set_xlim([xtks[0],xtks[-1]])
ax.legend()
ax.set_title("ENSO-GSI Lead-Lag, CESM1 Ens%02i" % (e+1))

figname = "%sCESM1_ENSO-GSI_LeadLag_%s_ens%02i.png" % (figpath,vname,e+1)
plt.savefig(figname,bbox_tight='inches',dpi=150,transparent=True)


#%% Loop the computation of Gulf Strea MIndex

for e in range(40):
    ncname = "%sCESM1_Crop_sst_ens%02i.nc" % (datpath,e+1)
    da_sst = xr.open_dataset(ncname)['sst'].load()
        
    # Compute Variables
    lat_max,max_gradient,sst_grads = gf.get_gs_coords_alltime(da_sst,n_roll,return_grad=True,
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
        
    
    # Make Data Array for max_gradient --------------------------------------------
    dims = {'time'     :sst_grads.time.values,
            'lon'      :sst_grads[lonname].values}
    attrs = {"long_name" :"max_gradient_along_gulfstream",
             "units"     :"Kelvin per kilometer",
             "varname"   :"sst",
             "n_rolll"   :n_roll,
             "lonslice"  :lonslice,
             "latslice"  :latslice,
             }
    
    da_maxgradient = xr.DataArray(max_gradient,
                                  dims=dims,
                                  coords = dims,
                                  attrs=attrs,
                                  name="max_gradient")
    
    savename = "%sCESM1_GSvariable_sst_maxgradient_ens%02i.nc" % (datpath,e+1)
    da_maxgradient.to_netcdf(savename)
    
    
    # Make Data Array for latitude ------------------------------------------------
    attrs = {"long_name" :"latitude_of_max_gradient",
             "units"     :"degrees North",
             "varname"   :"sst",
             "n_rolll"   :n_roll,
             "lonslice"  :lonslice,
             "latslice"  :latslice,
             }
    da_lat_max = xr.DataArray(lat_max_values,
                                  dims=dims,
                                  coords = dims,
                                  attrs=attrs,
                                  name="lat_max")
    savename = "%sCESM1_GSvariable_sst_latmax_ens%02i.nc" % (datpath,e+1)
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
    savename = "%sCESM1_GSvariable_sst_latmaxid_ens%02i.nc" % (datpath,e+1)
    da_lat_max.to_netcdf(savename)


#%%




# im = ax.pcolormesh(lons,lats,
#                    (nino_grad_mean-nina_grad_mean),
#                    transform=ccrs.PlateCarree(),
#                    vmin=-1,vmax=1,
#                    cmap="RdBu_r",)#norm=colors.CenteredNorm(0,0.8))#(0,0.0005))#,cmap='RdBu_r')

# stip = ax.contourf(lons,lats,sig*mask,transform=ccrs.PlateCarree(),
#                     hatches = ['..'],alpha=0,edgecolor='gray')

# ax.coastlines()
# 

# ax.set_extent((-76,-50,31,48))
# plt.colorbar(im,ax=ax,fraction=0.02, pad=0.04,extend='both',
#              label = cbar_lbl)
# ax.set_title('Difference in %s gradient (February, Nino-Nina), CESM1 Ens%02i' % (vname.upper(),e+1),fontsize=10)
# plt.tight_layout()

# figname = "%sCESM1_ENSOdiff_%s_ens%02i.png" % (figpath,vname,e+1)
# plt.savefig(figname,bbox_tight='inches',dpi=150,transparent=True)