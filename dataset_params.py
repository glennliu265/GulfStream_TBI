#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:28:40 2023

@author: gliu
"""

dict1 = {
    "modelname": "canesm2_lens",
    "sshf"     : "hfss",
    "sshf_path": "/stormtrack/data4/glliu/01_Data/CLIVAR_LE/canesm2_lens/Amon/hfss/",
    "sst"      : "ts",
    "sst_path" : "/stormtrack/data4/glliu/01_Data/CLIVAR_LE/canesm2_lens/Amon/ts/",
    "sst_units": "K",
    "latname"  : "lat",
    "lonname"  : "lon",
    "startyr"  : 1950,
    }

dict2 = {
    "modelname": "csiro_mk36_lens",
    "sshf"     : "hfss",
    "sshf_path": "/stormtrack/data4/glliu/01_Data/CLIVAR_LE/csiro_mk36_lens/Amon/ts",
    "sst"      : "ts",
    "sst_path" : "/stormtrack/data4/glliu/01_Data/CLIVAR_LE/csiro_mk36_lens/Amon/ts",
    "sst_units": "K",
    "latname"  : "lat",
    "lonname"  : "lon",
    "startyr"  : 1850,
    }

dict3 = {
    "modelname": "gfdl_esm2m_lens",
    "sshf"     : "hfss",
    "sshf_path": "/stormtrack/data4/glliu/01_Data/CLIVAR_LE/csiro_mk36_lens/Amon/ts",
    "sst"      : "ts",
    "sst_path" : "/stormtrack/data4/glliu/01_Data/CLIVAR_LE/gfdl_esm2m_lens/Amon/ts/",
    "sst_units": "K",
    "latname"  : "lat",
    "lonname"  : "lon",
    "startyr"  : 1850,
    }

indicts_datasets      = [dict1,dict2,]
indicts_keys     = [d["modelname"] for d in indicts_datasets]
vars_dict             = dict(zip(indicts_keys,indicts_datasets))