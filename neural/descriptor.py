
geometric = ['Di', 'Df', 'Dif', 'rho', 'VSA','GSA', 'VPOV', 'GPOV', 'POAV_vol_frac', 'PONAV_vol_frac', 'GPOAV','GPONAV', 'POAV', 'PONAV']
RAC = ['f-chi-0-all', 'f-chi-1-all', 'f-chi-2-all',
       'f-chi-3-all', 'f-Z-0-all', 'f-Z-1-all', 'f-Z-2-all', 'f-Z-3-all',
       'f-I-0-all', 'f-I-1-all', 'f-I-2-all', 'f-I-3-all', 'f-T-0-all',
       'f-T-1-all', 'f-T-2-all', 'f-T-3-all', 'f-S-0-all', 'f-S-1-all',
       'f-S-2-all', 'f-S-3-all', 'mc-chi-0-all', 'mc-chi-1-all',
       'mc-chi-2-all', 'mc-chi-3-all', 'mc-Z-0-all', 'mc-Z-1-all',
       'mc-Z-2-all', 'mc-Z-3-all', 'mc-I-0-all', 'mc-I-1-all', 'mc-I-2-all',
       'mc-I-3-all', 'mc-T-0-all', 'mc-T-1-all', 'mc-T-2-all', 'mc-T-3-all',
       'mc-S-0-all', 'mc-S-1-all', 'mc-S-2-all', 'mc-S-3-all',
       'D_mc-chi-0-all', 'D_mc-chi-1-all', 'D_mc-chi-2-all', 'D_mc-chi-3-all',
       'D_mc-Z-0-all', 'D_mc-Z-1-all', 'D_mc-Z-2-all', 'D_mc-Z-3-all',
       'D_mc-I-0-all', 'D_mc-I-1-all', 'D_mc-I-2-all', 'D_mc-I-3-all',
       'D_mc-T-0-all', 'D_mc-T-1-all', 'D_mc-T-2-all', 'D_mc-T-3-all',
       'D_mc-S-0-all', 'D_mc-S-1-all', 'D_mc-S-2-all', 'D_mc-S-3-all',
       'f-lig-chi-0', 'f-lig-chi-1', 'f-lig-chi-2', 'f-lig-chi-3', 'f-lig-Z-0',
       'f-lig-Z-1', 'f-lig-Z-2', 'f-lig-Z-3', 'f-lig-I-0', 'f-lig-I-1',
       'f-lig-I-2', 'f-lig-I-3', 'f-lig-T-0', 'f-lig-T-1', 'f-lig-T-2',
       'f-lig-T-3', 'f-lig-S-0', 'f-lig-S-1', 'f-lig-S-2', 'f-lig-S-3',
       'lc-chi-0-all', 'lc-chi-1-all', 'lc-chi-2-all', 'lc-chi-3-all',
       'lc-Z-0-all', 'lc-Z-1-all', 'lc-Z-2-all', 'lc-Z-3-all', 'lc-I-0-all',
       'lc-I-1-all', 'lc-I-2-all', 'lc-I-3-all', 'lc-T-0-all', 'lc-T-1-all',
       'lc-T-2-all', 'lc-T-3-all', 'lc-S-0-all', 'lc-S-1-all', 'lc-S-2-all',
        'lc-S-3-all', 'lc-alpha-0-all', 'lc-alpha-1-all', 'lc-alpha-2-all',
       'lc-alpha-3-all', 'D_lc-chi-0-all', 'D_lc-chi-1-all', 'D_lc-chi-2-all',
       'D_lc-chi-3-all', 'D_lc-Z-0-all', 'D_lc-Z-1-all', 'D_lc-Z-2-all',
       'D_lc-Z-3-all', 'D_lc-I-0-all', 'D_lc-I-1-all', 'D_lc-I-2-all',
       'D_lc-I-3-all', 'D_lc-T-0-all', 'D_lc-T-1-all', 'D_lc-T-2-all',
       'D_lc-T-3-all', 'D_lc-S-0-all', 'D_lc-S-1-all', 'D_lc-S-2-all',
       'D_lc-S-3-all', 'D_lc-alpha-0-all', 'D_lc-alpha-1-all',
       'D_lc-alpha-2-all', 'D_lc-alpha-3-all', 'func-chi-0-all',
       'func-chi-1-all', 'func-chi-2-all', 'func-chi-3-all', 'func-Z-0-all',
       'func-Z-1-all', 'func-Z-2-all', 'func-Z-3-all', 'func-I-0-all',
       'func-I-1-all', 'func-I-2-all', 'func-I-3-all', 'func-T-0-all',
       'func-T-1-all', 'func-T-2-all', 'func-T-3-all', 'func-S-0-all',
       'func-S-1-all', 'func-S-2-all', 'func-S-3-all', 'func-alpha-0-all',
       'func-alpha-1-all', 'func-alpha-2-all', 'func-alpha-3-all',
       'D_func-chi-0-all', 'D_func-chi-1-all', 'D_func-chi-2-all',
       'D_func-chi-3-all', 'D_func-Z-0-all', 'D_func-Z-1-all',
       'D_func-Z-2-all', 'D_func-Z-3-all', 'D_func-I-0-all', 'D_func-I-1-all',
       'D_func-I-2-all', 'D_func-I-3-all', 'D_func-T-0-all', 'D_func-T-1-all',
       'D_func-T-2-all', 'D_func-T-3-all', 'D_func-S-0-all', 'D_func-S-1-all',
       'D_func-S-2-all', 'D_func-S-3-all', 'D_func-alpha-0-all',
       'D_func-alpha-1-all', 'D_func-alpha-2-all', 'D_func-alpha-3-all']

metal_center = ['mc-chi-0-all', 'mc-chi-1-all',
       'mc-chi-2-all', 'mc-chi-3-all', 'mc-Z-0-all', 'mc-Z-1-all',
       'mc-Z-2-all', 'mc-Z-3-all', 'mc-I-0-all', 'mc-I-1-all', 'mc-I-2-all',
       'mc-I-3-all', 'mc-T-0-all', 'mc-T-1-all', 'mc-T-2-all', 'mc-T-3-all',
       'mc-S-0-all', 'mc-S-1-all', 'mc-S-2-all', 'mc-S-3-all',
       'D_mc-chi-0-all', 'D_mc-chi-1-all', 'D_mc-chi-2-all', 'D_mc-chi-3-all',
       'D_mc-Z-0-all', 'D_mc-Z-1-all', 'D_mc-Z-2-all', 'D_mc-Z-3-all',
       'D_mc-I-0-all', 'D_mc-I-1-all', 'D_mc-I-2-all', 'D_mc-I-3-all',
       'D_mc-T-0-all', 'D_mc-T-1-all', 'D_mc-T-2-all', 'D_mc-T-3-all',
       'D_mc-S-0-all', 'D_mc-S-1-all', 'D_mc-S-2-all', 'D_mc-S-3-all']

linker_center = ['lc-chi-0-all', 'lc-chi-1-all', 'lc-chi-2-all', 'lc-chi-3-all',
       'lc-Z-0-all', 'lc-Z-1-all', 'lc-Z-2-all', 'lc-Z-3-all', 'lc-I-0-all',
       'lc-I-1-all', 'lc-I-2-all', 'lc-I-3-all', 'lc-T-0-all', 'lc-T-1-all',
       'lc-T-2-all', 'lc-T-3-all', 'lc-S-0-all', 'lc-S-1-all', 'lc-S-2-all',
        'lc-S-3-all', 'lc-alpha-0-all', 'lc-alpha-1-all', 'lc-alpha-2-all',
       'lc-alpha-3-all', 'D_lc-chi-0-all', 'D_lc-chi-1-all', 'D_lc-chi-2-all',
       'D_lc-chi-3-all', 'D_lc-Z-0-all', 'D_lc-Z-1-all', 'D_lc-Z-2-all',
       'D_lc-Z-3-all', 'D_lc-I-0-all', 'D_lc-I-1-all', 'D_lc-I-2-all',
       'D_lc-I-3-all', 'D_lc-T-0-all', 'D_lc-T-1-all', 'D_lc-T-2-all',
       'D_lc-T-3-all', 'D_lc-S-0-all', 'D_lc-S-1-all', 'D_lc-S-2-all',
       'D_lc-S-3-all', 'D_lc-alpha-0-all', 'D_lc-alpha-1-all',
       'D_lc-alpha-2-all', 'D_lc-alpha-3-all']

functional_center = ['func-chi-0-all',
       'func-chi-1-all', 'func-chi-2-all', 'func-chi-3-all', 'func-Z-0-all',
       'func-Z-1-all', 'func-Z-2-all', 'func-Z-3-all', 'func-I-0-all',
       'func-I-1-all', 'func-I-2-all', 'func-I-3-all', 'func-T-0-all',
       'func-T-1-all', 'func-T-2-all', 'func-T-3-all', 'func-S-0-all',
       'func-S-1-all', 'func-S-2-all', 'func-S-3-all', 'func-alpha-0-all',
       'func-alpha-1-all', 'func-alpha-2-all', 'func-alpha-3-all',
       'D_func-chi-0-all', 'D_func-chi-1-all', 'D_func-chi-2-all',
       'D_func-chi-3-all', 'D_func-Z-0-all', 'D_func-Z-1-all',
       'D_func-Z-2-all', 'D_func-Z-3-all', 'D_func-I-0-all', 'D_func-I-1-all',
       'D_func-I-2-all', 'D_func-I-3-all', 'D_func-T-0-all', 'D_func-T-1-all',
       'D_func-T-2-all', 'D_func-T-3-all', 'D_func-S-0-all', 'D_func-S-1-all',
       'D_func-S-2-all', 'D_func-S-3-all', 'D_func-alpha-0-all',
       'D_func-alpha-1-all', 'D_func-alpha-2-all', 'D_func-alpha-3-all']


property_ = ["KVRH"]
'''
file name
"feature/toacco_geo2_mit_order.csv"
"feature/toacco_geo_chem_erase_mit_order.csv"

name
"geo"
"geo_RACs"
"geo_topology"
"geo_building_block"
...

'''