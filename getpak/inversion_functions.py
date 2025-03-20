# This module contains the functions that will be used to invert from reflectances to water quality parameters
# It was imported from the WaterQuality package (https://github.com/cordmaur/WaterQuality)
# The dictionary `functions`` will be used by the waterquality module:
# function = {
#   'name_of_the_parameter': {'function': any_function}, 'units': 'units to be displayed in the report',
#   'name_of_the_parameter2': .....
# }

# Any bands can be used to compute the final value
# Available bands in Sentinel2 are:
# Aerosol, Blue, Green, Red, RedEdg1, RedEdg2, RedEdg3, Nir, Nir2, Mir, Mir2

import numpy as np


#### CDOM ####
# Brezonik et al. 2005
def cdom_brezonik(Blue, RedEdge2):
    cdom = np.exp(1.872 - 0.830 * np.log(Blue / RedEdge2))
    return cdom


#### Chlorophyll-a ####
# 2-band ratio by Gilerson et al. (2010)
def chl_gilerson2(Red, RedEdg1, a=0.022, b=1.124):
    chl = (0.7864 * (RedEdg1 / Red) / a - 0.4245 / a) ** b
    return chl


# 3-band ratio by Gilerson et al. (2010)
def chl_gilerson3(Red, RedEdg1, RedEdg2, a=113.36, b=-16.45, c=1.124):
    chl = (a * (RedEdg2 / (Red - RedEdg1)) + b) ** c
    return chl


# Gitelson
def chl_gitelson(Red, RedEdg1, RedEdg2):
    chl = 23.1 + 117.4 * (1 / Red - 1 / RedEdg1) * RedEdg2
    return chl


# Gitelson and Kondratyev, Dall'Olmo et al. (2003)
def chl_gitelson2(Red, RedEdg1, a=61.324, b=-37.94):
    chl = a * (RedEdg1 / Red) + b
    return chl


# 2-band semi-analytical by Gons et al. (2003, 2005)
def chl_gons(Red, RedEdg1, RedEdg3, a=1.063, b=0.016, aw665=0.40, aw708=0.70):
    bb = 1.61 * RedEdg3 / (0.082 - 0.6 * RedEdg3)
    chl = ((RedEdg1 / Red) * (aw708 + bb) - aw665 - bb ** a) / b
    return chl


# 2-band squared band ration by Gurlin et al. (2011)
def chl_gurlin(Red, RedEdg1, a=25.28, b=14.85, c=-15.18):
    chl = (a * (RedEdg1 / Red) ** 2 + b * (RedEdg1 / Red) + c)
    return chl


# JM Hybride 1
def chl_h1(Red, RedEdge1, RedEdge2):
    res = RedEdge1 - Red  # B5 - B4
    chl = np.zeros_like(res)
    chl[res < 0] = 115.107 * RedEdge2[res < 0] * (1 / Red[res < 0] - 1 / RedEdge1[res < 0]) + 16.56
    chl[res >= 0] = 115.794 * RedEdge2[res >= 0] * (1 / Red[res >= 0] - 1 / RedEdge1[res >= 0]) + 20.678
    return chl


# JM Hybride 2
def chl_h2(Red, RedEdge1, RedEdge2):
    res = RedEdge1 - Red  # B5 - B4
    chl = np.zeros_like(res)
    chl[res < 0] = 46.859 * RedEdge1[res < 0] / Red[res < 0] - 29.916
    chl[res >= 0] = 115.794 * RedEdge2[res >= 0] * (1 / Red[res >= 0] - 1 / RedEdge1[res >= 0]) + 20.678
    return chl


# NDCI, Mishra and Mishra (2012)
def chl_ndci(Red, RedEdg1, a=14.039, b=86.115, c=194.325):
    index = (RedEdg1 - Red) / (RedEdg1 + Red)
    chl = (a + b * index + c * (index * index))
    return chl


# OC2
def chl_OC2(Blue, Green, a=0.2389, b=-1.9369, c=1.7627, d=-3.0777, e=-0.1054):
    X = np.log10(Blue / Green)
    chl = 10 ** (a + b * X + c * X ** 2 + d * X ** 3 + e * X ** 4)
    return chl


#### SPM ####

# Binding et al. (2010)
def spm_binding2010(RedEdge2):
    spm = 51.162 * (1 / (0.554 * 0.019)) * (2.8 * RedEdge2 * np.pi / ((0.54 * 128.123 / np.pi) + 0.48 * RedEdge2 * np.pi) - 0.00027)
    return spm

# Condé et al. (2019)
def spm_conde(Red, a=2.45, b=22.3):
    """
    Following the calibration at reservoirs in the Paranapanema river in Condé et al. (2019)
    Values between 1.9 and 48 NTU
    """
    turb = a * np.exp(b * Red * np.pi)  # from rho_w to Rrs
    return turb

# Dogliotti et al. (2015)
def spm_dogliotti(Red, Nir2):
    """Switching semi-analytical-algorithm computes turbidity from red and NIR band

    following Dogliotti et al., 2015
    :param Red : surface Reflectances Red  band [dl]
    :param Nir2: surface Reflectances NIR band [dl]
    :return: turbidity in FNU
    """

    limit_inf, limit_sup = 0.05 / np.pi, 0.07 / np.pi
    a_low, c_low = 228.1, 0.1641
    a_high, c_high = 3078.9, 0.2112

    t_low = spm_nechad(Red, a_low, c_low)
    t_high = spm_nechad(Nir2, a_high, c_high)
    w = (Red - limit_inf) / (limit_sup - limit_inf)
    t_mixing = (1 - w) * t_low + w * t_high

    t_low[Red >= limit_sup] = t_high[Red >= limit_sup]
    t_low[(Red >= limit_inf) & (Red < limit_sup)] = t_mixing[(Red >= limit_inf) & (Red < limit_sup)]
    # t_low[t_low > 4000] = 0
    return t_low


def spm_dogliotti_S2(Red, Nir2):
    """Switching semi-analytical-algorithm computes turbidity from red and NIR band
    following Dogliotti et al., 2015
    The coefficients were recalibrated for Sentinel-2 by Nechad et al. (2016)

    :return: turbidity in FNU
    """

    limit_inf, limit_sup = 0.05 / np.pi, 0.07 / np.pi
    a_low, c_low = 610.94, 0.2324
    a_high, c_high = 3030.32, 0.2115

    t_low = spm_nechad(Red, a_low, c_low)
    t_high = spm_nechad(Nir2, a_high, c_high)
    w = (Red - limit_inf) / (limit_sup - limit_inf)
    t_mixing = (1 - w) * t_low + w * t_high

    t_low[Red >= limit_sup] = t_high[Red >= limit_sup]
    t_low[(Red >= limit_inf) & (Red < limit_sup)] = t_mixing[(Red >= limit_inf) & (Red < limit_sup)]
    # t_low[t_low > 4000] = 0
    return t_low

# Nechad et al. (2010)
def spm_nechad(Red, a=610.94, c=0.2324):
    spm = a * Red / (1 - (Red / c))
    return spm

# Jiang et al. (2021)
# QAA based on turbidity OWT
def spm_jiang2021(Aerosol, Blue, Green, Red, RedEdge2, Nir2, mode='pixel'):
    # Constants of water absorption and backscattering
    aw = {"Aerosol": 0.00515124, "Blue": 0.01919594, "Green": 0.06299986, "Red": 0.41395333, "RedEdge1": 0.70385758,
          "RedEdge2": 2.71167020, "RedEdge3": 2.62000141, "Nir2": 4.61714226}
    bbw = {"Aerosol": 0.00215037, "Blue": 0.00138116, "Green": 0.00078491, "Red": 0.00037474, "RedEdge1": 0.00029185,
           "RedEdge2": 0.00023499, "RedEdge3": 0.00018516, "Nir2": 0.00012066}

    # Conversions and calculations:
    bands = [Aerosol, Blue, Green, Red, RedEdge2, Nir2]
    band_names = ['Aerosol', 'Blue', 'Green', 'Red', 'RedEdge2', 'Nir2']
    # subsurface remote sensing reflectance
    rrs = {wave: band / (0.52 + 1.7 * band) for wave, band in zip(band_names, bands)}
    # ratio of backscattering coefficient to the sum of backscattering and absorption coefficients
    u = {key: (-0.0895 + np.sqrt((0.089 ** 2) + 4 * 0.125 * value)) / (2 * 0.125) for key, value in rrs.items()}
    # estimation of Rrs(620) - empirical
    est620 = 1.693846e+02 * (Red ** 3) - 1.557556e+01 * (Red ** 2) + 1.316727e+00 * Red + 1.484814e-04

    # Functions
    def QAA_560(pos):
        x = np.log10((rrs["Aerosol"][pos[0], pos[1]] + rrs["Blue"][pos[0], pos[1]]) /
                     (rrs["Green"][pos[0], pos[1]] + 5 * rrs["Red"][pos[0], pos[1]] * rrs["Red"][pos[0], pos[1]] /
                      rrs["Blue"][pos[0], pos[1]]))
        a560 = aw["Green"] + 10 ** (-1.146 - 1.366 * x - 0.469 * (x ** 2))
        bbp560 = ((u["Green"][pos[0], pos[1]] * a560) / (1 - u["Green"][pos[0], pos[1]])) - bbw["Green"]
        one_tss = 94.48785 * bbp560
        wave = np.full(len(pos[0]), 560, dtype='float32')
        return np.array([a560, bbp560, wave, one_tss])

    def QAA_665(pos):
        a665 = aw["Red"] + 0.39 * ((Red[pos[0], pos[1]] / (Aerosol[pos[0], pos[1]] + Blue[pos[0], pos[1]])) ** 1.14)
        bbp665 = ((u["Red"][pos[0], pos[1]] * a665) / (1 - u["Red"][pos[0], pos[1]])) - bbw["Red"]
        one_tss = 113.87498 * bbp665
        wave = np.full(len(pos[0]), 665, dtype='float32')
        return np.array([a665, bbp665, wave, one_tss])

    def QAA_740(pos):
        bbp740 = (((u["RedEdge2"][pos[0], pos[1]] * aw["RedEdge2"]) / (1 - u["RedEdge2"][pos[0], pos[1]])) -
                  bbw["RedEdge2"])
        one_tss = 134.91845 * bbp740
        wave = np.full(len(pos[0]), 740, dtype='float32')
        aw740 = np.full(len(pos[0]), aw["RedEdge2"], dtype='float32')
        return np.array([aw740, bbp740, wave, one_tss])

    def QAA_865(pos):
        bbp865 = ((u["Nir2"][pos[0], pos[1]] * aw["Nir2"]) / (1 - u["Nir2"][pos[0], pos[1]])) - bbw["Nir2"]
        one_tss = 166.07382 * bbp865
        wave = np.full(len(pos[0]), 865, dtype='float32')
        aw865 = np.full(len(pos[0]), aw["Nir2"], dtype='float32')
        return np.array([aw865, bbp865, wave, one_tss])

    # Main function
    tss = np.zeros([4, Red.shape[0], Red.shape[1]], dtype='float32')
    # pixel-wise
    if mode == 'pixel':
        # generalisation
        ind = np.where(~np.isnan(Red))
        tss[:, ind[0], ind[1]] = QAA_740(ind)
        # first test
        ind = np.where(Blue > Green)
        tss[:, ind[0], ind[1]] = QAA_560(ind)
        # second test
        ind = np.where(Blue > est620)
        tss[:, ind[0], ind[1]] = QAA_665(ind)
        # third test
        ind = np.where((RedEdge2 > Blue) & (RedEdge2 > 0.010))
        tss[:, ind[0], ind[1]] = QAA_865(ind)
    # lake-wise: mean lake reflectance only for choosing a model to apply
    elif mode == 'polygon':
        ind = np.where(~np.isnan(Red))
        med = np.nanmedian(np.vstack((Aerosol.flatten(), Blue.flatten(), Green.flatten(), Red.flatten(),
                                      RedEdge2.flatten(), Nir2.flatten())), axis=1)
        est620 = 1.693846e+02 * (med[3] ** 3) - 1.557556e+01 * (med[3] ** 2) + 1.316727e+00 * med[3] + 1.484814e-04
        if med[1] > med[2]:  # Blue > Green
            tss[:, ind[0], ind[1]] = QAA_560(ind)
        elif med[1] > est620:  # Blue > est620
            tss[:, ind[0], ind[1]] = QAA_665(ind)
        elif (med[4] > med[1]) & (med[4] > 0.010):  # RedEdge2 > Blue and RedEdge2 > 0.010
            tss[:, ind[0], ind[1]] = QAA_865(ind)
        else:
            tss[:, ind[0], ind[1]] = QAA_740(ind)

    return tss[3, :, :]

# Jiang 2021 using only the green band (QAA 560 nm)
def spm_jiang2021_green(Aerosol, Blue, Green, Red):
    # Constants of water absorption and backscattering
    aw = {"Aerosol": 0.00515124, "Blue": 0.01919594, "Green": 0.06299986, "Red": 0.41395333}
    bbw = {"Aerosol": 0.00215037, "Blue": 0.00138116, "Green": 0.00078491, "Red": 0.00037474}

    # Conversions and calculations:
    bands = [Aerosol, Blue, Green, Red]
    band_names = ['Aerosol', 'Blue', 'Green', 'Red']
    # subsurface remote sensing reflectance
    rrs = {wave: band / (0.52 + 1.7 * band) for wave, band in zip(band_names, bands)}
    # ratio of backscattering coefficient to the sum of backscattering and absorption coefficients
    u = {key: (-0.0895 + np.sqrt((0.089 ** 2) + 4 * 0.125 * value)) / (2 * 0.125) for key, value in rrs.items()}

    x = np.log10((rrs["Aerosol"] + rrs["Blue"]) / (rrs["Green"] + 5 * rrs["Red"] * rrs["Red"] / rrs["Blue"]))
    a560 = aw["Green"] + 10 ** (-1.146 - 1.366 * x - 0.469 * (x ** 2))
    bbp560 = ((u["Green"] * a560) / (1 - u["Green"])) - bbw["Green"]
    tss = 94.48785 * bbp560

    return tss

# Jiang 2021 using only the red band (QAA 665 nm)
def spm_jiang2021_red(Aerosol, Blue, Green, Red):
    # Constants of water absorption and backscattering
    aw = {"Aerosol": 0.00515124, "Blue": 0.01919594, "Green": 0.06299986, "Red": 0.41395333}
    bbw = {"Aerosol": 0.00215037, "Blue": 0.00138116, "Green": 0.00078491, "Red": 0.00037474}

    # Conversions and calculations:
    bands = [Aerosol, Blue, Green, Red]
    band_names = ['Aerosol', 'Blue', 'Green', 'Red']
    # subsurface remote sensing reflectance
    rrs = {wave: band / (0.52 + 1.7 * band) for wave, band in zip(band_names, bands)}
    # ratio of backscattering coefficient to the sum of backscattering and absorption coefficients
    u = {key: (-0.0895 + np.sqrt((0.089 ** 2) + 4 * 0.125 * value)) / (2 * 0.125) for key, value in rrs.items()}

    a665 = aw["Red"] + 0.39 * ((Red / (Aerosol + Blue)) ** 1.14)
    bbp665 = ((u["Red"] * a665) / (1 - u["Red"])) - bbw["Red"]
    tss = 113.87498 * bbp665

    return tss

# SPM Alves e Santos 2024
def spm_madeira(Red, Nir2):
    spm = 945.1 * ((Nir2 / Red) ** 1.9463)

    return spm

# SPM SEN3R
def _spm_modis(Nir, Red):
    return 759.12 * ((Nir / Red) ** 1.92)

def _power(x, a, b, c):
    return a * x ** b + c

def spm_s3(Red, Nir2, cutoff_value=0.027, cutoff_delta=0.007, low_params=None, high_params=None):
    b665 = Red
    b865 = Nir2

    if cutoff_delta == 0:
        transition_coef = np.where(b665 <= cutoff_value, 0, 1)

    else:
        transition_range = (cutoff_value - cutoff_delta, cutoff_value + cutoff_delta)
        transition_coef = (b665 - transition_range[0]) / (transition_range[1] - transition_range[0])
        transition_coef = np.clip(transition_coef, 0, 1)

    low_params = [2.79101975e+05, 2.34858344e+00, 4.20023206e+00] if low_params is None else low_params
    high_params = [848.97770516, 1.79293191, 8.2788616] if high_params is None else high_params

    low = _power(b665, *low_params)
    high = _spm_modis(b865, b665)

    spm = (1 - transition_coef) * low + transition_coef * high
    return spm

# SPM hibrid
def spm_severo(Red, Nir2):
    if (Nir2 / Red) > 0.3:
        spm = 16.01 * 2.71828 ^ (4.99 * Nir2 / Red)
    else:
        spm = (2719.8 * Nir2) / (1 - (Nir2 / 21.1)) + 2.08

    return spm

def vectorized_spm_sev(Red, Nir2):
    # Ensure the inputs are numpy arrays (if not already)
    Red = np.asarray(Red)
    Nir2 = np.asarray(Nir2)
    # Avoid divide-by-zero errors
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = Nir2 / Red
        # Compute SPM values with vectorized operations
        spm = np.where(
            ratio > 0.3,
            16.01 * np.exp(4.99 * ratio),  # Use np.exp instead of ^
            (2719.8 * Nir2) / (1 - (Nir2 / 21.1)) + 2.08
        )
    return spm

# SPM Zhang et al. (2014)
def spm_zhang2014(RedEdge1, a=362507, b=2.3222):
    spm = a * (RedEdge1 ** b)
    return spm

# Secchi disk depth
# def secchi_lee():
#     """
#
#     """
#
#     return secchi


functions = {

    'CHL_Gitelson2': {
        'function': chl_gitelson2,
        'units': 'mg/m³'
    },

    'CHL_OC2': {
        'function': chl_OC2,
        'units': 'mg/m³'
    },

    'CHL_Gilerson2': {
        'function': chl_gilerson2,
        'units': 'mg/m³'
    },

    'CHL_Gilerson3': {
        'function': chl_gilerson3,
        'units': 'mg/m³'
    },

    'CHL_Gons': {
        'function': chl_gons,
        'units': 'mg/m³'
    },

    'CHL_Gurlin': {
        'function': chl_gurlin,
        'units': 'mg/m³'
    },

    'CHL_Hybrid1': {
        'function': chl_h1,
        'units': 'mg/m³'
    },

    'CHL_Hybrid2': {
        'function': chl_h2,
        'units': 'mg/m³'
    },

    'CHL_NDCI': {
        'function': chl_ndci,
        'units': 'mg/m³'
    },

    'CDOM_Brezonik': {
        'function': cdom_brezonik,
        'units': '',
    },

    'SPM_Nechad': {
        'function': spm_nechad,
        'units': 'mg/l'
    },

    'SPM_S3': {
        'function': spm_s3,
        'units': 'mg/l'
    },

    'TURB_Dogliotti': {
        'function': spm_dogliotti,
        'units': 'FNU'
    },

    'TURB_Dogliotti_S2': {
        'function': spm_dogliotti_S2,
        'units': 'FNU'
    },

    'TURB_Conde': {
        'function': spm_conde,
        'units': 'NTU'
    }
}
