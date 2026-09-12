# This module contains the functions that will be used to invert from reflectances to water quality parameters
# It was imported from the WaterQuality package (https://github.com/cordmaur/WaterQuality)

# Any bands can be used to compute the final value
# Available bands in Sentinel2 are:
# Aerosol, Blue, Green, Red, RedEdge1, RedEdge2, RedEdge3, Nir, Nir2, Swir1, Swir2

import numpy as np


_NUMERICAL_DIAGNOSTIC_KEYS = (
    "algorithm_evaluations",
    "missing_or_masked_input",
    "zero_denominator",
    "nonpositive_log_argument",
    "invalid_fractional_power_base",
    "invalid_square_root_radicand",
    "numerical_calculation_overflow",
)


def _new_diagnostics(shape):
    return {
        key: int(np.size(np.empty(shape))) if key == "algorithm_evaluations" else 0
        for key in _NUMERICAL_DIAGNOSTIC_KEYS
    }


def _finish(values, diagnostics, return_diagnostics):
    values = np.asarray(values, dtype=float)
    diagnostics["finite_output_count"] = int(np.count_nonzero(np.isfinite(values)))
    if return_diagnostics:
        return values, diagnostics
    return values


def _broadcast_float(*values):
    return np.broadcast_arrays(*(np.asarray(value, dtype=float) for value in values))


def _count(diagnostics, key, mask):
    diagnostics[key] += int(np.count_nonzero(mask))


def _safe_power(base, exponent, valid, diagnostics):
    result = np.full(base.shape, np.nan, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        np.power(base, exponent, out=result, where=valid)
    _count(diagnostics, "numerical_calculation_overflow", valid & ~np.isfinite(result))
    return result

def _real_power_domain(base, exponent):
    # Exact integer-ness is intentional; no fractional tolerance is used.
    # Zero to a non-positive exponent is undefined, including 0 ** 0.
    base, exponent = np.broadcast_arrays(
        np.asarray(base, dtype=float), np.asarray(exponent, dtype=float)
    )
    finite = np.isfinite(base) & np.isfinite(exponent)
    integer = finite & (exponent == np.trunc(exponent))
    return finite & (
        (base > 0)
        | ((base < 0) & integer)
        | ((base == 0) & (exponent > 0))
    )


def _safe_exp10(exponent, valid, diagnostics):
    result = np.full(exponent.shape, np.nan, dtype=float)
    representable = valid & (exponent <= np.log10(np.finfo(float).max))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        np.power(10.0, exponent, out=result, where=representable)
    _count(diagnostics, "numerical_calculation_overflow", valid & ~representable)
    _count(diagnostics, "numerical_calculation_overflow", representable & ~np.isfinite(result))
    return result

#### CDOM ####
# Brezonik et al. 2005
def cdom_brezonik(Blue, RedEdge2):
    cdom = np.exp(1.872 - 0.830 * np.log(Blue / RedEdge2))
    return cdom


#### Chlorophyll-a ####
# 2-band ratio by Gilerson et al. (2010)
def chl_gilerson2(Red, RedEdge1, a=0.022, b=1.124, return_diagnostics=False):
    Red, RedEdge1, a, b = _broadcast_float(Red, RedEdge1, a, b)
    diagnostics = _new_diagnostics(Red.shape)
    finite = np.isfinite(Red) & np.isfinite(RedEdge1)
    _count(diagnostics, "missing_or_masked_input", ~finite)
    _count(diagnostics, "zero_denominator", finite & (Red == 0))
    parameter_valid = np.isfinite(a) & (a != 0) & np.isfinite(b)
    ratio = np.full(Red.shape, np.nan, dtype=float)
    ratio_valid = finite & (Red != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(RedEdge1, Red, out=ratio, where=ratio_valid)
    base = np.full(Red.shape, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(0.7864 * ratio - 0.4245, a, out=base,
                  where=ratio_valid & parameter_valid)
    power_domain = _real_power_domain(base, b)
    power_valid = ratio_valid & parameter_valid & power_domain
    _count(
        diagnostics, "invalid_fractional_power_base",
        ratio_valid & parameter_valid & np.isfinite(base) & ~power_domain,
    )
    _count(diagnostics, "missing_or_masked_input", ratio_valid & ~parameter_valid)
    chl = _safe_power(base, b, power_valid, diagnostics)
    return _finish(chl, diagnostics, return_diagnostics)

# 3-band ratio by Gilerson et al. (2010)
def chl_gilerson3(Red, RedEdge1, RedEdge2, a=113.36, b=-16.45, c=1.124):
    chl = (a * (RedEdge2 / (Red - RedEdge1)) + b) ** c
    return chl


# Gitelson
def chl_gitelson(Red, RedEdge1, RedEdge2):
    chl = 23.1 + 117.4 * (1 / Red - 1 / RedEdge1) * RedEdge2
    return chl


# Gitelson and Kondratyev, Dall'Olmo et al. (2003)
def chl_gitelson2(Red, RedEdge1, a=61.324, b=-37.94):
    chl = a * (RedEdge1 / Red) + b
    return chl


# 2-band semi-analytical by Gons et al. (2003, 2005)
def chl_gons(Red, RedEdge1, RedEdge3, a=1.063, b=0.016, aw665=0.40, aw708=0.70,
             return_diagnostics=False):
    Red, RedEdge1, RedEdge3, a, b, aw665, aw708 = _broadcast_float(
        Red, RedEdge1, RedEdge3, a, b, aw665, aw708
    )
    diagnostics = _new_diagnostics(Red.shape)
    finite = np.isfinite(Red) & np.isfinite(RedEdge1) & np.isfinite(RedEdge3)
    _count(diagnostics, "missing_or_masked_input", ~finite)
    bb_den = 0.082 - 0.6 * RedEdge3
    bb_den_zero = finite & (bb_den == 0)
    _count(diagnostics, "zero_denominator", bb_den_zero)
    bb = np.full(Red.shape, np.nan, dtype=float)
    bb_valid = finite & ~bb_den_zero
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(1.61 * RedEdge3, bb_den, out=bb, where=bb_valid)
    red_den_zero = finite & (Red == 0)
    _count(diagnostics, "zero_denominator", red_den_zero)
    ratio = np.full(Red.shape, np.nan, dtype=float)
    ratio_valid = finite & (Red != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(RedEdge1, Red, out=ratio, where=ratio_valid)
    parameter_valid = (
        np.isfinite(a) & np.isfinite(b) & (b != 0)
        & np.isfinite(aw665) & np.isfinite(aw708)
    )
    power_domain = _real_power_domain(bb, a)
    power_valid = bb_valid & ratio_valid & parameter_valid & power_domain
    _count(
        diagnostics, "invalid_fractional_power_base",
        bb_valid & parameter_valid & np.isfinite(bb) & ~power_domain,
    )
    _count(
        diagnostics, "missing_or_masked_input",
        (bb_valid & ratio_valid) & ~parameter_valid,
    )
    bb_power = _safe_power(bb, a, power_valid, diagnostics)
    valid = power_valid & np.isfinite(bb_power)
    chl = np.full(Red.shape, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(
            ratio * (aw708 + bb) - aw665 - bb_power, b,
            out=chl, where=valid,
        )
    _count(diagnostics, "numerical_calculation_overflow", valid & ~np.isfinite(chl))
    return _finish(chl, diagnostics, return_diagnostics)

# 2-band squared band ratio by Gurlin et al. (2011)
def chl_gurlin(Red, RedEdge1, a=25.28, b=14.85, c=-15.18):
    chl = (a * (RedEdge1 / Red) ** 2 + b * (RedEdge1 / Red) + c)
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
def chl_ndci(Red, RedEdge1, a=14.039, b=86.115, c=194.325):
    index = (RedEdge1 - Red) / (RedEdge1 + Red)
    chl = (a + b * index + c * (index * index))
    return chl


# OC2
def chl_OC2(Blue, Green, a=0.2389, b=-1.9369, c=1.7627, d=-3.0777, e=-0.1054,
            return_diagnostics=False):
    Blue, Green = _broadcast_float(Blue, Green)
    diagnostics = _new_diagnostics(Blue.shape)
    finite = np.isfinite(Blue) & np.isfinite(Green)
    _count(diagnostics, "missing_or_masked_input", ~finite)
    green_zero = finite & (Green == 0)
    _count(diagnostics, "zero_denominator", green_zero)
    ratio = np.full(Blue.shape, np.nan, dtype=float)
    ratio_valid = finite & ~green_zero
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(Blue, Green, out=ratio, where=ratio_valid)
    log_valid = ratio_valid & np.isfinite(ratio) & (ratio > 0)
    _count(diagnostics, "nonpositive_log_argument", ratio_valid & np.isfinite(ratio) & (ratio <= 0))
    X = np.full(Blue.shape, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.log10(ratio, out=X, where=log_valid)
    with np.errstate(over="ignore", invalid="ignore"):
        exponent = (((e * X + d) * X + c) * X + b) * X + a
    valid = log_valid & np.isfinite(exponent)
    _count(diagnostics, "numerical_calculation_overflow", log_valid & ~np.isfinite(exponent))
    chl = _safe_exp10(exponent, valid, diagnostics)
    return _finish(chl, diagnostics, return_diagnostics)


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
    :param Red : surface Reflectances Red band [dl]
    :param Nir2: surface Reflectances NIR band [dl]
    :return: turbidity in FNU
    """

    limit_inf, limit_sup = 0.05 / np.pi, 0.07 / np.pi
    a_low, c_low = 228.1, 0.1641
    a_high, c_high = 3078.9, 0.2112

    t_low = spm_nechad(Red, a_low, 0, c_low)
    t_high = spm_nechad(Nir2, a_high, 0, c_high)
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

    t_low = spm_nechad(Red, a_low, 0, c_low)
    t_high = spm_nechad(Nir2, a_high, 0, c_high)
    w = (Red - limit_inf) / (limit_sup - limit_inf)
    t_mixing = (1 - w) * t_low + w * t_high

    t_low[Red >= limit_sup] = t_high[Red >= limit_sup]
    t_low[(Red >= limit_inf) & (Red < limit_sup)] = t_mixing[(Red >= limit_inf) & (Red < limit_sup)]
    # t_low[t_low > 4000] = 0
    return t_low

# Nechad et al. (2010)
def spm_nechad(Red, a=355.85, b=1.74, c=0.1728):
    spm = a * (Red * np.pi) / (1 - ((Red * np.pi) / c)) + b
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
def spm_jiang2021_green(Aerosol, Blue, Green, Red, return_diagnostics=False):
    aw_green, bbw_green = 0.06299986, 0.00078491
    Aerosol, Blue, Green, Red = _broadcast_float(Aerosol, Blue, Green, Red)
    diagnostics = _new_diagnostics(Aerosol.shape)
    finite = np.isfinite(Aerosol) & np.isfinite(Blue) & np.isfinite(Green) & np.isfinite(Red)
    _count(diagnostics, "missing_or_masked_input", ~finite)
    rrs = {}
    rrs_valid = finite.copy()
    for name, band in (("Aerosol", Aerosol), ("Blue", Blue), ("Green", Green), ("Red", Red)):
        denominator = 0.52 + 1.7 * band
        zero = finite & (denominator == 0)
        _count(diagnostics, "zero_denominator", zero)
        valid = finite & ~zero
        value = np.full(Aerosol.shape, np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(band, denominator, out=value, where=valid)
        rrs[name] = value
        rrs_valid &= valid & np.isfinite(value)
    radicand = 0.089 ** 2 + 4 * 0.125 * rrs["Green"]
    sqrt_valid = rrs_valid & (radicand >= 0)
    _count(diagnostics, "invalid_square_root_radicand", rrs_valid & (radicand < 0))
    u_green = np.full(Aerosol.shape, np.nan, dtype=float)
    with np.errstate(invalid="ignore"):
        np.divide(-0.0895 + np.sqrt(np.where(sqrt_valid, radicand, np.nan)), 2 * 0.125,
                  out=u_green, where=sqrt_valid)
    red_over_blue = np.full(Aerosol.shape, np.nan, dtype=float)
    red_over_blue_valid = rrs_valid & (rrs["Blue"] != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(rrs["Red"] * rrs["Red"], rrs["Blue"], out=red_over_blue, where=red_over_blue_valid)
    log_den = rrs["Green"] + 5 * red_over_blue
    log_num = rrs["Aerosol"] + rrs["Blue"]
    log_arg = np.full(Aerosol.shape, np.nan, dtype=float)
    log_ratio_valid = rrs_valid & (rrs["Blue"] != 0) & np.isfinite(log_den) & (log_den != 0)
    _count(diagnostics, "zero_denominator", rrs_valid & (rrs["Blue"] == 0))
    _count(diagnostics, "zero_denominator", rrs_valid & (rrs["Blue"] != 0) & (log_den == 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(log_num, log_den, out=log_arg, where=log_ratio_valid)
    log_valid = log_ratio_valid & np.isfinite(log_arg) & (log_arg > 0) & np.isfinite(u_green)
    _count(diagnostics, "nonpositive_log_argument", log_ratio_valid & (log_arg <= 0))
    x = np.full(Aerosol.shape, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.log10(log_arg, out=x, where=log_valid)
    with np.errstate(over="ignore", invalid="ignore"):
        exponent = -1.146 - 1.366 * x - 0.469 * x * x
    a560 = aw_green + _safe_exp10(exponent, log_valid & np.isfinite(exponent), diagnostics)
    bbp_den = 1 - u_green
    bbp_valid = log_valid & np.isfinite(a560) & (bbp_den != 0)
    _count(diagnostics, "zero_denominator", log_valid & (bbp_den == 0))
    tss = np.full(Aerosol.shape, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        tss[bbp_valid] = 94.48785 * ((u_green[bbp_valid] * a560[bbp_valid]) / bbp_den[bbp_valid] - bbw_green)
    _count(diagnostics, "numerical_calculation_overflow", bbp_valid & ~np.isfinite(tss))
    return _finish(tss, diagnostics, return_diagnostics)

# Jiang 2021 using only the red band (QAA 665 nm)
def spm_jiang2021_red(Aerosol, Blue, Green, Red, return_diagnostics=False):
    # Green remains in the public signature for caller compatibility, but the
    # red QAA branch never reads it or makes it a validity dependency.
    aw_red, bbw_red = 0.41395333, 0.00037474
    Aerosol, Blue, Red = _broadcast_float(Aerosol, Blue, Red)
    diagnostics = _new_diagnostics(Aerosol.shape)
    finite = np.isfinite(Aerosol) & np.isfinite(Blue) & np.isfinite(Red)
    _count(diagnostics, "missing_or_masked_input", ~finite)
    rrs = {}
    rrs_valid = finite.copy()
    for name, band in (("Aerosol", Aerosol), ("Blue", Blue), ("Red", Red)):
        denominator = 0.52 + 1.7 * band
        zero = finite & (denominator == 0)
        _count(diagnostics, "zero_denominator", zero)
        valid = finite & ~zero
        value = np.full(Aerosol.shape, np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(band, denominator, out=value, where=valid)
        rrs[name] = value
        rrs_valid &= valid & np.isfinite(value)
    radicand = 0.089 ** 2 + 4 * 0.125 * rrs["Red"]
    sqrt_valid = rrs_valid & (radicand >= 0)
    _count(diagnostics, "invalid_square_root_radicand", rrs_valid & (radicand < 0))
    u_red = np.full(Aerosol.shape, np.nan, dtype=float)
    with np.errstate(invalid="ignore"):
        np.divide(
            -0.0895 + np.sqrt(np.where(sqrt_valid, radicand, np.nan)),
            2 * 0.125, out=u_red, where=sqrt_valid,
        )
    denominator = Aerosol + Blue
    zero = finite & (denominator == 0)
    _count(diagnostics, "zero_denominator", zero)
    ratio = np.full(Aerosol.shape, np.nan, dtype=float)
    ratio_valid = rrs_valid & ~zero
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(Red, denominator, out=ratio, where=ratio_valid)
    power_domain = _real_power_domain(ratio, 1.14)
    power_valid = ratio_valid & np.isfinite(u_red) & power_domain
    _count(
        diagnostics, "invalid_fractional_power_base",
        ratio_valid & np.isfinite(ratio) & ~power_domain,
    )
    ratio_power = _safe_power(ratio, 1.14, power_valid, diagnostics)
    bbp_den = 1 - u_red
    bbp_valid = power_valid & (bbp_den != 0)
    _count(diagnostics, "zero_denominator", power_valid & (bbp_den == 0))
    a665 = aw_red + 0.39 * ratio_power
    tss = np.full(Aerosol.shape, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        tss[bbp_valid] = 113.87498 * (
            (u_red[bbp_valid] * a665[bbp_valid]) / bbp_den[bbp_valid]
            - bbw_red
        )
    _count(diagnostics, "numerical_calculation_overflow", bbp_valid & ~np.isfinite(tss))
    return _finish(tss, diagnostics, return_diagnostics)

# SPM Alves e Santos 2024
def spm_madeira(Red, Nir2):
    spm = 945.1 * ((Nir2 / Red) ** 1.9463)

    return spm

# SPM SEN3R
def _spm_modis(Nir, Red):
    return 759.12 * ((Nir / Red) ** 1.92)

def _power(x, a, b, c):
    return a * x ** b + c

def spm_s3(Red, Nir2, cutoff_value=0.027, cutoff_delta=0.007, low_params=None, high_params=None,
           return_diagnostics=False):
    b665, b865 = _broadcast_float(Red, Nir2)
    diagnostics = _new_diagnostics(b665.shape)
    finite = np.isfinite(b665) & np.isfinite(b865)
    _count(diagnostics, "missing_or_masked_input", ~finite)
    if cutoff_delta == 0:
        transition_coef = np.where(finite, (b665 > cutoff_value).astype(float), np.nan)
    else:
        transition_range = (cutoff_value - cutoff_delta, cutoff_value + cutoff_delta)
        transition_coef = np.full(b665.shape, np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(b665 - transition_range[0], transition_range[1] - transition_range[0],
                      out=transition_coef, where=finite)
        transition_coef = np.clip(transition_coef, 0, 1)
    low_params = [2.79101975e+05, 2.34858344e+00, 4.20023206e+00] if low_params is None else low_params
    high_params = [759.12, 1.92, 0.0] if high_params is None else high_params
    low_needed = finite & (transition_coef < 1)
    high_needed = finite & (transition_coef > 0)
    low_base_domain = _real_power_domain(b665, low_params[1])
    low_base_valid = low_needed & low_base_domain
    _count(diagnostics, "invalid_fractional_power_base", low_needed & ~low_base_domain)
    low_power = _safe_power(b665, low_params[1], low_base_valid, diagnostics)
    low = low_params[0] * low_power + low_params[2]
    high_den_zero = high_needed & (b665 == 0)
    _count(diagnostics, "zero_denominator", high_den_zero)
    ratio = np.full(b665.shape, np.nan, dtype=float)
    high_ratio_valid = high_needed & ~high_den_zero
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(b865, b665, out=ratio, where=high_ratio_valid)
    high_power_domain = _real_power_domain(ratio, high_params[1])
    high_power_valid = high_ratio_valid & high_power_domain
    _count(diagnostics, "invalid_fractional_power_base", high_ratio_valid & np.isfinite(ratio) & ~high_power_domain)
    high_power = _safe_power(ratio, high_params[1], high_power_valid, diagnostics)
    high = high_params[0] * high_power + high_params[2]
    spm = np.full(b665.shape, np.nan, dtype=float)
    low_only = low_needed & ~high_needed & low_base_valid & np.isfinite(low)
    high_only = high_needed & ~low_needed & high_power_valid & np.isfinite(high)
    mixed = low_needed & high_needed & low_base_valid & high_power_valid & np.isfinite(low) & np.isfinite(high)
    spm[low_only] = low[low_only]
    spm[high_only] = high[high_only]
    spm[mixed] = ((1 - transition_coef[mixed]) * low[mixed] + transition_coef[mixed] * high[mixed])
    _count(diagnostics, "numerical_calculation_overflow", (low_only | high_only | mixed) & ~np.isfinite(spm))
    return _finish(spm, diagnostics, return_diagnostics)

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