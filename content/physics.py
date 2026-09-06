"""
physics.py -- Python implementation of the exoplanet simulator physics.

This is the validation copy. It exists so the numbers can be checked in
an environment with real scientific libraries, against published figures,
without touching the browser.

An honest caveat, worth keeping in mind: this file and physics.js share a
lineage, so agreement between them proves that neither has DRIFTED. It
does not prove that both are right. The cases in test_cases.json marked
`"source": "published"` are the ones that check correctness, because they
compare against something outside this project. Add more of those.

Runs under CPython with numpy, and under Pyodide/JupyterLite unchanged.
"""

import json
import math
import numpy as np

# ---------------------------------------------------------------------
# Constants. Mirror physics.js exactly. If you change one, change both,
# and let the test suite tell you if you missed something.
# ---------------------------------------------------------------------

S0_SUN = 1361.0
TEFF_SUN = 5772.0
M_EARTH = 5.972e24
R_EARTH = 6.371e6
G_GRAV = 6.674e-11
G_EARTH = 9.807
SEC_PER_YEAR = 3.1557e7

A_OLR = 236.3   # raised when clouds became an explicit term
B_OLR = 2.0
A_GHG = 5.35
CO2_REF = 280.0
B_PRESSURE = 4.0
P_REF = 1.0

ALPHA_ICE = 0.62        # a SURFACE property; the cloud deck sits on top
ICE_T_CENTER = -10.0
ICE_T_WIDTH = 3.0

# Clouds. ALPHA_CLOUD is solved from Earth's planetary albedo of 0.30 at
# 67% cover over a clear-sky albedo of 0.15; LW_CLOUD is set so the
# longwave effect matches CERES at that cover.
ALPHA_CLOUD = 0.374
LW_CLOUD = 39.0

RHO_WATER = 1025.0
CP_WATER = 3994.0
MIXED_LAYER_M = 70.0
D_SCALE = 2.86
D_EARTH_REL = 0.20
DAY_HOURS_EARTH = 24.0
# Williams & Kasting 1997, following Farrell 1990. The exponent is
# disputed: Vladilo et al. found it unsupported by 3D models and Ramirez
# 2024 fitted coefficients to GCM runs instead. Kept at 2 because it is
# the value with a citation attached.
D_ROTATION_EXPONENT = 2.0
D_LOCKED_DEFAULT = 0.20

TIDAL_REF_AGE_GYR = 4.5
TIDAL_CONST = 0.3681

T_MIN_C, T_MAX_C = -100.0, 150.0

PLANET_TYPES = {
    "earth":  dict(label="Earth-like", surfaceAlbedo=0.15, cloudFraction=0.67,
                   transportFactor=1.0, mixedLayerM=8, massEarth=1.0),
    "desert": dict(label="Desert World", surfaceAlbedo=0.28, cloudFraction=0.30,
                   transportFactor=0.6, mixedLayerM=2, massEarth=0.3),
    "ocean":  dict(label="Ocean Super-Earth", surfaceAlbedo=0.09, cloudFraction=0.80,
                   transportFactor=1.6, mixedLayerM=40, massEarth=3.0),
}

DENSITY_PRESETS = dict(earth=5.51, mars=3.93, jupiter=1.33, saturn=0.69, neptune=1.64)


def clamp_c(t):
    return np.clip(t, T_MIN_C, T_MAX_C)


# ---------------------------------------------------------------------
# Stars
# ---------------------------------------------------------------------

def stellar_luminosity(mass):
    if mass < 0.14:
        return 2.20 * mass ** 3.45
    if mass < 0.43:
        return 0.23 * mass ** 2.3
    if mass < 2.0:
        return mass ** 4.0
    return 1.4 * mass ** 3.5


def stellar_radius(mass):
    return mass ** 0.9


def stellar_teff(mass):
    return TEFF_SUN * (stellar_luminosity(mass) / stellar_radius(mass) ** 2) ** 0.25


def stellar_lifespan_gyr(mass):
    return 10.0 * mass ** -2.5


# ---------------------------------------------------------------------
# Habitable zone (Kopparapu et al. 2013)
# ---------------------------------------------------------------------

HZ_COEFFS = {
    "recentVenus":       (1.7763, 1.4335e-4, 3.3954e-9, -7.6364e-12, -1.1950e-15),
    "runawayGreenhouse": (1.0385, 1.2456e-4, 1.4612e-8, -7.6345e-12, -1.7511e-15),
    "moistGreenhouse":   (1.0146, 8.1884e-5, 1.9394e-9, -4.3618e-12, -6.8260e-16),
    "maxGreenhouse":     (0.3507, 5.9578e-5, 1.6707e-9, -3.0058e-12, -5.1925e-16),
    "earlyMars":         (0.3207, 5.4471e-5, 1.5275e-9, -2.1709e-12, -3.8282e-16),
}
HZ_TEFF_MIN, HZ_TEFF_MAX = 2600.0, 7200.0


def seff_at(limit, teff):
    s, a, b, c, d = HZ_COEFFS[limit]
    t = np.clip(teff, HZ_TEFF_MIN, HZ_TEFF_MAX) - 5780.0
    return s + a * t + b * t ** 2 + c * t ** 3 + d * t ** 4


def habitable_zone(mass):
    lum = stellar_luminosity(mass)
    teff = stellar_teff(mass)
    dist = lambda limit: math.sqrt(lum / seff_at(limit, teff))
    return {
        "conservative": [dist("runawayGreenhouse"), dist("maxGreenhouse")],
        "optimistic": [dist("recentVenus"), dist("earlyMars")],
        "moistGreenhouse": dist("moistGreenhouse"),
        "teff": teff,
        "teffClamped": bool(teff < HZ_TEFF_MIN or teff > HZ_TEFF_MAX),
    }


# ---------------------------------------------------------------------
# Orbit, locking, dynamo
# ---------------------------------------------------------------------

def effective_s0(mass, a_au):
    return S0_SUN * stellar_luminosity(mass) / (a_au ** 2)


def tidal_lock_radius(star_mass, planet_mass_earth=1.0, age_gyr=TIDAL_REF_AGE_GYR):
    base = TIDAL_CONST * ((star_mass / 0.665) ** 2 / planet_mass_earth) ** (1 / 6)
    return base * (max(age_gyr, 0.01) / TIDAL_REF_AGE_GYR) ** (1 / 6)


def is_tidally_locked(star_mass, a_au, planet_mass_earth=1.0, age_gyr=TIDAL_REF_AGE_GYR):
    return bool(a_au <= tidal_lock_radius(star_mass, planet_mass_earth, age_gyr))


def magnetic_field_estimate(planet_mass_earth, age_gyr, is_locked):
    level = 2 if age_gyr < planet_mass_earth * 6.0 else (1 if age_gyr < planet_mass_earth * 10.0 else 0)
    if is_locked:
        level = max(0, level - 1)
    return {"level": level, "label": ["None", "Weak", "Strong"][level]}


# ---------------------------------------------------------------------
# Forcing and albedo
# ---------------------------------------------------------------------

def pressure_forcing(p_bar):
    return B_PRESSURE * math.log(max(p_bar, 0.01) / P_REF)


def greenhouse_forcing(co2_ppm, p_bar):
    return A_GHG * math.log(co2_ppm / CO2_REF) + pressure_forcing(p_bar)


def cloudy_albedos(surface_albedo, cloud_fraction):
    """Effective ice and clear-surface albedos under a cloud deck.

    alpha = (1-fc)*[ice*ALPHA_ICE + (1-ice)*surface] + fc*ALPHA_CLOUD,
    which rearranges into the same two-term form the solvers already use.
    """
    fc = min(max(cloud_fraction, 0.0), 1.0)
    base = (1 - fc) * surface_albedo + fc * ALPHA_CLOUD
    return {"ice": (1 - fc) * ALPHA_ICE + fc * ALPHA_CLOUD,
            "base": base, "planetary": base}


def cloud_longwave(cloud_fraction):
    """Extra IR trapping from a cloud deck, W/m^2."""
    return min(max(cloud_fraction, 0.0), 1.0) * LW_CLOUD


def diffusion_from(day_hours, pressure_bar=1.0, transport_factor=1.0):
    """Heat transport from rotation, pressure, and planet type."""
    day = max(day_hours, 0.1)
    return (D_EARTH_REL * max(pressure_bar, 0.01)
            * (day / DAY_HOURS_EARTH) ** D_ROTATION_EXPONENT * transport_factor)


def ice_fraction(t_c):
    return 0.5 * (1.0 - np.tanh((np.asarray(t_c, dtype=float) - ICE_T_CENTER) / ICE_T_WIDTH))


# ---------------------------------------------------------------------
# Insolation geometry
# ---------------------------------------------------------------------

def daily_mean_insolation(s0, declination_deg, lat_deg):
    phi = np.radians(np.asarray(lat_deg, dtype=float))
    delta = math.radians(declination_deg)
    tantan = np.tan(phi) * math.tan(delta)
    h0 = np.where(tantan >= 1, np.pi, np.where(tantan <= -1, 0.0, np.arccos(-np.clip(tantan, -1, 1))))
    return (s0 / np.pi) * (h0 * np.sin(phi) * math.sin(delta)
                           + np.cos(phi) * math.cos(delta) * np.sin(h0))


def latitude_grid(step=2):
    return np.arange(-90, 90 + step, step, dtype=float)


# ---------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------

def tridiag_solve(a, b, c, d):
    """Thomas algorithm. a is the sub-diagonal, c the super-diagonal."""
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def diffusion_operator(coords, d_rel, extra_diagonal):
    n = len(coords)
    step = abs(coords[1] - coords[0])
    dx = math.radians(step)
    d_phys = d_rel * D_SCALE
    lo = np.zeros(n)
    di = np.zeros(n)
    up = np.zeros(n)
    for i in range(n):
        cos_mid = max(math.cos(math.radians(coords[i])), 1e-4)
        cos_n = math.cos(math.radians(coords[i] + step / 2)) if i < n - 1 else 0.0
        cos_s = math.cos(math.radians(coords[i] - step / 2)) if i > 0 else 0.0
        dn = d_phys * cos_n / (cos_mid * dx * dx)
        ds = d_phys * cos_s / (cos_mid * dx * dx)
        lo[i] = -ds
        up[i] = -dn
        di[i] = extra_diagonal + dn + ds
    return lo, di, up


def solve_with_albedo_feedback(coords, q, d_rel, base_albedo, d_f,
                               tolerance=1e-4, max_iter=200, relaxation=0.5,
                               day_mask=None, ice_albedo=None):
    """Iterate the ice-albedo feedback to convergence and report whether it got there."""
    if ice_albedo is None:
        ice_albedo = ALPHA_ICE
    lo, di, up = diffusion_operator(coords, d_rel, B_OLR)
    frac = np.zeros(len(coords))
    temps = None
    converged = False
    iterations = 0
    for it in range(1, max_iter + 1):
        iterations = it
        f = frac if day_mask is None else np.where(day_mask, frac, 0.0)
        alpha = f * ice_albedo + (1 - f) * base_albedo
        rhs = q * (1 - alpha) + d_f - A_OLR
        temps = tridiag_solve(lo, di, up, rhs)
        target = ice_fraction(temps)
        new = frac + relaxation * (target - frac)
        delta = np.max(np.abs(new - frac))
        frac = new
        if delta < tolerance:
            converged = True
            break
    return {
        "coords": coords,
        "tempsC": clamp_c(temps),
        "tempsCRaw": temps,
        "iceFraction": frac,
        "converged": converged,
        "iterations": iterations,
        "outOfRange": bool(np.any((temps < T_MIN_C) | (temps > T_MAX_C))),
    }


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

def run_0d_ebm(T0_K=288.0, co2ppm=280.0, S0=S0_SUN, surfaceAlbedo=0.15,
               cloudFraction=0.67, pressureBar=1.0, years=300, dtYears=0.5,
               mixedLayerM=MIXED_LAYER_M, **_ignored):
    cl = cloudy_albedos(surfaceAlbedo, cloudFraction)
    heat_capacity = RHO_WATER * CP_WATER * mixedLayerM
    dt = dtYears * SEC_PER_YEAR
    d_f = greenhouse_forcing(co2ppm, pressureBar) + cloud_longwave(cloudFraction)
    t = T0_K
    times = [0.0]
    raw = [t]
    for i in range(int(years / dtYears)):
        f = float(ice_fraction(t - 273.15))
        alb = f * cl["ice"] + (1 - f) * cl["base"]
        asr = S0 * (1 - alb) / 4
        olr = A_OLR + B_OLR * (t - 273.15)
        t += dt * (asr - olr + d_f) / heat_capacity
        times.append((i + 1) * dtYears)
        raw.append(t)
    raw = np.array(raw)
    return {
        "times": np.array(times),
        "tempsK": np.clip(raw, T_MIN_C + 273.15, T_MAX_C + 273.15),
        "tempsKRaw": raw,
        "equilibriumC": float(np.clip(t, T_MIN_C + 273.15, T_MAX_C + 273.15) - 273.15),
        "equilibriumCRaw": float(t - 273.15),
        "outOfRange": bool(t - 273.15 < T_MIN_C or t - 273.15 > T_MAX_C),
    }


def lat_profile_equilibrium(S0=S0_SUN, co2ppm=280.0, obliquityDeg=23.44, planetType="earth",
                            dRel=None, pressureBar=1.0, declinationDeg=None,
                            dayHours=DAY_HOURS_EARTH):
    """Steady state at a FIXED declination. An upper bound, not a forecast."""
    t = PLANET_TYPES.get(planetType, PLANET_TYPES["earth"])
    d = diffusion_from(dayHours, pressureBar, t["transportFactor"]) if dRel is None else dRel
    decl = obliquityDeg if declinationDeg is None else declinationDeg
    lats = latitude_grid()
    cl = cloudy_albedos(t["surfaceAlbedo"], t["cloudFraction"])
    q = daily_mean_insolation(S0, decl, lats)
    out = solve_with_albedo_feedback(
        lats, q, d, cl["base"],
        greenhouse_forcing(co2ppm, pressureBar) + cloud_longwave(t["cloudFraction"]),
        ice_albedo=cl["ice"])
    out["lats"] = lats
    out["declinationDeg"] = decl
    return out


def lat_profile_seasonal(S0=S0_SUN, co2ppm=280.0, obliquityDeg=23.44, planetType="earth",
                         dRel=None, pressureBar=1.0, dayHours=DAY_HOURS_EARTH,
                         stepsPerYear=24, spinUpYears=200,
                         tolerance=0.02, mixedLayerM=None, albedoMemoryYears=5):
    """Time-stepping seasonal model with heat capacity. Backward Euler, so stable at any step."""
    t = PLANET_TYPES.get(planetType, PLANET_TYPES["earth"])
    d = diffusion_from(dayHours, pressureBar, t["transportFactor"]) if dRel is None else dRel
    depth = t["mixedLayerM"] if mixedLayerM is None else mixedLayerM
    cl = cloudy_albedos(t["surfaceAlbedo"], t["cloudFraction"])
    d_f = greenhouse_forcing(co2ppm, pressureBar) + cloud_longwave(t["cloudFraction"])

    lats = latitude_grid()
    n = len(lats)
    inertia = RHO_WATER * CP_WATER * depth / (SEC_PER_YEAR / stepsPerYear)
    lo, di, up = diffusion_operator(lats, d, B_OLR + inertia)

    q_year = [daily_mean_insolation(
        S0, obliquityDeg * math.cos(2 * math.pi * (s / stepsPerYear - 5 / 12)), lats)
        for s in range(stepsPerYear)]

    temps = np.full(n, 10.0)
    t_bar = temps.copy()
    memory_weight = 1.0 / (albedoMemoryYears * stepsPerYear)

    prev_year = None
    cycle = None
    converged = False
    years_run = 0
    for y in range(spinUpYears):
        years_run = y + 1
        cycle = np.zeros((stepsPerYear, n))
        for s in range(stepsPerYear):
            f = ice_fraction(t_bar)
            alpha = f * cl["ice"] + (1 - f) * cl["base"]
            rhs = q_year[s] * (1 - alpha) + d_f - A_OLR + inertia * temps
            temps = tridiag_solve(lo, di, up, rhs)
            t_bar = t_bar + memory_weight * (temps - t_bar)
            cycle[s] = temps
        if prev_year is not None and np.max(np.abs(cycle - prev_year)) < tolerance:
            converged = True
            prev_year = cycle
            break
        prev_year = cycle

    decls = [obliquityDeg * math.cos(2 * math.pi * (s / stepsPerYear - 5 / 12))
             for s in range(stepsPerYear)]
    solstice_step = int(np.argmax(decls))
    annual_mean = cycle.mean(axis=0)
    weights = np.cos(np.radians(lats))

    return {
        "lats": lats,
        "stepsPerYear": stepsPerYear,
        "months": clamp_c(cycle),
        "monthsRaw": cycle,
        "solsticeC": clamp_c(cycle[solstice_step]),
        "winterSolsticeC": clamp_c(cycle[(solstice_step + stepsPerYear // 2) % stepsPerYear]),
        "annualMeanC": clamp_c(annual_mean),
        "globalMeanC": float(np.clip(np.sum(annual_mean * weights) / np.sum(weights), T_MIN_C, T_MAX_C)),
        "solsticeStep": solstice_step,
        "converged": converged,
        "yearsRun": years_run,
        "outOfRange": bool(np.any((cycle < T_MIN_C) | (cycle > T_MAX_C))),
    }


def tidally_locked_profile(S0=S0_SUN, co2ppm=280.0, planetType="earth", dRel=None, pressureBar=1.0):
    t = PLANET_TYPES.get(planetType, PLANET_TYPES["earth"])
    # Deliberately NOT rotation-scaled: see D_LOCKED_DEFAULT.
    d = D_LOCKED_DEFAULT * t["transportFactor"] if dRel is None else dRel
    cl = cloudy_albedos(t["surfaceAlbedo"], t["cloudFraction"])
    angles = latitude_grid()
    q = np.where(angles > 0, S0 * np.sin(np.radians(angles)), 0.0)
    out = solve_with_albedo_feedback(
        angles, q, d, cl["base"],
        greenhouse_forcing(co2ppm, pressureBar) + cloud_longwave(t["cloudFraction"]),
        day_mask=(angles > 0), ice_albedo=cl["ice"])
    out["angles"] = angles
    out["substellarC"] = float(out["tempsC"][-1])
    out["antistellarC"] = float(out["tempsC"][0])
    out["terminatorC"] = float(out["tempsC"][len(angles) // 2])
    out["atmosphericCollapseRisk"] = bool(out["tempsCRaw"][0] < -140)
    return out


# ---------------------------------------------------------------------
# Calculators
# ---------------------------------------------------------------------

def kepler_period(a_au, star_mass):
    return math.sqrt(a_au ** 3 / star_mass)


def kepler_semi_major_axis(period_yr, star_mass):
    return (star_mass * period_yr ** 2) ** (1 / 3)


def kepler_star_mass(a_au, period_yr):
    return a_au ** 3 / period_yr ** 2


def planetary_properties(mass_earth, density_gcm3):
    rho = density_gcm3 * 1000.0
    m = mass_earth * M_EARTH
    r = ((3 * m) / (4 * math.pi * rho)) ** (1 / 3)
    g = G_GRAV * m / r ** 2
    return {"radiusM": r, "radiusEarth": r / R_EARTH,
            "gravityMS2": g, "gravityEarth": g / G_EARTH}


# ---------------------------------------------------------------------
# Derived quantities used by test_cases.json. These mirror
# test/harness.js; the names must match or the contract cannot be shared.
# ---------------------------------------------------------------------

def _idx(lat_deg):
    return int(round((lat_deg + 90) / 2))


def seasonalAmplitudeAt45N(planet_type):
    s = lat_profile_seasonal(planetType=planet_type)
    return float(s["solsticeC"][_idx(46)] - s["winterSolsticeC"][_idx(46)])


def tidalLockedMeanInsolationFraction():
    angles = latitude_grid()
    q = np.where(angles > 0, np.sin(np.radians(angles)), 0.0)
    w = np.cos(np.radians(angles))
    return float(np.sum(q * w) / np.sum(w))


def cloudPlanetaryAlbedo(surface_albedo, cloud_fraction):
    return cloudy_albedos(surface_albedo, cloud_fraction)["planetary"]


def cloudShortwaveEffect(surface_albedo, cloud_fraction):
    clear = cloudy_albedos(surface_albedo, 0.0)["planetary"]
    cloudy = cloudy_albedos(surface_albedo, cloud_fraction)["planetary"]
    return -(cloudy - clear) * S0_SUN / 4


def cloudNetEffect(surface_albedo, cloud_fraction):
    return cloudShortwaveEffect(surface_albedo, cloud_fraction) + cloud_longwave(cloud_fraction)


def iceAlbedoSwingRatio(surface_albedo, cloud_fraction):
    a = cloudy_albedos(surface_albedo, cloud_fraction)
    b = cloudy_albedos(surface_albedo, 0.0)
    return (a["ice"] - a["base"]) / (b["ice"] - b["base"])


def bistableAt(S0):
    warm = run_0d_ebm(T0_K=288.0, S0=S0)["equilibriumCRaw"]
    cold = run_0d_ebm(T0_K=215.0, S0=S0)["equilibriumCRaw"]
    return bool(abs(warm - cold) > 5)


def venusContrast():
    s = lat_profile_seasonal(S0=effective_s0(1.0, 0.723), pressureBar=92,
                             co2ppm=400, dayHours=24)
    return float(s["annualMeanC"][45] - s["annualMeanC"][90])


def lockedDayNightContrast():
    r = tidally_locked_profile(S0=effective_s0(0.122, 0.0485))
    return float(r["tempsCRaw"][-1] - r["tempsCRaw"][0])


def keplerPeriod(a_au, star_mass):
    return kepler_period(a_au, star_mass)


def keplerRoundTripError():
    worst = 0.0
    for a in (0.05, 0.1, 0.5, 1.0, 5.2, 30.0):
        for m in (0.089, 0.5, 1.0, 2.0):
            back = kepler_semi_major_axis(kepler_period(a, m), m)
            worst = max(worst, abs(back - a) / a)
    return worst


def planetRadiusEarth(mass_earth, density):
    return planetary_properties(mass_earth, density)["radiusEarth"]


def planetGravityEarth(mass_earth, density):
    return planetary_properties(mass_earth, density)["gravityEarth"]


def densityPreset(name):
    return DENSITY_PRESETS[name]


def magneticLabel(mass_earth, age_gyr, locked):
    return magnetic_field_estimate(mass_earth, age_gyr, locked)["label"]


def dailyMeanInsolation(s0, decl, lat):
    return float(daily_mean_insolation(s0, decl, lat))


def solsticePolarMinusEquator(s0, obliquity_deg):
    return float(daily_mean_insolation(s0, obliquity_deg, 90.0)
                 - daily_mean_insolation(s0, obliquity_deg, 0.0))


def hemisphereAsymmetry(obliquity_deg):
    lats = latitude_grid()
    north = lats[lats > 0]
    a = daily_mean_insolation(S0_SUN, obliquity_deg, north)
    b = daily_mean_insolation(S0_SUN, obliquity_deg, -north)
    return float(np.mean(np.abs(a - b)))



# ---------------------------------------------------------------------
# Climate zones
# ---------------------------------------------------------------------

CLIMATE_ZONES = [
    dict(key="scorching",   label="Too hot for liquid-water life", color="#7f1d1d"),
    dict(key="tropical",    label="Tropical (no cold season)",     color="#166534"),
    dict(key="subtropical", label="Subtropical (mild winter)",     color="#4d7c0f"),
    dict(key="temperate",   label="Temperate (freezing winter)",   color="#0e7490"),
    dict(key="continental", label="Continental (severe winter)",   color="#1e40af"),
    dict(key="tundra",      label="Tundra (brief cool summer)",    color="#6b21a8"),
    dict(key="polar",       label="Polar (never thaws)",           color="#334155"),
]
ZONE_BY_KEY = {z["key"]: z for z in CLIMATE_ZONES}
LIVABLE_ZONES = {"tropical", "subtropical", "temperate", "continental", "tundra"}


def classify_climate(warmest_c, coldest_c):
    if warmest_c > 50:
        return "scorching"
    if warmest_c < 0:
        return "polar"
    if warmest_c < 10:
        return "tundra"
    if coldest_c >= 18:
        return "tropical"
    if coldest_c >= 5:
        return "subtropical"
    if coldest_c >= -15:
        return "temperate"
    return "continental"


def climate_zones(coords, warmest_c, coldest_c):
    keys = [classify_climate(w, c) for w, c in zip(warmest_c, coldest_c)]
    bands = []
    start = 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[start]:
            z = ZONE_BY_KEY[keys[start]]
            bands.append(dict(key=z["key"], label=z["label"], color=z["color"],
                              **{"from": coords[start], "to": coords[i - 1]}))
            start = i
    w = np.cos(np.radians(np.asarray(coords, dtype=float)))
    livable = np.array([k in LIVABLE_ZONES for k in keys])
    return {"bands": bands, "keys": keys,
            "habitableFraction": float(np.sum(w[livable]) / np.sum(w))}


def seasonal_extremes(planet_type, **extra):
    s = lat_profile_seasonal(planetType=planet_type, **extra)
    return s["lats"], s["monthsRaw"].max(axis=0), s["monthsRaw"].min(axis=0)


def climateZoneAt(planet_type, lat_deg):
    lats, warm, cold = seasonal_extremes(planet_type)
    return climate_zones(lats, warm, cold)["keys"][_idx(lat_deg)]


def climateBandCount(planet_type):
    lats, warm, cold = seasonal_extremes(planet_type)
    return len(climate_zones(lats, warm, cold)["bands"])


def climateBandCountDifference(a, b):
    return climateBandCount(a) - climateBandCount(b)


def climateHabitableFraction(planet_type, extra=None):
    lats, warm, cold = seasonal_extremes(planet_type, **(extra or {}))
    return climate_zones(lats, warm, cold)["habitableFraction"]

# Names as they appear in test_cases.json -> callables here.
CONTRACT_FUNCTIONS = {
    "stellarLuminosity": stellar_luminosity,
    "stellarTeff": stellar_teff,
    "stellarRadius": stellar_radius,
    "stellarLifespanGyr": stellar_lifespan_gyr,
    "habitableZone": habitable_zone,
    "seffAt": seff_at,
    "effectiveS0": effective_s0,
    "tidalLockRadius": tidal_lock_radius,
    "isTidallyLocked": is_tidally_locked,
    "greenhouseForcing": greenhouse_forcing,
    "pressureForcing": pressure_forcing,
    "iceFraction": lambda t: float(ice_fraction(t)),
    "run0dEBM": lambda kw: run_0d_ebm(**kw),
    "latProfileEquilibrium": lambda kw: lat_profile_equilibrium(**kw),
    "latProfileSeasonal": lambda kw: lat_profile_seasonal(**kw),
    "tidallyLockedProfile": lambda kw: tidally_locked_profile(**kw),
    "dailyMeanInsolation": dailyMeanInsolation,
    "seasonalAmplitudeAt45N": seasonalAmplitudeAt45N,
    "tidalLockedMeanInsolationFraction": tidalLockedMeanInsolationFraction,
    "keplerPeriod": keplerPeriod,
    "keplerRoundTripError": keplerRoundTripError,
    "planetRadiusEarth": planetRadiusEarth,
    "planetGravityEarth": planetGravityEarth,
    "densityPreset": densityPreset,
    "magneticLabel": magneticLabel,
    "solsticePolarMinusEquator": solsticePolarMinusEquator,
    "hemisphereAsymmetry": hemisphereAsymmetry,
    "cloudyAlbedos": cloudy_albedos,
    "cloudLongwave": cloud_longwave,
    "diffusionFrom": diffusion_from,
    "cloudPlanetaryAlbedo": cloudPlanetaryAlbedo,
    "cloudShortwaveEffect": cloudShortwaveEffect,
    "cloudNetEffect": cloudNetEffect,
    "iceAlbedoSwingRatio": iceAlbedoSwingRatio,
    "bistableAt": bistableAt,
    "venusContrast": venusContrast,
    "lockedDayNightContrast": lockedDayNightContrast,
    "climateZoneAt": climateZoneAt,
    "climateBandCount": climateBandCount,
    "climateBandCountDifference": climateBandCountDifference,
    "climateHabitableFraction": climateHabitableFraction,
}


def _pluck(value, path):
    if not path:
        return value
    for key in path.split("."):
        value = value[int(key)] if key.isdigit() else value[key]
    return value


def run_contract(cases):
    """Run every case and return (results, n_pass, n_fail, n_skipped)."""
    results = []
    for case in cases:
        name = case["call"]["fn"]
        fn = CONTRACT_FUNCTIONS.get(name)
        if fn is None:
            results.append((case["id"], "SKIP", "not implemented in Python: " + name))
            continue
        try:
            actual = _pluck(fn(*case["call"].get("args", [])), case["expect"].get("path"))
            if isinstance(actual, (np.floating, np.integer)):
                actual = float(actual)
            exp = case["expect"]
            ok, detail = True, f"{actual}"
            if "value" in exp:
                if isinstance(exp["value"], (int, float)) and not isinstance(exp["value"], bool):
                    tol = exp.get("absTol", abs(exp["value"]) * exp.get("relTol", 1e-9))
                    ok = abs(actual - exp["value"]) <= tol
                    detail = f"{actual:.6g} vs {exp['value']:.6g} (tol {tol:.3g})"
                else:
                    ok = actual == exp["value"]
                    detail = f"{actual} vs {exp['value']}"
            if "min" in exp and ok:
                ok = actual >= exp["min"]
                detail = f"{actual:.6g} >= {exp['min']}"
            if "max" in exp and ok:
                ok = actual <= exp["max"]
                detail = f"{actual:.6g} <= {exp['max']}"
            results.append((case["id"], "PASS" if ok else "FAIL", detail))
        except Exception as exc:                                  # noqa: BLE001
            results.append((case["id"], "ERROR", repr(exc)))
    n_pass = sum(1 for r in results if r[1] == "PASS")
    n_fail = sum(1 for r in results if r[1] in ("FAIL", "ERROR"))
    n_skip = sum(1 for r in results if r[1] == "SKIP")
    return results, n_pass, n_fail, n_skip


def load_contract(path="test_cases.json"):
    """Load the contract from the local filesystem, or from a URL under Pyodide."""
    if path.startswith("http"):
        try:
            from pyodide.http import open_url
            return json.loads(open_url(path).read())
        except ImportError:
            from urllib.request import urlopen
            return json.loads(urlopen(path).read())
    with open(path) as handle:
        return json.load(handle)
