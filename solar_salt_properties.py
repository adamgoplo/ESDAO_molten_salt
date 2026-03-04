"""
Thermophysical properties of Solar Salt (60% NaNO3 / 40% KNO3)

Density and specific heat:
    Source: D'Aguanno et al., Scientific Reports 8, 10485 (2018)
    (Molecular dynamics + DSC study of nitrate molten salt mixtures)
    Valid range: ~530–890 K (liquid phase)

Thermal conductivity and viscosity:
    Source: Zavoico (2001), DLR Solar Salt Report (2022)
    Valid range: 220–600°C

Note on specific heat:
    D'Aguanno et al. (2018) show via MD simulations that cP is
    temperature-INDEPENDENT in the liquid phase for all K-Na nitrate
    mixtures, contradicting the commonly used increasing empirical law
    (Zavoico 2001). The constant value cP = 1.704 J/g·K is recommended
    for the solar salt (60% NaNO3 / 40% KNO3).
"""


def density(T_C):
    """
    Density [kg/m³], T in °C.

    Correlation: ρ(T) = α - β·T  [g/cm³], converted to kg/m³
    Parameters: α = 2.09 g/cm³, β = 6.369e-4 g/cm³/K
    Source: D'Aguanno et al. (2018), Eq. for solar salt liquid phase.
    (Original paper uses T in K; offset corrected for °C input.)
    Valid range: ~257–617°C (530–890 K, liquid phase)
    """
    T_K = T_C + 273.15
    rho_g_cm3 = 2.09 - 6.369e-4 * T_K
    return rho_g_cm3 * 1000  # convert g/cm³ → kg/m³


def specific_heat(T_C):
    """
    Specific heat capacity [J/kg·K], T in °C.

    Value: cP = 1.704 J/g·K = 1704 J/kg·K (temperature-independent)
    Source: D'Aguanno et al. (2018) — MD simulation result for solar salt
    liquid phase. Constant cP replaces the previously used increasing
    empirical law (Zavoico 2001: cP = 1443 + 0.172·T).
    Valid range: liquid phase (~257–617°C)
    """
    return 1704.0  # J/kg·K, constant (T-independent)


def thermal_conductivity(T_C):
    """
    Thermal conductivity [W/m·K], T in °C.
    Source: Zavoico (2001) / DLR Solar Salt Report (2022).
    Valid range: 220–600°C
    (D'Aguanno et al. 2018 does not provide transport properties.)
    """
    return 0.443 + 1.9e-4 * T_C


def viscosity(T_C):
    """
    Dynamic viscosity [Pa·s], T in °C.
    Source: Zavoico (2001) / DLR Solar Salt Report (2022).
    Valid range: 220–600°C
    (D'Aguanno et al. 2018 does not provide transport properties.)
    """
    return 2.2714e-2 - 1.2e-4 * T_C + 2.281e-7 * T_C**2 - 1.474e-10 * T_C**3
