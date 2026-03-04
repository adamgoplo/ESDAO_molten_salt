"""
Thermophysical properties of Solar Salt (60% NaNO3 / 40% KNO3)
Valid range: 220-600°C
Sources: Zavoico (2001), DLR Solar Salt Report (2022)
"""

def density(T):
    """Density [kg/m3], T in °C"""
    return 2090 - 0.636 * T

def specific_heat(T):
    """Specific heat capacity [J/kg·K], T in °C"""
    return 1443 + 0.172 * T

def thermal_conductivity(T):
    """Thermal conductivity [W/m·K], T in °C"""
    return 0.443 + 1.9e-4 * T

def viscosity(T):
    """Dynamic viscosity [Pa·s], T in °C"""
    return 2.2714e-2 - 1.2e-4 * T + 2.281e-7 * T**2 - 1.474e-10 * T**3
