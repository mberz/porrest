"""Backend implementations for the models.
"""

def johnson_champoux_allard(
        omega,
        phi, alpha_inf, k_s, lambda_s, lambda_t,
        rho_0, Pr, eta, gamma, P_stat,
    ):
    """Bulk modulus and density of Johnson-Champoux-Allard model.


    Parameters
    ----------
    omega : np.ndarray[float]
        Angular frequency.
    phi : float
        The porosity.
    alpha_inf : float
        The tortuosity high-frequency limit.
    k_s : float
        The viscous permeability.
    lambda_s : float
        The viscous characteristic length.
    lambda_t : float
        The thermal characteristic length.
    rho_0 : float
        The density of the medium saturating the pores.
    Pr : float
        The Prandtl number.
    eta : float
        The dynamic viscosity of the fluid saturating the pores.
    gamma : float
        The ratio of specific heats.
    P_stat : float
        The static pressure in the medium saturating the pores.

    Returns
    -------
    bulk_modulus : np.ndarray[complex]
        The bulk modulus of the equivalent fluid.
    density : np.ndarray[complex]
        The density of the equivalent fluid.

    Notes
    -----
    Code is auto-generated using sympy.

    """
    from numpy import sqrt

    x0 = phi**(-1.0)
    x1 = lambda_t**2
    x2 = 1j*omega*rho_0/eta
    x3 = 1j*eta/(omega*rho_0)
    return [
        P_stat*gamma*x0/(gamma - (gamma - 1)/(1 - 8*x3*sqrt((1/16)*Pr*x1*x2 + 1)/(Pr*x1))),  # noqa: E501
        rho_0*x0*(alpha_inf - phi*x3*sqrt(4*alpha_inf**2*k_s**2*x2/(lambda_s**2*phi**2) + 1)/k_s),  # noqa: E501
    ]


def johnson_champoux_allard_real_imag(
        omega,
        phi, alpha_inf, k_s, lambda_s, lambda_t,
        rho_0, Pr, eta, gamma, P_stat,
    ):
    """Bulk modulus and density of Johnson-Champoux-Allard model.

    Separated into real and imaginary parts.

    Parameters
    ----------
    omega : np.ndarray[float]
        Angular frequency.
    phi : float
        The porosity.
    alpha_inf : float
        The tortuosity high-frequency limit.
    k_s : float
        The viscous permeability.
    lambda_s : float
        The viscous characteristic length.
    lambda_t : float
        The thermal characteristic length.
    rho_0 : float
        The density of the medium saturating the pores.
    Pr : float
        The Prandtl number.
    eta : float
        The dynamic viscosity of the fluid saturating the pores.
    gamma : float
        The ratio of specific heats.
    P_stat : float
        The static pressure in the medium saturating the pores.

    Returns
    -------
    bulk_modulus_real : np.ndarray[float]
        The real part of the bulk modulus of the equivalent fluid.
    bulk_modulus_imag : np.ndarray[float]
        The imaginary part of the bulk modulus of the equivalent fluid.
    density_real : np.ndarray[float]
        The real part of the density of the equivalent fluid.
    density_imag : np.ndarray[float]
        The imaginary part of the density of the equivalent fluid.

    Notes
    -----
    Code is auto-generated using sympy.
    ----------

    """
    from pytensor.tensor import arctan, cos, sin, sqrt
    x0 = lambda_t**2
    x1 = omega*rho_0/eta
    x2 = (1/2)*arctan((1/16)*Pr*x0*x1)
    x3 = rho_0**(-1.0)
    x4 = Pr**2
    x5 = lambda_t**4
    x6 = eta**2
    x7 = omega**2
    x8 = rho_0**2
    x9 = x7*x8/x6
    x10 = (1/256)*x4*x5*x9 + 1
    x11 = eta/omega
    x12 = 8*x10**(1/4)*x11*x3/(Pr*x0)
    x13 = x12*sin(x2) + 1
    x14 = cos(x2)
    x15 = 64*sqrt(x10)*x14**2*x6/(x4*x5*x7*x8)
    x16 = x13**2 + x15
    x17 = 1 - gamma
    x18 = x17/x16
    x19 = gamma + x13*x18
    x20 = phi**(-1.0)
    x21 = P_stat*gamma*x20/(x15*x17**2/x16**2 + x19**2)
    x22 = (1/2)*arctan(4*alpha_inf**2*k_s**2*x1/(lambda_s**2*phi**2))
    x23 = x11*(16*alpha_inf**4*k_s**4*x9/(lambda_s**4*phi**4) + 1)**(1/4)/k_s
    return [
        x19*x21,
        -x12*x14*x18*x21,
        rho_0*x20*(alpha_inf + phi*x23*x3*sin(x22)),
        -x23*cos(x22),
    ]


def johnson_champoux_allard_lafarge(
        omega,
        phi, alpha_inf, k_s, k_t, lambda_s, lambda_t,
        rho_0, Pr, eta, gamma, P_stat,
    ):
    """Bulk modulus and density of Johnson-Champoux-Allard-Lafarge model.


    Parameters
    ----------
    omega : np.ndarray[float]
        Angular frequency.
    phi : float
        The porosity.
    alpha_inf : float
        The tortuosity high-frequency limit.
    k_s : float
        The viscous permeability.
    k_t : float
        The thermal permeability.
    lambda_s : float
        The viscous characteristic length.
    lambda_t : float
        The thermal characteristic length.
    rho_0 : float
        The density of the medium saturating the pores.
    Pr : float
        The Prandtl number.
    eta : float
        The dynamic viscosity of the fluid saturating the pores.
    gamma : float
        The ratio of specific heats.
    P_stat : float
        The static pressure in the medium saturating the pores.

    Returns
    -------
    bulk_modulus : np.ndarray[complex]
        The bulk modulus of the equivalent fluid.
    density : np.ndarray[complex]
        The density of the equivalent fluid.

    Notes
    -----
    Code is auto-generated using sympy.

    """
    from numpy import sqrt
    x0 = phi**(-1.0)
    x1 = 4*1j*omega*rho_0/(eta*phi**2)
    x2 = 1j*eta*phi/(omega*rho_0)
    return [
        P_stat*gamma*x0/(gamma - (gamma - 1)/(1 - x2*sqrt(Pr*k_t**2*x1/lambda_t**2 + 1)/(Pr*k_t))),  # noqa: E501
        rho_0*x0*(alpha_inf - x2*sqrt(alpha_inf**2*k_s**2*x1/lambda_s**2 + 1)/k_s),  # noqa: E501
    ]


def johnson_champoux_allard_lafarge_real_imag(
        omega,
        phi, alpha_inf, k_s, k_t, lambda_s, lambda_t,
        rho_0, Pr, eta, gamma, P_stat,
    ):
    """Bulk modulus and density of Johnson-Champoux-Allard-Lafarge model.

    Separated into real and imaginary parts.

    Parameters
    ----------
    omega : np.ndarray[float]
        Angular frequency.
    phi : float
        The porosity.
    alpha_inf : float
        The tortuosity high-frequency limit.
    k_s : float
        The viscous permeability.
    k_t : float
        The thermal permeability.
    lambda_s : float
        The viscous characteristic length.
    lambda_t : float
        The thermal characteristic length.
    rho_0 : float
        The density of the medium saturating the pores.
    Pr : float
        The Prandtl number.
    eta : float
        The dynamic viscosity of the fluid saturating the pores.
    gamma : float
        The ratio of specific heats.
    P_stat : float
        The static pressure in the medium saturating the pores.

    Returns
    -------
    bulk_modulus_real : np.ndarray[float]
        The real part of the bulk modulus of the equivalent fluid.
    bulk_modulus_imag : np.ndarray[float]
        The imaginary part of the bulk modulus of the equivalent fluid.
    density_real : np.ndarray[float]
        The real part of the density of the equivalent fluid.
    density_imag : np.ndarray[float]
        The imaginary part of the density of the equivalent fluid.

    Notes
    -----
    Code is auto-generated using sympy.

    """
    from pytensor.tensor import arctan, cos, sin, sqrt
    x0 = phi**(-1.0)
    x1 = k_t**2
    x2 = phi**2
    x3 = 4*omega*rho_0/(eta*x2)
    x4 = (1/2)*arctan(Pr*x1*x3/lambda_t**2)
    x5 = rho_0**(-1.0)
    x6 = phi*x5
    x7 = Pr**2
    x8 = eta**2
    x9 = omega**2
    x10 = rho_0**2
    x11 = 16*x10*x9/(phi**4*x8)
    x12 = k_t**4*x11*x7/lambda_t**4 + 1
    x13 = eta/omega
    x14 = x12**(1/4)*x13/(Pr*k_t)
    x15 = x14*x6*sin(x4) + 1
    x16 = cos(x4)
    x17 = sqrt(x12)*x16**2*x2*x8/(x1*x10*x7*x9)
    x18 = x15**2 + x17
    x19 = 1 - gamma
    x20 = x19/x18
    x21 = gamma + x15*x20
    x22 = P_stat*gamma/(x17*x19**2/x18**2 + x21**2)
    x23 = (1/2)*arctan(alpha_inf**2*k_s**2*x3/lambda_s**2)
    x24 = x13*(alpha_inf**4*k_s**4*x11/lambda_s**4 + 1)**(1/4)/k_s
    return [
        x0*x21*x22,
        -x14*x16*x20*x22*x5,
        rho_0*x0*(alpha_inf + x24*x6*sin(x23)),
        -x24*cos(x23),
    ]

# def impedance_wavenumber_mikis_model(f, phi, alpha_infty, sigma, c_0):
#     from pytensor.tensor import sqrt, pi
#     x0 = sqrt(alpha_infty)
#     x1 = f**0.632*sigma**(-0.632)
#     x2 = f**0.618*sigma**(-0.618)
#     return [
#         x0*(0.07*x1 - 0.107*1j*x1 + 1)/phi,
#         2*pi*f*x0*(0.109*x2 - 0.16*1j*x2 + 1)/c_0,
#     ]

def mikis_model(f, phi, alpha_infty, sigma, c_0, rho_0):
    """Bulk modulus and density of Miki's model.

    Separated into real and imaginary parts.

    Parameters
    ----------
    f : np.ndarray[float]
        The frequency.
    phi : float
        The porosity.
    alpha_infty : float
        The tortuosity high-frequency limit.
    sigma : float
        The flow resistivity.
    c_0 : float
        The speed of sound in the fluid saturating the pores.
    rho_0 : float
        The density of the medium saturating the pores.

    Returns
    -------
    bulk_modulus : np.ndarray[float]
        The bulk modulus of the equivalent fluid.
    density : np.ndarray[float]
        The density of the equivalent fluid.

    Notes
    -----
    Code is auto-generated using sympy.
    """
    x0 = f**(-0.618)*sigma**0.618
    x1 = 0.109*x0 - 0.16*1j*x0 + 1
    x2 = f**(-0.632)*sigma**0.632
    x3 = rho_0*(0.07*x2 - 0.107*1j*x2 + 1)/phi
    return [c_0**2*x3/x1, alpha_infty*x1*x3]


def mikis_model_real_imag(f, phi, alpha_infty, sigma, c_0, rho_0):
    """Bulk modulus and density of Miki's model.

    Separated into real and imaginary parts.

    Parameters
    ----------
    f : np.ndarray[float]
        The frequency.
    phi : float
        The porosity.
    alpha_infty : float
        The tortuosity high-frequency limit.
    sigma : float
        The flow resistivity.
    c_0 : float
        The speed of sound in the fluid saturating the pores.
    rho_0 : float
        The density of the medium saturating the pores.

    Returns
    -------
    bulk_modulus_real : np.ndarray[float]
        The real part of the bulk modulus of the equivalent fluid.
    bulk_modulus_imag : np.ndarray[float]
        The imaginary part of the bulk modulus of the equivalent fluid.
    density_real : np.ndarray[float]
        The real part of the density of the equivalent fluid.
    density_imag : np.ndarray[float]
        The imaginary part of the density of the equivalent fluid.

    Notes
    -----
    Code is auto-generated using sympy.

    """
    x0 = f**(-1.25)*sigma**1.25
    x1 = c_0**2
    x2 = f**(-0.618)
    x3 = sigma**0.618
    x4 = x2*x3
    x5 = 0.109*x4 + 1
    x6 = (0.0256*f**(-1.236)*sigma**1.236 + x5**2)**(-1.0)
    x7 = phi**(-1.0)
    x8 = rho_0*x7
    x9 = x1*x6*x8
    x10 = f**(-0.632)*sigma**0.632
    x11 = 0.07*x10
    x12 = x11 + 1
    x13 = 0.107*x10
    x14 = alpha_infty*x8
    return [0.01712*x0*x9 + x12*x5*x9, 0.16*rho_0*x1*x12*x2*x3*x6*x7 - x13*x5*x9, x14*(-0.00949*x0 + x11 + x5), x14*(-0.022863*x0 - x13 - 0.16*x4)]  # noqa: E501


def horoshenkov_jcal_parameters(
        porosity: float,
        median_pore_size: float,
        std_pore_size: float,
    ):
    import numpy as np
    tortuosity = np.exp(4*(std_pore_size*np.log(2))**2)

    viscous_permeability = (porosity*median_pore_size**2 /
        (8*tortuosity) *np.exp(-6*(std_pore_size*np.log(2))**2))

    thermal_permeability = (porosity*median_pore_size**2 /
        (8*tortuosity) * np.exp(6*(std_pore_size*np.log(2))**2))

    viscous_characteristic_length = (
        median_pore_size *
        np.exp(-2.5*(std_pore_size*np.log(2))**2))

    thermal_characteristic_length = (
        median_pore_size *np.exp(1.5*(std_pore_size*np.log(2))**2))

    return (
        tortuosity, viscous_permeability, thermal_permeability,
        viscous_characteristic_length, thermal_characteristic_length,
    )
