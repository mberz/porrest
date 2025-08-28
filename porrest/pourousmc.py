try:
    from aesara.tensor import sin, cos, arctan2, sqrt, arctan, sinh, cosh
except ImportError:
    from pytensor.tensor import sin, cos, arctan2, sqrt, arctan, sinh, cosh


def equivalent_fluid_to_impedance_wave_number(K_r, K_i, rho_r, rho_i, omega):
    """Convert from fluid parameters to impedance and wave number.

    The function converts the dynamic bulk modulus and the dynamic density
    of a porous material into its characteristic impedance and wave number.
    All parameters need to be provided for real and imaginary parts separately
    This function was auto-generated using sympy.

    Parameters
    ----------
    omega : array, float
        The angular frequenc
    K_r : float
        The real part of the bulk modulus
    K_i : float
        The imaginary part of the bulk modulus
    rho_r : _type_
        The real part of the density
    rho_i : _type_
        The imaginary part of the density


    Returns
    -------
    Z_m_re : float, array-like
        The real part of the characteristic impedance
    Z_m_im : float, array-like
        The imaginary part of the characteristic impedance
    k_m_re : float, array-like
        The real part of the wave number
    k_m_im : float, array-like
        The imaginary part of the wave number
    """
    x0 = K_i*rho_r
    x1 = K_r*rho_i
    x2 = x0 + x1
    x3 = K_i*rho_i
    x4 = K_r*rho_r
    x5 = x3 - x4
    x6 = (1/2)*arctan2(x2, -x5)
    x7 = (x2**2 + x5**2)**(1/4)
    x8 = x0 - x1
    x9 = K_i**2 + K_r**2
    x10 = x9**(-1.0)
    x11 = x3 + x4
    x12 = (1/2)*arctan2(-x10*x8, x10*x11)
    x13 = omega*(x11**2 + x8**2)**(1/4)/sqrt(x9)
    return [x7*cos(x6), x7*sin(x6), x13*cos(x12), x13*sin(x12)]


def limp_frame_approximation(rho_r, rho_i, rho_0, rho_m, phi):
    """Calculate equivalent density according to the limp frame approximation.

    Parameters
    ----------
    rho_r : float
        Real part of the equivalent density
    rho_i : float
        Imaginary part of the equivalent density
    rho_0 : flaot
        Density of the medium saturating the pores
    rho_m : float
        Density of the solid frame
    phi : float
        Porosity of the material

    Returns
    -------
    rho_limp_re : float
        Real part of the equivalent density
    rho_limp_im : float
        Imaginary part of the equivalent density
    """
    x0 = rho_i**2
    x1 = phi*rho_0 + rho_m
    x2 = -2*rho_0 + rho_r + x1
    x3 = (x0 + x2**2)**(-1.0)
    x4 = x3*(-rho_0**2 + rho_r*x1)
    return [x0*x1*x3 + x2*x4, rho_i*x1*x2*x3 - rho_i*x4]


def johnson_champoux_allard(
        phi, alpha_inf, sigma, lambda_s, lambda_t, omega, rho_0, Pr,
        eta, gamma, P_stat):
    """Calculate bulk modulus and density according to the JCA model.

    Parameters
    ----------
    omega : float, array-like
        The angular frequency
    phi : float
        The porosity
    alpha_inf : float
        The toruosity
    sigma : float
        The flow resistivity
    lambda_s : float
        The viscous characteristic length
    lambda_t : float
        The thermal characteristic length
    rho_0 : float
        The density of the viscous medium
    Pr : float
        The Prantel number
    eta : float
        Eta
    gamma : float
        Eta
    P_stat : float
        Static pressure

    Returns
    -------
    K_r : float
        The real part of the bulk modulus
    K_i : float
        The imaginary part of the bulk modulus
    rho_r : _type_
        The real part of the density
    rho_i : _type_
        The imaginary part of the density
    """
    x0 = lambda_t**2
    x1 = omega*rho_0
    x2 = (1/2)*arctan((1/16)*Pr*x0*x1/eta)
    x3 = omega**(-1.0)
    x4 = Pr**2
    x5 = eta**2
    x6 = lambda_t**4
    x7 = omega**2
    x8 = rho_0**2
    x9 = x7*x8
    x10 = (1/256)*x4*x6*x9/x5 + 1
    x11 = 8*eta*x10**(1/4)*x3/(Pr*rho_0*x0)
    x12 = x11*sin(x2) + 1
    x13 = cos(x2)
    x14 = 64*sqrt(x10)*x13**2*x5/(x4*x6*x7*x8)
    x15 = x12**2 + x14
    x16 = 1 - gamma
    x17 = x16/x15
    x18 = gamma + x12*x17
    x19 = phi**(-1.0)
    x20 = P_stat*gamma*x19/(x14*x16**2/x15**2 + x18**2)
    x21 = (1/2)*arctan(4*alpha_inf**2*eta*x1/(lambda_s**2*phi**2*sigma**2))
    x22 = sigma*x3*(16*alpha_inf**4*x5*x9/(lambda_s**4*phi**4*sigma**4) + 1)**(1/4)
    return [x18*x20, -x11*x13*x17*x20, alpha_inf*rho_0*x19 + x22*sin(x21), -x22*cos(x21)]


def reflection_factor(Z_r, Z_i, k_r, k_i, h):
    x0 = 2*h
    x1 = k_i*x0
    x2 = sinh(x1)
    x3 = k_r*x0
    x4 = sin(x3)
    x5 = cos(x3) + cosh(x1)
    x6 = x5**(-2.0)
    x7 = (x2**2*x6 + x4**2*x6)**(-1.0)
    x8 = x5**(-1.0)
    x9 = x7*x8
    x10 = Z_i*x9
    x11 = Z_r*x9
    x12 = x10*x2 + x11*x4
    x13 = -x12
    x14 = x11*x2
    x15 = x10*x4 - x14 + 1
    x16 = (x13**2 + x15**2)**(-1.0)
    x17 = x13*x16
    x18 = Z_i*x4*x7*x8 - x14 - 1
    return [-x12*x17 + x15*x16*x18, x12*x16*x18 + x15*x17]


def absorption_coefficient(Z_r, Z_i, k_r, k_i, h):
    # x0 = 2*h
    # x1 = k_r*x0
    # x2 = sin(x1)**2
    # x3 = cos(x1)
    # x4 = x3**2
    # x5 = k_i*x0
    # x6 = sinh(x5)**2
    # x7 = (x2 + x6)**(-2.0)
    # x8 = Z_i**2*x7
    # x9 = x4*x8
    # x10 = cosh(x5)
    # x11 = x10**2
    # x12 = x11*x8
    # x13 = Z_r**2*x7
    # x14 = x13*x4
    # x15 = x11*x13
    # x16 = 2*x10*x3
    # x17 = x16*x8
    # x18 = x13*x16
    # return -x12*x2 - x12*x6 - x14*x2 - x14*x6 - x15*x2 - x15*x6 - x17*x2 - x17*x6 - x18*x2 - x18*x6 - x2*x9 - x6*x9 + 1

    x0 = 2*h
    x1 = k_i*x0
    x2 = sinh(x1)
    x3 = k_r*x0
    x4 = cos(x3)
    x5 = cosh(x1)
    x6 = (x4 + x5)**(-2.0)
    x7 = sin(x3)
    x8 = x7**2
    x9 = x6*x8
    x10 = x2**2
    x11 = x10*x6
    x12 = x11*x4 + x11*x5 + x4*x9 + x5*x9
    x13 = x12**(-1.0)
    x14 = Z_i*x13
    x15 = Z_r*x13
    x16 = x14*x2 + x15*x7
    x17 = x16**2
    x18 = (x17 + (x14*x7 - x15*x2 + 1)**2)**(-2.0)
    x19 = x12**(-4.0)
    x20 = x18*x19
    x21 = Z_i**2
    x22 = x12**(-2.0)
    x23 = x21*x22
    x24 = 4*x18
    x25 = Z_r**2
    x26 = x22*x25
    x27 = 2*x17*x18
    return -Z_i**4*x20*x7**4 + 4*Z_i**3*Z_r*x18*x19*x2*x7**3 + 4*Z_i*Z_r**3*x18*x19*x2**3*x7 + 4*Z_i*Z_r*x17*x18*x2*x22*x7 - 12*Z_i*Z_r*x18*x2*x22*x7 - Z_r**4*x2**4*x20 + 2*x10*x18*x22*x25 - 6*x10*x20*x21*x25*x8 - x10*x23*x24 - x10*x26*x27 - x16**4*x18 + 2*x17*x18 + 2*x18*x21*x22*x8 - x18 - x23*x27*x8 - x24*x26*x8 + 1


def jcal(phi, alpha_inf, k_s, k_t, lambda_s, lambda_t, omega, rho_0, Pr, eta, gamma, P_stat):
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
    return [x0*x21*x22, -x14*x16*x20*x22*x5, rho_0*x0*(alpha_inf + x24*x6*sin(x23)), -x24*cos(x23)]



def jcapl(phi, alpha_inf, k_s, k_t, lambda_s, lambda_t, alpha_s_0, alpha_t_0, omega, rho_0, Pr, eta, gamma, P_stat):
    x0 = phi**(-1.0)
    x1 = Pr**2
    x2 = alpha_t_0 - 1
    x3 = eta**2
    x4 = omega**2
    x5 = rho_0**2
    x6 = x4*x5/x3
    x7 = (1 + lambda_t**4*x1*x2**4*x6/k_t**4)**(1/4)
    x8 = Pr**(-1.0)
    x9 = k_t**2
    x10 = x9**(-1.0)
    x11 = lambda_t**2
    x12 = omega*rho_0/eta
    x13 = (1/2)*arctan(Pr*x10*x11*x12*x2**2)
    x14 = 1/(x11*x2)
    x15 = eta/omega
    x16 = x15/rho_0
    x17 = 2*x16
    x18 = k_t*x14*x17*x7*x8*sin(x13) + 1
    x19 = 2*x0
    x20 = x14*x19*x9
    x21 = x20*x7*cos(x13) - x20 + 1
    x22 = phi**2*x10*x21**2*x3/(x1*x4*x5)
    x23 = x18**2 + x22
    x24 = 1 - gamma
    x25 = x24/x23
    x26 = gamma + x18*x25
    x27 = P_stat*gamma/(x22*x24**2/x23**2 + x26**2)
    x28 = -alpha_inf + alpha_s_0
    x29 = (1 + lambda_s**4*x28**4*x6/alpha_inf**4)**(1/4)
    x30 = alpha_inf**2
    x31 = lambda_s**2
    x32 = (1/2)*arctan(x12*x28**2*x31/x30)
    x33 = 1/(x28*x31)
    x34 = k_s*x19*x30*x33
    return [x0*x26*x27, -x16*x21*x25*x27*x8/k_t, alpha_inf*rho_0*x0*(alpha_inf*x17*x29*x33*sin(x32) + 1), -x15*(x29*x34*cos(x32) - x34 + 1)/k_s]


def dynamic_tortuosity_jcal(phi, alpha_inf, k_s, lambda_s, omega, rho_0, eta):
    """Calculate the dynamic tortuosity according to the JCA(L) models

    Parameters
    ----------
    phi : float
        The porosity
    alpha_inf : float
        The tortuosity in the high frequency limit
    k_s : float
        The viscous permeability
    lambda_s : float
        The viscous characteristic length
    omega : float
        The angular frequency
    rho_0 : float
        The density of the fluid saturating the pores
    eta : float
        The dynamic viscosity of the fluid

    Returns
    -------
    alpha_r : float
        The real part of the dynamic tortuosity
    alpha_i : float
        The imaginary part of the dynamic tortuosity

    """
    x0 = (1/2)*arctan(4*alpha_inf**2*k_s**2*omega*rho_0/(eta*lambda_s**2*phi**2))
    x1 = sqrt(eta)*(16*alpha_inf**4*k_s**4*omega**2*rho_0**2 + eta**2*lambda_s**4*phi**4)**(1/4)/(k_s*lambda_s*omega*rho_0)
    return [alpha_inf + x1*sin(x0), -x1*cos(x0)]


def bulk_modulus_jcal(
        phi, k_t, lambda_t, omega, rho_0, Pr, eta, gamma, P_stat):
    """Calculate the bulk modulus according to the JCAL model

    Parameters
    ----------
    phi : float
        The porosity
    k_t : float
        The thermal permeability
    lambda_t : float
        The thermal characteristic length
    omega : float
        The angular frequency
    rho_0 : float
        The density of the fluid saturating the pores
    Pr : float
        The Prantl number
    eta : float
        The dynamic viscosity of the fluid
    gamma : float
        The
    P_stat : float
        The static pressure

    Returns
    -------
    K_r : float
        The real part of the bulk modulus
    K_i : float
        The imaginary part of the bulk modulus

    """
    x0 = k_t**2
    x1 = phi**2
    x2 = (1/2)*arctan(4*Pr*omega*rho_0*x0/(eta*lambda_t**2*x1))
    x3 = Pr**2
    x4 = eta**2
    x5 = omega**2
    x6 = rho_0**2
    x7 = 16*k_t**4*x3*x5*x6/(lambda_t**4*phi**4*x4) + 1
    x8 = eta*phi*x7**(1/4)/(Pr*k_t*omega*rho_0)
    x9 = x8*sin(x2) + 1
    x10 = cos(x2)
    x11 = x1*x10**2*x4*sqrt(x7)/(x0*x3*x5*x6)
    x12 = x11 + x9**2
    x13 = 1 - gamma
    x14 = x13/x12
    x15 = gamma + x14*x9
    x16 = P_stat*gamma/(x11*x13**2/x12**2 + x15**2)
    return [x15*x16, -x10*x14*x16*x8]


def bulk_modulus_jca(omega, lambda_t, rho_0, Pr, eta, gamma, P_stat):
    """Calculate the bulk modulus according to the JCA model

    Parameters
    ----------
    lambda_t : float
        The thermal characteristic length
    omega : float
        The angular frequency
    rho_0 : float
        The density of the fluid saturating the pores
    Pr : float
        The Prantl number
    eta : float
        The dynamic viscosity of the fluid
    gamma : float
        The
    P_stat : float
        The static pressure

    Returns
    -------
    K_r : float
        The real part of the bulk modulus
    K_i : float
        The imaginary part of the bulk modulus
    """
    x0 = lambda_t**2
    x1 = (1/2)*arctan((1/16)*Pr*omega*rho_0*x0/eta)
    x2 = Pr**2
    x3 = eta**2
    x4 = lambda_t**4
    x5 = omega**2
    x6 = rho_0**2
    x7 = (1/256)*x2*x4*x5*x6/x3 + 1
    x8 = 8*eta*x7**(1/4)/(Pr*omega*rho_0*x0)
    x9 = x8*sin(x1) + 1
    x10 = cos(x1)
    x11 = 64*x10**2*x3*sqrt(x7)/(x2*x4*x5*x6)
    x12 = x11 + x9**2
    x13 = 1 - gamma
    x14 = x13/x12
    x15 = gamma + x14*x9
    x16 = P_stat*gamma/(x11*x13**2/x12**2 + x15**2)
    return [x15*x16, -x10*x14*x16*x8]


def wave_number_first_compressional_wave(
        phi, alpha_r, alpha_i, rho_0, rho_1, E_re, E_im, omega):
    """Calculate the wave number of the first compressional wave.


    Parameters
    ----------
    phi : float
        The porosity
    alpha_r : float
        Real part of the dynamic tortuosity
    alpha_i : float
        Imaginary part of the dynamic tortuosity
    rho_0 : float
        Density of the fluid saturating the pores
    rho_1 : float
        Density of the porous frame
    E_re : float
        Real part of Youngs modulus
    E_im : float
        Imaginary part of Youngs modulus
    omega : float
        Angular frequency

    Returns
    -------
    delta_1_re : float
        The real part of the wave number of the first compressional wave
    delta_1_im : float
        The imaginary part of the wave number of the first compressional wave
    """
    x0 = E_im**2
    x1 = alpha_i**2
    x2 = alpha_r**2
    x3 = E_re**2
    x4 = (x0*x1 + x0*x2 + x1*x3 + x2*x3)**(-1.0)
    x5 = E_im*rho_1
    x6 = phi*rho_0
    x7 = E_im*x6
    x8 = E_im*alpha_i
    x9 = E_re*rho_1
    x10 = E_re*alpha_r
    x11 = E_re*x6
    x12 = (1/2)*arctan2(x4*(E_im*alpha_r*phi*rho_0 + E_re*alpha_i*phi*rho_0 - x1*x5 - x1*x7 - x2*x5 - x2*x7), x4*(x1*x11 + x1*x9 - x10*x6 + x11*x2 + x2*x9 + x6*x8))
    x13 = E_im*alpha_r + E_re*alpha_i
    x14 = -x10 + x8
    x15 = alpha_r*rho_1
    x16 = alpha_i*rho_1
    x17 = x13*x6
    x18 = x14*x6
    x19 = omega*((alpha_i*x17 - alpha_r*x18 + x13*x16 - x14*x15 + x18)**2 + (alpha_i*x18 + alpha_r*x17 + x13*x15 + x14*x16 - x17)**2)**(1/4)/sqrt(x13**2 + x14**2)
    return [x19*cos(x12), x19*sin(x12)]


def wave_number_second_compressional_wave(
        phi, alpha_r, alpha_i, rho_0, rho_1, E_re, E_im, omega):
    """Calculate the wave number of the second compressional wave.

    Parameters
    ----------
    phi : float
        The porosity
    alpha_r : float
        Real part of the dynamic tortuosity
    alpha_i : float
        Imaginary part of the dynamic tortuosity
    rho_0 : float
        Density of the fluid saturating the pores
    rho_1 : float
        Density of the porous frame
    E_re : float
        Real part of Youngs modulus
    E_im : float
        Imaginary part of Youngs modulus
    omega : float
        Angular frequency

    Returns
    -------
    delta_2_re : float
        The real part of the wave number of the second compressional wave
    delta_2_im : float
        The imaginary part of the wave number of the second compressional wave
    """
    x0 = phi*(phi*rho_0 - 2*rho_0 + rho_1)
    x1 = E_im*x0 + rho_0*(E_im*alpha_r - E_re*alpha_i)
    x2 = E_im**2
    x3 = E_re**2
    x4 = x2 + x3
    x5 = 1/(phi*x4)
    x6 = E_re*x0 + rho_0*(E_im*alpha_i + E_re*alpha_r)
    x7 = (1/2)*arctan2(-x1*x5, x5*x6)
    x8 = omega*sqrt(x4)*(x1**2 + x6**2)**(1/4)/(sqrt(phi)*sqrt(E_im**4 + E_re**4 + 2*x2*x3))
    return [x8*cos(x7), x8*sin(x7)]


def wave_number_equivalent_fluid_phase(
        phi, alpha_r, alpha_i, rho_0, K_eq_re, K_eq_im, omega):
    """Calculate the wave number of the equivalent fluid phase.

    Parameters
    ----------
    phi : float
        The porosity
    alpha_r : float
        The real part of the dynamic tortuosity
    alpha_i : float
        The imaginary part of the dynamic tortuosity
    rho_0 : float
        The density of the fluid saturating the pores
    K_eq_re : float
        The real part of the equivalent bulk modulus
    K_eq_im : float
        The imaginary part of the equivalent bulk modulus
    omega : float
        The angular frequency

    Returns
    -------
    delta_eq_re : float
        The real part of the wave number of the equivalent fluid phase
    delta_eq_im : float
        The imaginary part of the wave number of the equivalent fluid phase

    """
    x0 = K_eq_im*alpha_r - K_eq_re*alpha_i
    x1 = K_eq_im**2 + K_eq_re**2
    x2 = x1**(-1.0)
    x3 = K_eq_im*alpha_i + K_eq_re*alpha_r
    x4 = (1/2)*arctan2(-x0*x2, x2*x3)
    x5 = omega*sqrt(rho_0)*(x0**2 + x3**2)**(1/4)/(sqrt(phi)*sqrt(x1))
    return [x5*cos(x4), x5*sin(x4)]


def wave_numbers_structural_to_fluid(
        delta_s1_re, delta_s1_im,
        delta_s2_re, delta_s2_im,
        delta_eq_re, delta_eq_im):
    """_summary_

    Parameters
    ----------
    delta_s1_re : _type_
        _description_
    delta_s1_im : _type_
        _description_
    delta_s2_re : _type_
        _description_
    delta_s2_im : _type_
        _description_
    delta_eq_re : _type_
        _description_
    delta_eq_im : _type_
        _description_

    Returns
    -------
    delta_1_re : _type_
        _description_
    delta_1_im : _type_
        _description_
    delta_2_re : _type_
        _description_
    delta_2_im : _type_
        _description_
    """
    x0 = delta_s1_im**2
    x1 = delta_s1_re**2
    x2 = delta_eq_im*delta_eq_re
    x3 = 8*x2
    x4 = delta_s2_im**2
    x5 = delta_s2_re**2
    x6 = delta_eq_im**2
    x7 = delta_s2_im*delta_s2_re
    x8 = 4*x6
    x9 = delta_eq_re**2
    x10 = delta_s1_im*delta_s1_re
    x11 = -4*delta_eq_im**3*delta_eq_re + 4*delta_eq_im*delta_eq_re**3 + 8*delta_eq_im*delta_eq_re*x0 + 4*delta_eq_im*delta_eq_re*x5 + 8*delta_s1_im*delta_s1_re*x6 - 4*delta_s2_im**3*delta_s2_re + 4*delta_s2_im*delta_s2_re**3 + 4*delta_s2_im*delta_s2_re*x9 - x1*x3 - 8*x10*x9 - 4*x2*x4 - x7*x8
    x12 = 2*x6
    x13 = 4*x9
    x14 = 2*x9
    x15 = delta_eq_im**4 + delta_eq_re**4 + delta_s2_im**4 + delta_s2_re**4 + x0*x13 - x0*x8 - x1*x13 + x1*x8 + 16*x10*x2 + x12*x4 - x12*x5 - x14*x4 + x14*x5 - x3*x7 - 6*x4*x5 - 6*x6*x9
    x16 = (1/2)*arctan2(x11, x15)
    x17 = (1/2)*(x11**2 + x15**2)**(1/4)
    x18 = x17*sin(x16)
    x19 = x2 + x7
    x20 = x18 + x19
    x21 = (1/2)*x6
    x22 = (1/2)*x4
    x23 = x17*cos(x16)
    x24 = -x21 - x22 + x23 + (1/2)*x5 + (1/2)*x9
    x25 = (1/2)*arctan2(x20, x24)
    x26 = (x20**2 + x24**2)**(1/4)
    x27 = -x18 + x19
    x28 = -x21 - x22 - x23 + (1/2)*x5 + (1/2)*x9
    x29 = (1/2)*arctan2(x27, x28)
    x30 = (x27**2 + x28**2)**(1/4)
    return [x26*cos(x25), x26*sin(x25), x30*cos(x29), x30*sin(x29)]


def impedance_poroelastic_dazel(
        delta_1_re, delta_1_im,
        delta_2_re, delta_2_im,
        delta_eq_re, delta_eq_im,
        K_eq_re, K_eq_im,
        phi, h, omega):
    """Calculate the impedance of a rigidly backed poroelastic material.

    The inputs are the wave-numbers of compressional waves, the
    equivalent of the fluid wave-number, the equivalent bulk modulus, the
    porosity, the thickness of the material, and the angular frequency.

    Parameters
    ----------
    delta_1_re : float
        The real part of wave-number of the first compressional wave
    delta_1_im : float
        The imaginary part of wave-number of the first compressional wave
    delta_2_re : float
        The real part of wave-number of the second compressional wave
    delta_2_im : float
        The imaginary part of wave-number of the second compressional wave
    delta_eq_re : float
        The real part of the equivalent fluid wave-number
    delta_eq_im : float
        The imaginary part of the equivalent fluid wave-number
    K_eq_re : float
        The real part of the equivalent bulk modulus
    K_eq_im : float
        The imaginary part of the equivalent bulk modulus
    phi : float
        The porosity
    h : float
        The thickness of the material layer
    omega : float
        The angular frequency

    Returns
    -------
    Z_re : float
        The real part of the impedance
    Z_im : float
        The imaginary part of the impedance

    """
    x0 = delta_1_im**2
    x1 = delta_1_re**2
    x2 = (x0 + x1)**(-1.0)
    x3 = delta_2_im**2
    x4 = 2*delta_1_re
    x5 = delta_1_im*x4
    x6 = 2*delta_2_re
    x7 = delta_2_im*x6
    x8 = x5 - x7
    x9 = delta_2_re**2
    x10 = -x0 + x1 + x3 - x9
    x11 = (x10**2 + x8**2)**(-1.0)
    x12 = h*x4
    x13 = 2*h
    x14 = delta_1_im*x13
    x15 = (cos(x12) + cosh(x14))**(-1.0)
    x16 = x11*x15
    x17 = x16*x3
    x18 = -x8
    x19 = sin(x12)
    x20 = x18*x19
    x21 = delta_eq_re**2
    x22 = x16*x21
    x23 = sinh(x14)
    x24 = x18*x23
    x25 = x16*x7
    x26 = 2*delta_eq_im*delta_eq_re
    x27 = x16*x26
    x28 = x16*x9
    x29 = delta_eq_im**2
    x30 = x16*x29
    x31 = x10*x23
    x32 = x10*x19
    x33 = x17*x20 + x17*x31 + x20*x22 - x20*x28 - x20*x30 + x22*x31 + x24*x25 - x24*x27 - x25*x32 + x27*x32 - x28*x31 - x30*x31
    x34 = x2*(2*delta_2_im*delta_2_re*x10*x11*x15*x23 + 2*delta_2_im*delta_2_re*x11*x15*x18*x19 + x10*x11*x15*x19*x21 + x10*x11*x15*x19*x3 + x11*x15*x18*x23*x29 + x11*x15*x18*x23*x9 - x17*x24 - x20*x27 - x22*x24 - x27*x31 - x28*x32 - x30*x32)
    x35 = (x3 + x9)**(-1.0)
    x36 = h*x6
    x37 = sin(x36)
    x38 = x18*x37
    x39 = delta_2_im*x13
    x40 = (cos(x36) + cosh(x39))**(-1.0)
    x41 = x11*x40
    x42 = x0*x41
    x43 = x21*x41
    x44 = sinh(x39)
    x45 = x18*x44
    x46 = x41*x5
    x47 = x10*x44
    x48 = x10*x37
    x49 = x26*x41
    x50 = 2*delta_1_im*delta_1_re*x10*x11*x37*x40 + 2*delta_eq_im*delta_eq_re*x11*x18*x40*x44 + x1*x10*x11*x40*x44 + x1*x11*x18*x37*x40 + x10*x11*x29*x40*x44 + x11*x18*x29*x37*x40 - x38*x42 - x38*x43 - x42*x47 - x43*x47 - x45*x46 - x48*x49
    x51 = x1*x41
    x52 = x29*x41
    x53 = x35*(-x38*x46 + x38*x49 + x42*x45 - x42*x48 + x43*x45 - x43*x48 - x45*x51 - x45*x52 - x46*x47 + x47*x49 + x48*x51 + x48*x52)
    x54 = delta_1_im*x2*x33 + delta_1_re*x34 + delta_2_im*x35*x50 + delta_2_re*x53
    x55 = delta_1_im*x34 - delta_1_re*x2*x33 + delta_2_im*x53 - delta_2_re*x35*x50
    x56 = 1/(omega*phi*(x54**2 + x55**2))
    x57 = K_eq_im*x56
    x58 = K_eq_re*x56
    return [x54*x57 + x55*x58, -x54*x58 + x55*x57]


def impedance_to_absorption_coefficient(Z_rigid_re, Z_rigid_im):
    """Calculate the absorption coefficient from the characteristic impedance.

    Parameters
    ----------
    Z_rigid_re : float
        Real part of the characteristic impedance
    Z_rigid_im : float
        Imaginary part of the characteristic impedance

    Returns
    -------
    float
        The absorption coefficient
    """
    return 4*Z_rigid_re/(Z_rigid_im**2 + Z_rigid_re**2 + 2*Z_rigid_re + 1)
