"""Backend implementations for the TMM module.
"""


def bulk_modulus_density_to_impedance_wave_number_real_imag(
        freq, K_r, K_i, rho_r, rho_i):
    r"""Convert from fluid parameters to impedance and wave number.

    Separated into real and imaginary parts.

    .. math::

        Z_m = \sqrt{\left(i K_{i} + K_{r}\right) \left(i \rho_{i} + \rho_{r}\right)} \\
        k_m = \omega \sqrt{\frac{i \rho_{i} + \rho_{r}}{i K_{i} + K_{r}}}

    Parameters
    ----------
    freq : array, float
        The frequency.
    K_r : array_like, float
        The real part of the bulk modulus.
    K_i : array_like, float
        The imaginary part of the bulk modulus.
    rho_r : array_like, float
        The real part of the density.
    rho_i : array_like, float
        The imaginary part of the density.


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

    Notes
    -----
    This function was auto-generated using the following sympy code:

    .. code-block:: python

        import sympy as sym
        omega, K_r, K_i, rho_r, rho_i = sym.symbols(
            'omega, K_r, K_i, rho_r, rho_i', real=True)
        Z_m = sym.sqrt((K_r+sym.I*K_i) * (rho_r+sym.I*rho_i))
        k_m = omega*sym.sqrt((rho_r+sym.I*rho_i) / (K_r+sym.I*K_i))

        func_K_rho_to_Z_k = sym.lambdify(
            [omega, K_r, K_i, rho_r, rho_i],
            [sym.re(Z_m), sym.im(Z_m), sym.re(k_m), sym.im(k_m)],
            modules=["numpy"], cse=True,
        )

    """  # noqa: E501
    from pytensor.tensor import sqrt, arctan2, cos, sin, pi
    omega = 2*pi*freq
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


def impedance_eq_fluid_rigid_termination_real_imag(Z_r, Z_i, k_r, k_i, h):
    r"""Surface impedance of an equivalent fluid layer with rigid termination.

    .. math::
        Z_{s} = -i \frac{Z_r + i Z_i}{\tan([k_r + i k_i] h)}

    Parameters
    ----------
    Z_r : array_like, float
        The real part of the characteristic impedance.
    Z_i : array_like, float
        The imaginary part of the characteristic impedance.
    k_r : array_like, float
        The real part of the wave number.
    k_i : array_like, float
        The imaginary part of the wave number.
    h : float
        The height of the layer.

    Returns
    -------
    Z_s_re : array-like, float
        The real part of the surface impedance.
    Z_s_im : array-like, float
        The imaginary part of the surface impedance.

    Notes
    -----
    This function was auto-generated using the following sympy code:

    .. code-block:: python

        import sympy as sym
        omega = sym.symbols('omega', real=True)
        Z_r, Z_i, k_r, k_i = sym.symbols('Z_r, Z_i, k_r, k_i', real=True)
        Z_s = -sym.I*(Z_r + sym.I*Z_i)/sym.tan((k_r + sym.I*k_i)*h)

        func_layer_rigid_term = sym.lambdify(
            [Z_r, Z_i, k_r, k_i, h],
            [sym.re(Z_s), sym.im(Z_s)],
            modules=["numpy"], cse=True,
        )

    """
    from numpy import sinh, cosh, sin, cos
    x0 = 2*h
    x1 = k_r*x0
    x2 = sin(x1)
    x3 = k_i*x0
    x4 = cos(x1) + cosh(x3)
    x5 = x4**(-2.0)
    x6 = sinh(x3)
    x7 = 1/(x4*(x2**2*x5 + x5*x6**2))
    x8 = Z_i*x7
    x9 = Z_r*x7
    return [x2*x8 - x6*x9, -x2*x9 - x6*x8]


def reflection_factor_eq_fluid_rigid_termination_real_imag(
        Z_r, Z_i, k_r, k_i, h, Z_0,
    ):
    r"""Reflection factor of an equivalent fluid layer with rigid termination.

    The surface impedance of an equivalent fluid layer is calculated as:

    .. math::
        Z_{s} = -i \frac{Z_r + i Z_i}{\tan([k_r + i k_i] h)}

    The reflection factor is calculated as:

    .. math::
        R = \frac{Z_s - Z_0}{Z_s + Z_0}

    Parameters
    ----------
    Z_r : array_like, float
        The real part of the characteristic impedance.
    Z_i : array_like, float
        The imaginary part of the characteristic impedance.
    k_r : array_like, float
        The real part of the wave number.
    k_i : array_like, float
        The imaginary part of the wave number.
    h : float
        The height of the layer.
    Z_0 : array_like, float
        The impedance of the medium.

    Returns
    -------
    R_re : array-like, float
        The real part of the reflection factor.
    R_im : array-like, float
        The imaginary part of the reflection factor.

    Notes
    -----
    This function was auto-generated using the following sympy code:

    .. code-block:: python

        import sympy as sym
        omega = sym.symbols('omega', real=True)
        Z_r, Z_i, k_r, k_i, Z_0 = sym.symbols(
            'Z_r, Z_i, k_r, k_i, Z_0', real=True)

        Z_rigid = -sym.I*(Z_r + sym.I*Z_i)/sym.tan((k_r + sym.I*k_i)*h)
        R = (Z_rigid - Z_0)/(Z_rigid + Z_0)

        func_R = sym.lambdify(
            [Z_r, Z_i, k_r, k_i, h, Z_0],
            [sym.re(R), sym.im(R)],
            modules=["numpy"], cse=True)

    """
    from numpy import sinh, cosh, sin, cos
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
    x15 = Z_0 + x10*x4 - x14
    x16 = (x13**2 + x15**2)**(-1.0)
    x17 = x13*x16
    x18 = -Z_0 + Z_i*x4*x7*x8 - x14
    return [-x12*x17 + x15*x16*x18, x12*x16*x18 + x15*x17]


def abs_coefficient_eq_fluid_rigid_termination_real_imag(
        Z_r, Z_i, k_r, k_i, h, Z_0,
    ):
    r"""Absorption coefficient of an equivalent fluid layer with rigid
    termination.

    The surface impedance of an equivalent fluid layer is calculated as:

    .. math::
        Z_{s} = -i \frac{Z_r + i Z_i}{\tan([k_r + i k_i] h)}

    The absorption coefficient is calculated as:

    .. math::
        \alpha = 1 - |{\frac{Z_{0} - Z_{s}}{Z_{0} + Z_{s}}}|^2

    Parameters
    ----------
    Z_r : array_like, float
        The real part of the characteristic impedance.
    Z_i : array_like, float
        The imaginary part of the characteristic impedance.
    k_r : array_like, float
        The real part of the wave number.
    k_i : array_like, float
        The imaginary part of the wave number.
    h : float
        The height of the layer.
    Z_0 : array_like, float
        The impedance of the medium.

    Returns
    -------
    alpha : array-like, float
        The absorption coefficient.

    Notes
    -----
    This function was auto-generated using the following sympy code:

    .. code-block:: python

        import sympy as sym
        omega = sym.symbols('omega', real=True)
        Z_r, Z_i, k_r, k_i, Z_0 = sym.symbols(
            'Z_r, Z_i, k_r, k_i, Z_0', real=True)

        Z_rigid = -sym.I*(Z_r + sym.I*Z_i)/sym.tan((k_r + sym.I*k_i)*h)
        R = (Z_rigid - Z_0)/(Z_rigid + Z_0)
        alpha = 1 - sym.Abs(R)**2

        func_alpha = sym.lambdify(
            [Z_r, Z_i, k_r, k_i, h, Z_0],
            alpha,
            modules=["numpy"], cse=True)

    """

    from pytensor.tensor import sinh, cosh, sin, cos
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
    x18 = (x17 + (Z_0 + x14*x7 - x15*x2)**2)**(-2.0)
    x19 = x12**(-4.0)
    x20 = x18*x19
    x21 = Z_0**2
    x22 = Z_i**2
    x23 = x12**(-2.0)
    x24 = x22*x23
    x25 = x18*x21
    x26 = 4*x25
    x27 = Z_r**2
    x28 = x23*x27
    x29 = 2*x17*x18
    return -Z_0**4*x18 - Z_i**4*x20*x7**4 + 4*Z_i**3*Z_r*x18*x19*x2*x7**3 + 4*Z_i*Z_r**3*x18*x19*x2**3*x7 + 4*Z_i*Z_r*x17*x18*x2*x23*x7 - 12*Z_i*Z_r*x2*x23*x25*x7 - Z_r**4*x2**4*x20 + 2*x10*x18*x21*x23*x27 - 6*x10*x20*x22*x27*x8 - x10*x24*x26 - x10*x28*x29 - x16**4*x18 + 2*x17*x18*x21 + 2*x18*x21*x22*x23*x8 - x24*x29*x8 - x26*x28*x8 + 1  # noqa: E501


def plane_wave_decomposition_transmission_setup(
        pressure_measurements, positions, L, surrounding_medium,
    ):
    """Setup for plane wave decomposition transmission.

    Parameters
    ----------
    pressure_measurements : pyfar.FrequencyData
        The pressure of the plane wave.
    positions : array_like, float
        The positions of the pressure measurements.
    L : float
        The length of the test specimen.
    surrounding_medium : EquivalentFluid
        The surrounding medium of the plane wave.

    """
    p_1 = pressure_measurements.freq[0]
    p_2 = pressure_measurements.freq[1]
    p_3 = pressure_measurements.freq[2]
    p_4 = pressure_measurements.freq[3]

    l_1 = positions[0]
    s_1 = positions[1] - l_1
    l_2 = positions[2]
    s_2 = positions[3] - l_2

    k_t = surrounding_medium.wave_number.freq

    from numpy import exp, sin

    # A
    p_i_plus = (
        1j*(p_1*exp(-1j*k_t*l_1) - p_2*exp(-1j*k_t*(l_1+s_1))) /
        (2*sin(k_t*s_1))
    )
    # B
    p_o_plus = (
        1j*(-p_1*exp(1j*k_t*l_1) + p_2*exp(1j*k_t*(l_1+s_1))) /
        (2*sin(k_t*s_1))
    )
    # D
    p_i_minus = (
        1j*(p_3*exp(-1j*k_t*(l_2+s_2)) - p_4*exp(-1j*k_t*l_2)) /
        (2*sin(-k_t*s_2))*
        exp(1j*k_t*(L))
    )
    # C
    p_o_minus = (
        1j*(-p_3*exp(1j*k_t*(l_2+s_2)) + p_4*exp(1j*k_t*l_2)) /
        (2*sin(-k_t*s_2))*
        exp(-1j*k_t*(L))
    )
    return p_i_plus, p_i_minus, p_o_plus, p_o_minus
