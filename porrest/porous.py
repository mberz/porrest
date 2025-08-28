# -*- coding: utf-8 -*-
import numpy as np
from numpy import sqrt
from scipy import optimize

from utils.constants import constants


def impedance_mikis_model(frequency, flow_resistivity, c=constants()['c']):
    """Calculate the impedance of a porous material using Miki's model.

    Parameters
    ----------
    frequency : float, array-like
        The frequencies for which the model is calculated
    flow_resistivity : float
        The flow resistivity inside the material
    c : float, optional
        The speed of sound, by default constants()['c']

    Returns
    -------
    complex, array
        The complex material impedance
    complex, array
        The complex propagation constant

    References
    ----------
    [#] Y. Miki, “Acoustical properties of porous materials. Modifications of
        Delany-Bazley models,” Journal of the Acoustical Society of Japan,
        vol. 11, no. 1, pp. 19-24, 1990.


    """
    omega = 2*np.pi*frequency

    imp_real = 1 + 0.0699*(frequency/flow_resistivity)**(-0.632)
    imp_imag = -0.1070*(frequency/flow_resistivity)**(-0.632)
    prop_const_real = omega/c * 0.160*(frequency/flow_resistivity)**(-0.618)
    prop_const_imag = omega/c * (1 + 0.109*(frequency/flow_resistivity)**(-0.618))
    impedance_porous = imp_real + 1j*imp_imag
    prop_const = prop_const_real + 1j*prop_const_imag

    return impedance_porous, prop_const


def impedance_komatsu_model(frequency, flow_resistivity, c=constants()['c']):
    """Calculate impedance of a porous material using Komatsu's model.

    Parameters
    ----------
    frequency : float, array
        The frequencies for which the model is to be calculated
    flow_resistivity : float
        Flow resisitivity of the material
    c : float, optional
        The speed of sound in the surrounding material, by default 343.6

    Returns
    -------
    complex, array
        The complex material impedance
    complex, array
        The complex propagation constant

    References
    ----------
    [#] T. Komatsu, “Improvement of the Delany-Bazley and Miki models for
        fibrous sound-absorbing materials,” Acoust. Sci. & Tech., vol. 29,
        no. 2, pp. 121-129, 2008, doi: 10/fnx4zd.
    """

    omega = 2*np.pi*frequency
    log_term = 2-np.log10(frequency/flow_resistivity)
    imp_real = 1 + 0.00027*log_term**6.2
    imp_imag = -0.0047*log_term**4.1
    prop_const_real = omega/c * 0.0069*log_term**4.1
    prop_const_imag = omega/c * (1 + 0.0004*log_term**6.2)
    impedance_porous = imp_real + 1j*imp_imag
    prop_const = prop_const_real + 1j*prop_const_imag

    return impedance_porous, prop_const


def impedance_champoux_allard(
        frequency, flow_resistivity,
        c=constants()['c'], rho=constants()['rho_0'], P_stat=101325,):
    """Calculate the impedance of a material using the Champoux, Allard model.

    Parameters
    ----------
    frequency : float, array
        The frequencies for which the model is to be calculated
    flow_resistivity : float
        Flow resistivity inside the material
    c : float, optional
        The speed of sound, by default constants()['c']
    rho : float, optional
        The density of the medium, by default constants()['rho_0']

    Returns
    -------
    complex, array
        The complex material impedance
    complex, array
        The complex propagation constant

    References
    ----------
    [#] J. Allard and Y. Champoux, “New empirical equations for sound
        propagation in rigid frame fibrous materials,” The Journal of the
        Acoustical Society of America, vol. 91, no. 6, pp. 3346-3353, 1992,
        doi: 10.1121/1.402824.

    """

    omega = 2*np.pi*frequency
    X = rho*frequency/flow_resistivity

    dynamic_density = 1.2 + np.sqrt(-0.0364/X**2 - 1j*0.1144/X)
    sqrt_term = np.sqrt(2.82/X**2 + 1j*24.9/X)
    bulk_modulus = P_stat * (1j*29.64 + sqrt_term) / (1j*21.17 + sqrt_term)

    impedance_porous = np.sqrt(dynamic_density*bulk_modulus)/rho/c
    prop_const = 1j*omega*np.sqrt(dynamic_density/bulk_modulus)

    return impedance_porous, prop_const


def impedance_jcal(
        frequency, phi, alpha_inf, k_s,  k_t, lambda_s, lambda_t,
        rho_frame=None, temperature=20, humidity=0.5, P_stat=101325):

    const = constants(temperature, humidity, atmospheric_pressure=P_stat)
    c = const['c']
    rho_0 = const['rho_0']
    Pr = const['Pr']
    eta = const['eta']
    gamma = const['kappa']

    omega = 2*np.pi*frequency
    nu = eta/rho_0

    # alpha_inf = tortuosity

    sqrt_term = np.sqrt(1+1j*omega/nu*(2*alpha_inf*k_s/phi/lambda_s)**2)
    dynamic_tortuosity = alpha_inf + 1/1j*nu/omega*phi/k_s*sqrt_term

    nu_t = nu/Pr
    sqrt_term = np.sqrt(1+1j*omega/nu_t*(2*k_t/phi/lambda_t)**2)
    thermal_tortuosity = 1 + 1/1j*nu_t*phi/omega/k_t*sqrt_term

    rho_eq = rho_0/phi*dynamic_tortuosity
    K_eq = gamma*P_stat/phi / (gamma - (gamma-1)/thermal_tortuosity)

    bulk_modulus = K_eq
    if rho_frame is not None:
        rho_t = rho_frame + phi*rho_0
        dynamic_density = (rho_t*rho_eq-rho_0**2) / (rho_t+rho_eq-2*rho_0)
    else:
        dynamic_density = rho_eq

    zeta_m = np.sqrt(dynamic_density*bulk_modulus)/rho_0/c
    prop_const = 1j*omega*np.sqrt(dynamic_density/bulk_modulus)
    k_m = -1j*prop_const

    return zeta_m, prop_const


def bulk_modulus_jcal(
        frequency, phi, k_t, lambda_t,
        temperature=20, humidity=0.5, P_stat=101325, equivalent_fluid=True):

    const = constants(temperature, humidity, atmospheric_pressure=P_stat)
    rho_0 = const['rho_0']
    Pr = const['Pr']
    eta = const['eta']
    gamma = const['kappa']

    omega = 2*np.pi*frequency
    nu = eta/rho_0

    nu_t = nu/Pr
    sqrt_term = np.sqrt(1+1j*omega/nu_t*(2*k_t/phi/lambda_t)**2)
    thermal_tortuosity = 1 + 1/1j*nu_t*phi/omega/k_t*sqrt_term

    K_eq = gamma*P_stat / (gamma - (gamma-1)/thermal_tortuosity)

    if equivalent_fluid:
        return K_eq/phi
    else:
        return K_eq


def impedance_johnson_champoux_allard(
        frequency, phi, alpha_inf, sigma, lambda_s, lambda_t, rho_frame=None,
        temperature=20, humidity=0.5, P_stat=101325):
    """Calculate the characteristic impedance and propagation constant
    of a porous material using the Johnson, Champoux, Allard model.

    Implementation of the model described in [#]_.

    Parameters
    ----------
    frequency : ndarray, float
        The frequencies for which the model is to be calculated.
    phi : float
        Porosity of the material in the interval [0, 1].
    alpha_inf : float
        Tortuosity of the material.
    sigma : float
        Flow resistivity of the material.
    lambda_s : float
        Characteristic length of the material.
    lambda_t : float
        Characteristic thermal length of the material.
    rho_frame : float, optional
        The density of the frame, by default None
    temperature : float, optional
        Temperature in degrees Celsius, by default 20
    humidity : float, optional
        Relative humidity of air in the interval [0, 1], by default 0.5.
    P_stat : int, optional
        Static air pressure, by default 101325

    Returns
    -------
    zeta_m : complex, ndarray
        Specific impedance.
    prop_const : complex, ndarray
        Propagation constant of the material.


    References
    ----------
    [#] J.-F. Allard and N. Atalla, Propagation of sound in porous media:
        modelling sound absorbing materials, 2nd ed. Hoboken, N.J: Wiley, 2009.

    """

    const = constants(temperature, humidity, atmospheric_pressure=P_stat)
    # speed of sound
    c = const['c']
    # density of air
    rho_0 = const['rho_0']
    # Pratt number
    Pr = const['Pr']
    # viscosity of air
    eta = const['eta']
    # heat ratio
    gamma = const['kappa']

    omega = 2*np.pi*frequency

    sqrt_term = np.sqrt(
        1 + 1j*4*alpha_inf**2*eta*rho_0*omega/(sigma*lambda_s*phi)**2)
    rho_eq = alpha_inf*rho_0/phi + sigma/1j/omega*sqrt_term

    sqrt_term = np.sqrt(1 + 1j*rho_0*omega*Pr*lambda_t**2/(16*eta))
    factor = 8*eta/(1j*rho_0*omega*Pr*lambda_t**2)

    K_eq = gamma*P_stat/phi/(gamma-(gamma-1)/(1+factor*sqrt_term))

    # sqrt_term = np.sqrt(1+1j*omega/nu*(2*alpha_inf*k_s/phi/lambda_s)**2)
    # dynamic_tortuosity = alpha_inf + 1/1j*nu/omega*phi/k_s*sqrt_term

    # nu_t = nu/Pr
    # sqrt_term = np.sqrt(1+1j*omega/nu_t*(2*k_t/phi/lambda_t)**2)
    # thermal_tortuosity = 1 + 1/1j*nu_t*phi/omega/k_t*sqrt_term

    # rho_eq = rho_0/phi*dynamic_tortuosity
    # K_eq = gamma*P_stat/phi / (gamma - (gamma-1)/thermal_tortuosity)

    bulk_modulus = K_eq

    if rho_frame is not None:
        rho_t = rho_frame + phi*rho_0
        dynamic_density = (rho_t*rho_eq-rho_0**2) / (rho_t+rho_eq-2*rho_0)
    else:
        dynamic_density = rho_eq

    zeta_m = np.sqrt(dynamic_density*bulk_modulus)/rho_0/c
    prop_const = 1j*omega*np.sqrt(dynamic_density/bulk_modulus)
    k_m = -1j*prop_const

    return zeta_m, prop_const


def jacobian_jca_wavenumber_density(
        sigma, alpha_inf, phi, lambda_s, lambda_t,
        omega, rho_0, Pr, eta, gamma, P_stat):
    """Calculate the Jacobian of the wave number and the equivalent density
    according to the JCA model.

    Parameters
    ----------
    sigma : float
        Flow resistivity [Pa s / m^2]
    alpha_inf : float
        Tortuosity [-]
    phi : float
        Porosity [-]
    lambda_s : float
        Characteristic length of the solid phase [m]
    lambda_t : float
        Characteristic thermal length [m]
    omega : float, array
        Frequency [rad/s]
    rho_0 : float
        Density of air [kg/m^3]
    Pr : float
        Prandtl number [-]
    eta : float
        _description_
    gamma : float
        _description_
    P_stat : float
        _description_

    Returns
    -------
    J_sigma_k_eq : complex, array
        Jacobian of k_eq with respect to sigma
    J_alpha_k_eq : complex, array
        Jacobian of k_eq with respect to alpha_inf
    J_phi_k_eq : complex, array
        Jacobian of k_eq with respect to phi
    J_lambda_s_k_eq : complex, array
        Jacobian of k_eq with respect to lambda_s
    J_lambda_t_k_eq : complex, array
        Jacobian of k_eq with respect to lambda_t
    J_sigma_rho_eq : complex, array
        Jacobian of rho_eq with respect to sigma
    J_alpha_rho_eq : complex, array
        Jacobian of rho_eq with respect to alpha_inf
    J_phi_rho_eq : complex, array
        Jacobian of rho_eq with respect to phi
    J_lambda_s_rho_eq : complex, array
        Jacobian of rho_eq with respect to lambda_s
    J_lambda_t_rho_eq : complex, array
        Jacobian of rho_eq with respect to lambda_t
    """
    x0 = omega*rho_0
    x1 = alpha_inf**2
    x2 = 4*eta
    x3 = x1*x2
    x4 = x0*x3
    x5 = lambda_s**2
    x6 = phi**2
    x7 = sigma**2
    x8 = x5*x6*x7
    x9 = 1j*x4 + x8
    x10 = 1/sqrt(P_stat)
    x11 = 1/sqrt(gamma)
    x12 = x10*x11/sqrt(lambda_s)
    x13 = (1/2)*sqrt(omega)
    x14 = sqrt(x9)
    x15 = x14**(-1.0)
    x16 = alpha_inf*rho_0
    x17 = omega*x16
    x18 = lambda_s*x17
    x19 = 1j*x14
    x20 = x18 - x19
    x21 = x20**(-1.0)
    x22 = lambda_t**2
    x23 = Pr*x22
    x24 = alpha_inf*lambda_s*omega**2*rho_0**2*x23
    x25 = x0*x23
    x26 = 2*1j
    x27 = sqrt(eta)
    x28 = 16*eta + 1j*x25
    x29 = sqrt(x28)
    x30 = x27*x29
    x31 = x26*x30
    x32 = x25 - x31
    x33 = x32**(-1.0)
    x34 = gamma - 1
    x35 = x33*x34
    x36 = x15*x21*sqrt(1j*Pr*omega*rho_0*x14*x22*x33*x34 + alpha_inf*gamma*lambda_s*omega*rho_0 - gamma*x19 - x24*x35)
    x37 = alpha_inf*x2
    x38 = lambda_s*x14
    x39 = gamma*x32 - x25*x34
    x40 = omega**(3/2)*rho_0
    x41 = x12*x40*sqrt(x20*x33*x39)
    x42 = lambda_s**(3/2)
    x43 = 1j*phi
    x44 = x10*x11
    x45 = rho_0/phi
    x46 = x15/lambda_s
    x47 = x6**(-1.0)
    return [-x12*x13*x36*(x4 + 1j*x9)/sigma, (1/2)*x15*x21*x41*(x37 + x38), -x13*x42*x43*x44*x7*sqrt(-x33*(2*gamma*x14*x30 + gamma*x18*x31 + x19*x25 - x24))/(x17*x38 + x4 - 1j*x8), -2*eta*x1*x36*x40*x44/x42, Pr*lambda_t*x27*x35*x41*(x25 + x26*x28)/(x29*x39), -lambda_s*sigma*x15*x43/omega, x37*x45*x46 + x45, -rho_0*x3*x46*x47 - x16*x47, -x15*x3*x45/x5, 0]


def tortuosity_jcal(
        frequency, phi, alpha_inf, k_s, lambda_s,
        temperature=20, humidity=0.5, P_stat=101325):

    const = constants(temperature, humidity, atmospheric_pressure=P_stat)
    rho_0 = const['rho_0']
    eta = const['eta']

    nu = eta/rho_0
    sigma = eta/k_s
    omega = 2*np.pi*frequency

    G_j = np.sqrt(
        1 + 1j*4*alpha_inf**2*eta*rho_0*omega/(sigma*lambda_s*phi)**2)
    return alpha_inf + nu*phi/1j/omega/k_s*G_j


def impedance_dazel_elastic(
        frequency, phi, tortuosity, bulk_modulus, rho_1,
        youngs_modulus, shear_modulus, h,
        temperature=20, humidity=0.5, P_stat=101325,
        normalize=False):

    const = constants(temperature, humidity, atmospheric_pressure=P_stat)
    if normalize:
        c = const['c']
    rho_0 = const['rho_0']
    gamma = const['kappa']

    omega = 2*np.pi*frequency

    rho_12 = phi*rho_0*(1 - tortuosity)

    rho_2 = phi*rho_0
    rho_11 = rho_1 - rho_12
    rho_22 = rho_2 - rho_12
    rho_eq = rho_22/phi**2

    gamma = phi*(rho_12/rho_22 - (1-phi)/phi)

    rho = rho_11 - rho_12**2/rho_22
    rho_s = rho + gamma**2*rho_eq

    K_b = youngs_modulus/3/(1-2*shear_modulus)
    N = K_b*3/2*(1-2*shear_modulus)/(1+shear_modulus)
    P_hat = K_b + 4*N/3

    delta_s1 = omega*np.sqrt(rho/P_hat)
    delta_s2 = omega*np.sqrt(rho_s/P_hat)

    K_eq = bulk_modulus/phi
    # K_eq = K_jca/phi
    k_m = omega*np.sqrt(rho_eq/K_eq)
    delta_eq = k_m

    delta_1 = np.sqrt(((delta_s2**2 + delta_eq**2) + np.sqrt(
        (delta_s2**2 + delta_eq**2)**2 - 4*delta_s1**2*delta_eq**2)) / 2)
    delta_2 = np.sqrt(((delta_s2**2 + delta_eq**2) - np.sqrt(
        (delta_s2**2 + delta_eq**2)**2 - 4*delta_s1**2*delta_eq**2)) / 2)

    mu_1 = (delta_2**2 - delta_eq**2) / (delta_2**2 - delta_1**2)
    mu_2 = (delta_1**2 - delta_eq**2) / (delta_1**2 - delta_2**2)

    Z = bulk_modulus/1j/omega/phi * 1/(
        mu_2/delta_2*np.tan(delta_2*h) + mu_1/delta_1*np.tan(delta_1*h))

    if normalize:
        Z = Z/c/rho_0

    return Z




def model_parameter_estimation_dbm(
        R_meas, freqs, height_absorber, x0,
        model=impedance_mikis_model,
        temperature=20, humidity=0.5, P_stat=101325):

    def residuum(flow_res, freqs, R, c=constants()['c']):

        zeta_m, gamma_m_komatsu = model(
            freqs, flow_res, c=c)
        k_m = -1j*gamma_m_komatsu

        zeta_a = -1j*zeta_m/np.tan(k_m*height_absorber)
        R_a = (zeta_a-1)/(zeta_a+1)

        res_real = np.real(R_a - R)/np.sum(np.abs(R.real))
        res_imag = np.imag(R_a - R)/np.sum(np.abs(R.imag))
        return res_real + res_imag

    c = constants(temperature, humidity, P_stat)['c']

    res = optimize.least_squares(
        residuum,
        x0,
        xtol=1e-15, ftol=1e-15, gtol=1e-15,
        args=(freqs, R_meas, c))

    return res


def model_parameter_estimation_jcal(
        R_meas, freqs, height_absorber, x0=None,
        temperature=20, humidity=0.5, P_stat=101325,
        frame='rigid',
        flow_res_est=None, verbose=False):

    def residuum_jcal_minimize(
                x,
                freqs, R, temperature=20, humidity=0.5, P_stat=101325):
        phi = x[0]
        alpha_inf = x[1]
        k_s = x[2]
        k_t = x[3]
        lambda_s = x[4]
        lambda_t = x[5]
        rho_frame = x[-1] if x.size == 7 else None
        zeta_m, gamma_m = impedance_jcal(
            freqs, phi, alpha_inf, k_s,  k_t, lambda_s, lambda_t,
            rho_frame=rho_frame,
            temperature=temperature, humidity=humidity, P_stat=P_stat)
        k_m = -1j*gamma_m
        zeta_a = -1j*zeta_m/phi/np.tan(k_m*height_absorber)
        R_a = (zeta_a-1)/(zeta_a+1)

        f_real = np.sum((R_a.real - R.real)**2)/np.sum(np.abs(R.real)**2)
        f_imag = np.sum((R_a.imag - R.imag)**2)/np.sum(np.abs(R.imag)**2)

        return f_real + f_imag

    if x0 is None:
        x0 = np.array([0.5, 5, 1e-9, 1e-9, 1e-6, 1e-6])

    if frame == 'limp':
        x0 = np.r_[x0, 30]

    eta = constants()['eta']
    error = 1e9
    for _ in range(100):
        x0 *= (np.abs(np.random.randn(x0.size)/2 + 1))

        if flow_res_est is not None:
            x0[2] = eta / flow_res_est

        bounds = np.array([
            [0, 1, 0.1e-9, 0.1e-9, 1e-6, 1e-6],
            [1, 10, 100e-9, 100e-9, 20000e-6, 20000e-6]])

        if frame == 'limp':
            bounds = np.hstack((bounds, np.atleast_2d([10, 100]).T))

        mask_bounds = ~(x0 < bounds[1]) | ~(x0 > bounds[0])
        x0[mask_bounds] = bounds[0][mask_bounds]
        res = optimize.minimize(
            residuum_jcal_minimize,
            x0,
            method='Nelder-Mead',
            bounds=optimize.Bounds(bounds[0], bounds[1]),
            options={'xatol': 1e-12, 'fatol': 1e-12, 'maxfev': 400*6},
            args=(
                freqs, R_meas,
                temperature, humidity, P_stat),
        )

        new_error = residuum_jcal_minimize(
            res.x, freqs, R_meas,
            temperature, humidity, 101325)

        if new_error < error:
            if verbose:
                print(eta/res.x[2])
            res_opt = res
            error = new_error

    return res_opt


def model_parameter_estimation_jca(
        R_meas, freqs, height_absorber, x0=None,
        frame='rigid',
        temperature=20, humidity=0.5, P_stat=101325,
        flow_res_est=None, verbose=False, n_tries=100,
        method='Nelder-Mead',
        opts={'xatol': 1e-9, 'fatol': 1e-9, 'maxfev': 400*6}):

    if x0 is None:
        x0 = np.array([0.9, 1.5, 8e3, 50e-6, 300e-6])

    if frame == 'limp':
        x0 = np.r_[x0, 30]

    rng = np.random.default_rng()

    error = 1e9
    # for phi_range in np.linspace(0.1, 1, 10):
    # np.logspace()
    phi_logspace = np.logspace(2, 5, 4)
    for idx_logspace in range(phi_logspace.size-1):
        phi_range = [phi_logspace[idx_logspace], phi_logspace[idx_logspace+1]]

        from joblib import Parallel, delayed

        results = Parallel(n_jobs=8)(delayed(parallel_helper)(
            R_meas, freqs, height_absorber, frame, temperature,
            humidity, P_stat, flow_res_est, method, opts, rng, phi_range)
                for _ in range(n_tries))
        # for _ in range(n_tries):

            # res = parallel_helper(
            #     R_meas, freqs, height_absorber, frame, temperature,
            #     humidity, P_stat, flow_res_est, method, opts, rng, phi_range)

        for res in results:

            new_error = residuum_jca_minimize(
                res.x, freqs, R_meas, height_absorber,
                temperature, humidity, 101325)

            if new_error < error:
                if verbose:
                    print(res.x[2])
                res_opt = res
                error = new_error

    return res_opt

def parallel_helper(
        R_meas, freqs, height_absorber, frame, temperature, humidity,
        P_stat, flow_res_est, method, opts, rng, phi_range):
    lower = [1e-5, 1, phi_range[0], 1e-6, 1e-6]
    upper = [1, 100, phi_range[1], 2e-3, 2e-3]
    x0 = [rng.uniform(l, u) for l, u in zip(lower, upper)]
    x0[3] = x0[4]/2

    if flow_res_est is not None:
        x0[2] = flow_res_est

    bounds = np.array([
                lower,
                upper])

    if frame == 'limp':
        x0 = np.r_[x0, 30]
        bounds = np.hstack((bounds, np.atleast_2d([10, 100]).T))

    return optimize.minimize(
        residuum_jca_minimize,
        x0,
        method=method,
        bounds=optimize.Bounds(bounds[0], bounds[1]),
        options=opts,
        args=(freqs, R_meas, height_absorber, temperature, humidity, P_stat),
    )


def weight_func(x):
    return np.log10(1-1/(1+x))


def residuum_jca_minimize(
        x, freqs, R, height_absorber,
        temperature=20, humidity=0.5, P_stat=101325):
    phi = x[0]
    alpha_inf = x[1]
    sigma = x[2]
    lambda_s = x[3]
    lambda_t = x[4]
    rho_frame = x[-1] if x.size == 6 else None
    zeta_m, gamma_m = impedance_johnson_champoux_allard(
        freqs, phi, alpha_inf, sigma, lambda_s, lambda_t,
        rho_frame=rho_frame,
        temperature=temperature, humidity=humidity, P_stat=P_stat)
    k_m = -1j*gamma_m
    zeta_a = -1j*zeta_m/np.tan(k_m*height_absorber)
    R_a = (zeta_a-1)/(zeta_a+1)

    N = R_a.shape[-1]

    f_real = np.sum((R_a.real - R.real)**2)/np.sum(np.abs(R.real)**2)/N
    f_imag = np.sum((R_a.imag - R.imag)**2)/np.sum(np.abs(R.imag)**2)/N

    residual = f_real + f_imag
    if lambda_s > lambda_t:
        residual += 1e9

    # return weight_func(f_real + f_imag + f_alpha)
    return residual


def model_parameter_estimation_jca_dual_annealing(
        R_meas, freqs, height_absorber, x0=None,
        frame='rigid',
        temperature=20, humidity=0.5, P_stat=101325,
        flow_res_est=None, verbose=False, n_tries=100, n_iter=1000):

    if x0 is None:
        x0 = np.array([0.9, 1.5, 8e3, 50e-6, 300e-6])

    if frame == 'limp':
        x0 = np.r_[x0, 30]

    error = 1e20
    for _ in range(n_tries):
        x0 *= (np.abs(np.random.randn(x0.size)/2 + 1))

        if flow_res_est is not None:
            x0[2] = flow_res_est

        bounds = np.array([
            [1e-5, 1, 1000, 1e-6, 1e-6],
            [1, 100, 200e3, 20e-3, 20e-3]])

        if frame == 'limp':
            bounds = np.hstack((bounds, np.atleast_2d([10, 100]).T))

        mask_bounds = ~(x0 < bounds[1]) | ~(x0 > bounds[0])
        x0[mask_bounds] = bounds[0][mask_bounds]
        res = optimize.dual_annealing(
            residuum_jca_minimize,
            x0=x0,
            bounds=optimize.Bounds(bounds[0], bounds[1]),
            args=(
                freqs, R_meas, height_absorber,
                temperature, humidity, P_stat),
            maxiter=n_iter,
        )

        new_error = residuum_jca_minimize(
            res.x, freqs, R_meas, height_absorber,
            temperature, humidity, 101325)

        if new_error < error:
            if verbose:
                print(res.x[2])
            res_opt = res
            error = new_error

    return res_opt

