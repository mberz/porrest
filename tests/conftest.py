import pytest
from sympy import lambdify
import sympy as sym
from porrest.constants import Air


@pytest.fixture
def standard_air():
    return Air(temperature=20, humidity=50/100, static_pressure=101325)

@pytest.fixture
def miki_test_dict():
    return {
        'phi': 0.99,
        'alpha_inf': 1.01,
        'sigma': 12.9e3,
        'k_s': 14.3e-10,
    }

@pytest.fixture
def jca_test_dict():
    return {
        'phi': 0.99,
        'alpha_inf': 1.01,
        'k_s': 14.3e-10,
        'lambda_s': 184e-6,
        'lambda_t': 82e-6,
    }

@pytest.fixture
def jcal_test_dict():
    return {
        'phi': 0.99,
        'alpha_inf': 1.01,
        'k_s': 12e-10,
        'k_t': 32e-10,
        'lambda_s': 184e-6,
        'lambda_t': 82e-6,
    }

@pytest.fixture
def air_params_dict():
    return {
        'rho_0': 1.2,
        'Pr': 0.715,
        'eta': 1.85e-5,
        'gamma': 1.4,
        'P_stat': 101325,
    }

@pytest.fixture
def symbolic_jca_bulk_modulus_density():

    Z_r, Z_i, gamma_r, gamma_i, omega, c, f, sigma, Z_w, h = sym.symbols(
        'Z_r, Z_i, gamma_r, gamma_i, omega, c, f, sigma, Z_w, h',
        real=True, positive=True)

    f, phi, alpha_inf, k_s,  k_t, lambda_s, lambda_t = sym.symbols(
        'f, phi, alpha_inf, k_s,  k_t, lambda_s, lambda_t',
        real=True, positive=True)

    T_c, rH, P_s, c, rho_0, Pr, eta, gamma, P_stat = sym.symbols(
        'T_c, rH, P_s, c, rho_0, Pr, eta, gamma, P_stat',
        real=True, positive=True)

    nu = eta/rho_0
    nu_t = nu/Pr
    G_jca = sym.sqrt(1 + (2*alpha_inf*k_s/phi/lambda_s)**2 * sym.I*omega/nu)
    G_dash_jca = sym.sqrt(1 + (lambda_t/4)**2 * sym.I*omega/nu_t)

    K_eq_jca = (gamma*P_stat/phi/(gamma-(gamma-1)/(1 + nu_t*phi/sym.I/omega/k_t*G_dash_jca))).subs(k_t, phi*lambda_t**2/8)

    rho_eq_jca = rho_0/phi*(alpha_inf + nu*phi/sym.I/omega/k_s*G_jca)

    x = [
        omega,
        phi,
        alpha_inf,
        k_s,
        lambda_s,
        lambda_t,
        rho_0,
        Pr,
        eta,
        gamma,
        P_stat]

    func_jca = lambdify(
        x,
        [K_eq_jca, rho_eq_jca],
        modules=['numpy'], cse=True)

    return func_jca


@pytest.fixture
def symbolic_jcal_bulk_modulus_density():

    Z_r, Z_i, gamma_r, gamma_i, omega, c, f, sigma, Z_w, h = sym.symbols(
        'Z_r, Z_i, gamma_r, gamma_i, omega, c, f, sigma, Z_w, h',
        real=True, positive=True)

    f, phi, alpha_inf, k_s,  k_t, lambda_s, lambda_t = sym.symbols(
        'f, phi, alpha_inf, k_s,  k_t, lambda_s, lambda_t',
        real=True, positive=True)

    T_c, rH, P_s, c, rho_0, Pr, eta, gamma, P_stat = sym.symbols(
        'T_c, rH, P_s, c, rho_0, Pr, eta, gamma, P_stat',
        real=True, positive=True)

    nu = eta/rho_0
    nu_t = nu/Pr
    G_jca = sym.sqrt(1 + (2*alpha_inf*k_s/phi/lambda_s)**2 * sym.I*omega/nu)
    G_dash_jcal = sym.sqrt(1 + (2*k_t/phi/lambda_t)**2 * sym.I*omega/nu_t)

    K_eq_jcal = (gamma*P_stat/phi/(gamma-(gamma-1)/(1 + nu_t*phi/sym.I/omega/k_t*G_dash_jcal)))

    rho_eq_jcal = rho_0/phi*(alpha_inf + nu*phi/sym.I/omega/k_s*G_jca)

    x = [
        omega,
        phi,
        alpha_inf,
        k_s,
        k_t,
        lambda_s,
        lambda_t,
        rho_0,
        Pr,
        eta,
        gamma,
        P_stat]

    func_jcal = lambdify(
        x,
        [K_eq_jcal, rho_eq_jcal],
        modules=['numpy'], cse=True)

    return func_jcal
