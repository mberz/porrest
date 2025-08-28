import numpy as np

import pytensor
from pytensor import tensor as pt
from porrest.constants import Air

from porrest.eqfluid.implementations import (
    johnson_champoux_allard,
    johnson_champoux_allard_real_imag,
    johnson_champoux_allard_lafarge,
    johnson_champoux_allard_lafarge_real_imag,
)


def test_jca_bulk_modulus_density(
        symbolic_jca_bulk_modulus_density,
        jca_test_dict,
        air_params_dict,
    ):
    jca_params = jca_test_dict
    air_params = air_params_dict
    freqs = np.linspace(100, 5e3, 10)
    K_eq_ref, rho_eq_ref = symbolic_jca_bulk_modulus_density(
        2*np.pi*freqs, **jca_params, **air_params,
    )

    K_eq, rho_eq = johnson_champoux_allard(
        2*np.pi*freqs, **jca_params, **air_params,
    )

    np.testing.assert_allclose(K_eq, K_eq_ref)

    np.testing.assert_allclose(rho_eq, rho_eq_ref)


def test_jca_model_pytensor(
        jca_test_dict,
        symbolic_jca_bulk_modulus_density,
    ):

    freqs = np.linspace(100, 8e3, 1000)

    jca_params = jca_test_dict
    air = Air()

    K_eq_ref, rho_eq_ref = symbolic_jca_bulk_modulus_density(
        2*np.pi*freqs,
        jca_params['phi'],
        jca_params['alpha_inf'],
        jca_params['k_s'],
        jca_params['lambda_s'],
        jca_params['lambda_t'],
        air.density,
        air.prandtl_number,
        air.viscosity,
        air.heat_capacity,
        air.static_pressure,
    )

    omega = pt.vector(dtype=float, shape=freqs.shape)
    phi = pt.dscalar()
    alpha_inf = pt.dscalar()
    k_s = pt.dscalar()
    lambda_s = pt.dscalar()
    lambda_t = pt.dscalar()
    air_density = pt.dscalar()
    air_prandtl_number = pt.dscalar()
    air_viscosity = pt.dscalar()
    air_heat_capacity = pt.dscalar()
    air_static_pressure = pt.dscalar()

    model_func = johnson_champoux_allard_real_imag(
            omega,
            phi,
            alpha_inf,
            k_s,
            lambda_s,
            lambda_t,
            air_density,
            air_prandtl_number,
            air_viscosity,
            air_heat_capacity,
            air_static_pressure,
    )

    pytensor_func = pytensor.function(
        [
            omega,
            phi,
            alpha_inf,
            k_s,
            lambda_s,
            lambda_t,
            air_density,
            air_prandtl_number,
            air_viscosity,
            air_heat_capacity,
            air_static_pressure,
        ],
        model_func,
    )

    K_eq_r, K_eq_i, rho_eq_r, rho_eq_i = pytensor_func(
        2*np.pi*freqs,
        jca_params['phi'],
        jca_params['alpha_inf'],
        jca_params['k_s'],
        jca_params['lambda_s'],
        jca_params['lambda_t'],
        air.density,
        air.prandtl_number,
        air.viscosity,
        air.heat_capacity,
        air.static_pressure,
    )

    np.testing.assert_allclose(np.real(K_eq_ref), K_eq_r)

    np.testing.assert_allclose(np.imag(K_eq_ref), K_eq_i)

    np.testing.assert_allclose(np.real(rho_eq_ref), rho_eq_r)

    np.testing.assert_allclose(np.imag(rho_eq_ref), rho_eq_i)


def test_jcal_bulk_modulus_density(
        symbolic_jcal_bulk_modulus_density,
        jcal_test_dict,
        air_params_dict,
    ):
    jcal_params = jcal_test_dict
    air_params = air_params_dict
    freqs = np.linspace(100, 5e3, 10)
    K_eq_ref, rho_eq_ref = symbolic_jcal_bulk_modulus_density(
        2*np.pi*freqs, **jcal_params, **air_params,
    )

    K_eq, rho_eq = johnson_champoux_allard_lafarge(
        2*np.pi*freqs, **jcal_params, **air_params,
    )

    np.testing.assert_allclose(K_eq, K_eq_ref)

    np.testing.assert_allclose(rho_eq, rho_eq_ref)


def test_jcal_model_pytensor(
        jcal_test_dict,
        symbolic_jcal_bulk_modulus_density,
    ):

    jcal_params = jcal_test_dict

    freqs = np.linspace(100, 8e3, 1000)

    air = Air()

    K_eq_ref, rho_eq_ref = symbolic_jcal_bulk_modulus_density(
        2*np.pi*freqs,
        jcal_params['phi'],
        jcal_params['alpha_inf'],
        jcal_params['k_s'],
        jcal_params['k_t'],
        jcal_params['lambda_s'],
        jcal_params['lambda_t'],
        air.density,
        air.prandtl_number,
        air.viscosity,
        air.heat_capacity,
        air.static_pressure,
    )

    omega = pt.vector(dtype=float, shape=freqs.shape)
    phi = pt.dscalar()
    alpha_inf = pt.dscalar()
    k_s = pt.dscalar()
    k_t = pt.dscalar()
    lambda_s = pt.dscalar()
    lambda_t = pt.dscalar()
    air_density = pt.dscalar()
    air_prandtl_number = pt.dscalar()
    air_viscosity = pt.dscalar()
    air_heat_capacity = pt.dscalar()
    air_static_pressure = pt.dscalar()

    model_func = johnson_champoux_allard_lafarge_real_imag(
        omega,
        phi,
        alpha_inf,
        k_s,
        k_t,
        lambda_s,
        lambda_t,
        air_density,
        air_prandtl_number,
        air_viscosity,
        air_heat_capacity,
        air_static_pressure,
    )

    pytensor_func = pytensor.function(
        [
            omega,
            phi,
            alpha_inf,
            k_s,
            k_t,
            lambda_s,
            lambda_t,
            air_density,
            air_prandtl_number,
            air_viscosity,
            air_heat_capacity,
            air_static_pressure,
        ],
        model_func,
    )

    K_eq_r, K_eq_i, rho_eq_r, rho_eq_i = pytensor_func(
        2*np.pi*freqs,
        jcal_params['phi'],
        jcal_params['alpha_inf'],
        jcal_params['k_s'],
        jcal_params['k_t'],
        jcal_params['lambda_s'],
        jcal_params['lambda_t'],
        air.density,
        air.prandtl_number,
        air.viscosity,
        air.heat_capacity,
        air.static_pressure,
    )

    np.testing.assert_allclose(np.real(K_eq_ref), K_eq_r)

    np.testing.assert_allclose(np.imag(K_eq_ref), K_eq_i)

    np.testing.assert_allclose(np.real(rho_eq_ref), rho_eq_r)

    np.testing.assert_allclose(np.imag(rho_eq_ref), rho_eq_i)
