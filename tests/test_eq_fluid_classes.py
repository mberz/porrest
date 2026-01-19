import pytest
import numpy as np

from porrest.eqfluid.classes import (
    Miki,
    Horoshenkov,
    JohnsonChampouxAllard,
    JohnsonChampouxAllardLafarge,
)


def test_miki_model(miki_test_dict, standard_air):
    freqs = np.array([1000, 2000, 3000, 4000, 5000])
    porosity = miki_test_dict['phi']
    tortuosity = miki_test_dict['alpha_inf']
    flow_resistivity = miki_test_dict['sigma']

    eqf = Miki(
        freqs, porosity=porosity, tortuosity=tortuosity,
        flow_resistivity=flow_resistivity, saturating_fluid=standard_air)

    assert eqf.porosity == porosity
    assert eqf.tortuosity == tortuosity
    assert eqf.flow_resistivity == flow_resistivity


def test_jca(jca_test_dict, standard_air):
    freqs = np.array([1000, 2000, 3000, 4000, 5000])

    porosity=jca_test_dict['phi']
    tortuosity=jca_test_dict['alpha_inf']
    viscous_permeability=jca_test_dict['k_s']
    viscous_characteristic_length=jca_test_dict['lambda_s']
    thermal_characteristic_length=jca_test_dict['lambda_t']

    eqf = JohnsonChampouxAllard(
        freqs,
        porosity=porosity,
        tortuosity=tortuosity,
        viscous_permeability=viscous_permeability,
        viscous_characteristic_length=viscous_characteristic_length,
        thermal_characteristic_length=thermal_characteristic_length,
        saturating_fluid=standard_air,
    )

    assert eqf.porosity == porosity
    assert eqf.tortuosity == tortuosity
    assert eqf.viscous_permeability == viscous_permeability
    assert eqf.viscous_characteristic_length == viscous_characteristic_length
    assert eqf.thermal_characteristic_length == thermal_characteristic_length

def test_jca_eq_fluid_backend_consistency(
        symbolic_jcal_bulk_modulus_density,
        jcal_test_dict,
        standard_air,
    ):
    jca_params = jcal_test_dict
    jca_params['k_t'] = jca_params['phi'] * jca_params['lambda_t']**2/8
    print(jca_params)
    air_params = {
        'rho_0': standard_air.density,
        'Pr': standard_air.prandtl_number,
        'eta': standard_air.viscosity,
        'gamma': standard_air.heat_capacity,
        'P_stat': standard_air.static_pressure,
    }

    freqs = np.linspace(100, 5e3, 10)
    K_eq_ref, rho_eq_ref = symbolic_jcal_bulk_modulus_density(
        2*np.pi*freqs, **jca_params, **air_params,
    )

    porosity=jca_params['phi']
    tortuosity=jca_params['alpha_inf']
    viscous_permeability=jca_params['k_s']
    viscous_characteristic_length=jca_params['lambda_s']
    thermal_characteristic_length=jca_params['lambda_t']

    eqf = JohnsonChampouxAllard(
        freqs,
        porosity=porosity,
        tortuosity=tortuosity,
        viscous_permeability=viscous_permeability,
        viscous_characteristic_length=viscous_characteristic_length,
        thermal_characteristic_length=thermal_characteristic_length,
        saturating_fluid=standard_air,
    )
    np.testing.assert_almost_equal(
        eqf.saturating_fluid.density, air_params['rho_0'])
    np.testing.assert_almost_equal(
        eqf.saturating_fluid.prandtl_number, air_params['Pr'])
    np.testing.assert_almost_equal(
        eqf.saturating_fluid.viscosity, air_params['eta'])
    np.testing.assert_almost_equal(
        eqf.saturating_fluid.heat_capacity, air_params['gamma'])
    np.testing.assert_almost_equal(
        eqf.saturating_fluid.static_pressure, air_params['P_stat'])

    K_eq_model = eqf.bulk_modulus
    rho_eq_model = eqf.density
    np.testing.assert_allclose(
        np.real(K_eq_ref), np.real(np.squeeze(K_eq_model.freq)))
    np.testing.assert_allclose(
        np.imag(K_eq_ref), np.imag(np.squeeze(K_eq_model.freq)))
    np.testing.assert_allclose(
        np.real(rho_eq_ref), np.real(np.squeeze(rho_eq_model.freq)))
    np.testing.assert_allclose(
        np.imag(rho_eq_ref), np.imag(np.squeeze(rho_eq_model.freq)))


def test_jcal(jcal_test_dict, standard_air):
    freqs = np.array([1000, 2000, 3000, 4000, 5000])

    porosity=jcal_test_dict['phi']
    tortuosity=jcal_test_dict['alpha_inf']
    viscous_permeability=jcal_test_dict['k_s']
    viscous_characteristic_length=jcal_test_dict['lambda_s']
    thermal_characteristic_length=jcal_test_dict['lambda_t']
    thermal_permeability=jcal_test_dict['k_s']

    eqf = JohnsonChampouxAllardLafarge(
        freqs,
        porosity=porosity,
        tortuosity=tortuosity,
        viscous_permeability=viscous_permeability,
        viscous_characteristic_length=viscous_characteristic_length,
        thermal_characteristic_length=thermal_characteristic_length,
        thermal_permeability=thermal_permeability,
        saturating_fluid=standard_air,
    )

    assert eqf.porosity == porosity
    assert eqf.tortuosity == tortuosity
    assert eqf.viscous_permeability == viscous_permeability
    assert eqf.viscous_characteristic_length == viscous_characteristic_length
    assert eqf.thermal_characteristic_length == thermal_characteristic_length
    assert eqf.thermal_permeability == thermal_permeability


def test_jcal_eq_fluid_backend_consistency(
        jcal_test_dict, standard_air, symbolic_jcal_bulk_modulus_density,
    ):
    freqs = np.linspace(100, 5e3, 20)

    jcal_params = jcal_test_dict
    air_params = {
        'rho_0': standard_air.density,
        'Pr': standard_air.prandtl_number,
        'eta': standard_air.viscosity,
        'gamma': standard_air.heat_capacity,
        'P_stat': standard_air.static_pressure,
    }
    K_eq_ref, rho_eq_ref = symbolic_jcal_bulk_modulus_density(
        2*np.pi*freqs, **jcal_params, **air_params,
    )

    porosity=jcal_test_dict['phi']
    tortuosity=jcal_test_dict['alpha_inf']
    viscous_permeability=jcal_test_dict['k_s']
    viscous_characteristic_length=jcal_test_dict['lambda_s']
    thermal_characteristic_length=jcal_test_dict['lambda_t']
    thermal_permeability=jcal_test_dict['k_t']

    eqf = JohnsonChampouxAllardLafarge(
        freqs,
        porosity=porosity,
        tortuosity=tortuosity,
        viscous_permeability=viscous_permeability,
        viscous_characteristic_length=viscous_characteristic_length,
        thermal_characteristic_length=thermal_characteristic_length,
        thermal_permeability=thermal_permeability,
        saturating_fluid=standard_air,
    )

    assert eqf.porosity == porosity
    assert eqf.tortuosity == tortuosity
    assert eqf.viscous_permeability == viscous_permeability
    assert eqf.viscous_characteristic_length == viscous_characteristic_length
    assert eqf.thermal_characteristic_length == thermal_characteristic_length
    assert eqf.thermal_permeability == thermal_permeability

    K_eq_model = eqf.bulk_modulus
    rho_eq_model = eqf.density

    np.testing.assert_allclose(
        np.real(K_eq_ref), np.real(np.squeeze(K_eq_model.freq)))
    np.testing.assert_allclose(
        np.imag(K_eq_ref), np.imag(np.squeeze(K_eq_model.freq)))
    np.testing.assert_allclose(
        np.real(rho_eq_ref), np.real(np.squeeze(rho_eq_model.freq)))
    np.testing.assert_allclose(
        np.imag(rho_eq_ref), np.imag(np.squeeze(rho_eq_model.freq)))


def test_horoshenkov_model(standard_air):
    freqs = np.array([1000, 2000, 3000, 4000, 5000])

    porosity = 0.998
    median_pore_size = 147e-6  # in m
    std_pore_size = 0.325

    eqf = Horoshenkov(
        freqs,
        porosity=porosity,
        median_pore_size=median_pore_size,
        std_pore_size=std_pore_size,
        saturating_fluid=standard_air,
    )

    assert eqf.porosity == porosity
    assert eqf.median_pore_size == median_pore_size
    assert eqf.std_pore_size == std_pore_size

    exp_arg_std_poresize = (std_pore_size*np.log(2))**2
    tortuosity = np.exp(4*exp_arg_std_poresize)
    np.testing.assert_approx_equal(
        eqf.tortuosity,
        tortuosity,
    )

    np.testing.assert_approx_equal(
        eqf.viscous_permeability,
        median_pore_size**2*porosity/8/tortuosity*np.exp(-6*exp_arg_std_poresize),
    )

    np.testing.assert_approx_equal(
        eqf.thermal_permeability,
        median_pore_size**2*porosity/8/tortuosity*np.exp(6*exp_arg_std_poresize),
    )

    np.testing.assert_almost_equal(
        eqf.viscous_characteristic_length,
        median_pore_size*np.exp(-5/2*exp_arg_std_poresize),
    )

    np.testing.assert_almost_equal(
        eqf.thermal_characteristic_length,
        median_pore_size*np.exp(3/2*exp_arg_std_poresize),
    )

    with pytest.raises(AttributeError):
        eqf.flow_resistivity = 1

    with pytest.raises(AttributeError):
        eqf.viscous_permeability = 1

    with pytest.raises(AttributeError):
        eqf.viscous_characteristic_length = 1

    with pytest.raises(AttributeError):
        eqf.thermal_characteristic_length = 1

    with pytest.raises(AttributeError):
        eqf.thermal_permeability = 1

