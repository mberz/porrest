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


def test_jca(jcal_test_dict, standard_air):
    freqs = np.array([1000, 2000, 3000, 4000, 5000])

    porosity=jcal_test_dict['phi']
    tortuosity=jcal_test_dict['alpha_inf']
    viscous_permeability=jcal_test_dict['k_s']
    viscous_characteristic_length=jcal_test_dict['lambda_s']
    thermal_characteristic_length=jcal_test_dict['lambda_t']

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

