import pytest
import numpy as np


def test_constants():
    """Test the Constants class."""
    from porrest.constants import Constants

    # Test default values
    constants = Constants()
    assert constants.T_ref == 20
    assert constants.T_zero_kelvin == 273.15
    assert constants.p_ref_atmospheric == 101325

    # Test custom values
    custom_constants = Constants(
        T_ref=25, T_zero_kelvin=300, p_ref_atmospheric=100000)
    assert custom_constants.T_ref == 25
    assert custom_constants.T_zero_kelvin == 300
    assert custom_constants.p_ref_atmospheric == 100000


def test_air():
    """Test the Air class."""
    from porrest.constants import Air

    # Test default values
    air = Air()
    assert air.temperature == 20
    assert air.temperature_kelvin == 293.15
    np.testing.assert_approx_equal(air.speed_of_sound, 343.34, significant=2)
    np.testing.assert_approx_equal(air.density, 1.2041, significant=2)

    #
    freqs = np.array([1000])
    # Test frequency, wavelength, and wavenumber properties
    air.frequencies = freqs
    assert np.array_equal(air.frequencies, freqs)
    np.testing.assert_approx_equal(air.wavelength, 0.34334, significant=2)
    np.testing.assert_approx_equal(air.wavenumber, 18.3, significant=2)

    # Test custom values
    custom_air = Air(temperature=25, frequencies=np.array([1000, 2000, 3000]))
    assert custom_air.temperature == 25
    np.testing.assert_array_equal(
        custom_air.frequencies, np.array([1000, 2000, 3000]))


