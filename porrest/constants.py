r"""This module defines media and their respective properties.

For example, the respective properties of air with a temperature of
:math:`20^\circ` and relative humidity of :math:`50\%`
can be calculated as:

.. code-block:: python

    from porrest import Air
    air = Air(temperature=20, humidity=0.5)

    c = air.speed_of_sound  # Speed of sound in m/s
    lambda_ = air.wavelength  # Wavelength in m
    k = air.wavenumber  # Wavenumber in 1/m
    rho_0 = air.density  # Density in kg/m^3

"""
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from numbers import Number


@dataclass(frozen=True)
class Constants:
    """Class for storing constants.

    Attributes
    ----------
    T_zero_kelvin : float
        The zero point of the Kelvin scale in degrees Celsius.
    T_ref : float
        The reference temperature (20 deg. Celsius) in Kelvin.
    p_ref_atmospheric : float
        The reference atmospheric pressure in Pascals (101325 Pa).

    """

    T_ref: float = 20
    T_zero_kelvin: float = 273.15
    T_ref_kelvin: float = T_ref + T_zero_kelvin
    p_ref_atmospheric: float = 101325


class _Medium(ABC):
    """Base class defining a medium.
    """

    def __init__(
            self,
            frequencies: np.ndarray[float] | None = None):
        self._frequencies = frequencies if frequencies is not None else np.array([])

    @property
    def wavelength(self):
        """The wavelength."""
        if self.frequencies.size == 0 or self.frequencies is None:
            raise ValueError(
                "Frequencies must be set to calculate the wavelength.")
        return self.speed_of_sound/self.frequencies

    @property
    def wavenumber(self):
        """The wavenumber."""
        if self.frequencies.size == 0 or self.frequencies is None:
            raise ValueError(
                "Frequencies must be set to calculate the wavenumber.")
        return 2*np.pi/self.wavelength

    @property
    def frequencies(self):
        """The frequencies for which models are evaluated.
        """
        return self._frequencies

    @frequencies.setter
    def frequencies(self, frequencies: np.ndarray[float] | Number):
        """The frequencies for which models are evaluated.
        """
        self._frequencies = np.asarray(frequencies)

    @property
    @abstractmethod
    def speed_of_sound(self):
        """The speed of sound in the medium."""
        pass

    @property
    @abstractmethod
    def density(self):
        """The density of the medium."""
        pass

    @property
    @abstractmethod
    def bulk_modulus(self):
        """The bulk modulus of the medium."""
        pass


class Fluid(_Medium):
    """Base class for gases.

    """

    def __init__(
            self,
            temperature: float = 20.,
            frequencies: np.ndarray[float] | None = None,
        ):
        super().__init__(frequencies)
        self._temperature = temperature


    @property
    def temperature(self):
        """The temperature of the medium in degrees Celsius."""
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        """Set the temperature of the medium in degrees Celsius."""
        self._temperature = temperature

    @property
    def temperature_kelvin(self):
        """The temperature of the medium in Kelvin."""
        return self.temperature + Constants.T_zero_kelvin

    @property
    @abstractmethod
    def viscosity(self):
        """The dynamic viscosity of the medium."""
        pass


class Air(Fluid):
    """The medium air.

    Parameters
    ----------
    temperature : float
        The temperature of the air in degrees Celsius.
    humidity : float
        The relative humidity of the air in the interval [0, 1].
    atmospheric_pressure : float
        The atmospheric pressure in Pascals.
    frequencies : np.ndarray[float] | None
        The frequencies for which frequency dependent models are evaluated.
        If None, an empty array is used which will raise an error when trying
        to calculate the wavelength or wavenumber. Default is None.
    """

    def __init__(
            self,
            temperature: float = 20,
            humidity: float = 0.5,
            static_pressure: float = 101325,
            frequencies: np.ndarray[float] | None = None,
            ):
        super().__init__(temperature, frequencies=frequencies)
        self.humidity = humidity
        self.static_pressure = static_pressure

    @property
    def humidity(self):
        """The relative humidity of the air in the interval [0, 1]."""
        return self._humidity

    @humidity.setter
    def humidity(self, humidity):
        """Set the relative humidity of the air in the interval [0, 1]."""
        if not (0 <= humidity <= 1):
            raise ValueError("Humidity must be in the interval [0, 1].")
        self._humidity = humidity

    @property
    def static_pressure(self):
        """The static pressure of the air in Pascals."""
        return self._static_pressure

    @static_pressure.setter
    def static_pressure(self, static_pressure):
        """Set the static pressure of the air in Pascals."""
        if static_pressure <= 0:
            raise ValueError("Static pressure must be greater than 0.")
        self._static_pressure = static_pressure

    @property
    def bulk_modulus(self):
        """The bulk modulus."""
        return self.density * self.speed_of_sound**2

    @property
    def prandtl_number(self):
        """The Prandtl number."""
        return 1e9/(
            1.1*self.temperature**3 -
            120*self.temperature**2 +
            322000*self.temperature +
            1.393e9)

    @property
    def density(self):
        """The density."""
        return self.static_pressure / (
            self.gas_constant * self.temperature_kelvin)

    @property
    def viscosity(self):
        """The dynamic viscosity."""
        # Sutherland's formula
        return 1.485e-6*self.temperature_kelvin**(3/2)/(
            self.temperature_kelvin + 110.4)

    @property
    def heat_capacity(self):
        """The heat capacity ratio."""
        # heat capacity ratio
        kappa = 1.4
        return kappa

    @property
    def gas_constant(self):
        """The gas constant for air with relative humidity."""
        T_ref = Constants.T_ref_kelvin
        V = -6.8346*(T_ref / self.temperature_kelvin)**1.261 + 4.6151

        # saturation vapor pressure
        p_ref = Constants.p_ref_atmospheric
        p_sat = p_ref*10**V

        # molar mass of dry air
        M_r = 0.0289644

        # molar gas constant for air
        R_mol = 8.31

        # gas constant for dry air [J/(kg*K)]
        R_l = R_mol/M_r

        # gas constant of water vapor
        R_d = 461

        # gas constant for air with relative humidity phi [J/(kg K)]
        h = 100*self.humidity*p_sat/self.static_pressure

        R_f = R_l/(1-(h/100)*(1-R_l/R_d))

        return R_f

    @property
    def speed_of_sound(self):
        """The speed of sound."""
        return np.sqrt(
            self.heat_capacity*self.gas_constant*self.temperature_kelvin)
