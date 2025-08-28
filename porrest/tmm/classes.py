"""Classes for the transfer matrix method (TMM)."""
import numpy as np
import pyfar as pf
from porrest.eqfluid import EquivalentFluidData


class SurfaceImpedance():
    """Surface impedance boundary data.
    """

    def __init__(self, impedance, surrounding_medium, frequencies=None):
        if isinstance(impedance, pf.FrequencyData):
            impedance = impedance.freq
            frequencies = impedance.frequencies
        self._impedance = impedance
        self._surrounding_medium = surrounding_medium
        self._frequencies = frequencies

    @property
    def frequencies(self):
        """Frequencies of the surface impedance.
        """
        return self._frequencies

    @classmethod
    def from_equivalent_fluid_layer(
            cls,
            fluid,
            layer_height,
            backing='rigid',
        ):
        """Create a surface impedance from a fluid layer.
        """
        Z_m = fluid.characteristic_impedance
        if isinstance(Z_m, pf.FrequencyData):
            Z_m = Z_m.freq

        k_m = fluid.wavenumber
        if isinstance(k_m, pf.FrequencyData):
            k_m = k_m.freq

        if backing == 'rigid':
            Z_rigid = -1j*Z_m/np.tan(k_m*layer_height)

        return cls(Z_rigid, fluid.saturating_fluid, fluid.frequencies)

    @property
    def reflection_factor(self):
        """Reflection factor of the surface impedance.
        """
        c_0 = self._surrounding_medium.speed_of_sound
        rho_0 = self._surrounding_medium.density
        Z_0 = c_0 * rho_0
        if isinstance(Z_0, pf.FrequencyData):
            Z_0 = Z_0.freq

        return pf.FrequencyData(
            (self._impedance - Z_0)/(self._impedance + Z_0),
            frequencies=self.frequencies,
        )

    @property
    def absorption_coefficient(self):
        """Absorption coefficient of the surface impedance.
        """
        return pf.FrequencyData(
            1 - np.abs(self.reflection_factor.freq)**2,
            frequencies=self.frequencies,
        )


    @property
    def impedance(self):
        """Impedance of the surface impedance.
        """
        return pf.FrequencyData(
            self._impedance,
            frequencies=self.frequencies,
        )

    @property
    def surrounding_medium(self):
        """Surrounding medium of the surface impedance.
        """
        return self._surrounding_medium

    @property
    def normalized_impedance(self):
        """Normalized impedance of the surface impedance.
        """
        c_0 = self._surrounding_medium.speed_of_sound
        rho_0 = self._surrounding_medium.density
        Z_0 = c_0 * rho_0

        return pf.FrequencyData(
            self._impedance / Z_0,
            frequencies=self.frequencies,
        )



class SymmetricScatteringMatrix():
    """Scattering matrix for a transfer matrix method (TMM) layer.
    """

    def __init__(self, matrix, frequencies=None):
        if isinstance(matrix, pf.FrequencyData):
            matrix = matrix.freq
            frequencies = matrix.frequencies
        self._matrix = matrix
        self._frequencies = frequencies

    @property
    def frequencies(self):
        """Frequencies of the scattering matrix.
        """
        return self._frequencies

    @property
    def matrix(self):
        """Scattering matrix.
        """
        return pf.FrequencyData(
            self._matrix,
            frequencies=self.frequencies,
        )

    @classmethod
    def from_reflection_transmission(
            cls,
            reflection_coefficient,
            transmission_coefficient,
            frequencies=None,
        ):
        """Create a symmetric scattering matrix from reflection and
        transmission coefficients.
        """
        if isinstance(reflection_coefficient, pf.FrequencyData):
            frequencies = reflection_coefficient.frequencies
            reflection_coefficient = reflection_coefficient.freq
        if isinstance(transmission_coefficient, pf.FrequencyData):
            frequencies = transmission_coefficient.frequencies
            transmission_coefficient = transmission_coefficient.freq

        matrix = np.array([
            [reflection_coefficient, transmission_coefficient],
            [transmission_coefficient, reflection_coefficient],
        ])

        return cls(matrix, frequencies)


    def to_equivalent_fluid_layer(
            self,
            surrounding_medium,
            layer_height,
        ):
        """Convert the scattering matrix to an equivalent fluid layer.
        """
        rho0 = surrounding_medium.density
        c0 = surrounding_medium.speed_of_sound
        Z_0 = c0 * rho0
        if isinstance(Z_0, pf.FrequencyData):
            Z_0 = Z_0.freq

        R = self.matrix[0, 0].freq
        T = self.matrix[0, 1].freq

        Z_m = np.sqrt(((1+R)**2 - T**2)/((1-R)**2 - T**2) )*Z_0
        exp_km = 1/T*(1 + (Z_0 - Z_m)/(Z_0 + Z_m)*R)
        k_m = np.log(exp_km) / layer_height / 1j

        return EquivalentFluidData.from_impedance_wavenumber(
            Z_c=pf.FrequencyData(Z_m, self.frequencies),
            k=pf.FrequencyData(k_m, self.frequencies),
            saturating_fluid=surrounding_medium,
        )
