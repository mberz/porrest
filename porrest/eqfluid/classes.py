"""Class implementations for equivalent fluid models.
"""
from porrest.constants import Air, Fluid
from porrest.constants import _Medium
from abc import abstractmethod, ABC
import numpy as np
import pyfar as pf
from numbers import Number
from .implementations import (
    johnson_champoux_allard_lafarge,
    mikis_model,
)


class EquivalentFluidBase(_Medium, ABC):
    """Base class for equivalent fluid models.

    """

    def __init__(
            self,
            frequencies=None,
            porosity: Number | None = None,
            saturating_fluid: Fluid | None = None,
        ):
        super().__init__(frequencies)
        self.porosity = porosity
        self.saturating_fluid = saturating_fluid

    @property
    def saturating_fluid(self):
        """The fluid saturating the pores."""
        return self._saturating_fluid

    @saturating_fluid.setter
    def saturating_fluid(self, saturating_fluid: Fluid | None):
        """The fluid saturating the pores."""
        if saturating_fluid is None:
            self._saturating_fluid = None
            return
        if not isinstance(saturating_fluid, Fluid):
            raise TypeError(
                f"Expected a Fluid object, got {type(saturating_fluid)}",
            )
        self._saturating_fluid = saturating_fluid

    @property
    @abstractmethod
    def bulk_modulus(self):
        r"""The bulk modulus of the equivalent fluid :math:`K_\mathrm{eq}`."""
        pass

    @property
    @abstractmethod
    def density(self):
        r"""The dynamic density :math:`\rho_\mathrm{eq}`."""
        pass

    @property
    def characteristic_impedance(self):
        r"""The characteristic impedance :math:`Z_\mathrm{eq}`."""
        if isinstance(self.bulk_modulus, pf.FrequencyData):
            bulk_modulus = self.bulk_modulus.freq
        if isinstance(self.density, pf.FrequencyData):
            density = self.density.freq
        Z_c = np.sqrt(bulk_modulus*density)
        return pf.FrequencyData(Z_c, frequencies=self.frequencies)

    @property
    def wavenumber(self):
        r"""The wavenumber of the equivalent fluid :math:`k_\mathrm{eq}`."""
        omega = 2*np.pi*self.frequencies
        if isinstance(self.bulk_modulus, pf.FrequencyData):
            bulk_modulus = self.bulk_modulus.freq
        if isinstance(self.density, pf.FrequencyData):
            density = self.density.freq
        wavenum = omega*np.sqrt(density/bulk_modulus)
        return pf.FrequencyData(wavenum, frequencies=self.frequencies)

    @property
    def speed_of_sound(self):
        r"""The speed of sound in the equivalent fluid :math:`c_\mathrm{eq}`.
        """
        if isinstance(self.wavenumber, pf.FrequencyData):
            wavenumber = self.wavenumber.freq
            c = 2*np.pi*self.frequencies/wavenumber
        return pf.FrequencyData(c, frequencies=self.frequencies)

    @property
    def porosity(self):
        r"""The porosity :math:`\phi`."""
        return self._porosity

    @porosity.setter
    def porosity(self, value: Number):
        """Set the porosity of the equivalent fluid."""
        if value < 0 or value > 1:
            raise ValueError(
                "Porosity must be between 0 and 1.",
            )
        self._porosity = value


class PhenomenologicalBase(EquivalentFluidBase, ABC):
    def __init__(
            self,
            frequencies: np.ndarray[float],
            porosity: Number | None = None,
            saturating_fluid: Fluid = Air(),
        ):
        super().__init__(
            frequencies=frequencies,
            porosity=porosity,
            saturating_fluid=saturating_fluid,
        )
        self._tortuosity = None
        self._viscous_permeability = None
        self._flow_resistivity = None

    @property
    def tortuosity(self):
        r"""The tortuosity :math:`\alpha_\infty`."""
        return self._tortuosity

    @property
    def viscous_permeability(self):
        """The viscous permeability :math:`q_0`."""
        return self._viscous_permeability

    @property
    def flow_resistivity(self):
        r"""The flow resistivity :math:`\sigma`."""
        return self.saturating_fluid.viscosity / self.viscous_permeability


class Miki(PhenomenologicalBase):
    r"""Miki's phenomenological equivalent fluid model.

    The model is implemented as described in [#]_ with the modifications
    proposed in [#]_.

    The model is calculated using the following equations for the
    characteristic impedance :math:`Z_m` and wavenumber :math:`k_m`:

    .. math::

        Z_m = \frac{\sqrt{\alpha_{\infty}} c_{0} \rho_{0} \left(f^{0.632} + 0.07 \sigma^{0.632} - 0.107 i \sigma^{0.632}\right)}{f^{0.632} \phi}

        k_m = \frac{\pi \sqrt{\alpha_{\infty}} \left(f^{0.382} \sigma^{0.618} \left(0.218 - 0.32 i\right) + 2.0 f^{1.0}\right)}{c_{0}}


    Parameters
    ----------
    frequencies : np.ndarray[float]
        The frequencies at which the model is evaluated.
    porosity : Number | None
        The porosity :math:`\phi`.
    tortuosity : Number | None
        The tortuosity :math:`\alpha_\infty`.
    flow_resistivity : Number | None
        The flow resistivity :math:`\sigma`.
        Either this or the `viscous_permeability`
        must be set.
    viscous_permeability : Number | None
        The viscous permeability :math:`q_0`.
        Either this or the `flow_resistivity` must be set.
    saturating_fluid : Fluid, optional
        The medium saturating the pores, by default `Air()`


    References
    ----------
    .. [#]  Y. Miki, “Acoustical properties of porous materials.
            Modifications of Delany-Bazley models,” Journal of the Acoustical
            Society of Japan, vol. 11, no. 1, pp. 19–24, 1990.
    .. [#]  Y. Miki, “Acoustical properties of porous materials:
            Generalizations of empirical models.,” Journal of the Acoustical
            Society of Japan, vol. 11, no. 1, pp. 25-28, 1990.


    """  # noqa: E501

    def __init__(
            self,
            frequencies: np.ndarray[float],
            porosity: Number | None = None,
            tortuosity: Number | None = None,
            flow_resistivity: Number | None = None,
            viscous_permeability: Number | None = None,
            saturating_fluid: Fluid = Air(),
        ):
        """Initialize the Miki equivalent fluid model.
        """
        super().__init__(frequencies, porosity, saturating_fluid)
        self.tortuosity = tortuosity

        if (flow_resistivity and viscous_permeability):
            raise ValueError(
                "Either viscous_permeability or flow_resistivity must be set,",
            )
        if viscous_permeability is not None:
            self.viscous_permeability = viscous_permeability
        elif flow_resistivity is not None:
            permeability = saturating_fluid.viscosity/flow_resistivity
            self.viscous_permeability = permeability
        else:
            raise ValueError(
                "Either viscous_permeability or flow_resistivity must be set.",
            )

    @PhenomenologicalBase.tortuosity.setter
    def tortuosity(self, value: Number):
        """Set the tortuosity of the equivalent fluid."""
        if value < 1:
            raise ValueError(
                "Tortuosity must be greater than or equal to 1.",
            )
        self._tortuosity = value

    @PhenomenologicalBase.viscous_permeability.setter
    def viscous_permeability(self, value: Number):
        """Set the viscous permeability of the equivalent fluid."""
        if value < 0:
            raise ValueError(
                "Viscous permeability must be greater than 0.",
            )
        self._viscous_permeability = value

    @PhenomenologicalBase.flow_resistivity.setter
    def flow_resistivity(self, value: Number):
        """Set the flow resistivity of the equivalent fluid."""
        if value < 0:
            raise ValueError(
                "Flow resistivity must be greater than 0.",
            )
        self.viscous_permeability = self.saturating_fluid.viscosity/value

    @property
    def bulk_modulus(self):
        r"""The bulk modulus :math:`K_\mathrm{eq}`."""
        K_eq = mikis_model(
            self.frequencies,
            self.porosity,
            self.tortuosity,
            self.flow_resistivity,
            self.saturating_fluid.speed_of_sound,
            self.saturating_fluid.density,
        )[0]

        return pf.FrequencyData(K_eq, frequencies=self.frequencies)

    @property
    def density(self):
        r"""The dynamic density :math:`\rho_\mathrm{eq}`."""
        rho_eq = mikis_model(
            self.frequencies,
            self.porosity,
            self.tortuosity,
            self.flow_resistivity,
            self.saturating_fluid.speed_of_sound,
            self.saturating_fluid.density,
        )[1]

        return pf.FrequencyData(rho_eq, frequencies=self.frequencies)



class SemiPhenomenologicalBase(PhenomenologicalBase):
    r"""Johnson-Champoux-Allard-Lafarge and derived equivalent fluid model.

    This is a base class for the Johnson-Champoux-Allard-Lafarge equivalent
    fluid model, which implements the common properties and methods for the
    model. The actual implementation is done in the
    `JohnsonChampouxAllardLafarge` class.

    """

    def __init__(
            self,
            frequencies: np.ndarray[float],
            porosity: Number | None = None,
            saturating_fluid: Fluid = Air(),
        ):
        super().__init__(
            frequencies,
            porosity=porosity,
            saturating_fluid=saturating_fluid,
        )
        self._viscous_characteristic_length = None
        self._thermal_characteristic_length = None
        self._thermal_permeability = None

    @property
    def viscous_characteristic_length(self):
        r"""The viscous characteristic length :math:`\Lambda`."""
        return self._viscous_characteristic_length

    @property
    def thermal_characteristic_length(self):
        r"""The thermal characteristic length :math:`\Lambda^\prime`."""
        return self._thermal_characteristic_length

    @property
    def viscous_characteristic_frequency(self):
        """The viscous characteristic frequency."""
        nu = self.saturating_fluid.viscosity / self.saturating_fluid.density
        return (1/2/np.pi*self.porosity*nu /
            self.tortuosity/self.viscous_permeability)

    @property
    def thermal_permeability(self):
        """The thermal permeability of the equivalent fluid."""
        return self._thermal_permeability

    @property
    def thermal_characteristic_frequency(self):
        """The viscous characteristic frequency of the equivalent fluid."""
        nu = self.saturating_fluid.viscosity / self.saturating_fluid.density
        nu_dash = nu / self.saturating_fluid.prandtl_number

        return (1/2/np.pi*self.porosity*nu_dash /self.thermal_permeability)

    @property
    def bulk_modulus(self):
        r"""The bulk modulus of the equivalent fluid :math:`K_\mathrm{eq}`.
        """
        K_eq = johnson_champoux_allard_lafarge(
            2*np.pi*self.frequencies,
            self.porosity,
            self.tortuosity,
            self.viscous_permeability,
            self.thermal_permeability,
            self.viscous_characteristic_length,
            self.thermal_characteristic_length,
            self.saturating_fluid.density,
            self.saturating_fluid.prandtl_number,
            self.saturating_fluid.viscosity,
            self.saturating_fluid.heat_capacity,
            self.saturating_fluid.static_pressure,
        )[0]

        return pf.FrequencyData(K_eq, frequencies=self.frequencies)

    @property
    def density(self):
        r"""The density of the equivalent fluid :math:`\rho_\mathrm{eq}`.
        """
        rho_eq = johnson_champoux_allard_lafarge(
            2*np.pi*self.frequencies,
            self.porosity,
            self.tortuosity,
            self.viscous_permeability,
            self.thermal_permeability,
            self.viscous_characteristic_length,
            self.thermal_characteristic_length,
            self.saturating_fluid.density,
            self.saturating_fluid.prandtl_number,
            self.saturating_fluid.viscosity,
            self.saturating_fluid.heat_capacity,
            self.saturating_fluid.static_pressure,
        )[1]

        return pf.FrequencyData(rho_eq, frequencies=self.frequencies)



class JohnsonChampouxAllard(SemiPhenomenologicalBase, Miki):
    r"""Johnson-Champoux-Allard equivalent fluid model.

    The model is calculated using the following equations for the
    bulk modulus :math:`K_\mathrm{eq}` (see [#]_) and dynamic density
    :math:`\rho_\mathrm{eq}` (see [#]_):

    .. math::

        K_{\mathrm{eq}} =\frac{P_{0} \gamma}{\phi \left(\gamma - \frac{\gamma - 1}{1 - \frac{8 i \eta \sqrt{\frac{i Pr \left(\Lambda^{\prime}\right)^{2} \omega \rho_{0}}{16 \eta} + 1}}{Pr \left(\Lambda^{\prime}\right)^{2} \omega \rho_{0}}}\right)}

        \rho_{\mathrm{eq}} = \frac{\alpha_{\infty} \rho_{0}}{\phi} - \frac{i \sqrt{\eta} \sqrt{\Lambda^{2} \eta \phi^{2} + 4 i \alpha_{\infty}^{2} \omega q_0^{2} \rho_{0}}}{\Lambda \omega \phi q_0}

    These are taken from Section. 8.6 of Ref. [#]_.


    Parameters
    ----------
    frequencies : np.ndarray[float]
        The frequencies at which the model is evaluated.
    porosity : Number | None
        The porosity :math:`\phi`.
    tortuosity : Number | None
        The tortuosity :math:`\alpha_\infty`.
    viscous_permeability : Number | None
        The viscous permeability :math:`q_0`.
        Either this or the `flow_resistivity` must be set.
    viscous_characteristic_length : Number | None
        The viscous characteristic length :math:`\Lambda`.
    thermal_characteristic_length : Number | None
        The thermal characteristic length :math:`\Lambda^\prime`.
    flow_resistivity : Number | None
        The flow resistivity :math:`\sigma`.
        Either this or the `viscous_permeability`
        must be set.
    saturating_fluid : Fluid, optional
        The medium saturating the pores, by default `Air()`

    References
    ----------
    .. [#]  D. L. Johnson, J. Koplik, and R. Dashen, “Theory of dynamic
            permeability and tortuosity in fluids saturated porous media,”
            J. Fluid Mech., vol. 176, pp. 379-402, Mar. 1987,
            doi: 10.1017/S0022112087000727.
    .. [#]  Y. Champoux and J. Allard, “Dynamic tortuosity and bulk modulus in
            air-saturated porous media,” J. Appl. Phys., vol. 70, no. 4,
            pp. 1975-1979, Aug. 1991, doi: 10.1063/1.349482.
    .. [#]  J.-F. Allard and N. Atalla, Propagation of sound in porous media:
            modelling sound absorbing materials, 2nd ed.
            Hoboken, N.J: Wiley, 2009.



    """  # noqa: E501

    def __init__(
            self,
            frequencies: np.ndarray[float],
            porosity: Number | None = None,
            tortuosity: Number | None = None,
            viscous_permeability: Number | None = None,
            viscous_characteristic_length: Number | None = None,
            thermal_characteristic_length: Number | None = None,
            flow_resistivity: Number | None = None,
            saturating_fluid: Fluid = Air(),
        ):
        r"""Init the Johnson-Champoux-Allard equivalent fluid model.

        Parameters
        ----------
        frequencies : np.ndarray[float]
            The frequencies at which the model is evaluated.
        porosity : Number | None
            The porosity :math:`\phi`.
        tortuosity : Number | None
            The tortuosity :math:`\alpha_\infty`.
        viscous_permeability : Number | None
            The viscous permeability :math:`q_0`.
            Either this or the `flow_resistivity` must be set.
        viscous_characteristic_length : Number | None
            The viscous characteristic length :math:`\Lambda`.
        thermal_characteristic_length : Number | None
            The thermal characteristic length :math:`\Lambda^\prime`.
        flow_resistivity : Number | None
            The flow resistivity :math:`\sigma`.
            Either this or the `viscous_permeability`
            must be set.
        saturating_fluid : Fluid, optional
            The medium saturating the pores, by default `Air()`
        """
        Miki.__init__(
            self=self,
            frequencies=frequencies,
            porosity=porosity,
            tortuosity=tortuosity,
            viscous_permeability=viscous_permeability,
            flow_resistivity=flow_resistivity,
            saturating_fluid=saturating_fluid,
        )

        self.viscous_characteristic_length = viscous_characteristic_length
        self.thermal_characteristic_length = thermal_characteristic_length

    @SemiPhenomenologicalBase.viscous_characteristic_length.setter
    def viscous_characteristic_length(
            self,
            value: Number,
        ):
        """Set the viscous characteristic length of the equivalent fluid."""
        if value < 0:
            raise ValueError(
                "Viscous characteristic length must be greater than 0.",
            )
        self._viscous_characteristic_length = value

    @SemiPhenomenologicalBase.thermal_characteristic_length.setter
    def thermal_characteristic_length(
            self,
            value: Number,
        ):
        """Set the thermal characteristic length of the equivalent fluid."""
        if value < 0:
            raise ValueError(
                "Thermal characteristic length must be greater than 0.",
            )
        self._thermal_characteristic_length = value

    @property
    def thermal_permeability(self):
        r"""The thermal permeability :math:`q^\prime` of the equivalent fluid.
        Approximated as :math:`\frac{\phi (\Lambda^\prime)^2}{8}`.
        """
        return self.porosity*self.thermal_characteristic_length**2 / 8

    @property
    def bulk_modulus(self):
        r"""The bulk modulus :math:`K_\mathrm{eq}`."""
        return SemiPhenomenologicalBase.bulk_modulus.fget(self)

    @property
    def density(self):
        r"""The density :math:`\rho_\mathrm{eq}`."""
        return SemiPhenomenologicalBase.density.fget(self)


class JohnsonChampouxAllardLafarge(
        JohnsonChampouxAllard, SemiPhenomenologicalBase,
    ):
    r"""Johnson-Champoux-Allard-Lafarge equivalent fluid model.

    The model is calculated using the following equations for the
    bulk modulus :math:`K_\mathrm{eq}` (see [#]_ and [#]_) and dynamic density
    :math:`\rho_\mathrm{eq}` (see [#]_):

    .. math::

        K_{\mathrm{eq}} = \frac{P_{0} \gamma}{\phi \left(\gamma - \frac{\gamma - 1}{1 - \frac{i \eta \phi \sqrt{\frac{4 i Pr \omega \left(q_0^{\prime}\right)^{2} \rho_{0}}{\left(\Lambda^{\prime}\right)^{2} \eta \phi^{2}} + 1}}{Pr \omega q_0^{\prime} \rho_{0}}}\right)}

        \rho_{\mathrm{eq}} = \frac{\alpha_{\infty} \rho_{0}}{\phi} - \frac{i \sqrt{\eta} \sqrt{\Lambda^{2} \eta \phi^{2} + 4 i \alpha_{\infty}^{2} \omega q_0^{2} \rho_{0}}}{\Lambda \omega \phi q_0}

    The exact equations are taken from Section. 8.6 of Ref. [#]_.

    References
    ----------
    .. [#]  D. L. Johnson, J. Koplik, and R. Dashen, “Theory of dynamic
            permeability and tortuosity in fluids saturated porous media,”
            J. Fluid Mech., vol. 176, pp. 379-402, Mar. 1987,
            doi: 10.1017/S0022112087000727.
    .. [#]  D. Lafarge, P. Lemarinier, J. F. Allard, and V. Tarnow, “Dynamic
            compressibility of air in porous structures at audible frequencies,
            ”J. Acoust. Soc. Am., vol. 102, no. 4, pp. 1995-2006, Oct. 1997,
            doi: 10.1121/1.419690.
    .. [#]  Y. Champoux and J. Allard, “Dynamic tortuosity and bulk modulus in
            air-saturated porous media,” J. Appl. Phys., vol. 70, no. 4,
            pp. 1975-1979, Aug. 1991, doi: 10.1063/1.349482.
    .. [#]  J.-F. Allard and N. Atalla, Propagation of sound in porous media:
            modelling sound absorbing materials, 2nd ed.
            Hoboken, N.J: Wiley, 2009.

    """  # noqa: E501

    def __init__(
            self,
            frequencies: np.ndarray[float],
            porosity: Number | None = None,
            tortuosity: Number | None = None,
            viscous_permeability: Number | None = None,
            thermal_permeability: Number | None = None,
            viscous_characteristic_length: Number | None = None,
            thermal_characteristic_length: Number | None = None,
            flow_resistivity: Number | None = None,
            saturating_fluid: Fluid = Air(),
        ):
        JohnsonChampouxAllard.__init__(
            self=self,
            frequencies=frequencies,
            porosity=porosity,
            tortuosity=tortuosity,
            viscous_permeability=viscous_permeability,
            viscous_characteristic_length=viscous_characteristic_length,
            thermal_characteristic_length=thermal_characteristic_length,
            flow_resistivity=flow_resistivity,
            saturating_fluid=saturating_fluid,
        )
        self.thermal_permeability = thermal_permeability


    @SemiPhenomenologicalBase.thermal_permeability.setter
    def thermal_permeability(self, value: Number):
        """Set the thermal permeability of the equivalent fluid."""
        if value < 0:
            raise ValueError(
                "Thermal permeability must be greater than 0.",
            )
        self._thermal_permeability = value


class Horoshenkov(SemiPhenomenologicalBase):
    r"""Initialize the Horoshenkov et al. equivalent fluid model.

    Implemented as defined in [#]_ as a modification of the
    Johnson-Champoux-Allard-Lafarge model [#]_.

    Parameters
    ----------
    frequencies : np.ndarray[float]
        The frequencies at which the model is evaluated.
    porosity : Number | None
        The porosity :math:`\phi`.
    median_pore_size : Number | None
        The median pore size :math:`\bar{s}`.
    std_pore_size : Number | None
        The standard deviation of the pore size distribution
        :math:`\sigma_{s}`.

    References
    ----------
    .. [#] K. V. Horoshenkov, A. Hurrell, and J.-P. Groby, “A three-parameter
           analytical model for the acoustical properties of porous media,”
           J. Acoust. Soc. Am., vol. 145, no. 4, pp. 2512-2517, Apr. 2019,
           doi: 10.1121/1.5098778.
    .. [#]  J.-F. Allard and N. Atalla, Propagation of sound in porous media:
            modelling sound absorbing materials, 2nd ed.
            Hoboken, N.J: Wiley, 2009.

    """

    def __init__(
            self,
            frequencies: np.ndarray[float],
            porosity: Number | None = None,
            median_pore_size: Number | None = None,
            std_pore_size: Number | None = None,
            saturating_fluid: Fluid | None = None,
        ):
        """Initialize the Horoshenkov equivalent fluid model.
        """
        super().__init__(frequencies, porosity, saturating_fluid)
        self.median_pore_size = median_pore_size
        self.std_pore_size = std_pore_size

    @property
    def median_pore_size(self):
        r"""The median pore size :math:`\bar{s}`."""
        return self._median_pore_size

    @median_pore_size.setter
    def median_pore_size(self, value: Number):
        """Set the median pore size of the equivalent fluid."""
        if value < 0:
            raise ValueError(
                "Median pore size must be greater than 0.",
            )
        self._median_pore_size = value

    @property
    def std_pore_size(self):
        r"""The standard deviation of the pore size distribution
        :math:`\sigma_{s}`.
        """
        return self._std_deviation_pore_size

    @std_pore_size.setter
    def std_pore_size(self, value: Number):
        """Set the standard deviation of the pore size distribution."""
        if value < 0:
            raise ValueError(
                "Standard deviation of the pore size distribution must be "
                "greater than 0.",
            )
        self._std_deviation_pore_size = value


    @property
    def tortuosity(self):
        r"""The tortuosity :math:`\alpha_\infty = \exp(4[\sigma_s \log(2)]^2)`.
        """
        return np.exp(4*(self.std_pore_size*np.log(2))**2)

    @property
    def viscous_permeability(self):
        r"""The viscous permeability :math:`q_0 =
        \frac{\phi \bar{s}^2}{8 \alpha_\infty} \exp(-6[\sigma_s\log(2)]^2)`.
        """
        return (
            self.porosity*self.median_pore_size**2 /
            (8*self.tortuosity) *
            np.exp(-6*(self.std_pore_size*np.log(2))**2))

    @property
    def viscous_characteristic_length(self):
        r"""The viscous characteristic length :math:`\Lambda =
        \bar{s}\exp(-\frac{5}{2}[\sigma_s\log(2)]^2)`.
        """
        return (
            self.median_pore_size *
            np.exp(-2.5*(self.std_pore_size*np.log(2))**2)
        )

    @property
    def thermal_characteristic_length(self):
        r"""The thermal characteristic length :math:`\Lambda^\prime =
        \bar{s}\exp(\frac{3}{2}[\sigma_s\log(2)]^2)`.
        """
        return (
            self.median_pore_size *
            np.exp(1.5*(self.std_pore_size*np.log(2))**2)
        )

    @property
    def thermal_permeability(self):
        r"""The thermal permeability :math:`q^\prime =
        \frac{\phi (\Lambda^\prime)^2}{8}\exp(6[\sigma_s\log(2)]^2)`.
        """
        return (
            self.porosity*self.median_pore_size**2 /
            (8*self.tortuosity) *
            np.exp(6*(self.std_pore_size*np.log(2))**2))


class MicroPerforatedPanel(SemiPhenomenologicalBase):
    r"""Micro-perforated panel equivalent fluid model.

    The model represents a micro-perforated panel of thickness :math:`h` with
    cylindrical holes of radius :math:`r` as an equivalent fluid.
    The model adapts the JCA model as described in [#]_.

    The original JCA model parameters are modified as follows:

    - The tortuosity is calculated as
      :math:`\alpha_\infty = 1 + 2 \cdot 0.82 \cdot \frac{r}{h}`,
    - The viscous permeability is calculated as
        :math:`q_0 = \frac{\phi r^2}{8}`,
    - The viscous and thermal characteristic lengths are calculated as
        :math:`\Lambda = r` and :math:`\Lambda^\prime = r`.

    References
    ----------
    .. [#] N. Atalla and F. Sgard, “Modeling of perforated plates and screens
           using rigid frame porous models,” J. Sound Vib., vol. 303, no.
           1-2, pp. 195-208, Jun. 2007, doi: 10.1016/j.jsv.2007.01.012.

    """

    def __init__(
            self,
            frequencies: np.ndarray[float],
            porosity: Number | None = None,
            perforation_radius: Number | None = None,
            panel_thickness: Number | None = None,
            saturating_fluid: Fluid = Air(),
        ):
        """Initialize the micro-perforated panel equivalent fluid model.
        """
        super().__init__(frequencies, porosity, saturating_fluid)
        self.perforation_radius = perforation_radius
        self.panel_thickness = panel_thickness

    @property
    def perforation_radius(self):
        r"""The perforation radius :math:`r`."""
        return self._perforation_radius

    @perforation_radius.setter
    def perforation_radius(self, value: Number):
        """Set the perforation radius of the equivalent fluid."""
        if value < 0:
            raise ValueError(
                "Perforation radius must be greater than 0.",
            )
        self._perforation_radius = value

    @property
    def panel_thickness(self):
        r"""The panel thickness :math:`t`."""
        return self._panel_thickness

    @panel_thickness.setter
    def panel_thickness(self, value: Number):
        """Set the panel thickness of the equivalent fluid."""
        if value < 0:
            raise ValueError(
                "Panel thickness must be greater than 0.",
            )
        self._panel_thickness = value

    @property
    def tortuosity(self):
        r"""
        The tortuosity
        :math:`\alpha_\infty = 1 + 2 \cdot 0.82 \cdot \frac{r}{h}`.
        """
        r = self.perforation_radius
        h = self.panel_thickness
        return 1 + 2 * 0.82 * r / h

    @property
    def viscous_permeability(self):
        r"""The viscous permeability :math:`q_0 = \frac{\phi r^2}{8}`."""
        return self.porosity * self.perforation_radius**2 / 8

    @property
    def flow_resistivity(self):
        r"""The flow resistivity :math:`\sigma = \frac{\eta}{q_0}`."""
        return self.saturating_fluid.viscosity / self.viscous_permeability

    @property
    def thermal_permeability(self):
        r"""The thermal permeability :math:`q^\prime` of the equivalent fluid.
        Approximated as :math:`\frac{\phi (\Lambda^\prime)^2}{8}`.
        """
        return self.porosity*self.thermal_characteristic_length**2 / 8

    @property
    def viscous_characteristic_length(self):
        r"""The viscous characteristic length :math:`\Lambda = r`."""
        return self.perforation_radius

    @property
    def thermal_characteristic_length(self):
        r"""The thermal characteristic length :math:`\Lambda^\prime = r`."""
        return self.perforation_radius


class EquivalentFluidData(EquivalentFluidBase):
    """Base class for equivalent fluid data.

    """

    def __init__(
            self,
            bulk_modulus: pf.FrequencyData,
            density: pf.FrequencyData,
            saturating_fluid: Fluid | None = None,
        ):
        """Initialize the equivalent fluid data.
        """
        if not np.array_equal(
            bulk_modulus.frequencies, density.frequencies):
            raise ValueError(
                "Bulk modulus and density must share the same frequencies.")

        frequencies = bulk_modulus.frequencies
        super().__init__(
            frequencies,
            porosity=1,
            saturating_fluid=saturating_fluid)
        self._bulk_modulus = bulk_modulus
        self._density = density

    @classmethod
    def from_impedance_wavenumber(
            cls,
            Z_c: pf.FrequencyData,
            k: pf.FrequencyData,
            saturating_fluid: Fluid | None = None,
        ):
        """Create object from the characteristic impedance and wavenumber.

        Parameters
        ----------
        Z_c : FrequencyData
            The characteristic impedance of the equivalent fluid.
        k : FrequencyData
            The wavenumber of the equivalent fluid.
        saturating_fluid : Fluid, optional
            The fluid saturating the pores.

        Returns
        -------
        EquivalentFluidData
            The equivalent fluid data.

        """
        if not np.array_equal(Z_c.frequencies, k.frequencies):
            raise ValueError(
                "Characteristic impedance and wavenumber must " \
                "share the same frequencies.",
            )

        omega = 2*np.pi*Z_c.frequencies
        rho_eq = pf.FrequencyData((Z_c.freq*k.freq)/omega, Z_c.frequencies)
        K_eq = pf.FrequencyData((Z_c.freq*omega)/k.freq, Z_c.frequencies)

        return cls(K_eq, rho_eq, saturating_fluid)

    @classmethod
    def from_bulk_modulus_density(
            cls,
            bulk_modulus: pf.FrequencyData,
            density: pf.FrequencyData,
            saturating_fluid: Fluid | None = None,
        ):
        """Create object from the bulk modulus and density.

        Parameters
        ----------
        bulk_modulus : FrequencyData
            The bulk modulus of the equivalent fluid.
        density : FrequencyData
            The density of the equivalent fluid.
        saturating_fluid : Fluid, optional
            The fluid saturating the pores.

        Returns
        -------
        EquivalentFluidData
            The equivalent fluid data.

        """
        return cls(bulk_modulus, density, saturating_fluid)

    @property
    def bulk_modulus(self):
        r"""The bulk modulus of the equivalent fluid :math:`K_\mathrm{eq}`."""
        return self._bulk_modulus

    @property
    def density(self):
        r"""The dynamic density :math:`\rho_\mathrm{eq}`."""
        return self._density
