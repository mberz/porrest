"""This module implements stochastic inference methods for equivalent fluid
models using pymc.

"""
from typing import Literal
from porrest.constants import Fluid
import numpy as np
import pymc as pm
from porrest.eqfluid.implementations import (
    johnson_champoux_allard_real_imag,
    johnson_champoux_allard_lafarge_real_imag,
    mikis_model_real_imag,
    horoshenkov_jcal_parameters,
)
from porrest.tmm.implementations import (
    bulk_modulus_density_to_impedance_wave_number_real_imag,
    abs_coefficient_eq_fluid_rigid_termination_real_imag,
    reflection_factor_eq_fluid_rigid_termination_real_imag,
    impedance_eq_fluid_rigid_termination_real_imag,
)


def initialize_prior_variables(
        eq_fluid_model: str,
        context: pm.Model | None = None,
    ):
    """Setup the model variables for the inference.

    Parameters
    ----------
    eq_fluid_model : str
        The equivalent fluid model to use for the inference. Can be one of
        either 'JCA' or 'JCAL'.
    context : pm.Model
        The pymc model context. If None, a new model context will be created.
        Defaults to None.
    """

    if context is None:
        context = pm.Model()

    with context:
        porosity = pm.TruncatedNormal(
            'phi', mu=0.9, sigma=1, lower=0.01, upper=1)
        if eq_fluid_model in ['JCA', 'JCAL', 'Miki']:
            viscous_permeability = pm.Uniform(
                'q_s', lower=0.1e-10, upper=1e-6)
            tortuosity = pm.TruncatedNormal(
                'alpha_infty', mu=1.1, sigma=0.5, lower=1, upper=10)
            priors = {
                'viscous_permeability': viscous_permeability,
                'porosity': porosity,
                'tortuosity': tortuosity,
            }
        if eq_fluid_model in ['JCA', 'JCAL']:
            thermal_length = pm.Uniform(
                'Lambda_t', lower=10e-6, upper=1000e-6)
            viscous_length = pm.Uniform(
                'Lambda_s', lower=10e-6, upper=thermal_length)

            priors['thermal_length'] = thermal_length
            priors['viscous_length'] = viscous_length

        if eq_fluid_model == 'JCAL':
            thermal_permeability = pm.Uniform(
                'q_t', lower=viscous_permeability, upper=1e-6)
            priors['thermal_permeability'] = thermal_permeability


        priors = {
            'porosity': porosity,
        }

        if eq_fluid_model == 'Horoshenkov':
            median_pore_size = pm.Uniform(
                's', lower=1e-6, upper=10000e-6)
            std_pore_size = pm.Uniform(
                'sigma_s', lower=0., upper=1)

            alpha_inf, q_s, q_t, lambda_s, lambda_t = horoshenkov_jcal_parameters(
                    porosity,
                    median_pore_size,
                    std_pore_size)

            tortuosity = pm.Deterministic('alpha_infty', alpha_inf)
            viscous_permeability = pm.Deterministic('q_s', q_s)
            thermal_length = pm.Deterministic('Lambda_t', lambda_t)
            viscous_length = pm.Deterministic('Lambda_s', lambda_s)
            thermal_permeability = pm.Deterministic('q_t', q_t)

            priors['median_pore_size'] = median_pore_size
            priors['std_pore_size'] = std_pore_size

        if eq_fluid_model != 'Miki':
            priors['thermal_length'] = thermal_length
            priors['viscous_length'] = viscous_length
        if eq_fluid_model in ['JCAL', 'Horoshenkov']:
            priors['thermal_permeability'] = thermal_permeability
        priors['viscous_permeability'] = viscous_permeability
        priors['tortuosity'] = tortuosity

    return priors, context


def likelihood_function_fluid_parameters(
        eq_fluid_model: str,
        priors: dict,
        frequencies: np.ndarray[float],
        saturating_fluid: Fluid,
        target: Literal[
                'bulk_modulus_density',
                'impedance_wave_number',
            ] = 'bulk_modulus_density',
        context: pm.Model | None = None,
    ):
    """Setup the likelihood function for the inference.
    """
    if context is None:
        context = pm.modelcontext()

    rho_0 = float(saturating_fluid.density)
    Pr = float(saturating_fluid.prandtl_number)
    eta = float(saturating_fluid.viscosity)
    gamma = float(saturating_fluid.heat_capacity)
    P_stat = float(saturating_fluid.static_pressure)
    c_0 = float(saturating_fluid.speed_of_sound)

    Z_0 = c_0 * rho_0

    with context:
        if eq_fluid_model == 'JCA':
            K_r, K_i, rho_r, rho_i = johnson_champoux_allard_real_imag(
                2*np.pi*frequencies,
                priors['porosity'],
                priors['tortuosity'],
                priors['viscous_permeability'],
                priors['thermal_length'],
                priors['viscous_length'],
                rho_0, Pr, eta, gamma, P_stat,
            )
        elif eq_fluid_model in ['JCAL', 'Horoshenkov']:
            K_r, K_i, rho_r, rho_i = johnson_champoux_allard_lafarge_real_imag(
                2*np.pi*frequencies,
                priors['porosity'],
                priors['tortuosity'],
                priors['viscous_permeability'],
                priors['thermal_permeability'],
                priors['thermal_length'],
                priors['viscous_length'],
                rho_0, Pr, eta, gamma, P_stat,
            )
        elif eq_fluid_model == 'Miki':
            K_r, K_i, rho_r, rho_i = mikis_model_real_imag(
                frequencies,
                priors['porosity'],
                priors['tortuosity'],
                eta/priors['viscous_permeability'],
                c_0, rho_0,
            )

        if target == 'impedance_wave_number':
            Z_r, Z_i, k_r, k_i = bulk_modulus_density_to_impedance_wave_number_real_imag(  # noqa: E501
                frequencies, K_r, K_i, rho_r, rho_i)
            return [Z_r, Z_i, k_r, k_i]

        elif target == 'bulk_modulus_density':
            return [K_r, K_i, rho_r, rho_i]
        else:
            raise ValueError(
                f"Unknown target '{target}'. "
                "Must be one of 'bulk_modulus_density' or "
                "'impedance_wave_number'.")


def likelihood_function_rigid_termination(
        eq_fluid_model: str,
        priors: dict,
        frequencies: np.ndarray[float],
        layer_height: float,
        saturating_fluid: Fluid,
        target: Literal[
            'absorption', 'reflection', 'impedance'] = 'absorption',
        context: pm.Model | None = None,
    ):
    """Setup the likelihood function for the inference.

    Parameters
    ----------
    eq_fluid_model : str
        The equivalent fluid model to use for the inference. Can be one of
        either 'JCA' or 'JCAL'.
    priors : dict
        The model priors.
    frequencies : np.ndarray[float]
        The frequencies at which the model is evaluated.
    layer_height : float
        The height of the equivalent fluid layer.
    saturating_fluid : Fluid
        The saturating fluid.
    target : str
        The target variable to compute. Can be one of either 'absorption',
        'reflection', or 'impedance'.
    context : pm.Model
        The pymc model context.
    """

    if context is None:
        context = pm.modelcontext()

    rho_0 = float(saturating_fluid.density)
    c_0 = float(saturating_fluid.speed_of_sound)

    Z_0 = c_0 * rho_0

    Z_r, Z_i, k_r, k_i = likelihood_function_fluid_parameters(
        eq_fluid_model,
        priors,
        frequencies,
        saturating_fluid,
        target='impedance_wave_number',
        context=context,
    )

    with context:

        if target == 'absorption':
            alpha = abs_coefficient_eq_fluid_rigid_termination_real_imag(
                Z_r, Z_i, k_r, k_i, layer_height, Z_0)
            return alpha

        elif target == 'reflection':
            R_r, R_i =  reflection_factor_eq_fluid_rigid_termination_real_imag(
                Z_r, Z_i, k_r, k_i, layer_height, Z_0)
            return [R_r, R_i]

        elif target == 'impedance':
            Z_s_r, Z_s_i =  impedance_eq_fluid_rigid_termination_real_imag(
                Z_r, Z_i, k_r, k_i, layer_height)
            return [Z_s_r, Z_s_i]



