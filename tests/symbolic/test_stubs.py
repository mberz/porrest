import numpy as np

def test_jca(
        symbolic_jca_bulk_modulus_density,
        jca_test_dict,
        air_params_dict,
    ):
    jca_params = jca_test_dict
    air_params = air_params_dict
    freqs = np.linspace(100, 5e3, 10)
    K_eq, rho_eq = symbolic_jca_bulk_modulus_density(
        2*np.pi*freqs, **jca_params, **air_params,
    )




