# %%
# %%
import matplotlib.pyplot as plt
import numpy as np
from porrest.eqfluid.classes import (
    JohnsonChampouxAllard, JohnsonChampouxAllardLafarge,
)
import sympy as sym
# %%
z_f, k_f, z_s, gamma_f = sym.symbols(
    'z_f, k_f, z_s, gamma_f', complex=True)

alpha_inf, phi, sigma, f, h = sym.symbols(
    r'alpha_infty, phi, sigma, f, h', real=True, positive=True)

c_0 = sym.symbols('c', real=True, positive=True)
# %%
def impedance_wavenumber_to_bulk_modulus_density(z, k, freqs):
    omega = 2*sym.pi*freqs
    K = z/k*omega
    rho = k/omega*z
    return K, rho


a = sym.symbols('a', real=True, positive=True)
c_0 = sym.symbols('c_0', real=True, positive=True)
rho_0 = sym.symbols('rho_0', real=True, positive=True)


def mikis_model(phi, alpha_inf, sigma, frequency, c_0=c_0):

    # z = sym.sqrt(alpha_inf)/phi * (
    #     1+0.07*(frequency/sigma)**-0.632 + sym.I*0.107*(frequency/sigma)**-0.632)

    # k = 2*sym.pi*frequency*sym.sqrt(alpha_inf)/c_0 * (
    #     1 + 0.109*(frequency/sigma)**-0.618 - sym.I*0.16*(frequency/sigma)**-0.618)
    ratio_freq_sigma = frequency/sigma
    ratio_freq_sigma = a

    z = sym.sqrt(alpha_inf)/phi * (
        1+0.07*(ratio_freq_sigma)**-0.632 - sym.I*0.107*(ratio_freq_sigma)**-0.632)

    k = 2*sym.pi*frequency*sym.sqrt(alpha_inf)/c_0 * (
        1 + 0.109*(ratio_freq_sigma)**-0.618 - sym.I*0.16*(ratio_freq_sigma)**-0.618)

    return z, k
# %%
z, k = mikis_model(phi, alpha_inf, sigma, f)
K, rho = impedance_wavenumber_to_bulk_modulus_density(z*c_0*rho_0, k, f)

# %%
func_miki_z_k = sym.lambdify(
    [f, phi, alpha_inf, sigma, c_0],
    [z.subs(a, sigma/f).doit(), k.subs(a, sigma/f).doit()],
    modules='numpy',
    cse=True,
)

func_miki_K_rho = sym.lambdify(
    [f, phi, alpha_inf, sigma, c_0, rho_0],
    [K.subs(a, sigma/f).doit(), rho.subs(a, sigma/f).doit()],
    modules='numpy',
    cse=True,
)
# %%
sym.limit(rho.subs(a, sigma/f), f, 0)
sym.limit(sym.re(rho.subs(a, sigma/f).simplify()), f, sym.oo)
import inspect
print(inspect.getsource(func_miki_z_k))
print(inspect.getsource(func_miki_K_rho))

# %%
f_numeric = np.linspace(100, 5e3, 100)
z_numeric, k_numeric = func_miki_z_k(f_numeric, 1, 1, 10e3, 343)
plt.plot(f_numeric, np.real(z_numeric), label='z_r')
plt.plot(f_numeric, np.imag(z_numeric), label='z_i')
plt.plot(f_numeric, np.real(k_numeric), label='z_r')
plt.plot(f_numeric, np.imag(k_numeric), label='z_i')

K_numeric, rho_numeric = func_miki_K_rho(
    f_numeric, 1, 1, 10e3, 343, 1.2)

plt.plot(f_numeric, np.real(K_numeric), label='z_r')
plt.plot(f_numeric, np.imag(K_numeric), label='z_i')
plt.plot(f_numeric, np.real(rho_numeric), label='z_r')
plt.plot(f_numeric, np.imag(rho_numeric), label='z_i')
# %%
sym.print_latex(z.subs(a, sigma/f))
z.subs(a, sigma/f)

sym.print_latex(k.subs(a, sigma/f))
# %%
sym.print_latex((rho.simplify()).subs(a, sigma/f))

sym.print_latex(K.simplify().subs(a, sigma/f))
# rho.subs(sigma/f, a)
# %%

K.simplify()
# %%
sym.simplify(sym.conjugate(gamma_f - sym.I * sym.conjugate(k_f))/sym.I)
# %%
omega = 2*sym.pi*f
ratio = f/sigma
ratio = sym.symbols('R', real=True, positive=True)
omega = sym.symbols('omega', real=True, positive=True)
alpha_m = omega/c_0*(.16*ratio**-0.618)
beta_m = omega/c_0*(1+.109*ratio**-0.618)
alpha_m, beta_m = sym.symbols('alpha_m beta_m', real=True)
gamma_f_miki = alpha_m + sym.I*beta_m

k_f_miki = sym.simplify(sym.expand(sym.I*sym.conjugate(gamma_f_miki)))
# %%
import porrest
freqs = np.linspace(100, 7e3, 1000)
saturating_fluid = porrest.constants.Air()
jca_test_dict = {
    'phi': 0.99,
    'alpha_inf': 1.01,
    'k_s': 14.3e-10,
    'lambda_s': 184e-6,
    'lambda_t': 82e-6,
}


# %%
jca_model = JohnsonChampouxAllard(
    freqs,
    porosity=jca_test_dict['phi'],
    tortuosity=jca_test_dict['alpha_inf'],
    viscous_characteristic_length=jca_test_dict['lambda_s'],
    thermal_characteristic_length=jca_test_dict['lambda_t'],
    viscous_permeability=jca_test_dict['k_s'],
    saturating_fluid=saturating_fluid,
)

# %%
def bulk_modulus_density_mikis_model(f, phi, alpha_infty, sigma, c_0, rho_0):
    x0 = f**0.618*sigma**(-0.618)
    x1 = 0.109*x0 - 0.16*1j*x0 + 1
    x2 = f**0.632*sigma**(-0.632)
    x3 = rho_0*(0.07*x2 - 0.107*1j*x2 + 1)/phi
    return [c_0**2*x3/x1, alpha_infty*x1*x3]

# %%
K_miki, rho_miki = bulk_modulus_density_mikis_model(
    freqs,
    jca_test_dict['phi'],
    jca_test_dict['alpha_inf'],
    saturating_fluid.viscosity/jca_test_dict['k_s'],
    saturating_fluid.speed_of_sound,
    saturating_fluid.density,
)

# %%
_, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[1].plot(freqs, np.real(jca_model.density.freq.T/saturating_fluid.density), color='C0')
axs[1].plot(freqs, np.imag(jca_model.density.freq.T/saturating_fluid.density), color='C0', linestyle='-')
axs[1].plot(freqs, np.real(rho_miki/saturating_fluid.density), linestyle='-.', color='grey')
axs[1].plot(freqs, np.imag(rho_miki/saturating_fluid.density), linestyle='-.', color='grey')

axs[1].set_xscale('linear')
axs[1].set_ylim(-20, 10)

axs[0].plot(freqs, np.real(jca_model.bulk_modulus.freq.T/saturating_fluid.static_pressure), color='C0')
axs[0].plot(freqs, np.imag(jca_model.bulk_modulus.freq.T/saturating_fluid.static_pressure), color='C0')
axs[0].plot(freqs, np.real(K_miki/saturating_fluid.static_pressure), linestyle='-.', color='grey')
axs[0].plot(freqs, np.imag(K_miki/saturating_fluid.static_pressure), linestyle='-.', color='grey')

for ax in axs:
    ax.axvline(jca_model.viscous_characteristic_frequency, color='k', linestyle=':')
    ax.axvline(jca_model.thermal_characteristic_frequency, color='k', linestyle=':')

# %%
