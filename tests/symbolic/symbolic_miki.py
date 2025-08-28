# %%
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
    # ratio_freq_sigma = a

    z = sym.sqrt(alpha_inf)/phi * (
        1+0.07*(ratio_freq_sigma)**(-0.632) - sym.I*0.107*(ratio_freq_sigma)**(-0.632))

    k = 2*sym.pi*frequency*sym.sqrt(alpha_inf)/c_0 * (
        1 + 0.109*(ratio_freq_sigma)**(-0.618) - sym.I*0.16*(ratio_freq_sigma)**(-0.618))

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
    [K, rho],
    modules='numpy',
    cse=True,
)
import inspect
print(inspect.getsource(func_miki_z_k))
print(inspect.getsource(func_miki_K_rho))

# %%
sym.print_latex((rho_0*c_0*z).simplify())
sym.print_latex(k.simplify())
# %%

func_miki_K_rho_re_imag = sym.lambdify(
    [f, phi, alpha_inf, sigma, c_0, rho_0],
    [
        sym.re(K),
        sym.im(K),
        sym.re(rho),
        sym.im(rho),
    ],
    modules='numpy',
    cse=True,
)

print(inspect.getsource(func_miki_K_rho_re_imag))
# %%
import matplotlib.pyplot as plt
import numpy as np
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
