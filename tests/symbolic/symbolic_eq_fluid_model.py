# %%
import inspect
from sympy import lambdify

import sympy as sym
from porrest.constants import Air

# %%
Z_r, Z_i, gamma_r, gamma_i, omega, c, f, sigma, Z_w, h = sym.symbols(
    'Z_r, Z_i, gamma_r, gamma_i, omega, c, f, sigma, Z_w, h',
    real=True, positive=True)
# %%
f, phi, alpha_inf, k_s,  k_t, lambda_s, lambda_t = sym.symbols(
    r'f, phi, alpha_inf, q,  q_T, Lambda, Lambda_T',
    real=True, positive=True)

# %%
# constants
T_c, rH, P_s, c, rho_0, Pr, eta, gamma, P_stat = sym.symbols(
    'T_c, rH, P_s, c, rho_0, Pr, eta, gamma, P_stat',
    real=True, positive=True)

K_r, K_i, rho_r, rho_i = sym.symbols(
    'K_r, K_i, rho_r, rho_i',
    real=True)

K_r, K_i, rho_r = sym.symbols(
    'K_r, K_i, rho_r',
    real=True, positive=True)

rho_i = sym.symbols('rho_i', real=True)

# %% [Johnson Champoux Allard]
#
sqrt_term = sym.sqrt(
    1 + sym.I*4*alpha_inf**2*eta*rho_0*omega/(sigma*lambda_s*phi)**2)
rho_eq_symbol = alpha_inf*rho_0/phi + sigma/sym.I/omega*sqrt_term
sqrt_term = sym.sqrt(1 + sym.I*rho_0*omega*Pr*lambda_t**2/(16*eta))
factor = 8*eta/(sym.I*rho_0*omega*Pr*lambda_t**2)

nu = eta/rho_0
nu_t = nu/Pr
G_jca = sym.sqrt(1 + (2*alpha_inf*k_s/phi/lambda_s)**2 * sym.I*omega/nu)
G_dash_jca = sym.sqrt(1 + (lambda_t/4)**2 * sym.I*omega/nu_t)

K_eq_jca = (gamma*P_stat/phi/(gamma-(gamma-1)/(1 + nu_t*phi/sym.I/omega/k_t*G_dash_jca))).subs(k_t, phi*lambda_t**2/8)
# K_eq_jca
phi*lambda_t**2/8

rho_eq_jca = rho_0/phi*(alpha_inf + nu*phi/sym.I/omega/k_s*G_jca)

sym.print_latex(rho_eq_jca.simplify())
sym.print_latex(K_eq_jca.simplify())

# K_eq = gamma*P_stat/phi/(gamma-(gamma-1)/(1+factor*sqrt_term))

# (rho_eq_jca.subs(k_s, eta/sigma) - rho_eq).simplify()
# K_eq - K_eq_jca
# %%
x = [
    omega,
    phi,
    alpha_inf,
    k_s,
    lambda_s,
    lambda_t,
    rho_0,
    Pr,
    eta,
    gamma,
    P_stat]

func = lambdify(
    x,
    [sym.re(K_eq_jca), sym.im(K_eq_jca), sym.re(rho_eq_jca), sym.im(rho_eq_jca)],
    modules=['numpy'], cse=True)
print(inspect.getsource(func))
# %%

func_jca = lambdify(
    x,
    [K_eq_jca, rho_eq_jca],
    modules=['numpy'], cse=True)
print(inspect.getsource(func_jca))
# %% [markdown]
# ## Analytic inversion
#

# K_eq_symbol = K_r + sym.I*K_i
# rho_eq_symbol = rho_r + sym.I*rho_i

K_eq_symbol = sym.symbols('K_eq', complex=True, positive=True)
K_eq_symbol = sym.symbols('K_eq', complex=True)
rho_eq_symbol = sym.symbols('rho_eq', complex=True)

lambda_t_expressions = sym.solve(K_eq_symbol - K_eq_jca, lambda_t)
alpha_inf_expressions = sym.solve(rho_eq_symbol - rho_eq_jca, alpha_inf)
lambda_s_expressions = sym.solve(rho_eq_symbol - rho_eq_jca, lambda_s)

# %%
# lambda_t_expressions[3].simplify()
# res_ = sym.solve(
#     [sym.re(K_eq_jca) - K_r, sym.im(K_eq_jca) - K_i],
#     [lambda_t, alpha_inf],
# )
# %%
# res_rho_jca = sym.solve(
#     [sym.re(rho_eq_jca) - rho_r, sym.im(rho_eq_jca) - rho_i],
#     [alpha_inf, lambda_s],
# )

# %%
alpha_inf_expressions[0].simplify()
lambda_t_expressions[0].simplify()
lambda_s_expressions[0].simplify().subs(rho_eq_symbol, rho_r + sym.I*rho_i).expand().simplify()

# %%
x_diff = [
    phi,
    alpha_inf,
    k_s,
    lambda_s,
    lambda_t,
]

param = x_diff[0]

jca_jacobian_K_eq = []
jca_jacobian_rho_eq = []
for param in x_diff:
    jca_jacobian_K_eq.append(sym.diff(K_eq_jca, param).simplify())
    jca_jacobian_rho_eq.append(sym.diff(rho_eq_jca, param).simplify())
# %%
jca_jacobian_K_eq[-1]

# %%
import pyfar as pf
import numpy as np
from matplotlib import pyplot as plt
freqs = pf.dsp.filter.fractional_octave_frequencies(
    12, frequency_range=[100, 3e3])[1]
air = Air(frequencies=freqs)

# pore_params = {
#     'phi': 0.37,
#     'alpha_inf': 1.37,
#     'sigma': air.viscosity/1.23e-10,
#     'lambda_s': 31e-6,
#     'lambda_t': 90e-6,
# }

pore_params_industry_modus = {
    'phi': 0.99,
    'alpha_inf': 1.01,
    'k_s': 14.3e-10,
    'lambda_s': 184e-6,
    'lambda_t': 82e-6,
}

K_eq_numeric_jca, rho_eq_numeric_jca = func_jca(
    2*np.pi*freqs,
    phi=pore_params_industry_modus['phi'],
    alpha_inf=pore_params_industry_modus['alpha_inf'],
    q=pore_params_industry_modus['k_s'],
    Lambda=pore_params_industry_modus['lambda_s'],
    Lambda_T=pore_params_industry_modus['lambda_t'],
    rho_0=air.density, Pr=air.prandtl_number,
    eta=air.viscosity, gamma=air.heat_capacity,
    P_stat=air.static_pressure)

# %%
_, axs = plt.subplots(2, 2)
axs[0, 0].plot(freqs, np.real(K_eq_numeric_jca/air.static_pressure))
axs[1, 0].plot(freqs, np.imag(K_eq_numeric_jca/air.static_pressure))

axs[0, 1].plot(freqs, np.real(rho_eq_numeric_jca/air.density))
axs[1, 1].plot(freqs, np.imag(rho_eq_numeric_jca/air.density))
for ax in axs.flatten():
    ax.set_xscale('log')
    ax.grid(True)
plt.tight_layout()
# %%

func = lambdify(
    x,
    [sym.re(K_eq_jca*phi), sym.im(K_eq_jca*phi)],
    modules=['numpy'], cse=True)
print(inspect.getsource(func))
# %%
from porrest.eqfluid.classes import JohnsonChampouxAllard, JohnsonChampouxAllardLafarge
eqfluid = JohnsonChampouxAllard(
    freqs,
    porosity=pore_params_industry_modus['phi'],
    tortuosity=pore_params_industry_modus['alpha_inf'],
    viscous_permeability=pore_params_industry_modus['k_s'],
    viscous_characteristic_length=pore_params_industry_modus['lambda_s'],
    thermal_characteristic_length=pore_params_industry_modus['lambda_t'],
    saturating_fluid=air)
# %%
pf.plot.freq_phase(eqfluid.bulk_modulus/eqfluid.saturating_fluid.static_pressure, dB=False)
pf.plot.freq_phase(eqfluid.density/eqfluid.saturating_fluid.density, dB=False)


# %%

bulk_modulus = K_eq_jca

dynamic_density = rho_eq_jca

zeta_m = sym.sqrt(dynamic_density*bulk_modulus)/rho_0/c
prop_const = sym.I*omega*sym.sqrt(dynamic_density/bulk_modulus)
k_m = omega*sym.sqrt(dynamic_density/bulk_modulus)

# %% [Johnson Champoux Allard Lafarge]
# omega = 2*sym.pi*f
nu = eta/rho_0
nu_t = nu/Pr

# %%
sqrt_term = sym.sqrt(1+sym.I*omega/nu*(2*alpha_inf*k_s/phi/lambda_s)**2)
dynamic_tortuosity = alpha_inf + 1/sym.I*nu/omega*phi/k_s*sqrt_term

sqrt_term = sym.sqrt(1+sym.I*omega/nu_t*(2*k_t/phi/lambda_t)**2)
thermal_tortuosity = 1 + 1/sym.I*nu_t*phi/omega/k_t*sqrt_term

rho_eq_symbol = rho_0/phi*dynamic_tortuosity
K_eq_symbol = gamma*P_stat/phi / (gamma - (gamma-1)/thermal_tortuosity)


G_jcal = sym.sqrt(1 + (2*alpha_inf*k_s/phi/lambda_s)**2 * sym.I*omega/nu)
G_dash_jcal = sym.sqrt(1 + (2*k_t/phi/lambda_t)**2 * sym.I*omega/nu_t)

K_eq_jcal = (gamma*P_stat/phi/(gamma-(gamma-1)/(1 + nu_t*phi/sym.I/omega/k_t*G_dash_jcal)))
K_eq_jcal

rho_eq_jcal = rho_0/phi*(alpha_inf + nu*phi/sym.I/omega/k_s*G_jca)
rho_eq_jcal.simplify()
sym.print_latex(rho_eq_jcal.simplify().subs(k_s, eta/sigma))

K_eq_symbol - K_eq_jcal
rho_eq_symbol - rho_eq_jcal


bulk_modulus = K_eq_symbol
dynamic_density = rho_eq_symbol
zeta_m = sym.sqrt(dynamic_density*bulk_modulus)/rho_0/c
prop_const = sym.I*omega*sym.sqrt(dynamic_density/bulk_modulus)
k_m = omega*sym.sqrt(dynamic_density/bulk_modulus)
# %%

sym.print_latex(rho_eq_jcal.simplify())
sym.print_latex(K_eq_jcal.simplify())

# %%
k_m.simplify()
# %%
re_k_m = sym.re(k_m)
im_k_m = sym.im(k_m)

re_zeta_m = sym.re(zeta_m)
im_zeta_m = sym.im(zeta_m)


Z_rigid = -sym.I*(zeta_m)/sym.tan((k_m)*h)

R = (Z_rigid - 1)/(Z_rigid + 1)

R_r = sym.re(R)
R_i = sym.im(R)

alpha = 1 - sym.Abs(R)**2

# %%
x = [
    omega,
    phi,
    alpha_inf,
    k_s,
    k_t,
    lambda_s,
    lambda_t,
    rho_0,
    Pr,
    eta,
    gamma,
    P_stat]

func_jcal = lambdify(
    x,
    [sym.re(K_eq_jcal), sym.im(K_eq_jcal), sym.re(rho_eq_jcal), sym.im(rho_eq_jcal)],
    modules=["numpy"], cse=True)
print(inspect.getsource(func_jcal))
# %%
func_jcal = lambdify(
    x,
    [K_eq_jcal, rho_eq_jcal],
    modules=["numpy"], cse=True)
print(inspect.getsource(func_jcal))

# %%
K_eq_jcal

rho_eq_jcal


# %%
pore_params_industry_modus = {
    'phi': 0.99,
    'alpha_inf': 1.01,
    'k_s': 14.3e-10,
    'k_t': 31e-10,
    'lambda_s': 184e-6,
    'lambda_t': 82e-6,
}

k_t_approx = pore_params_industry_modus['phi']*pore_params_industry_modus['lambda_t']**2/8

pore_params_industry_modus_approx = {
    'phi': 0.99,
    'alpha_inf': 1.01,
    'k_s': 14.3e-10,
    'k_t': k_t_approx,
    'lambda_s': 184e-6,
    'lambda_t': 82e-6,
}

K_eq_numeric_jcal, rho_eq_numeric_jcal = func_jcal(
    2*np.pi*freqs, **pore_params_industry_modus,
    rho_0=air.density, Pr=air.prandtl_number,
    eta=air.viscosity, gamma=air.heat_capacity,
    P_stat=air.static_pressure)

# %%
_, axs = plt.subplots(2, 2)
axs[0, 0].plot(freqs, np.real(K_eq_numeric_jca/air.static_pressure))
axs[1, 0].plot(freqs, np.imag(K_eq_numeric_jca/air.static_pressure))

axs[0, 1].plot(freqs, np.real(rho_eq_numeric_jca/air.density))
axs[1, 1].plot(freqs, np.imag(rho_eq_numeric_jca/air.density))

axs[0, 0].plot(freqs, np.real(K_eq_numeric_jcal/air.static_pressure))
axs[1, 0].plot(freqs, np.imag(K_eq_numeric_jcal/air.static_pressure))

axs[0, 1].plot(freqs, np.real(rho_eq_numeric_jcal/air.density))
axs[1, 1].plot(freqs, np.imag(rho_eq_numeric_jcal/air.density))
for ax in axs.flatten():
    ax.set_xscale('log')
    ax.grid(True)
plt.tight_layout()
# %%

func = lambdify(
    x,
    [sym.re(dynamic_tortuosity).simplify(), sym.im(dynamic_tortuosity).simplify()],
    modules=["numpy"], cse=True)
print(inspect.getsource(func))


func = lambdify(
    x,
    [sym.re(K_eq_symbol*phi), sym.im(K_eq_symbol*phi)],
    modules=["numpy"], cse=True)
print(inspect.getsource(func))

# %%

func = lambdify(
    x,
    # [K_eq, rho_eq],
    alpha,
    modules=["numpy"], cse=True)
print(inspect.getsource(func))
# %%

F = sym.Matrix([alpha])
jcal_jacobian_K_eq = F.jacobian([phi, alpha_inf, k_s, k_t, lambda_s, lambda_t])
# jcal_jacobian_K_eq[0].simplify()
# %%

func = lambdify(
    x,
    jcal_jacobian_K_eq,
    modules=["numpy"], cse=True)
print(inspect.getsource(func))
# %%

func = lambdify(
    x,
    # [K_eq, rho_eq],
    K_eq_symbol,
    modules=["numpy"], cse=True)
print(inspect.getsource(func))

F = sym.Matrix([zeta_m, k_m])
jcal_jacobian_K_eq = F.jacobian([phi, alpha_inf, k_s, k_t, lambda_s, lambda_t])


func = lambdify(
    x,
    jcal_jacobian_K_eq,
    modules=["numpy"], cse=True)
print(inspect.getsource(func))
# jcal_jacobian_K_eq[0].simplify()
# %%

sym.re(rho_eq_symbol).simplify()
sym.limit(sym.re(rho_eq_symbol).simplify(), omega, 0)


sym.limit(sym.im(rho_eq_symbol).simplify(), omega, sym.oo)

sym.im(rho_eq_symbol).simplify()

sym.re(K_eq_symbol).simplify()

sym.im(K_eq_symbol).simplify()

print(sym.limit(sym.re(K_eq_symbol), omega, 0))
print(sym.limit(sym.im(K_eq_symbol), omega, 0))

sym.limit(sym.re(K_eq_symbol), omega, sym.oo)
sym.limit(sym.im(K_eq_symbol), omega, sym.oo)

# %%
rho_t = rho_m + phi*rho_0
rho_limp = (rho_t * rho_eq_symbol - rho_0**2) / (rho_t + rho_eq_symbol - 2*rho_0)

x = [
    phi,
    alpha_inf,
    k_s,
    k_t,
    lambda_s,
    lambda_t,
    rho_m,
    omega,
    rho_0,
    Pr,
    eta,
    gamma,
    P_stat]
# func = lambdify(x, [re_zeta_m, im_zeta_m, re_k_m, im_k_m], modules=["numpy"])

func = lambdify(
    x,
    [sym.re(K_eq_symbol), sym.im(K_eq_symbol), sym.re(rho_limp), sym.im(rho_limp)],
    modules=["numpy"], cse=True)
print(inspect.getsource(func))

# %%


# %% [Johnson Champoux Allard Pride Lafarge]
# omega = 2*sym.pi*f

alpha_s_0, alpha_t_0 = sym.symbols('alpha_s_0, alpha_t_0', real=True, positive=True)

nu = eta/rho_0
nu_t = nu/Pr

# %%
sqrt_term = sym.sqrt(1+sym.I*omega/nu*(2*alpha_inf*k_s/phi/lambda_s)**2)
dynamic_tortuosity = alpha_inf + 1/sym.I*nu/omega*phi/k_s*sqrt_term

sqrt_term = sym.sqrt(1+sym.I*omega/nu_t*(2*k_t/phi/lambda_t)**2)
thermal_tortuosity = 1 + 1/sym.I*nu_t*phi/omega/k_t*sqrt_term

M_t = 8*k_t/phi/lambda_t**2
M_s = 8*k_s*alpha_inf/phi/lambda_s**2

S_T_t = lambda_t*sym.sqrt(sym.I*omega*Pr*rho_0/eta)
S_T_s = lambda_s*sym.sqrt(sym.I*omega*rho_0/eta)

q_s = 1/(alpha_s_0 - alpha_inf)*2*k_s*alpha_inf**2/phi/lambda_s**2
q_t = 1/(alpha_t_0 - 1)*2*k_t**2/phi/lambda_t**2

alpha_s = alpha_inf*(1+8/M_s/S_T_s**2*(1-q_s + q_s*sym.sqrt(1+M_s**2/16/q_s**2*S_T_s**2)))
alpha_t = 1 + 8/M_t/S_T_t**2*(1-q_t + q_t*sym.sqrt(1+M_t**2/16/q_t**2*S_T_t**2))

rho_eq_symbol = rho_0/phi*alpha_s
K_eq_symbol = P_stat*gamma/phi / (gamma - (gamma-1)/alpha_t)
K_eq_jca - K_eq_symbol.subs(k_t, phi*lambda_t**2/8)
(rho_eq_jca - rho_eq_symbol).simplify()

# %%
k_m.simplify()
# %%
# sym.re(prop_const).simplify()
re_k_m = sym.re(k_m)
im_k_m = sym.im(k_m)

re_zeta_m = sym.re(zeta_m)
im_zeta_m = sym.im(zeta_m)

# %%
x = [
    phi,
    alpha_inf,
    k_s,
    k_t,
    lambda_s,
    lambda_t,
    alpha_s_0,
    alpha_t_0,
    omega,
    rho_0,
    Pr,
    eta,
    gamma,
    P_stat]
# func = lambdify(x, [re_zeta_m, im_zeta_m, re_k_m, im_k_m], modules=["numpy"])

func = lambdify(
    x,
    [sym.re(K_eq_symbol), sym.im(K_eq_symbol), sym.re(rho_eq_symbol), sym.im(rho_eq_symbol)],
    modules=["numpy"], cse=True)
print(inspect.getsource(func))

# %%
