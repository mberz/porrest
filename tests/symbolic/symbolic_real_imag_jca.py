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
    'f, phi, alpha_inf, k_s,  k_t, lambda_s, lambda_t',
    real=True, positive=True)

# %%
# constants
T_c, rH, P_s, c, rho_0, Pr, eta, gamma, P_stat = sym.symbols(
    'T_c, rH, P_s, c, rho_0, Pr, eta, gamma, P_stat',
    real=True, positive=True)

K_r, K_i, rho_r, rho_i = sym.symbols(
    'K_r, K_i, rho_r, rho_i',
    real=True)

bulk_modulus = K_r + sym.I*K_i
dynamic_density = rho_r + sym.I*rho_i

zeta_m = sym.sqrt(dynamic_density*bulk_modulus)/rho_0/c
prop_const = sym.I*omega*sym.sqrt(dynamic_density/bulk_modulus)
k_m = omega*sym.sqrt(dynamic_density/bulk_modulus)

# S_m, S_p = sym.symbols('S_m, S_p', real=True)
# subs_Z_m = [
#     (-K_i*rho_i + K_r*rho_r, S_m),
#     (K_i*rho_r + K_r*rho_i, S_p)]

# %% [markdown]
# ## Characteristic impedance calculation

import sympy as sym
omega, K_r, K_i, rho_r, rho_i = sym.symbols(
    'omega, K_r, K_i, rho_r, rho_i', real=True)
Z_m = sym.sqrt((K_r+sym.I*K_i) * (rho_r+sym.I*rho_i))
k_m = omega*sym.sqrt((rho_r+sym.I*rho_i) / (K_r+sym.I*K_i))

func_K_rho_to_Z_k = lambdify(
    [omega, K_r, K_i, rho_r, rho_i],
    [sym.re(Z_m), sym.im(Z_m), sym.re(k_m), sym.im(k_m)],
    modules=["numpy"], cse=True,
)

print(inspect.getsource(func_K_rho_to_Z_k))

# %%
sym.print_latex(Z_m)
sym.print_latex(k_m)

# %% [markdown]
# ## Surface impedance calculation

import sympy as sym
omega = sym.symbols('omega', real=True)
Z_r, Z_i, k_r, k_i = sym.symbols('Z_r, Z_i, k_r, k_i', real=True)
Z_s = -sym.I*(Z_r + sym.I*Z_i)/sym.tan((k_r + sym.I*k_i)*h)

func_layer_rigid_term = lambdify(
    [Z_r, Z_i, k_r, k_i, h],
    [sym.re(Z_s), sym.im(Z_s)],
    modules=["numpy"], cse=True,
)

print(inspect.getsource(func_layer_rigid_term))

# %%
sym.print_latex(Z_s)


# %% [markdown]
# ## Reflection factor rigid termination

import sympy as sym
omega = sym.symbols('omega', real=True)
Z_r, Z_i, k_r, k_i, Z_0 = sym.symbols(
    'Z_r, Z_i, k_r, k_i, Z_0', real=True)

Z_rigid = -sym.I*(Z_r + sym.I*Z_i)/sym.tan((k_r + sym.I*k_i)*h)
R = (Z_rigid - Z_0)/(Z_rigid + Z_0)

func_R = sym.lambdify(
    [Z_r, Z_i, k_r, k_i, h, Z_0],
    [sym.re(R), sym.im(R)],
    modules=["numpy"], cse=True)

print(inspect.getsource(func_R))



# %% [markdown]
# ## Absorption coefficient rigid termination
import inspect

import sympy as sym
omega = sym.symbols('omega', real=True)
Z_r, Z_i, k_r, k_i, Z_0, h = sym.symbols('Z_r, Z_i, k_r, k_i, Z_0, h', real=True)

Z_rigid = -sym.I*(Z_r + sym.I*Z_i)/sym.tan((k_r + sym.I*k_i)*h)
R = (Z_rigid - Z_0)/(Z_rigid + Z_0)
R_r = sym.re(R)
R_i = sym.im(R)
alpha = 1 - sym.Abs(R_r + sym.I*R_i)**2

func_alpha = sym.lambdify(
    [Z_r, Z_i, k_r, k_i, h, Z_0],
    # sym.re(alpha),
    alpha,
    modules=["numpy"], cse=True)

print(inspect.getsource(func_alpha))


# %%
sym.print_latex(alpha)

# %%








# ----------------------

# %%

R_r, R_i, Z_r, Z_i, k_r, k_i = sym.symbols(
    'R_r, R_i, Z_r, Z_i, k_r, k_i', real=True)


# %%

rho_m = sym.symbols('rho_m', real=True)
rho_t = rho_m + phi*rho_0
rho_limp = (rho_t * (rho_r+sym.I*rho_i) - rho_0**2) / (rho_t + (rho_r+sym.I*rho_i) - 2*rho_0)

func_rho_limp = lambdify(
    [rho_r, rho_i, rho_0, rho_m, phi],
    [sym.re(rho_limp), sym.im(rho_limp)],
    modules=["numpy"], cse=False)

print(inspect.getsource(func_rho_limp))

# %%
sym.limit(rho_eq, omega, sym.oo)*phi

sym.limit(K_eq, omega, sym.oo)*phi
sym.limit(K_eq, omega, 0)*phi
# %%
(sym.limit(sym.re(rho_eq), omega, 0)*phi).simplify()
# %%
sym.re(rho_limp).simplify().subs(phi, 1).simplify().doit().subs(rho_i, 0).subs(rho_r, rho_0).simplify()


# %%
R_r, R_i, Z_r, Z_i, k_r, k_i, Z_0 = sym.symbols(
    'R_r, R_i, Z_r, Z_i, k_r, k_i, Z_0', real=True)

Z_rigid = -sym.I*(Z_r + sym.I*Z_i)/sym.tan((k_r + sym.I*k_i)*h)

R = (Z_rigid - Z_0)/(Z_rigid + Z_0)

R = (Z_rigid - Z_0)/(Z_rigid + Z_0)
alpha = 1 - sym.Abs(R)**2

func_alpha = lambdify(
    [Z_r, Z_i, k_r, k_i, h, Z_0],
    alpha,
    modules=["numpy"], cse=True)
print(inspect.getsource(func_alpha))

Z_s_re, Z_s_im = sym.symbols('Z_s_re, Z_s_im', real=True)

Z_s = sym.symbols('Z_s', real=False)

R = (Z_s - Z_0)/(Z_s + Z_0)
alpha = 1 - sym.Abs(R)**2
sym.print_latex(alpha)
# %%

func_Z_to_alpha = lambdify(
    [Z_rigid, ],
    alpha,
    modules=["numpy"], cse=True)
print(inspect.getsource(func_R))
# %%
re_k_m = sym.re(k_m)
im_k_m = sym.im(k_m)

re_zeta_m = sym.re(zeta_m)
im_zeta_m = sym.im(zeta_m)

# %%
x = [
    phi,
    alpha_inf,
    sigma,
    lambda_s,
    lambda_t,
    omega,
    rho_0,
    Pr,
    eta,
    gamma,
    P_stat]
func = lambdify(x, [re_zeta_m, im_zeta_m, re_k_m, im_k_m], modules=["numpy"])

# inspect.getmodule(func)
inspect.getsource(func)

# %% [Johnson Champoux Allard Lafarge]
# omega = 2*sym.pi*f
nu = eta/rho_0
nu_t = nu/Pr

# %%
sqrt_term = sym.sqrt(1+sym.I*omega/nu*(2*alpha_inf*k_s/phi/lambda_s)**2)
dynamic_tortuosity = alpha_inf + 1/sym.I*nu/omega*phi/k_s*sqrt_term

sqrt_term = sym.sqrt(1+sym.I*omega/nu_t*(2*k_t/phi/lambda_t)**2)
thermal_tortuosity = 1 + 1/sym.I*nu_t*phi/omega/k_t*sqrt_term

rho_eq = rho_0/phi*dynamic_tortuosity
K_eq = gamma*P_stat/phi / (gamma - (gamma-1)/thermal_tortuosity)


G_jcal = sym.sqrt(1 + (2*alpha_inf*k_s/phi/lambda_s)**2 * sym.I*omega/nu)
G_dash_jcal = sym.sqrt(1 + (2*k_t/phi/lambda_t)**2 * sym.I*omega/nu_t)

K_eq_jcal = (gamma*P_stat/phi/(gamma-(gamma-1)/(1 + nu_t*phi/sym.I/omega/k_t*G_dash_jcal)))
K_eq_jcal

rho_eq_jcal = rho_0/phi*(alpha_inf + nu*phi/sym.I/omega/k_s*G_jca)
rho_eq_jcal.simplify()
sym.print_latex(rho_eq_jcal.simplify().subs(k_s, eta/sigma))

K_eq - K_eq_jcal
rho_eq - rho_eq_jcal


bulk_modulus = K_eq
dynamic_density = rho_eq
zeta_m = sym.sqrt(dynamic_density*bulk_modulus)/rho_0/c
prop_const = sym.I*omega*sym.sqrt(dynamic_density/bulk_modulus)
k_m = omega*sym.sqrt(dynamic_density/bulk_modulus)
# %%
k_m.simplify()
# %%
# sym.re(prop_const).simplify()
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
    phi,
    alpha_inf,
    k_s,
    k_t,
    lambda_s,
    lambda_t,
    omega,
    rho_0,
    Pr,
    eta,
    gamma,
    P_stat]
# func = lambdify(x, [re_zeta_m, im_zeta_m, re_k_m, im_k_m], modules=["numpy"])

func_jcal = lambdify(
    x,
    [sym.re(K_eq), sym.im(K_eq), sym.re(rho_eq), sym.im(rho_eq)],
    modules=["numpy"], cse=True)
print(inspect.getsource(func))
# %%
func_jcal = lambdify(
    x,
    [K_eq, rho_eq],
    modules=["numpy"], cse=True)
print(inspect.getsource(func_jcal))

# %%
func_jcal
# %%

func = lambdify(
    x,
    [sym.re(dynamic_tortuosity).simplify(), sym.im(dynamic_tortuosity).simplify()],
    modules=["numpy"], cse=True)
print(inspect.getsource(func))


func = lambdify(
    x,
    [sym.re(K_eq*phi), sym.im(K_eq*phi)],
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
    K_eq,
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

sym.re(rho_eq).simplify()
sym.limit(sym.re(rho_eq).simplify(), omega, 0)


sym.limit(sym.im(rho_eq).simplify(), omega, sym.oo)

sym.im(rho_eq).simplify()

sym.re(K_eq).simplify()

sym.im(K_eq).simplify()

print(sym.limit(sym.re(K_eq), omega, 0))
print(sym.limit(sym.im(K_eq), omega, 0))

sym.limit(sym.re(K_eq), omega, sym.oo)
sym.limit(sym.im(K_eq), omega, sym.oo)

# %%
rho_t = rho_m + phi*rho_0
rho_limp = (rho_t * rho_eq - rho_0**2) / (rho_t + rho_eq - 2*rho_0)

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
    [sym.re(K_eq), sym.im(K_eq), sym.re(rho_limp), sym.im(rho_limp)],
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

rho_eq = rho_0/phi*alpha_s
K_eq = P_stat*gamma/phi / (gamma - (gamma-1)/alpha_t)
K_eq_jca - K_eq.subs(k_t, phi*lambda_t**2/8)
(rho_eq_jca - rho_eq).simplify()

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
    [sym.re(K_eq), sym.im(K_eq), sym.re(rho_eq), sym.im(rho_eq)],
    modules=["numpy"], cse=True)
print(inspect.getsource(func))

# %%
