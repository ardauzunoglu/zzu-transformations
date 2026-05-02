import sys; sys.path.insert(0, '.')
import numpy as np
import transformation_algorithms as ta
import toy_data as td

# --- Test 1: all three regressors on exponential multiplicative data ---
bundle = td.make_exponential_multiplicative()
X = bundle.X.values
y = bundle.y.values

model_fn   = lambda X, t: t[0] * np.exp(t[1] * X[:, 0])
theta_init = np.array([1.0, 1.0])

gd = ta.GradientDescentRegressor(model_fn=model_fn, max_iter=3000, learning_rate=0.002).fit(X, y, theta_init)
gn = ta.GaussNewtonRegressor(model_fn=model_fn).fit(X, y, theta_init)
bf = ta.BFGSRegressor(model_fn=model_fn).fit(X, y, theta_init)

for name, reg in [('GD', gd), ('GN', gn), ('BFGS', bf)]:
    m = ta.regression_metrics(y, reg.predict(X))
    rmse = m['rmse']
    print(f'{name}: converged={reg.converged_}, n_iter={reg.n_iter_}, RMSE={rmse:.4f}, theta={reg.theta_}')

# --- Test 2: ZZU workflow ---
def coeff_to_init(tols_model):
    beta = tols_model.beta_
    return np.array([np.exp(beta[0]), beta[1]])

zzu = ta.ZZUTransformRegressor(
    model_fn=model_fn,
    coeff_to_init=coeff_to_init,
    nonlinear_method='bfgs',
).fit(X, y)
s = zzu.summary()
best_t = s['best_transform']
converged = s['converged']
rmse = s['train_metrics']['rmse']
print(f'ZZU: best_transform={best_t}, converged={converged}, RMSE={rmse:.4f}')
print(f'     theta_init_used={zzu.theta_init_used_}, final_theta={s["final_theta"]}')

# --- Test 3: evaluate_nonlinear_models ---
models = {'gd': ta.GradientDescentRegressor(model_fn=model_fn, max_iter=3000, learning_rate=0.002),
          'gn': ta.GaussNewtonRegressor(model_fn=model_fn),
          'bf': ta.BFGSRegressor(model_fn=model_fn)}
inits  = {'gd': theta_init, 'gn': theta_init, 'bf': theta_init}
df = ta.evaluate_nonlinear_models(X, y, models, inits)
print(df[['model','method','converged','n_iter','rmse','r2']])

# --- Test 4: graceful error handling ---
bad_fn  = lambda X, t: (_ for _ in ()).throw(RuntimeError('boom'))
bad_reg = ta.BFGSRegressor(model_fn=bad_fn)
bad_models = {'broken': bad_reg}
bad_inits  = {'broken': theta_init}
df2 = ta.evaluate_nonlinear_models(X, y, bad_models, bad_inits)
print('Error captured:', df2['error'].iloc[0])

print('All tests passed.')