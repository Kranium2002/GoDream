"""p01c: does warm_start fit() reset Adam optimizer state; does partial_fit keep it; partial_fit with solver='lbfgs' quirk. FIXTURE data."""
import warnings
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)
rng = np.random.default_rng(0); X = rng.normal(size=(64, 5)); y = X[:, 0] - X[:, 1]
m = MLPRegressor(hidden_layer_sizes=(8,), solver="adam", warm_start=True, max_iter=5, random_state=0).fit(X, y)
o1 = m._optimizer; t1 = o1.t
m.fit(X, y); o2 = m._optimizer
print(f"warm_start fit(): optimizer object replaced={o1 is not o2}  adam.t before={t1} after={o2.t}  (new Adam => moments ms/vs reset)  n_iter_={m.n_iter_} len(loss_curve_)={len(m.loss_curve_)}")
m.partial_fit(X, y); o3 = m._optimizer
print(f"partial_fit():      optimizer object kept={o2 is o3}  adam.t={o3.t}  n_iter_={m.n_iter_} len(loss_curve_)={len(m.loss_curve_)}")
L = MLPRegressor(hidden_layer_sizes=(8,), solver="lbfgs", max_iter=50, random_state=0)
print("MLPRegressor(solver='lbfgs') hasattr partial_fit:", hasattr(L, "partial_fit"),
      "| MLPClassifier(solver='lbfgs') hasattr partial_fit:", hasattr(__import__("sklearn.neural_network", fromlist=["MLPClassifier"]).MLPClassifier(solver="lbfgs"), "partial_fit"))
L.partial_fit(X, y); i1 = L.n_iter_; c1 = L.coefs_[0].copy()
L.partial_fit(X, y)
print(f"lbfgs partial_fit call1 n_iter_={i1} call2 n_iter_={L.n_iter_} (a full L-BFGS run of up to max_iter each call, continuing from current weights; weights changed={not np.array_equal(c1, L.coefs_[0])})")
