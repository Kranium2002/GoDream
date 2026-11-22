"""p01: sklearn MLPRegressor API contract. ALL ARRAYS ARE SYNTHETIC FIXTURES (API/shape only, not performance evidence)."""
import inspect, io, pickle, sys, warnings, hashlib, subprocess
import numpy as np
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

print("sklearn", sklearn.__version__)
print("SIGNATURE MLPRegressor", inspect.signature(MLPRegressor.__init__))
print("has partial_fit:", hasattr(MLPRegressor, "partial_fit"))
print("partial_fit signature:", inspect.signature(MLPRegressor.partial_fit))

def fixture(n, d=10, k=1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    W = rng.normal(size=(d, k))
    y = X @ W + 0.1 * rng.normal(size=(n, k))
    return X, (y[:, 0] if k == 1 else y)

def try_fit(label, est, X, y):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            est.fit(X, y)
            msg = f"OK n_iter_={est.n_iter_}"
            if getattr(est, "validation_scores_", None) is not None:
                msg += f" len(validation_scores_)={len(est.validation_scores_)} best_validation_score_={est.best_validation_score_!r}"
        except Exception as e:
            msg = f"EXC {type(e).__name__}: {e}"
    cw = [f"{x.category.__name__}: {str(x.message)[:160]}" for x in w]
    print(f"[{label}] {msg} | warnings={cw}")

print("\n== early_stopping=True (adam, default validation_fraction=0.1) at small n ==")
for n in [2, 3, 5, 10, 20, 40, 2000]:
    X, y = fixture(n)
    try_fit(f"es n={n}", MLPRegressor(hidden_layer_sizes=(16,), early_stopping=True, random_state=0, max_iter=200), X, y)

print("\n== early_stopping with solver=lbfgs (ignored?) ==")
X, y = fixture(5)
try_fit("lbfgs es n=5", MLPRegressor(hidden_layer_sizes=(16,), solver="lbfgs", early_stopping=True, random_state=0), X, y)

print("\n== solver lbfgs vs adam, default max_iter, n in {5,10,20,40,2000} ==")
for solver in ["lbfgs", "adam"]:
    for n in [5, 10, 20, 40, 2000]:
        X, y = fixture(n)
        try_fit(f"{solver} n={n}", MLPRegressor(hidden_layer_sizes=(32,), solver=solver, random_state=0), X, y)

print("\n== partial_fit availability per solver ==")
X, y = fixture(20)
for solver in ["lbfgs", "adam", "sgd"]:
    m = MLPRegressor(hidden_layer_sizes=(8,), solver=solver, random_state=0)
    try:
        m.partial_fit(X, y)
        print(f"partial_fit solver={solver}: OK n_iter_={m.n_iter_} t_={m.t_}")
    except Exception as e:
        print(f"partial_fit solver={solver}: EXC {type(e).__name__}: {e}")

print("\n== multi-output y (2-D) ==")
X, Y = fixture(40, k=3)
m = MLPRegressor(hidden_layer_sizes=(16,), solver="lbfgs", random_state=0, max_iter=500).fit(X, Y)
print("Y.shape", Y.shape, "predict shape", m.predict(X).shape, "n_outputs_", m.n_outputs_, "out_activation_", m.out_activation_,
      "coefs_ shapes", [c.shape for c in m.coefs_], "intercepts_ shapes", [b.shape for b in m.intercepts_])
X1, y1 = fixture(40, k=1)
m1 = MLPRegressor(hidden_layer_sizes=(16,), solver="lbfgs", random_state=0, max_iter=500).fit(X1, y1)
print("1-D y predict shape", m1.predict(X1).shape, "single row predict shape", m1.predict(X1[:1]).shape)
m1b = MLPRegressor(hidden_layer_sizes=(16,), solver="lbfgs", random_state=0, max_iter=500).fit(X1, y1.reshape(-1, 1))
print("(n,1) y predict shape", m1b.predict(X1).shape)

print("\n== determinism: same random_state, same process ==")
X, y = fixture(200)
def digest_model(m):
    h = hashlib.sha256()
    for a in m.coefs_ + m.intercepts_:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]
for solver in ["lbfgs", "adam"]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        a = MLPRegressor(hidden_layer_sizes=(32, 32), solver=solver, random_state=7).fit(X, y)
        b = MLPRegressor(hidden_layer_sizes=(32, 32), solver=solver, random_state=7).fit(X, y)
        c = MLPRegressor(hidden_layer_sizes=(32, 32), solver=solver, random_state=8).fit(X, y)
    print(f"{solver}: digest rs7={digest_model(a)} rs7again={digest_model(b)} rs8={digest_model(c)} same7={digest_model(a)==digest_model(b)}")
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        pass

print("\n== ConvergenceWarning behaviour ==")
X, y = fixture(40)
for solver, mi in [("adam", 5), ("lbfgs", 5)]:
    try_fit(f"{solver} max_iter={mi}", MLPRegressor(hidden_layer_sizes=(16,), solver=solver, max_iter=mi, random_state=0), X, y)
print("ConvergenceWarning class path:", ConvergenceWarning.__module__ + "." + ConvergenceWarning.__name__)

print("\n== scaling: Pipeline(StandardScaler, MLP) inside TransformedTargetRegressor(StandardScaler) ==")
rng = np.random.default_rng(1)
Xs = rng.normal(size=(40, 10)) * np.array([1, 10, 100, 0.01, 1, 1, 5, 1, 1, 1000.0])
Ys = np.c_[Xs[:, 0] * 50 + 300, Xs[:, 1] * 0.01]
model = TransformedTargetRegressor(
    regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32,), solver="lbfgs", max_iter=2000, random_state=0)),
    transformer=StandardScaler(),
)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    model.fit(Xs, Ys)
print("fit warnings:", [f"{x.category.__name__}: {str(x.message)[:120]}" for x in w])
p = model.predict(Xs[:3])
print("TTR predict shape (3 rows, 2 targets):", p.shape, "single row:", model.predict(Xs[:1]).shape)
blob = pickle.dumps(model)
model2 = pickle.loads(blob)
print("pickle bytes", len(blob), "round-trip predict identical:", np.array_equal(model2.predict(Xs), model.predict(Xs)))
print("inner mlp access: model.regressor_.named_steps ->", list(model.regressor_.named_steps), "transformer_ mean_", model.transformer_.mean_.shape)
# 1-D target with TTR + StandardScaler
model_1d = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(8,), solver="lbfgs", max_iter=2000, random_state=0)), transformer=StandardScaler())
model_1d.fit(Xs, Ys[:, 0])
print("TTR 1-D target predict shape:", model_1d.predict(Xs[:4]).shape)

print("\n== warm_start semantics ==")
X, y = fixture(40)
m = MLPRegressor(hidden_layer_sizes=(16,), solver="adam", warm_start=True, max_iter=50, random_state=0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    m.fit(X, y); c0 = m.coefs_[0].copy(); it0 = m.n_iter_; t0 = m.t_
    m.fit(X, y)
print(f"warm_start adam: after fit1 n_iter_={it0} t_={t0}; after fit2 n_iter_={m.n_iter_} t_={m.t_}; first-layer weights changed={not np.array_equal(c0, m.coefs_[0])}")
print("  _optimizer object kept across warm_start fits? (has attr):", hasattr(m, "_optimizer"))
# warm_start + changed output dimension
X3, Y3 = fixture(40, k=3)
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m.fit(X3, Y3)
    print("warm_start out-dim change 1->3: OK? coefs_[-1].shape", m.coefs_[-1].shape)
except Exception as e:
    print(f"warm_start out-dim change 1->3: EXC {type(e).__name__}: {e}")
# warm_start + changed input dimension
Xd, yd = fixture(40, d=12)
m2 = MLPRegressor(hidden_layer_sizes=(16,), solver="adam", warm_start=True, max_iter=20, random_state=0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    m2.fit(*fixture(40))
try:
    m2.fit(Xd, yd)
    print("warm_start in-dim change 10->12: OK?", m2.coefs_[0].shape)
except Exception as e:
    print(f"warm_start in-dim change 10->12: EXC {type(e).__name__}: {e}")
# partial_fit with changed output dim
m3 = MLPRegressor(hidden_layer_sizes=(16,), solver="adam", random_state=0)
m3.partial_fit(*fixture(40))
try:
    m3.partial_fit(X3, Y3)
    print("partial_fit out-dim change: OK?", m3.coefs_[-1].shape)
except Exception as e:
    print(f"partial_fit out-dim change: EXC {type(e).__name__}: {e}")
# warm_start with lbfgs
m4 = MLPRegressor(hidden_layer_sizes=(16,), solver="lbfgs", warm_start=True, max_iter=30, random_state=0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    m4.fit(X, y); c0 = m4.coefs_[0].copy(); i0 = m4.n_iter_
    m4.fit(X, y)
print(f"warm_start lbfgs: refit continues from previous weights (changed={not np.array_equal(c0, m4.coefs_[0])}), n_iter_ fit1={i0} fit2={m4.n_iter_}")

print("\n== partial_fit semantics: one epoch per call? ==")
m5 = MLPRegressor(hidden_layer_sizes=(16,), solver="adam", batch_size=8, random_state=0, learning_rate_init=1e-3)
X, y = fixture(40)
for i in range(3):
    m5.partial_fit(X, y)
    print(f"  call {i+1}: n_iter_={m5.n_iter_} t_={m5.t_} loss_={m5.loss_:.6f}")
print("  partial_fit with early_stopping=True:")
m6 = MLPRegressor(hidden_layer_sizes=(16,), solver="adam", early_stopping=True, random_state=0)
try:
    m6.partial_fit(X, y)
    print("   OK (validation_scores_ =", m6.validation_scores_, ")")
except Exception as e:
    print(f"   EXC {type(e).__name__}: {e}")

print("\n== partial_fit through a Pipeline? ==")
pipe = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(8,), random_state=0))
print("Pipeline has partial_fit:", hasattr(pipe, "partial_fit"))
try:
    pipe.partial_fit(X, y)
    print("Pipeline.partial_fit OK")
except Exception as e:
    print(f"Pipeline.partial_fit EXC {type(e).__name__}: {e}")
ttr = TransformedTargetRegressor(regressor=MLPRegressor(hidden_layer_sizes=(8,), random_state=0), transformer=StandardScaler())
print("TransformedTargetRegressor has partial_fit:", hasattr(ttr, "partial_fit"))

print("\n== dream rehearsal fixture: shared model, one-hot skill id, refit-from-scratch on union vs partial_fit on mixed batches ==")
rng = np.random.default_rng(3)
def skill_rows(skill_idx, n, n_skills=3, d=8):
    Xs_ = rng.normal(size=(n, d)); oh = np.zeros((n, n_skills)); oh[:, skill_idx] = 1
    y_ = np.c_[np.sin(Xs_[:, 0] + skill_idx), Xs_[:, 1] * (skill_idx + 1)]
    return np.c_[Xs_, oh], y_
XA, yA = skill_rows(0, 300); XB, yB = skill_rows(1, 300)
shared = MLPRegressor(hidden_layer_sizes=(64, 64), solver="adam", random_state=0, max_iter=300)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    shared.fit(XA, yA)
    print("  fit on A: input dim", shared.n_features_in_, "outputs", shared.n_outputs_)
    for epoch in range(5):
        idx = rng.permutation(600)
        Xmix = np.r_[XA, XB][idx]; ymix = np.r_[yA, yB][idx]
        shared.partial_fit(Xmix, ymix)
    print("  partial_fit on mixed A(dream)+B(real): OK, n_iter_", shared.n_iter_, "t_", shared.t_)
print("  NOTE fixed input width needed: one-hot for ALL planned skills must be allocated up-front (n_features_in_ cannot grow).")
