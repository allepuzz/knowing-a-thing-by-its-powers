# Experiments of "Knowing a Thing by Its Powers". Written for Colab with a GPU; it also runs on CPU.
# Phases: ajuste (tuning: picks weight decay and dimension on validation) -> principal (main: curves, probes, intervention)
# -> largo (long training, H3). prueba is a smoke test.
# Resumable: every finished run is saved to <salida>/<fase>.jsonl and skipped when the command is launched again.
import argparse, json, os, time, numpy as np, torch, torch.nn as nn
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from scipy.spatial.distance import pdist

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N, KZ, R, NNEW, J = 200, 4, 6, 40, 8      # objects, latent dimension, relations, new objects, observed powers of each new object
D, K = 64, 16                             # model dimension, powers in the context of OV
PS = (0.005, 0.01, 0.02, 0.05, 0.1)       # fraction of the pairs of known objects used for training
WDS = (1e-3, 1e-2, 1e-1)
SALIDA = None                             # results folder; when set, the object vectors of every run are saved
LECTORES = ('OV', 'TRR')                  # models that read the context of the object

# ---------------- worlds ----------------
class Mundo:
    """gen = 'mlp' (a fixed random MLP per relation), 'bilineal' (bilinear) or 'par' (a random label per fact)."""
    def __init__(s, gen, p, seed):
        g = np.random.default_rng(seed); s.gen = gen
        s.z = g.normal(size=(N, KZ))
        if gen == 'mlp':
            s.W1 = g.normal(size=(R, 2 * KZ, 32)) / np.sqrt(2 * KZ); s.b1 = g.normal(size=(R, 32)) * .5; s.W2 = g.normal(size=(R, 32)) / np.sqrt(32)
        elif gen == 'bilineal':
            s.M = g.normal(size=(R, KZ, KZ)); s.u = g.normal(size=(R, KZ)); s.v = g.normal(size=(R, KZ))
        a, r, b = (x.ravel() for x in np.meshgrid(np.arange(N), np.arange(R), np.arange(N), indexing='ij'))
        ok = a != b; T = np.c_[a[ok], r[ok], b[ok]]
        if gen == 'par': y = g.random(len(T)) < .5
        else:
            f = s.f(s.z[T[:, 0]], T[:, 1], s.z[T[:, 2]])
            s.umbral = np.array([np.median(f[T[:, 1] == k]) for k in range(R)])  # balanced classes in every relation
            y = f > s.umbral[T[:, 1]]
        es_n = np.zeros(N, bool); es_n[g.permutation(N)[:NNEW]] = True; s.nuevo = es_n
        ikk = np.flatnonzero(~es_n[T[:, 0]] & ~es_n[T[:, 2]]); g.shuffle(ikk)
        val_par, te_par, resto = ikk[:3000], ikk[3000:8000], ikk[8000:]
        tr = list(resto[:int(p * len(ikk))])
        iu = np.flatnonzero(es_n[T[:, 0]] ^ es_n[T[:, 2]]); g.shuffle(iu); usados = np.zeros(N, int); fuera = []
        for i in iu:
            o = T[i, 0] if es_n[T[i, 0]] else T[i, 2]
            if usados[o] < J: tr.append(i); usados[o] += 1
            else: fuera.append(i)
        tr = np.array(tr); s.T, s.y = T[tr], y[tr].astype(np.float32)
        s.eval = {'val_par': (T[val_par], y[val_par]), 'test_par': (T[te_par], y[te_par]),
                  'val_obj': (T[fuera[:3000]], y[fuera[:3000]]), 'test_obj': (T[fuera[3000:8000]], y[fuera[3000:8000]]),
                  'train': (s.T[:5000], s.y[:5000] > .5)}
        # context of each object: (relation, direction, partner, outcome) of its training facts
        filas = [[] for _ in range(N)]
        for (a_, r_, b_), yy in zip(s.T, s.y):
            filas[a_].append((r_, 0, b_, int(yy))); filas[b_].append((r_, 1, a_, int(yy)))
        L = max(len(f) for f in filas); C = np.zeros((N, L, 4), np.int64); n = np.zeros(N, np.int64)
        for o, f in enumerate(filas):
            if f: C[o, :len(f)] = f; n[o] = len(f)
        s.C, s.n = torch.as_tensor(C, device=DEV), torch.as_tensor(n, device=DEV)
        s.Tg, s.yg = torch.as_tensor(s.T, device=DEV), torch.as_tensor(s.y, device=DEV)
    def f(s, za, r, zb):
        if s.gen == 'mlp':
            h = np.tanh(np.einsum('ni,nih->nh', np.c_[za, zb], s.W1[r]) + s.b1[r]); return (h * s.W2[r]).sum(1)
        return np.einsum('ni,nij,nj->n', za, s.M[r], zb) + (s.u[r] * za).sum(1) + (s.v[r] * zb).sum(1)
    def etiqueta(s, za, r, zb): return s.f(za, r, zb) > s.umbral[r]
    def ctx(s, obj, otro):
        """Up to K powers of each object, WITHOUT replacement; if it has K or fewer, all of them are read.
        Those that involve the partner of the query are excluded before choosing (no leakage) and take no slot.
        (Version 2. Version 1 sampled with replacement: repeated powers and powers left unread.)"""
        Co = s.C[obj]; L = Co.shape[1]
        valida = (torch.arange(L, device=DEV)[None] < s.n[obj][:, None]) & (Co[..., 2] != otro[:, None])
        llave = torch.where(valida, torch.rand(len(obj), L, device=DEV), torch.full((), -1., device=DEV))
        idx = llave.topk(min(K, L), 1).indices        # valid ones have a key >= 0: they come out before any invalid one
        return Co.gather(1, idx[..., None].expand(-1, -1, 4)), valida.gather(1, idx)

# ---------------- models ----------------
def bloque(capas): return nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 4, 4 * D, 0., batch_first=True, norm_first=True), capas, enable_nested_tensor=False)
class Emb(nn.Module):
    """Models with a stored vector: one embedding row per identifier."""
    def rep(s, obj, otro, mundo): return s.o(obj)
    def reps(s, mundo): return s.o.weight.detach()
    def forward(s, T, mundo): return s.cabeza(s.rep(T[:, 0], T[:, 2], mundo), T[:, 1], s.rep(T[:, 2], T[:, 0], mundo))
class TR(Emb):
    def __init__(s, d=D):
        super().__init__(); s.o = nn.Embedding(N, D); s.r = nn.Embedding(R, D); s.pos = nn.Parameter(torch.randn(3, D) * .02); s.tr = bloque(2); s.h = nn.Linear(3 * D, 1)
    def cabeza(s, ra, r, rb): return s.h(s.tr(torch.stack([ra, s.r(r), rb], 1) + s.pos).flatten(1)).squeeze(1)
class MLP(Emb):
    def __init__(s, d=D):
        super().__init__(); s.o = nn.Embedding(N, D); s.r = nn.Embedding(R, D)
        s.f = nn.Sequential(nn.Linear(3 * D, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
    def cabeza(s, ra, r, rb): return s.f(torch.cat([ra, s.r(r), rb], 1)).squeeze(1)
class FAC(Emb):
    def __init__(s, d=32):
        super().__init__(); s.o = nn.Embedding(N, d); nn.init.normal_(s.o.weight, std=.3)
        s.W = nn.Parameter(torch.randn(R, d, d) * .1); s.ua = nn.Parameter(torch.zeros(R, d)); s.ub = nn.Parameter(torch.zeros(R, d)); s.c = nn.Parameter(torch.zeros(R))
    def cabeza(s, ra, r, rb): return torch.einsum('ni,nij,nj->n', ra, s.W[r], rb) + (s.ua[r] * ra).sum(1) + (s.ub[r] * rb).sum(1) + s.c[r]
class OV(nn.Module):
    """Inferred object vector: no embedding of its own; it is computed by attention over its observed powers."""
    def __init__(s, d=D):
        super().__init__(); s.otro = nn.Embedding(N, D); s.rr = nn.Embedding(R, D); s.dir = nn.Embedding(2, D); s.res = nn.Embedding(2, D)
        s.cls = nn.Parameter(torch.randn(1, 1, D) * .02); s.enc = bloque(1); s.dec = TR()
        del s.dec.o
    def vec(s, F, m):
        x = s.rr(F[..., 0]) + s.dir(F[..., 1]) + s.otro(F[..., 2]) + s.res(F[..., 3])
        x = torch.cat([s.cls.expand(len(x), 1, D), x], 1); m = torch.cat([torch.ones(len(m), 1, dtype=torch.bool, device=m.device), m], 1)
        return s.enc(x, src_key_padding_mask=~m)[:, 0]
    def rep(s, obj, otro, mundo): return s.vec(*mundo.ctx(obj, otro))
    @torch.no_grad()
    def reps(s, mundo, muestras=8):
        o = torch.arange(N, device=DEV); sin = torch.full((N,), -1, device=DEV)
        return torch.stack([s.rep(o, sin, mundo) for _ in range(muestras)]).mean(0)
    def cabeza(s, ra, r, rb): return s.dec.cabeza(ra, r, rb)
    def forward(s, T, mundo): return s.cabeza(s.rep(T[:, 0], T[:, 2], mundo), T[:, 1], s.rep(T[:, 2], T[:, 0], mundo))
class TRR(OV):
    """Additional model. Stored row + reader: the vector of an object is the sum of its embedding row
    and of the output of the reader of OV. It differs from TR in the reader and from OV in the row."""
    def __init__(s, d=D):
        super().__init__(); s.o = nn.Embedding(N, D)
    def rep(s, obj, otro, mundo): return s.vec(*mundo.ctx(obj, otro)) + s.o(obj)
MODELOS = {'TR': TR, 'MLP': MLP, 'FAC': FAC, 'OV': OV, 'TRR': TRR}
PRINCIPALES = ('TR', 'MLP', 'FAC', 'OV')  # the four models of the pre-registered experiment

# ---------------- training and measurements ----------------
@torch.no_grad()
def mide(m, mundo, T, y, bs=8192):
    m.eval(); lg = torch.cat([m(torch.as_tensor(T[i:i + bs], device=DEV), mundo) for i in range(0, len(T), bs)])
    yt = torch.as_tensor(np.asarray(y, np.float32), device=DEV)
    return float(((lg > 0).float() == yt).float().mean()), float(nn.functional.binary_cross_entropy_with_logits(lg, yt))

def sondas(m, mundo, semilla):
    """R² of a linear probe (cross validation over objects) towards z and towards a random control; intervention by editing z."""
    E = m.reps(mundo).float().cpu().numpy(); z = mundo.z; cv = KFold(5, shuffle=True, random_state=semilla)
    r2 = lambda Y: float(r2_score(Y, cross_val_predict(RidgeCV(np.logspace(-3, 3, 13)), E, Y, cv=cv)))
    ctrl = np.random.default_rng(semilla + 999).normal(size=z.shape)
    out = {'r2_z': r2(z), 'r2_control': r2(ctrl), '_E': E}
    # nonlinear probe (additional): Gaussian kernel regression, same folds and same control; hyperparameters by internal validation
    med = float(np.median(pdist(E)))
    if med > 0:
        rej = {'alpha': list(np.logspace(-3, 3, 13)), 'gamma': [1 / (2 * (med * w) ** 2) for w in np.logspace(-1, 1, 7)]}
        kr = lambda Y: float(r2_score(Y, cross_val_predict(GridSearchCV(KernelRidge(kernel='rbf'), rej, cv=5), E, Y, cv=cv)))
        out.update(r2_z_kernel=kr(z), r2_control_kernel=kr(ctrl))
    # intervention: move the representation of a along the linear map z -> rep, from z_a to z_c, and see whether the model follows the edited world
    A = Ridge(1.0).fit(z, E); g = np.random.default_rng(semilla + 7)
    conoc = np.flatnonzero(~mundo.nuevo); a = g.choice(conoc, 40, replace=False); c = g.choice(conoc, 40, replace=False)
    b = np.array([x for x in conoc]); Ra = torch.as_tensor(E, device=DEV)
    acc_ed, acc_or, cambian = [], [], []
    with torch.no_grad():
        m.eval()
        for ai, ci in zip(a, c):
            if ai == ci: continue
            bb = b[b != ai]; rr = np.repeat(np.arange(R), len(bb)); bb = np.tile(bb, R)
            nuevo = Ra[ai] + torch.as_tensor((z[ci] - z[ai]) @ A.coef_.T, dtype=Ra.dtype, device=DEV)
            pred = (m.cabeza(nuevo.expand(len(bb), -1), torch.as_tensor(rr, device=DEV), Ra[torch.as_tensor(bb, device=DEV)]) > 0).cpu().numpy()
            y_ed = mundo.etiqueta(np.repeat(z[ci][None], len(bb), 0), rr, z[bb]); y_or = mundo.etiqueta(np.repeat(z[ai][None], len(bb), 0), rr, z[bb])
            acc_ed.append((pred == y_ed).mean()); acc_or.append((pred == y_or).mean()); cambian.append((y_ed != y_or).mean())
    out.update(interv_acc_editado=float(np.mean(acc_ed)), interv_acc_original=float(np.mean(acc_or)), interv_frac_cambia=float(np.mean(cambian)))
    return out

def entrena(cfg):
    global K; K = cfg.get('k', 16)  # protocol v2: OV reads up to 64 interactions
    torch.manual_seed(cfg['seed']); t0 = time.time()
    mundo = Mundo(cfg['gen'], cfg['p'], cfg['seed'])
    m = MODELOS[cfg['modelo']](**({'d': cfg['d']} if cfg.get('d') else {})).to(DEV)
    o = torch.optim.AdamW(m.parameters(), cfg.get('lr', 1e-3), weight_decay=cfg['wd'])
    cos = cfg.get('sched') == 'cos'  # protocol v2: after the warmup, cosine decay to 0
    sch = torch.optim.lr_scheduler.LambdaLR(o, lambda k: min(1., (k + 1) / 500) * (0.5 * (1 + np.cos(np.pi * min(k, cfg['pasos']) / cfg['pasos'])) if cos else 1.))
    curva, cada = [], max(1, cfg['pasos'] // 10)
    for paso in range(1, cfg['pasos'] + 1):
        m.train(); i = torch.randint(len(mundo.Tg), (cfg.get('bs', 256),), device=DEV)
        l = nn.functional.binary_cross_entropy_with_logits(m(mundo.Tg[i], mundo), mundo.yg[i])
        o.zero_grad(); l.backward()
        if cfg.get('clip'): nn.utils.clip_grad_norm_(m.parameters(), cfg['clip'])  # protocol v2: gradient clipping
        o.step(); sch.step()
        if paso % cada == 0 or paso == cfg['pasos']:
            r = {'paso': paso}
            for k, (T, y) in mundo.eval.items(): r[k], r[k + '_loss'] = mide(m, mundo, T, y)
            curva.append(r)
    res = dict(cfg, n_train=int(len(mundo.T)), curva=curva, **{k: v for k, v in curva[-1].items() if k != 'paso'})
    if cfg.get('sondas') and cfg['gen'] != 'par':
        s_ = sondas(m, mundo, cfg['seed']); E = s_.pop('_E'); res.update(s_)
        if SALIDA:  # object vectors on disk: any new probe can be computed later without training again
            os.makedirs(os.path.join(SALIDA, 'reps'), exist_ok=True)
            nombre = f"{cfg['gen']}_{cfg['modelo']}_p{cfg['p']}_s{cfg['seed']}{'_v2' if cfg.get('sched') else ''}.npz"
            np.savez_compressed(os.path.join(SALIDA, 'reps', nombre), E=E, z=mundo.z, nuevo=mundo.nuevo)
    res['seg'] = time.time() - t0; res['dev'] = DEV; return res

# ---------------- phases ----------------
CTX_VER = 2  # version of the reading of OV; runs of OV made with another version are repeated and ignored by the analysis
def clave(c):
    d = {k: c[k] for k in sorted(c) if k not in ('sondas',)}
    if c['modelo'] in LECTORES: d['ctx'] = CTX_VER
    return json.dumps(d, sort_keys=True)
def vigente(r): return r['modelo'] not in LECTORES or r.get('ctx') == CTX_VER
def corre(cfgs, ruta):
    hechas = set()
    if os.path.exists(ruta):
        for l in open(ruta): hechas.add(json.loads(l)['clave'])
    pend = [c for c in cfgs if clave(c) not in hechas]
    print(f'{ruta}: {len(cfgs)} runs, {len(pend)} pending, device {DEV}', flush=True); t0 = time.time()
    for i, c in enumerate(pend, 1):
        r = entrena(c); r['clave'] = clave(c)
        if c['modelo'] in LECTORES: r['ctx'] = CTX_VER
        with open(ruta, 'a') as f: f.write(json.dumps(r) + '\n')
        print(f"[{i}/{len(pend)}] {time.time() - t0:6.0f}s {c['gen']} {c['modelo']} p={c['p']} seed={c['seed']} "
              f"test_par={r['test_par']:.3f} test_obj={r['test_obj']:.3f} train={r['train']:.3f} ({r['seg']:.0f}s)", flush=True)

def elegidos(salida):
    """Best (wd, d) for each model by mean validation accuracy in the tuning phase."""
    rs = [r for r in (json.loads(l) for l in open(os.path.join(salida, 'ajuste.jsonl'))) if vigente(r)]; mejor = {}
    for r in rs:
        v = (r['val_par'] + r['val_obj']) / 2
        if r['modelo'] not in mejor or v > mejor[r['modelo']][0]: mejor[r['modelo']] = (v, r['wd'], r.get('d'))
    return {k: {'wd': w, 'd': d} for k, (_, w, d) in mejor.items()}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fase', required=True, choices=['ajuste', 'principal', 'largo', 'prueba'])
    ap.add_argument('--salida', default='resultados'); ap.add_argument('--semillas', default='0,1,2')
    ap.add_argument('--pasos', type=int, default=10000); ap.add_argument('--pasos_largo', type=int, default=50000)
    ap.add_argument('--sin_bilineal', action='store_true')
    ap.add_argument('--ps', default=None, help='additional: fractions p separated by commas, e.g. 0.001,0.002 (default: those of the protocol)')
    ap.add_argument('--modelos', default=None, help='additional: models separated by commas, e.g. TRR (default: TR,MLP,FAC,OV)')
    ap.add_argument('--v2', action='store_true', help='protocol v2: OV reads up to 64, cosine decay, clipping 1.0 (use it with another --salida)')
    a = ap.parse_args(); os.makedirs(a.salida, exist_ok=True); sem = [int(x) for x in a.semillas.split(',')]
    SALIDA = a.salida
    ps = tuple(float(x) for x in a.ps.split(',')) if a.ps else PS
    mods = tuple(a.modelos.split(',')) if a.modelos else PRINCIPALES
    assert all(m in MODELOS for m in mods), f'unknown model in {mods}'
    V2 = dict(sched='cos', clip=1.0) if a.v2 else {}
    def v2(c):
        if a.v2: c.update(V2); c.update({'k': 64} if c['modelo'] in LECTORES else {})
        return c
    if a.fase == 'prueba':  # smoke test: the whole path with very few steps
        cfgs = [dict(gen=g, p=.02, modelo=m, seed=0, wd=1e-2, pasos=40, sondas=True) for g in ('mlp', 'par') for m in mods]
        corre([v2(c) for c in cfgs], os.path.join(a.salida, 'prueba.jsonl'))
    elif a.fase == 'ajuste':  # seeds 100+ so that tuning does not use those of the experiment
        cfgs = [dict(gen='mlp', p=.02, modelo=m, seed=100, wd=w, pasos=a.pasos, **({'d': d} if m == 'FAC' else {}))
                for m in mods for w in WDS for d in ((8, 32) if m == 'FAC' else (None,))]
        corre([v2(c) for c in cfgs], os.path.join(a.salida, 'ajuste.jsonl'))
        json.dump(elegidos(a.salida), open(os.path.join(a.salida, 'elegidos.json'), 'w'), indent=1); print(elegidos(a.salida))
    elif a.fase == 'principal':
        h = json.load(open(os.path.join(a.salida, 'elegidos.json'))); assert all(m in h for m in mods), 'the tuning phase of some model is missing'; gens = ('mlp',) if a.sin_bilineal else ('mlp', 'bilineal')
        cfgs = [dict(gen=g, p=p, modelo=m, seed=s, pasos=a.pasos, sondas=True, **{k: v for k, v in h[m].items() if v is not None})
                for s in sem for g in gens for p in ps for m in mods]
        cfgs += [dict(gen='par', p=.02, modelo=m, seed=s, pasos=a.pasos, **{k: v for k, v in h[m].items() if v is not None}) for s in sem for m in mods]
        corre([v2(c) for c in cfgs], os.path.join(a.salida, 'principal.jsonl'))
    elif a.fase == 'largo':  # H3: long training with high weight decay
        cfgs = [dict(gen='mlp', p=.02, modelo=m, seed=s, wd=.1, pasos=a.pasos_largo) for s in sem for m in ('TR', 'OV')]
        corre([v2(c) for c in cfgs], os.path.join(a.salida, 'largo.jsonl'))
