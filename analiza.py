# Analysis of the experiments. Usage: python analiza.py --salida results_v2
# Writes <salida>/report.md and PNG figures. Intervals: Student's t at 95% over seeds; ratio with a bootstrap over seeds.
import argparse, json, os, numpy as np
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

MOD = ['TR', 'MLP', 'FAC', 'OV', 'TRR']; COL = {'TR': '#1f77b4', 'MLP': '#ff7f0e', 'FAC': '#2ca02c', 'OV': '#d62728', 'TRR': '#9467bd'}
NOMBRE = {'TR': 'TR', 'MLP': 'MLP', 'FAC': 'FAC', 'OV': 'OV', 'TRR': 'TRR (stored row and reader)'}  # the names used in the paper
PRE = (0.005, 0.01, 0.02, 0.05, 0.1)  # sizes of the pre-registered protocol: the original ratio uses only these
LECTORES = ('OV', 'TRR')

CTX_VER = 2  # as in exp.py: runs of OV made with the old reading (with replacement) are ignored
def carga(ruta):
    rs = [json.loads(l) for l in open(ruta)] if os.path.exists(ruta) else []
    return [r for r in rs if r['modelo'] not in LECTORES or r.get('ctx') == CTX_VER]
def ic(x):
    x = np.asarray(x, float)
    if len(x) < 2: return float(x.mean()), float('nan')
    return float(x.mean()), float(stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
def fmt(x): m, h = ic(x); return f'{m:.3f} ± {h:.3f}' if h == h else f'{m:.3f}'

def n_para(ns, err, eps):
    """Training size needed to reach the error eps, interpolating in log(n); inf if it is not reached."""
    ns, err = np.asarray(ns, float), np.minimum.accumulate(np.asarray(err, float))
    if err[0] <= eps: return ns[0]
    for i in range(1, len(ns)):
        if err[i] <= eps:
            t = (err[i - 1] - eps) / (err[i - 1] - err[i]); return float(np.exp(np.log(ns[i - 1]) + t * (np.log(ns[i]) - np.log(ns[i - 1]))))
    return np.inf

def ratio(rs, gen, split, a='TR', b='OV', B=2000):
    """n_a(eps)/n_b(eps) for each seed. eps = error of a with the largest size (a reaches it by definition)."""
    por = {}
    for r in rs:
        if r['gen'] == gen and r['modelo'] in (a, b) and r['p'] in PRE: por.setdefault((r['seed'], r['modelo']), []).append((r['n_train'], 1 - r[split]))
    vals = []
    for s in sorted({k[0] for k in por}):
        if (s, a) not in por or (s, b) not in por: continue
        A, Bm = sorted(por[(s, a)]), sorted(por[(s, b)]); eps = A[-1][1]
        na, nb = n_para([x for x, _ in A], [e for _, e in A], eps), n_para([x for x, _ in Bm], [e for _, e in Bm], eps)
        vals.append(na / nb)
    vals = np.array(vals)
    if len(vals) == 0: return None
    if np.any(vals <= 0): return vals, float('nan'), (float('nan'), float('nan'))  # b does not reach eps in some seed
    g = np.random.default_rng(0); boot = [np.exp(np.mean(np.log(g.choice(vals, len(vals))))) for _ in range(B)]
    return vals, float(np.exp(np.mean(np.log(vals)))), np.percentile(boot, [2.5, 97.5])

def ratio_acc(rs, gen, split, acc, a='TR', b='OV'):
    """Post hoc analysis: n_a/n_b to reach a fixed accuracy, for each seed (None if either does not reach it)."""
    out = []
    for s in sorted({r['seed'] for r in rs if r['gen'] == gen}):
        cur = {}
        for m in (a, b):
            f = sorted([r for r in rs if r['gen'] == gen and r['modelo'] == m and r['seed'] == s and r['p'] in PRE], key=lambda r: r['n_train'])
            cur[m] = n_para([r['n_train'] for r in f], [1 - r[split] for r in f], 1 - acc) if f else np.inf
        out.append(round(cur[a] / cur[b], 2) if np.isfinite(cur[a]) and np.isfinite(cur[b]) else None)
    return out

def ratio_curva(rs, gen, split, a='TR', b='OV'):
    """Additional: rho(n) = n_a(eps_b(n)) / n for every size n, for each seed. eps_b(n) is the error of b trained on n.
    Returns {p: (n, [(value, mark) for each seed])}; mark '' exact, '>' lower bound (a does not reach eps), '<' upper bound."""
    out = {}
    for s in sorted({r['seed'] for r in rs if r['gen'] == gen}):
        A = sorted((r['n_train'], 1 - r[split]) for r in rs if r['gen'] == gen and r['modelo'] == a and r['seed'] == s)
        Bm = sorted((r['n_train'], 1 - r[split], r['p']) for r in rs if r['gen'] == gen and r['modelo'] == b and r['seed'] == s)
        if not A or not Bm: continue
        ns, err = [x for x, _ in A], [e for _, e in A]
        for n, eps, p in Bm:
            na = n_para(ns, err, eps)
            if not np.isfinite(na): v = (ns[-1] / n, '>')
            elif err[0] <= eps: v = (ns[0] / n, '<')
            else: v = (na / n, '')
            out.setdefault(p, (n, []))[1].append(v)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--salida', default='resultados'); a = ap.parse_args(); S = a.salida
    P, L, A = carga(f'{S}/principal.jsonl'), carga(f'{S}/largo.jsonl'), carga(f'{S}/ajuste.jsonl'); inf = ['# Automatic report', '']
    if A:
        inf += ['## Tuning (mean validation accuracy, seed 100)', '', '| Model | wd | d | val |', '|---|---|---|---|']
        inf += [f"| {r['modelo']} | {r['wd']} | {r.get('d', '')} | {(r['val_par'] + r['val_obj']) / 2:.3f} |" for r in A]; inf.append('')
    for gen in sorted({r['gen'] for r in P}):
        R = [r for r in P if r['gen'] == gen]; ps = sorted({r['p'] for r in R}); MOD = [m for m in COL if any(r['modelo'] == m for r in R)]
        inf += [f'## World {gen}', '']
        for split in ('train', 'test_par', 'test_obj'):
            inf += [f'**{split}** (mean ± 95% CI)', '', '| p | n | ' + ' | '.join(MOD) + ' |', '|---' * (len(MOD) + 2) + '|']
            for p in ps:
                fila = [r for r in R if r['p'] == p]; n = int(np.mean([r['n_train'] for r in fila]))
                inf.append(f'| {p} | {n} | ' + ' | '.join(fmt([r[split] for r in fila if r['modelo'] == m]) for m in MOD) + ' |')
            inf.append('')
        if gen != 'par':
            for split in ('test_par', 'test_obj'):
                for b in ('OV', 'MLP'):
                    x = ratio(R, gen, split, 'TR', b)
                    if x is not None:
                        v, gm, (lo, hi) = x
                        inf.append(f'- Ratio n_TR(ε)/n_{b}(ε), {split} (0 = {b} does not reach ε): geometric mean {gm:.2f}, bootstrap CI [{lo:.2f}, {hi:.2f}], by seed {np.round(v, 2).tolist()}')
            for split in ('test_par', 'test_obj'):
                for acc in (0.70, 0.75, 0.80):
                    inf.append(f'- (post hoc) n_TR/n_OV for accuracy {acc:.2f}, {split}, by seed: {ratio_acc(R, gen, split, acc)}')
            for split in ('test_par', 'test_obj'):
                for b in [m for m in ('OV', 'MLP', 'TRR') if m in MOD]:
                    cur = ratio_curva(R, gen, split, 'TR', b)
                    if not cur: continue
                    inf += ['', f'**(additional) ρ(n) = n_TR(ε_{b}(n)) / n, {split}** (> lower bound: TR does not reach that error; < upper bound)', '',
                            '| p | n | by seed | geometric mean |', '|---|---|---|---|']
                    for p in sorted(cur):
                        n, vs = cur[p]; exacto = all(m == '' for _, m in vs)
                        gm = f"{np.exp(np.mean(np.log([v for v, _ in vs]))):.2f}" if exacto else 'with bounds'
                        inf.append(f"| {p} | {n} | {', '.join(f'{m}{v:.2f}' for v, m in vs)} | {gm} |")
            if any('r2_z' in r for r in R):
                nl = any('r2_z_kernel' in r for r in R)
                inf += ['', '**(additional) Probes by size** (mean ± 95% CI over seeds)', '',
                        '| p | Model | R² z linear | R² control linear |' + (' R² z kernel | R² control kernel |' if nl else ''), '|---|---|---|---|' + ('---|---|' if nl else '')]
                for p in ps:
                    for m in MOD:
                        f = [r for r in R if r['modelo'] == m and r['p'] == p and 'r2_z' in r]
                        if not f: continue
                        fila = f"| {p} | {m} | {fmt([r['r2_z'] for r in f])} | {fmt([r['r2_control'] for r in f])} |"
                        fk = [r for r in f if 'r2_z_kernel' in r]
                        if nl: fila += (f" {fmt([r['r2_z_kernel'] for r in fk])} | {fmt([r['r2_control_kernel'] for r in fk])} |" if fk else ' | |')
                        inf.append(fila)
            inf += ['', '**Probes and intervention** (mean ± 95% CI over seeds and sizes)', '', '| Model | R² z | R² control | accuracy edited world | accuracy original world | fraction that changes |', '|---|---|---|---|---|---|']
            for m in MOD:
                f = [r for r in R if r['modelo'] == m and 'r2_z' in r and r['p'] in PRE]  # only the pre-registered sizes
                if f: inf.append(f"| {m} | {fmt([r['r2_z'] for r in f])} | {fmt([r['r2_control'] for r in f])} | {fmt([r['interv_acc_editado'] for r in f])} | {fmt([r['interv_acc_original'] for r in f])} | {fmt([r['interv_frac_cambia'] for r in f])} |")
            inf.append('')
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
            for k, split in enumerate(('test_par', 'test_obj')):
                for m in MOD:
                    xs, ys, hs = [], [], []
                    for p in ps:
                        v = [r[split] for r in R if r['p'] == p and r['modelo'] == m]
                        if v: mu, h = ic(v); xs.append(np.mean([r['n_train'] for r in R if r['p'] == p])); ys.append(mu); hs.append(0 if h != h else h)
                    ax[k].errorbar(xs, ys, yerr=hs, marker='o', capsize=3, color=COL[m], label=NOMBRE[m])
                ax[k].set_xscale('log'); ax[k].set_title({'test_par': 'Unseen pairs of known objects', 'test_obj': 'New objects (8 observed interactions)'}[split])
                ax[k].set_xlabel('Training interactions'); ax[k].axhline(.5, color='gray', lw=.8, ls=':')
            ax[0].set_ylabel('Test accuracy'); ax[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(f'{S}/curvas_{gen}.png', dpi=150); plt.close(fig)
            inf.append(f'![curves](curvas_{gen}.png)\n')
    if L:
        inf += ['## Long training (H3)', '']
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for m in ('TR', 'OV'):
            f = [r for r in L if r['modelo'] == m]
            if not f: continue
            pasos = [c['paso'] for c in f[0]['curva']]; Y = np.array([[c['test_par'] for c in r['curva']] for r in f])
            ax.plot(pasos, Y.mean(0), color=COL[m], label=NOMBRE[m]); ax.fill_between(pasos, Y.min(0), Y.max(0), color=COL[m], alpha=.2)
            inf.append(f"- {m}: final test_par {fmt([r['test_par'] for r in f])}, train {fmt([r['train'] for r in f])}")
        ax.set_xlabel('Step'); ax.set_ylabel('Test accuracy (unseen pairs)'); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f'{S}/largo.png', dpi=150); plt.close(fig)
        inf.append('\n![long](largo.png)\n')
    open(f'{S}/report.md', 'w', encoding='utf-8').write('\n'.join(inf)); print(f'Written {S}/report.md')
