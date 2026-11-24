#!/usr/bin/env python3
"""
Auditoria de QMEs: consistência entre execuções e comparação entre conjuntos.

Uso (na pasta que contém as subpastas dos conjuntos):
    py analise.py                 -> usa a pasta do script
    py analise.py "C:\\caminho"   -> usa outra pasta

Cada subpasta com um *relatorio_auditoria*.csv é um conjunto (o sufixo FINAL é removido
do nome). Saída em ./resultados/:

  1_consistencia_intra_conjunto/<CONJUNTO>/   execução 1 x 2 do próprio conjunto
      κ de Cohen (+IC 95% bootstrap), PABAK, McNemar por diretriz; Jaccard e Hamming por
      questão; κ da resposta cega, do confere_gabarito e do nível de Bloom; Wilcoxon,
      ICC(3,1) e Bland-Altman da plausibilidade.

  2_comparacao_entre_conjuntos/              conjuntos entre si
      Diretrizes: prevalência (consenso das 2 execuções; união como sensibilidade),
      qui-quadrado por permutação + V de Cramér com correção de Holm, pós-teste de Fisher.
      Taxonomia de Bloom: qui-quadrado por permutação, V de Cramér, resíduos ajustados,
      Kruskal-Wallis no nível ordinal. Consistência intra comparada entre conjuntos.
"""
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------- utilidades
def normaliza_bloom(x: str) -> str:
    """'ANÁLISE', 'Analisar', 'APLICAR' -> 'analisar', 'aplicar'."""
    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower().strip()
    return MAPA_BLOOM.get(s, s)


# Taxonomia de Bloom revisada (Anderson & Krathwohl, 2001), em ordem hierárquica
ORDEM_BLOOM = ["lembrar", "compreender", "aplicar", "analisar", "avaliar", "criar"]
MAPA_BLOOM = {
    "lembrar": "lembrar", "recordar": "lembrar", "conhecimento": "lembrar", "conhecer": "lembrar",
    "entender": "compreender", "compreender": "compreender", "compreensao": "compreender",
    "aplicar": "aplicar", "aplicacao": "aplicar",
    "analisar": "analisar", "analise": "analisar",
    "avaliar": "avaliar", "avaliacao": "avaliar",
    "criar": "criar", "criacao": "criar", "sintese": "criar", "sintetizar": "criar",
}


def conjunto_diretrizes(x) -> frozenset:
    if pd.isna(x) or str(x).strip() == "":
        return frozenset()
    return frozenset(d.strip().upper() for d in str(x).split(",") if d.strip())


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def cohen_kappa(r1, r2) -> float:
    """Kappa de Cohen para dois avaliadores (categorias quaisquer)."""
    r1, r2 = np.asarray(r1), np.asarray(r2)
    cats = np.union1d(r1, r2)
    po = np.mean(r1 == r2)
    pe = sum(np.mean(r1 == c) * np.mean(r2 == c) for c in cats)
    if np.isclose(pe, 1.0):
        return np.nan  # ambos constantes: kappa indefinido
    return (po - pe) / (1 - pe)


def pabak(r1, r2) -> float:
    """Prevalence-Adjusted Bias-Adjusted Kappa (binário): 2*Po - 1."""
    return 2 * np.mean(np.asarray(r1) == np.asarray(r2)) - 1


def interpreta_kappa(k: float) -> str:
    """Escala de Landis & Koch (1977)."""
    if np.isnan(k):
        return "indefinido"
    if k < 0:
        return "pobre"
    for lim, rot in [(0.20, "leve"), (0.40, "razoável"), (0.60, "moderada"),
                     (0.80, "substancial"), (1.00, "quase perfeita")]:
        if k <= lim:
            return rot
    return "quase perfeita"


def mcnemar(a, b):
    """McNemar exato (binomial) para pares binários."""
    a, b = np.asarray(a).astype(int), np.asarray(b).astype(int)
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n = n01 + n10
    p = 1.0 if n == 0 else stats.binomtest(n01, n, 0.5).pvalue
    return n01, n10, p


def icc_3_1(x, y) -> float:
    """ICC(3,1): consistência, two-way mixed, medida única (Shrout & Fleiss)."""
    m = np.column_stack([x, y]).astype(float)
    n, k = m.shape
    gm = m.mean()
    ss_lin = k * np.sum((m.mean(axis=1) - gm) ** 2)
    ss_col = n * np.sum((m.mean(axis=0) - gm) ** 2)
    ss_tot = np.sum((m - gm) ** 2)
    ms_lin = ss_lin / (n - 1)
    ms_err = (ss_tot - ss_lin - ss_col) / ((n - 1) * (k - 1))
    den = ms_lin + (k - 1) * ms_err
    return np.nan if den == 0 else (ms_lin - ms_err) / den


def bootstrap_ic(func, r1, r2, n_boot=5000, alpha=0.05, seed=42):
    """IC percentil por bootstrap pareado (reamostra questões, mantendo os pares)."""
    rng = np.random.default_rng(seed)
    r1, r2 = np.asarray(r1), np.asarray(r2)
    n = len(r1)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = func(r1[idx], r2[idx])
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


# ----------------------------------------------------------------- gráficos
# Paleta: azul/laranja categóricos, cinza neutro, rampa sequencial de azul.
AZUL, LARANJA, CINZA, TINTA, TINTA2 = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
FAIXAS_LK = [(-1, 0, "Pobre"), (0, .20, "Leve"), (.20, .40, "Razoável"),
             (.40, .60, "Moderada"), (.60, .80, "Substancial"), (.80, 1.0, "Quase perfeita")]


def _estilo():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.edgecolor": "#c9c8c3", "axes.labelcolor": TINTA, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
        "xtick.color": TINTA2, "ytick.color": TINTA2,
        "legend.frameon": False, "figure.dpi": 100,
        "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def _salvar(fig, out: Path, nome: str):
    fig.savefig(out / f"{nome}.png")
    fig.savefig(out / f"{nome}.pdf")  # vetorial, para o LaTeX/Word


def _nota(ax, texto, dy=-38):
    """Nota de rodapé a uma distância fixa (em pontos) do eixo, independente da altura."""
    ax.annotate(texto, xy=(0, 0), xycoords="axes fraction", xytext=(0, dy),
                textcoords="offset points", fontsize=7.5, color=TINTA2, va="top")


def _faixas_landis_koch(ax, horizontal=True, xmin=-0.2, rotulos=True):
    tons = ["#f4f3ef", "#ffffff"]
    for i, (a, b, rot) in enumerate(FAIXAS_LK):
        a = max(a, xmin)
        if horizontal:
            ax.axvspan(a, b, color=tons[i % 2], zorder=0, lw=0)
            if rotulos:
                    ax.text((a + b) / 2, 1.01, rot, transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=7.5, color=TINTA2)


def gerar_graficos(e1, e2, por_q, por_g, globais_ic, cols_g, out: Path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    _estilo()
    fig_dir = out / "graficos"
    fig_dir.mkdir(exist_ok=True)

    # ---------- Fig. 1: forest plot do kappa por diretriz (IC 95% bootstrap)
    d = por_g.dropna(subset=["kappa"]).sort_values("kappa").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(d) + 1.4))
    _faixas_landis_koch(ax)
    y = np.arange(len(d))
    off = 0.14  # desloca as duas medidas para não se sobreporem
    ax.hlines(y + off, d["kappa_ic_inf"], d["kappa_ic_sup"], color=AZUL, lw=2, zorder=2)
    ax.plot(d["kappa"], y + off, "o", ms=8, color=AZUL, mec="white", mew=2, zorder=3)
    ax.plot(d["pabak"], y - off, "D", ms=6, color=LARANJA, mec="white", mew=1.5, zorder=3)
    rotulos = [f"{g}  (n₁={a}, n₂={b})" for g, a, b in zip(d["diretriz"], d["freq_exec1"], d["freq_exec2"])]
    ax.set_yticks(y, rotulos)
    ax.set_xlim(-0.2, 1.05)
    ax.axvline(0, color=CINZA, lw=0.8)
    ax.set_xlabel("Coeficiente de concordância")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[
        Line2D([], [], color=AZUL, marker="o", ms=7, lw=2, label="κ de Cohen (IC 95%, bootstrap)"),
        Line2D([], [], color=LARANJA, marker="D", ms=6, lw=0, label="PABAK"),
    ], loc="upper left", frameon=True, facecolor="white", edgecolor="#e6e5e0")
    ax.set_title("Concordância entre execuções por diretriz", pad=18)
    _nota(ax, "n₁, n₂ = nº de questões em que a diretriz foi apontada na execução 1 e 2. "
          "Diretrizes nunca apontadas foram omitidas.")
    _salvar(fig, fig_dir, "fig1_kappa_por_diretriz"); plt.close(fig)

    # ---------- Fig. 2: Bland-Altman da plausibilidade média
    p1 = e1["plausibilidade_media"].values.astype(float)
    p2 = e2["plausibilidade_media"].values.astype(float)
    media, dif = (p1 + p2) / 2, p2 - p1
    vies, dp = dif.mean(), dif.std(ddof=1)
    loa_inf, loa_sup = vies - 1.96 * dp, vies + 1.96 * dp
    pts = pd.DataFrame({"m": np.round(media, 4), "d": np.round(dif, 4)}).value_counts().reset_index(name="n")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(pts["m"], pts["d"], s=30 + 22 * pts["n"], color=AZUL, alpha=0.85,
               edgecolor="white", linewidth=1.5, zorder=3)
    for _, r in pts[pts["n"] > 1].iterrows():
        if r["n"] >= 10:  # bolha grande: número dentro
            ax.text(r["m"], r["d"], f"{int(r['n'])}", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold", zorder=4)
        else:
            ax.annotate(f"{int(r['n'])}", (r["m"], r["d"]), xytext=(8, 6),
                        textcoords="offset points", fontsize=8, color=TINTA2)
    for val, estilo, rot in [(vies, "-", f"Viés = {vies:+.3f}"),
                             (loa_sup, "--", f"LSC = {loa_sup:+.3f}"),
                             (loa_inf, "--", f"LIC = {loa_inf:+.3f}")]:
        ax.axhline(val, color=LARANJA if estilo == "-" else TINTA2, ls=estilo, lw=1.2, zorder=2)
        ax.text(1.01, val, rot, transform=ax.get_yaxis_transform(), va="center", fontsize=8.5, color=TINTA)
    ax.axhline(0, color=CINZA, lw=0.8, zorder=1)
    ax.set_xlabel("Média das execuções (plausibilidade média)")
    ax.set_ylabel("Diferença (execução 2 − execução 1)")
    ax.set_title("Bland-Altman: plausibilidade média dos distratores")
    ax.text(0, -0.30, "Tamanho do ponto = nº de questões sobrepostas. Limites de concordância = viés ± 1,96·DP.\n"
            "As diferenças são discretas porque a plausibilidade é a média de 3 a 6 distratores pontuados de 1 a 5.",
            transform=ax.transAxes, fontsize=8, color=TINTA2)
    _salvar(fig, fig_dir, "fig2_bland_altman_plausibilidade"); plt.close(fig)

    # ---------- Fig. 3: distribuição do Jaccard por questão
    jac = por_q["jaccard_diretrizes"]
    cont = jac.round(3).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(cont.index, cont.values, width=0.045, color=AZUL, edgecolor="white", linewidth=2, zorder=2)
    for x, v in cont.items():
        ax.text(x, v + 0.4, str(v), ha="center", fontsize=8, color=TINTA2)
    ax.axvline(jac.mean(), color=LARANJA, lw=1.5, label=f"Média = {jac.mean():.3f}")
    ax.axvline(jac.median(), color=TINTA2, lw=1.2, ls="--", label=f"Mediana = {jac.median():.3f}")
    ax.set_xlim(-0.03, 1.05)
    ax.set_xlabel("Índice de Jaccard das diretrizes violadas (execução 1 × 2)")
    ax.set_ylabel("Nº de questões")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title(f"Similaridade das diretrizes por questão (n = {len(jac)})")
    _salvar(fig, fig_dir, "fig3_distribuicao_jaccard"); plt.close(fig)

    # ---------- Fig. 4: tabela de contingência do gabarito do agente
    ordem = [c for c in ["A", "B", "C", "D", "E", "MULTIPLOS", "NENHUM"]
             if c in set(e1["gabarito_agente"]) | set(e2["gabarito_agente"])]
    tab = pd.crosstab(pd.Categorical(e1["gabarito_agente"], ordem),
                      pd.Categorical(e2["gabarito_agente"], ordem), dropna=False)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.imshow(tab.values, cmap="Blues", vmin=0, vmax=max(tab.values.max(), 1) * 1.15)
    vmax = tab.values.max()
    for i in range(len(ordem)):
        for j in range(len(ordem)):
            v = tab.values[i, j]
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=10,
                        color="white" if v > vmax * 0.55 else TINTA,
                        fontweight="bold" if i == j else "normal")
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=LARANJA, lw=1.5))
    ax.set_xticks(range(len(ordem)), ordem, rotation=30, ha="right")
    ax.set_yticks(range(len(ordem)), ordem)
    ax.set_xlabel("Execução 2"); ax.set_ylabel("Execução 1")
    ax.grid(False)
    k, (li, ls) = globais_ic["gabarito_agente"]
    ax.set_title(f"Resposta da resolução cega: execução 1 × 2\n"
                 f"concordância = {np.trace(tab.values) / tab.values.sum():.1%}; "
                 f"κ = {k:.3f} (IC 95% {li:.3f}–{ls:.3f})", fontsize=10)
    _salvar(fig, fig_dir, "fig4_contingencia_gabarito_agente"); plt.close(fig)

    # ---------- Fig. 5: McNemar para confere_gabarito
    c1, c2 = e1["confere_gabarito"].values.astype(int), e2["confere_gabarito"].values.astype(int)
    m = np.array([[np.sum((c1 == 1) & (c2 == 1)), np.sum((c1 == 1) & (c2 == 0))],
                  [np.sum((c1 == 0) & (c2 == 1)), np.sum((c1 == 0) & (c2 == 0))]])
    _, _, p = mcnemar(c1, c2)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() * 1.15)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, m[i, j], ha="center", va="center", fontsize=13,
                    color="white" if m[i, j] > m.max() * 0.55 else TINTA)
    for (i, j) in [(0, 1), (1, 0)]:  # células discordantes (usadas no teste)
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=LARANJA, lw=2))
    ax.set_xticks([0, 1], ["Confere", "Não confere"]); ax.set_yticks([0, 1], ["Confere", "Não confere"])
    ax.set_xlabel("Execução 2"); ax.set_ylabel("Execução 1"); ax.grid(False)
    ax.set_title(f"Gabarito oficial confirmado pela resolução cega\n"
                 f"McNemar exato: p = {p:.4f} (células em destaque)", fontsize=10)
    _salvar(fig, fig_dir, "fig5_mcnemar_confere_gabarito"); plt.close(fig)

    # ---------- Fig. 6: resumo dos kappas globais
    nomes = {"diretrizes_micro": "Diretrizes (todas as células)",
             "gabarito_agente": "Resposta da resolução cega",
             "confere_gabarito": "Confere com o gabarito oficial",
             "bloom": "Nível de Bloom"}
    itens = [(nomes[k_], *globais_ic[k_]) for k_ in nomes if not np.isnan(globais_ic[k_][0])]
    itens.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(itens) + 1.4))
    _faixas_landis_koch(ax)
    for i, (rot, k, (li, ls)) in enumerate(itens):
        if not np.isnan(li):
            ax.hlines(i, li, ls, color=AZUL, lw=2, zorder=2)
        ax.plot(k, i, "o", ms=9, color=AZUL, mec="white", mew=2, zorder=3)
        ax.text(1.07, i, f"{k:.2f}", va="center", fontsize=9, color=TINTA,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(itens)), [t[0] for t in itens])
    ax.set_xlim(-0.2, 1.05); ax.grid(axis="y", visible=False)
    ax.set_xlabel("κ de Cohen (IC 95%, bootstrap)")
    ax.set_title("Reprodutibilidade global entre as execuções", pad=18)
    _salvar(fig, fig_dir, "fig6_kappas_globais"); plt.close(fig)



# ----------------------------------------------------------------- principal
def analisar_conjunto(caminho_csv, pasta_saida, nome="conjunto"):
    """Consistência intra-conjunto: execução 1 x execução 2 do MESMO conjunto."""
    out = Path(pasta_saida)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(caminho_csv)
    # resposta vazia na resolução cega = nenhuma alternativa correta
    df["gabarito_agente"] = df["gabarito_agente"].fillna("NENHUM").astype(str)
    df["bloom_norm"] = df["nivel_bloom"].map(normaliza_bloom)
    df["dir_set"] = df["diretrizes_violadas"].map(conjunto_diretrizes)
    cols_g = [c for c in df.columns if c.lower().startswith("g") and c[1:2].isdigit()]

    e1 = df[df["execucao"] == 1].set_index("id_questao")
    e2 = df[df["execucao"] == 2].set_index("id_questao")
    ids = [q for q in e1.index if q in e2.index]
    ids.sort(key=lambda q: (int("".join(ch for ch in str(q) if ch.isdigit()) or 0), str(q)))
    e1, e2 = e1.loc[ids], e2.loc[ids]
    n = len(ids)

    # ---------------- comparação por questão (Q1 x Q1)
    linhas = []
    for q in ids:
        a, b = e1.loc[q], e2.loc[q]
        s1, s2 = a["dir_set"], b["dir_set"]
        linhas.append({
            "id_questao": q,
            "jaccard_diretrizes": round(jaccard(s1, s2), 4),
            "hamming_g": int(np.sum(a[cols_g].values != b[cols_g].values)),
            "so_exec1": ", ".join(sorted(s1 - s2)),
            "so_exec2": ", ".join(sorted(s2 - s1)),
            "em_ambas": ", ".join(sorted(s1 & s2)),
            "gabarito_oficial": a["gabarito_oficial"],
            "agente_exec1": a["gabarito_agente"],
            "agente_exec2": b["gabarito_agente"],
            "agente_concorda": int(a["gabarito_agente"] == b["gabarito_agente"]),
            "confere_exec1": int(a["confere_gabarito"]),
            "confere_exec2": int(b["confere_gabarito"]),
            "confere_concorda": int(a["confere_gabarito"] == b["confere_gabarito"]),
            "plaus_exec1": a["plausibilidade_media"],
            "plaus_exec2": b["plausibilidade_media"],
            "delta_plaus": round(b["plausibilidade_media"] - a["plausibilidade_media"], 4),
            "bloom_concorda": int(a["bloom_norm"] == b["bloom_norm"]),
        })
    por_q = pd.DataFrame(linhas)
    # 'estavel' = mesma decisão de gabarito e mesmas diretrizes
    por_q["totalmente_estavel"] = (
        (por_q["jaccard_diretrizes"] == 1) & (por_q["agente_concorda"] == 1)
    ).astype(int)
    por_q.to_csv(out / "comparacao_por_questao.csv", index=False)

    # ---------------- concordância por diretriz (kappa)
    linhas_g = []
    for g in cols_g:
        r1, r2 = e1[g].values.astype(int), e2[g].values.astype(int)
        if r1.sum() == 0 and r2.sum() == 0:
            continue  # diretriz nunca acionada em nenhuma execução
        k = cohen_kappa(r1, r2)
        ic_inf, ic_sup = bootstrap_ic(cohen_kappa, r1, r2)
        n01, n10, p = mcnemar(r1, r2)
        linhas_g.append({
            "diretriz": g.upper(),
            "freq_exec1": int(r1.sum()),
            "freq_exec2": int(r2.sum()),
            "concordancia_obs": round(np.mean(r1 == r2), 4),
            "kappa": round(k, 4) if not np.isnan(k) else np.nan,
            "kappa_ic_inf": round(ic_inf, 4),
            "kappa_ic_sup": round(ic_sup, 4),
            "interpretacao": interpreta_kappa(k),
            "pabak": round(pabak(r1, r2), 4),
            "so_exec2(0->1)": n01,
            "so_exec1(1->0)": n10,
            "mcnemar_p": round(p, 4),
        })
    por_g = pd.DataFrame(linhas_g).sort_values("kappa", na_position="last")
    por_g.to_csv(out / "concordancia_por_diretriz.csv", index=False)

    # ---------------- métricas globais
    k_ag = cohen_kappa(e1["gabarito_agente"], e2["gabarito_agente"])
    k_conf = cohen_kappa(e1["confere_gabarito"], e2["confere_gabarito"])
    k_bloom = cohen_kappa(e1["bloom_norm"], e2["bloom_norm"])
    n01, n10, p_mc = mcnemar(e1["confere_gabarito"], e2["confere_gabarito"])
    p1, p2 = e1["plausibilidade_media"].values, e2["plausibilidade_media"].values
    try:
        w = stats.wilcoxon(p1, p2, zero_method="wilcox")
        w_stat, w_p = w.statistic, w.pvalue
    except ValueError:
        w_stat, w_p = np.nan, 1.0
    icc = icc_3_1(p1, p2)

    # kappa "micro": todas as células questão×diretriz empilhadas
    m1 = e1[cols_g].values.ravel().astype(int)
    m2 = e2[cols_g].values.ravel().astype(int)

    jac = por_q["jaccard_diretrizes"]
    glob = [
        ("Questões pareadas", n, ""),
        ("Jaccard médio das diretrizes", round(jac.mean(), 4), "1 = mesmas diretrizes"),
        ("Jaccard mediano", round(jac.median(), 4), ""),
        ("Jaccard desvio-padrão", round(jac.std(ddof=1), 4), ""),
        ("Questões com diretrizes idênticas", f"{(jac == 1).sum()}/{n}", ""),
        ("Hamming médio (diretrizes alteradas por questão)", round(por_q["hamming_g"].mean(), 4), ""),
        ("Kappa diretrizes (micro, todas as células)", round(cohen_kappa(m1, m2), 4),
         interpreta_kappa(cohen_kappa(m1, m2))),
        ("Kappa médio por diretriz (macro)", round(por_g["kappa"].mean(), 4),
         interpreta_kappa(por_g["kappa"].mean())),
        ("Concordância gabarito_agente", round(por_q["agente_concorda"].mean(), 4), ""),
        ("Kappa gabarito_agente", round(k_ag, 4), interpreta_kappa(k_ag)),
        ("Taxa confere_gabarito exec1", round(e1["confere_gabarito"].mean(), 4), ""),
        ("Taxa confere_gabarito exec2", round(e2["confere_gabarito"].mean(), 4), ""),
        ("Kappa confere_gabarito", round(k_conf, 4), interpreta_kappa(k_conf)),
        ("McNemar confere (0->1 / 1->0)", f"{n01} / {n10}", f"p = {p_mc:.4f}"),
        ("Kappa nível de Bloom", round(k_bloom, 4) if not np.isnan(k_bloom) else "NaN",
         interpreta_kappa(k_bloom)),
        ("Plausibilidade média exec1", round(p1.mean(), 4), ""),
        ("Plausibilidade média exec2", round(p2.mean(), 4), ""),
        ("Wilcoxon plausibilidade", round(w_stat, 4) if not np.isnan(w_stat) else "NaN",
         f"p = {w_p:.4f}"),
        ("ICC(3,1) plausibilidade", round(icc, 4), ""),
        ("Questões totalmente estáveis", f"{por_q['totalmente_estavel'].sum()}/{n}",
         "mesmas diretrizes e mesma resposta"),
    ]
    resumo = pd.DataFrame(glob, columns=["metrica", "valor", "observacao"])
    resumo.to_csv(out / "resumo_global.csv", index=False)

    # ---------------- IC 95% dos kappas globais + gráficos
    pares = {
        "diretrizes_micro": (m1, m2),
        "gabarito_agente": (e1["gabarito_agente"].values, e2["gabarito_agente"].values),
        "confere_gabarito": (e1["confere_gabarito"].values, e2["confere_gabarito"].values),
        "bloom": (e1["bloom_norm"].values, e2["bloom_norm"].values),
    }
    globais_ic = {}
    for chave, (a, b) in pares.items():
        if chave == "diretrizes_micro":
            # reamostra QUESTÕES (blocos de diretrizes), não células isoladas
            A = e1[cols_g].values.astype(int); B = e2[cols_g].values.astype(int)
            rng = np.random.default_rng(42); vals = []
            for _ in range(5000):
                idx = rng.integers(0, n, n)
                vals.append(cohen_kappa(A[idx].ravel(), B[idx].ravel()))
            vals = np.array(vals); vals = vals[~np.isnan(vals)]
            ic = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        else:
            ic = bootstrap_ic(cohen_kappa, a, b)
        globais_ic[chave] = (cohen_kappa(a, b), ic)
    pd.DataFrame([{"variavel": k_, "kappa": round(v[0], 4), "ic95_inf": round(v[1][0], 4),
                   "ic95_sup": round(v[1][1], 4)} for k_, v in globais_ic.items()]
                 ).to_csv(out / "kappas_globais_ic95.csv", index=False)
    gerar_graficos(e1, e2, por_q, por_g, globais_ic, cols_g, out)

    print(f"  [{nome}] {n} questões | κ diretrizes = {globais_ic['diretrizes_micro'][0]:.3f} | "
          f"Jaccard médio = {jac.mean():.3f} | κ resposta = {k_ag:.3f}")
    return {"nome": nome, "n": n, "e1": e1, "e2": e2, "cols_g": cols_g, "por_q": por_q,
            "por_g": por_g, "globais_ic": globais_ic}


# ================================================================
#  COMPARAÇÃO ENTRE CONJUNTOS
# ================================================================
CORES_CONJ = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def holm(pvals):
    """Correção de Holm-Bonferroni (controla o FWER)."""
    p = np.asarray(pvals, float)
    ordem = np.argsort(p)
    m = len(p)
    aj = np.empty(m)
    acum = 0.0
    for i, idx in enumerate(ordem):
        acum = max(acum, (m - i) * p[idx])
        aj[idx] = min(acum, 1.0)
    return aj


def _tabela(lab, val, nl, nv):
    return np.bincount(lab * nv + val, minlength=nl * nv).reshape(nl, nv)


def _chi2_stat(t):
    t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 0.0
    esp = t.sum(axis=1, keepdims=True) * t.sum(axis=0, keepdims=True) / t.sum()
    return float(((t - esp) ** 2 / esp).sum())


def qui2_permutacao(rotulos, valores, n_perm=10000, seed=42):
    """Qui-quadrado de independência com p-valor por permutação (Monte Carlo).
    Válido mesmo com frequências esperadas < 5, onde o qui-quadrado assintótico falha."""
    lab = pd.factorize(pd.Series(rotulos))[0]
    val = pd.factorize(pd.Series(valores))[0]
    nl, nv = lab.max() + 1, val.max() + 1
    t = _tabela(lab, val, nl, nv)
    obs = _chi2_stat(t)
    rng = np.random.default_rng(seed)
    maior = sum(_chi2_stat(_tabela(rng.permutation(lab), val, nl, nv)) >= obs - 1e-12
                for _ in range(n_perm))
    p = (maior + 1) / (n_perm + 1)
    n = t.sum()
    k = min(t[t.sum(axis=1) > 0].shape[0], t[:, t.sum(axis=0) > 0].shape[1])
    v = np.sqrt(obs / (n * (k - 1))) if k > 1 else np.nan  # V de Cramér
    return obs, p, v, t


def residuos_ajustados(t):
    """Resíduos padronizados ajustados de Haberman: |r| > 1.96 → célula difere (α = 0.05)."""
    t = np.asarray(t, float)
    n = t.sum()
    lin, col = t.sum(axis=1, keepdims=True), t.sum(axis=0, keepdims=True)
    esp = lin * col / n
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (t - esp) / np.sqrt(esp * (1 - lin / n) * (1 - col / n))
    return np.nan_to_num(r)


def _bootstrap_media(x, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _kruskal_pares(grupos: dict, nome_var: str):
    """Kruskal-Wallis + ε² e Mann-Whitney par a par com Holm."""
    nomes = list(grupos)
    vals = [np.asarray(grupos[k], float) for k in nomes]
    try:
        H, p = stats.kruskal(*vals)
    except ValueError:  # todos os valores iguais
        H, p = 0.0, 1.0
    if np.isnan(H):
        H, p = 0.0, 1.0
    n = sum(len(v) for v in vals)
    eps2 = H / ((n ** 2 - 1) / (n + 1))
    linhas = [{"variavel": nome_var, "teste": "Kruskal-Wallis", "comparacao": " x ".join(nomes),
               "estatistica": round(H, 4), "p": round(p, 4), "efeito": round(eps2, 4),
               "medida_efeito": "epsilon²"}]
    pares, ps = [], []
    for i in range(len(nomes)):
        for j in range(i + 1, len(nomes)):
            try:
                U, pp = stats.mannwhitneyu(vals[i], vals[j], alternative="two-sided")
            except ValueError:
                U, pp = np.nan, 1.0
            r = 1 - 2 * U / (len(vals[i]) * len(vals[j]))  # correlação rank-bisserial
            pares.append((nomes[i], nomes[j], U, r)); ps.append(pp)
    for (a, b, U, r), pa in zip(pares, holm(ps) if ps else []):
        linhas.append({"variavel": nome_var, "teste": "Mann-Whitney (Holm)", "comparacao": f"{a} x {b}",
                       "estatistica": round(U, 4), "p": round(pa, 4), "efeito": round(r, 4),
                       "medida_efeito": "rank-bisserial"})
    return pd.DataFrame(linhas)


def comparar_conjuntos(res: list, out: Path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap
    _estilo()
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "graficos"
    fig_dir.mkdir(exist_ok=True)
    nomes = [r["nome"] for r in res]
    cor = {nm: CORES_CONJ[i % len(CORES_CONJ)] for i, nm in enumerate(nomes)}
    cols = sorted({c for r in res for c in r["cols_g"]},
                  key=lambda c: (int("".join(ch for ch in c if ch.isdigit()) or 0), c))

    # ---- base por questão: diretriz em CONSENSO (apontada nas duas execuções)
    #      e em UNIÃO (em pelo menos uma) — a união serve de análise de sensibilidade
    blocos = []
    for r in res:
        e1, e2 = r["e1"].copy(), r["e2"].copy()
        for c in cols:
            if c not in e1:
                e1[c] = 0
            if c not in e2:
                e2[c] = 0
        A, B = e1[cols].values.astype(int), e2[cols].values.astype(int)
        d = pd.DataFrame(A & B, columns=cols, index=e1.index).add_prefix("cons_")
        d = d.join(pd.DataFrame(A | B, columns=cols, index=e1.index).add_prefix("uniao_"))
        d["conjunto"] = r["nome"]
        d["bloom"] = e1["bloom_norm"].values
        d["n_diretrizes"] = (A & B).sum(axis=1)
        blocos.append(d.reset_index())
    base = pd.concat(blocos, ignore_index=True)
    base.to_csv(out / "base_por_questao.csv", index=False)

    # ================= 1. DIRETRIZES: prevalência por conjunto
    linhas = []
    for c in cols:
        if base[f"uniao_{c}"].sum() == 0:
            continue
        lin = {"diretriz": c.upper()}
        for nm in nomes:
            sub = base[base["conjunto"] == nm]
            lin[f"{nm}_n"] = int(sub[f"cons_{c}"].sum())
            lin[f"{nm}_%"] = round(100 * sub[f"cons_{c}"].mean(), 2)
        chi, p, v, _ = qui2_permutacao(base["conjunto"], base[f"cons_{c}"])
        _, p_u, _, _ = qui2_permutacao(base["conjunto"], base[f"uniao_{c}"])
        lin.update({"qui2": round(chi, 4), "p_perm": p,
                    "V_cramer": round(v, 4) if not np.isnan(v) else np.nan, "p_perm_uniao": p_u})
        linhas.append(lin)
    prev = pd.DataFrame(linhas)
    prev["p_holm"] = holm(prev["p_perm"]).round(4)
    prev["p_holm_uniao"] = holm(prev["p_perm_uniao"]).round(4)
    prev["significativo"] = np.where(prev["p_holm"] < 0.05, "sim", "não")
    prev["p_perm"] = prev["p_perm"].round(4)
    prev["p_perm_uniao"] = prev["p_perm_uniao"].round(4)
    prev.to_csv(out / "diretrizes_prevalencia_teste.csv", index=False)

    # pós-teste par a par (Fisher exato + Holm) para as diretrizes significativas
    pos = []
    for _, lin in prev[prev["significativo"] == "sim"].iterrows():
        c = lin["diretriz"].lower()
        pares, ps = [], []
        for i in range(len(nomes)):
            for j in range(i + 1, len(nomes)):
                a = base[base["conjunto"] == nomes[i]][f"cons_{c}"]
                b = base[base["conjunto"] == nomes[j]][f"cons_{c}"]
                t = [[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]]
                orr, pp = stats.fisher_exact(t)
                pares.append((nomes[i], nomes[j], orr)); ps.append(pp)
        for (x, y, orr), pa in zip(pares, holm(ps)):
            pos.append({"diretriz": lin["diretriz"], "comparacao": f"{x} x {y}",
                        "odds_ratio": round(orr, 4), "p_holm": round(pa, 4)})
    pd.DataFrame(pos, columns=["diretriz", "comparacao", "odds_ratio", "p_holm"]).to_csv(
        out / "diretrizes_pos_teste_pares.csv", index=False)

    testes = [_kruskal_pares({nm: base[base["conjunto"] == nm]["n_diretrizes"] for nm in nomes},
                             "nº de diretrizes violadas por questão (consenso)")]

    # ================= 2. TAXONOMIA DE BLOOM
    niveis = [b for b in ORDEM_BLOOM if b in set(base["bloom"])] + \
             sorted(set(base["bloom"]) - set(ORDEM_BLOOM))
    tab_b = pd.crosstab(pd.Categorical(base["conjunto"], nomes),
                        pd.Categorical(base["bloom"], niveis), dropna=False)
    tab_b.index, tab_b.columns = nomes, niveis
    chi_b, p_b, v_b, _ = qui2_permutacao(base["conjunto"], base["bloom"])
    res_b = pd.DataFrame(residuos_ajustados(tab_b.values), index=nomes, columns=niveis)
    pct_b = (tab_b.div(tab_b.sum(axis=1), axis=0) * 100).round(2)
    tab_b.to_csv(out / "bloom_contagem.csv")
    pct_b.to_csv(out / "bloom_percentual.csv")
    res_b.round(3).to_csv(out / "bloom_residuos_ajustados.csv")
    ordinal = base["bloom"].map({b: i + 1 for i, b in enumerate(ORDEM_BLOOM)})
    if ordinal.notna().all():
        testes.append(_kruskal_pares({nm: ordinal[base["conjunto"] == nm] for nm in nomes},
                                     "nível de Bloom (ordinal)"))
    testes.append(pd.DataFrame([{"variavel": "nível de Bloom (categórico)",
                                 "teste": "Qui-quadrado (permutação)", "comparacao": " x ".join(nomes),
                                 "estatistica": round(chi_b, 4), "p": round(p_b, 4),
                                 "efeito": round(v_b, 4) if not np.isnan(v_b) else np.nan,
                                 "medida_efeito": "V de Cramér"}]))

    # ================= 3. DIRETRIZ × BLOOM (por conjunto, consenso)
    db = []
    for nm in nomes:
        sub = base[base["conjunto"] == nm]
        for b in niveis:
            sb = sub[sub["bloom"] == b]
            for c in prev["diretriz"].str.lower():
                db.append({"conjunto": nm, "bloom": b, "diretriz": c.upper(), "n_questoes": len(sb),
                           "n_violacoes": int(sb[f"cons_{c}"].sum()),
                           "%": round(100 * sb[f"cons_{c}"].mean(), 2) if len(sb) else np.nan})
    pd.DataFrame(db).to_csv(out / "diretriz_por_bloom.csv", index=False)

    # ================= 4. CONSISTÊNCIA intra-conjunto comparada entre conjuntos
    cons = []
    for r in res:
        jac = r["por_q"]["jaccard_diretrizes"]
        lj, uj = _bootstrap_media(jac)
        for chave, rot in [("diretrizes_micro", "κ diretrizes"), ("gabarito_agente", "κ resposta cega"),
                           ("confere_gabarito", "κ confere gabarito"), ("bloom", "κ nível de Bloom")]:
            k, (li, ls) = r["globais_ic"][chave]
            cons.append({"conjunto": r["nome"], "metrica": rot, "valor": round(k, 4),
                         "ic95_inf": round(li, 4), "ic95_sup": round(ls, 4)})
        cons.append({"conjunto": r["nome"], "metrica": "Jaccard médio", "valor": round(jac.mean(), 4),
                     "ic95_inf": round(lj, 4), "ic95_sup": round(uj, 4)})
    cons = pd.DataFrame(cons)
    cons.to_csv(out / "consistencia_por_conjunto.csv", index=False)
    testes.append(_kruskal_pares({r["nome"]: r["por_q"]["jaccard_diretrizes"] for r in res},
                                 "Jaccard intra-conjunto (exec1 x exec2)"))
    kg = pd.concat([r["por_g"][["diretriz", "freq_exec1", "freq_exec2", "kappa", "kappa_ic_inf",
                                 "kappa_ic_sup", "pabak"]]
                    .assign(conjunto=r["nome"]) for r in res], ignore_index=True)
    kg.to_csv(out / "kappa_por_diretriz_por_conjunto.csv", index=False)
    testes = pd.concat(testes, ignore_index=True)
    testes.to_csv(out / "testes_entre_conjuntos.csv", index=False)

    # ---------------------------------------------------------------- gráficos
    def legenda_conj(ax, y=-0.14):
        ax.legend(handles=[Line2D([], [], color=cor[nm], marker="o", lw=0, ms=7, label=nm)
                           for nm in nomes], loc="upper center", bbox_to_anchor=(0.5, y),
                  ncol=len(nomes))

    desloc = {nm: (i - (len(nomes) - 1) / 2) * 0.22 for i, nm in enumerate(nomes)}
    rng = np.random.default_rng(0)

    # Fig. A — prevalência de cada diretriz por conjunto (dot plot de Cleveland)
    d = prev.copy()
    d["_max"] = d[[f"{nm}_%" for nm in nomes]].max(axis=1)
    d = d[d["_max"] > 0].sort_values("_max").reset_index(drop=True)  # oculta diretrizes sem consenso
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(d) + 2))
    y = np.arange(len(d))
    for i in y:
        vals = [d.loc[i, f"{nm}_%"] for nm in nomes]
        ax.hlines(i, min(vals), max(vals), color="#d6d5d0", lw=1.2, zorder=1)
    for nm in nomes:
        ax.plot(d[f"{nm}_%"], y + desloc[nm] * 0.6, "o", ms=8, color=cor[nm],
                mec="white", mew=1.5, zorder=3)
    rot = [f"{g} *" if s == "sim" else g for g, s in zip(d["diretriz"], d["significativo"])]
    ax.set_yticks(y, rot)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-2, 102)
    ax.set_xlabel("Questões com a diretriz violada (%, consenso das 2 execuções)")
    ax.set_title("Prevalência das violações por diretriz em cada conjunto")
    ax.legend(handles=[Line2D([], [], color=cor[nm], marker="o", lw=0, ms=7, label=nm) for nm in nomes],
              loc="lower right", frameon=True, facecolor="white", edgecolor="#e6e5e0")
    _nota(ax, "* diferença significativa entre conjuntos (qui-quadrado por permutação, "
          "correção de Holm, α = 0.05). Diretrizes sem violação em consenso omitidas (ver CSV).")
    _salvar(fig, fig_dir, "figA_prevalencia_diretrizes"); plt.close(fig)

    # Fig. B — distribuição dos níveis de Bloom por conjunto (barra 100% empilhada, rampa sequencial)
    rampa = LinearSegmentedColormap.from_list("seq", ["#cfe0f6", "#2a78d6", "#0b3f80"])
    tons = [rampa(x) for x in np.linspace(0, 1, len(niveis))] if len(niveis) > 1 else [rampa(0.6)]
    fig, ax = plt.subplots(figsize=(7.5, 0.7 * len(nomes) + 2))
    for i, nm in enumerate(nomes):
        esq = 0
        for b, tom in zip(niveis, tons):
            v = pct_b.loc[nm, b]
            if v <= 0:
                continue
            ax.barh(i, v, left=esq, color=tom, edgecolor="white", linewidth=2, height=0.6)
            if v >= 8:
                lum = 0.299 * tom[0] + 0.587 * tom[1] + 0.114 * tom[2]
                ax.text(esq + v / 2, i, f"{v:.0f}%\n(n={tab_b.loc[nm, b]})", ha="center",
                        va="center", fontsize=7.5, color="white" if lum < 0.55 else TINTA)
            esq += v
    ax.set_yticks(range(len(nomes)), [f"{nm} (n={tab_b.loc[nm].sum()})" for nm in nomes])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.grid(False)
    ax.set_xlabel("Questões (%)")
    ax.set_title(f"Níveis da taxonomia de Bloom por conjunto\n"
                 f"χ² = {chi_b:.2f}, p (permutação) = {p_b:.4f}, V de Cramér = {v_b:.3f}", fontsize=10)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=t) for t in tons],
              labels=[b.capitalize() for b in niveis], loc="upper center",
              bbox_to_anchor=(0.5, -0.25), ncol=len(niveis))
    _salvar(fig, fig_dir, "figB_bloom_por_conjunto"); plt.close(fig)

    # Fig. C — resíduos ajustados conjunto × Bloom (divergente, meio neutro)
    div = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f2f1ed", "#2a78d6"])
    lim = max(2.5, float(np.abs(res_b.values).max()))
    fig, ax = plt.subplots(figsize=(1.3 * len(niveis) + 2.5, 0.8 * len(nomes) + 1.8))
    ax.imshow(res_b.values, cmap=div, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(nomes)):
        for j in range(len(niveis)):
            v = res_b.values[i, j]
            sig = abs(v) > 1.96
            ax.text(j, i, f"{v:+.2f}" + (" *" if sig else ""), ha="center", va="center",
                    fontsize=9, color=TINTA, fontweight="bold" if sig else "normal")
    ax.set_xticks(range(len(niveis)), [b.capitalize() for b in niveis])
    ax.set_yticks(range(len(nomes)), nomes)
    ax.grid(False)
    ax.set_title("Resíduos ajustados: conjunto × nível de Bloom", fontsize=10)
    fig.text(0.02, -0.04, "Azul = mais questões que o esperado; laranja = menos. "
             "* |r| > 1.96 (α = 0.05).", fontsize=7.5, color=TINTA2)
    _salvar(fig, fig_dir, "figC_residuos_bloom"); plt.close(fig)

    # Fig. D — nº de diretrizes violadas por questão
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, nm in enumerate(nomes):
        v = base[base["conjunto"] == nm]["n_diretrizes"].values
        ax.boxplot(v, positions=[i], widths=0.45, showfliers=False,
                   boxprops=dict(color=TINTA2), medianprops=dict(color=TINTA, lw=2),
                   whiskerprops=dict(color=TINTA2), capprops=dict(color=TINTA2))
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v + rng.uniform(-0.12, 0.12, len(v)),
                   s=14, color=cor[nm], alpha=0.6, edgecolor="none", zorder=3)
    kw = testes[testes["variavel"].str.startswith("nº de diretrizes")
                & (testes["teste"] == "Kruskal-Wallis")].iloc[0]
    ax.set_xticks(range(len(nomes)), nomes)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Diretrizes violadas por questão (consenso)")
    ax.set_title(f"Carga de violações por questão\nKruskal-Wallis: H = {kw['estatistica']:.2f}, "
                 f"p = {kw['p']:.4f}, ε² = {kw['efeito']:.3f}", fontsize=10)
    ax.text(0, -0.14, "Caixa = mediana e quartis. Pontos com pequeno deslocamento aleatório "
            "para evitar sobreposição.", transform=ax.transAxes, fontsize=7.5, color=TINTA2)
    _salvar(fig, fig_dir, "figD_n_diretrizes_por_questao"); plt.close(fig)

    # Fig. E — consistência intra-conjunto comparada (κ com IC 95%)
    mets = ["κ diretrizes", "κ resposta cega", "κ confere gabarito", "κ nível de Bloom"]
    fig, ax = plt.subplots(figsize=(7.5, 0.8 * len(mets) + 2))
    _faixas_landis_koch(ax)
    for j, m in enumerate(mets):
        for nm in nomes:
            lin = cons[(cons["conjunto"] == nm) & (cons["metrica"] == m)]
            if lin.empty or pd.isna(lin["valor"].iloc[0]):
                continue
            yy = j + desloc[nm]
            ax.hlines(yy, lin["ic95_inf"], lin["ic95_sup"], color=cor[nm], lw=2, zorder=2)
            ax.plot(lin["valor"], yy, "o", ms=7, color=cor[nm], mec="white", mew=1.5, zorder=3)
    ax.set_yticks(range(len(mets)), mets)
    ax.set_ylim(len(mets) - 0.5, -0.5)
    ax.set_xlim(-0.2, 1.05)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("κ de Cohen entre execução 1 e 2 (IC 95%, bootstrap)")
    ax.set_title("Reprodutibilidade de cada conjunto", pad=18)
    legenda_conj(ax, y=-0.16)
    _nota(ax, "Ponto sem barra: concordância perfeita em todas as questões; o IC bootstrap é degenerado ([1,00; 1,00]).\n"
          "IC amplo com κ baixo e concordância observada alta indica prevalência extrema (paradoxo do κ).", dy=-62)
    _salvar(fig, fig_dir, "figE_consistencia_por_conjunto"); plt.close(fig)

    # Fig. F — κ por diretriz, um painel por conjunto (IC 95% bootstrap; n nos rótulos)
    MIN_CASOS = 5  # critério declarado em Materiais e Métodos: apontada em >= 5 questões em alguma execução
    kf = kg.dropna(subset=["kappa"]).copy()
    kf["n_max"] = kf[["freq_exec1", "freq_exec2"]].max(axis=1)
    kf["suficiente"] = kf["n_max"] >= MIN_CASOS
    # ordem: κ médio entre conjuntos, considerando só os casos suficientes (os demais vão ao fim)
    ordem_k = kf[kf["suficiente"]].groupby("diretriz")["kappa"].mean()
    resto = sorted(set(kf["diretriz"]) - set(ordem_k.index), key=lambda g: (len(g), g))
    ordem = list(ordem_k.sort_values().index) + resto
    pos = {g: i for i, g in enumerate(ordem)}
    fig, axs = plt.subplots(1, len(nomes), sharey=True,
                            figsize=(3.1 * len(nomes) + 0.8, 0.42 * len(ordem) + 2.2))
    axs = np.atleast_1d(axs)
    for ax, nm in zip(axs, nomes):
        _faixas_landis_koch(ax, rotulos=False)
        sub = kf[kf["conjunto"] == nm]
        for _, r in sub.iterrows():
            yy = pos[r["diretriz"]]
            cheio = bool(r["suficiente"])
            ax.hlines(yy, r["kappa_ic_inf"], r["kappa_ic_sup"], color=cor[nm],
                      lw=2 if cheio else 1, alpha=1 if cheio else 0.45, zorder=2)
            ax.plot(r["kappa"], yy, "o", ms=7, mew=1.5, zorder=3,
                    color=cor[nm], mfc=cor[nm] if cheio else "white", mec=cor[nm] if not cheio else "white")
            ax.annotate(f"{int(r['freq_exec1'])}/{int(r['freq_exec2'])}", (1.05, yy),
                        xycoords=("data", "data"), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7, color=TINTA2, annotation_clip=False)
        ax.axvline(0, color=CINZA, lw=0.8)
        ax.set_xlim(-0.2, 1.05)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(axis="y", visible=False)
        ax.set_title(nm, color=cor[nm], loc="center", pad=8)
        if "n₁/n₂" not in [t.get_text() for t in ax.texts]:
            ax.text(1.05, -0.9, "n₁/n₂", fontsize=7, color=TINTA2, ha="left", va="bottom",
                    transform=ax.transData, clip_on=False)
    axs[0].set_yticks(range(len(ordem)), ordem)
    axs[0].set_ylim(len(ordem) - 0.5, -1.2)
    fig.supxlabel("κ de Cohen entre execução 1 e 2 (IC 95%, bootstrap)", fontsize=10, y=0.06)
    fig.suptitle("Estabilidade de cada diretriz, por conjunto", x=0.02, ha="left",
                 fontsize=11, fontweight="bold")
    fig.text(0.02, 0.0,
             f"Símbolo vazado: diretriz apontada em menos de {MIN_CASOS} questões nas duas execuções (κ instável). "
             "n₁/n₂ = nº de questões em que a diretriz foi apontada na execução 1 e 2.\n"
             "Faixas sombreadas segundo Landis e Koch (1977): < 0 pobre; 0–0,20 leve; 0,20–0,40 razoável; "
             "0,40–0,60 moderada; 0,60–0,80 substancial; 0,80–1,00 quase perfeita.\n"
             "Ponto ausente: diretriz nunca apontada naquele conjunto. Ponto sem barra: IC degenerado "
             "(mesma decisão em todas as reamostragens).",
             fontsize=7.5, color=TINTA2, va="bottom")
    fig.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.16, wspace=0.22)
    _salvar(fig, fig_dir, "figF_kappa_diretriz_por_conjunto"); plt.close(fig)

    # Fig. G — Jaccard intra-conjunto
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, r in enumerate(res):
        v = r["por_q"]["jaccard_diretrizes"].values
        ax.boxplot(v, positions=[i], widths=0.45, showfliers=False,
                   boxprops=dict(color=TINTA2), medianprops=dict(color=TINTA, lw=2),
                   whiskerprops=dict(color=TINTA2), capprops=dict(color=TINTA2))
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=14, color=cor[r["nome"]],
                   alpha=0.6, edgecolor="none", zorder=3)
        ax.plot(i, v.mean(), "D", ms=7, color=TINTA, zorder=4)
    kw = testes[testes["variavel"].str.startswith("Jaccard")
                & (testes["teste"] == "Kruskal-Wallis")].iloc[0]
    ax.set_xticks(range(len(nomes)), nomes)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Jaccard das diretrizes (exec. 1 × 2)")
    ax.set_title(f"Similaridade entre execuções, por conjunto\nKruskal-Wallis: "
                 f"H = {kw['estatistica']:.2f}, p = {kw['p']:.4f}", fontsize=10)
    ax.text(0, -0.14, "◆ = média. Caixa = mediana e quartis.", transform=ax.transAxes,
            fontsize=7.5, color=TINTA2)
    _salvar(fig, fig_dir, "figG_jaccard_por_conjunto"); plt.close(fig)

    print(f"  Bloom: χ² = {chi_b:.2f}, p = {p_b:.4f}, V de Cramér = {v_b:.3f}")
    print("  Diretrizes com diferença significativa entre conjuntos (Holm): "
          f"{', '.join(prev.loc[prev['significativo'] == 'sim', 'diretriz']) or 'nenhuma'}")


def nome_conjunto(pasta: str) -> str:
    import re
    limpo = re.sub(r"[\s_-]*FINAL$", "", pasta.strip(), flags=re.IGNORECASE).strip()
    return limpo or pasta


def encontrar_conjuntos(base: Path):
    """Cada subpasta com um *relatorio_auditoria*.csv vira um conjunto."""
    achados = []
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        if sub.name.lower().startswith("resultados") or sub.name.startswith("."):
            continue
        csvs = [c for c in sorted(sub.rglob("*relatorio_auditoria*.csv"))
                if not any(x.lower().startswith("resultados") for x in c.relative_to(sub).parts)]
        if csvs:
            if len(csvs) > 1:
                print(f"Aviso: {sub.name} tem {len(csvs)} relatórios; usando {csvs[0].name}")
            achados.append((nome_conjunto(sub.name), csvs[0]))
    return achados


if __name__ == "__main__":
    base = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else Path(__file__).resolve().parent
    saida = base / "resultados"
    conjuntos = encontrar_conjuntos(base)

    if not conjuntos:  # nenhuma subpasta com relatório: analisa o CSV da própria pasta
        csvs = sorted(base.glob("*relatorio_auditoria*.csv"))
        if not csvs:
            sys.exit(f"Nenhum relatorio_auditoria*.csv encontrado em {base} nem nas subpastas.")
        conjuntos = [(base.name, csvs[0])]

    print("Conjuntos encontrados:")
    for nm, c in conjuntos:
        print(f"  - {nm}: {c}")

    print("\n1) Consistência intra-conjunto (execução 1 x 2)")
    resultados = [analisar_conjunto(c, saida / "1_consistencia_intra_conjunto" / nm, nm)
                  for nm, c in conjuntos]

    if len(resultados) >= 2:
        print("\n2) Comparação entre conjuntos")
        comparar_conjuntos(resultados, saida / "2_comparacao_entre_conjuntos")
    print(f"\nTudo salvo em: {saida}")