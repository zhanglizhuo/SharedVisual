"""Generate publication-quality figures for SharedVisual paper.

Style: IEEE Access / llm-annotation visual language — serif fonts, semantic
colour palette, thin axes, no top/right spines, subtle gridlines.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────
PAPER_ROOT = Path(__file__).resolve().parent
OUT = PAPER_ROOT / 'figures'
RESULTS = PAPER_ROOT.parent / 'results'
os.makedirs(OUT, exist_ok=True)

# ── Unified palette: single accent (blue) + grays ────────────────────────
PRIMARY_BLUE  = "#376092"   # the paper's single accent
NEUTRAL_FILL  = "#B7B7B7"   # references, controls, secondary elements
MID_GRAY      = "#6B7280"   # neutral hypothesis boxes
LINE_DARK     = "#1F2937"   # regression / emphasis lines
TEXT_COLOR     = "#1F2937"
EDGE_COLOR     = "#374151"
GRID_COLOR     = "#E5E7EB"
BACKGROUND     = "#FFFFFF"
HEATMAP_CMAP   = "Blues_r"  # single-hue sequential, grayscale-legible

# ── Global style ─────────────────────────────────────────────────────────
def configure_paper_figure(font_size=8.0):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['STIX Two Text', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': font_size,
        'text.color': TEXT_COLOR,
        'axes.labelcolor': TEXT_COLOR,
        'xtick.color': TEXT_COLOR,
        'ytick.color': TEXT_COLOR,
        'axes.facecolor': BACKGROUND,
        'figure.facecolor': BACKGROUND,
        'axes.edgecolor': EDGE_COLOR,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

configure_paper_figure()

# ── Utilities ─────────────────────────────────────────────────────────────
def load_json(name):
    p = os.path.join(RESULTS, name)
    with open(p) as f:
        return json.load(f)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f'{name}.pdf')
    plt.close(fig)


def lighten(hex_color: str, blend: float = 0.28) -> tuple:
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(c * (1.0 - blend) + blend for c in rgb)


def text_color_for(val, vmin, vmax, cmap_name=HEATMAP_CMAP):
    """Black/white annotation color by perceptual luminance of the cell color."""
    cmap = plt.get_cmap(cmap_name)
    frac = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    r, g, b, _ = cmap(frac)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'black' if luminance > 0.5 else 'white'


def style_axis(ax, grid_axis: str | None = 'y', tick_size: float = 8.0):
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.5,
                alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine_name in ['left', 'bottom']:
        ax.spines[spine_name].set_color(EDGE_COLOR)
        ax.spines[spine_name].set_linewidth(0.6)
    ax.tick_params(labelsize=tick_size, colors=TEXT_COLOR, width=0.6, length=3.0)


# ── Figure 1: Causal Chain ───────────────────────────────────────────────
def fig1_phenomenon_overview():
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(6.5, 3.85))
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, 3.85)
    ax.axis('off')

    BOX_COLOR = PRIMARY_BLUE
    ARROW_COLOR = '#555555'
    EVIDENCE_COLOR = '#555555'

    def draw_box(x, y, w, h, text, fontsize=8.5, color=BOX_COLOR):
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                             boxstyle="round,pad=0.06",
                             facecolor=color, edgecolor='white', linewidth=1.0,
                             alpha=0.85, zorder=5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color='white', zorder=6)

    def draw_arrow(x1, x2, y):
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                    lw=1.8), zorder=4)

    def draw_attack(x, y_bottom, y_top, color):
        ax.annotate('', xy=(x, y_top), xytext=(x, y_bottom),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.3,
                                    linestyle='dashed'), zorder=2)

    # ── Main chain (4 boxes) ──
    box_w, box_h = 1.08, 0.62
    margin = 0.08
    first_cx = margin + box_w / 2
    last_cx = 6.5 - margin - box_w / 2
    spacing = (last_cx - first_cx) / 3
    centers_x = [first_cx + i * spacing for i in range(4)]
    y_main = 2.95

    labels = [
        'Web-Scale\nTraining',
        'Shared Class-\nNeighborhood\nGeometry\n(Level 1)',
        'Shared Visual\nDifficulty\n(Level 2)',
        'CLIP Predicts\nMLLM Difficulty',
    ]
    for cx, lab in zip(centers_x, labels):
        draw_box(cx, y_main, box_w, box_h, lab)

    for i in range(3):
        draw_arrow(centers_x[i] + box_w / 2, centers_x[i + 1] - box_w / 2,
                   y_main)

    # ── Evidence annotations ABOVE the chain ──
    y_ev = y_main + box_h / 2 + 0.10
    evidence = [
        'All on\nweb-scale data',
        'Mantel ρ = 0.90–0.98 (prototype level)\nCKA 0.79–0.87 (feature level)',
        'Mediation (S40):\nindirect CIs exclude 0',
        'SigLIP 0.72 > CLIP 0.46 (n=4)\nρ = 0.82 (S40)',
    ]
    for cx, ev in zip(centers_x, evidence):
        ax.text(cx, y_ev, ev, ha='center', va='bottom', fontsize=8.5,
                color=EVIDENCE_COLOR, style='italic', zorder=3)

    # ── Hypothesis verdicts: one row beneath the chain, vertically aligned ──
    hyps = [
        (centers_x[0], '(H2) Shared Data\n(WEAKENED)', MID_GRAY, True),
        (centers_x[1], '(H3) Shared Geometry\n(SUPPORTED — ours)', BOX_COLOR,
         False),
        (centers_x[2], '(H4) Image Difficulty\n(CONSTRAINED)', MID_GRAY,
         True),
        (centers_x[3], '(H1) Shared Encoder\n(RULED OUT)', MID_GRAY, True),
    ]
    y_hyp_top = y_main - box_h / 2 - 0.16
    box_hyp_h = 0.52
    for xb, lab, col, has_arrow in hyps:
        if has_arrow:
            draw_attack(xb, y_hyp_top - 0.10, y_hyp_top, '#555555')
        box = FancyBboxPatch((xb - 0.75, y_hyp_top - box_hyp_h), 1.5,
                             box_hyp_h, boxstyle="round,pad=0.05",
                             facecolor=col, edgecolor='white', linewidth=1,
                             alpha=0.78, zorder=5)
        ax.add_patch(box)
        ax.text(xb, y_hyp_top - box_hyp_h / 2, lab, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='white', zorder=6)

    fig.tight_layout(pad=0.3)
    save(fig, 'fig1_phenomenon_overview')
    print('  ✓ Figure 1 saved (two-level chain diagram)')


# ── Figure 2: Not the Shared Encoder ────────────────────────────────────
def fig2_not_shared_encoder():
    from matplotlib.patches import Patch

    kw = load_json('kruskal_4groups.json')
    groups_data = kw['groups']

    GROUP_COLORS = [NEUTRAL_FILL, PRIMARY_BLUE, PRIMARY_BLUE, PRIMARY_BLUE]
    group_names = ['CLIP\n(LLaVA)', 'SigLIP\n(Gemma)', 'QwenViT\n(Qwen3.x)', 'Qwen2-VL\nViT']
    group_keys = ['CLIP', 'SigLIP', 'QwenViT', 'Qwen2-VL-ViT']

    groups = {}
    for name, key, color in zip(group_names, group_keys, GROUP_COLORS):
        inds = groups_data[key]
        groups[name] = {
            'mean': np.mean(inds), 'individuals': inds,
            'n': len(inds), 'color': color,
        }

    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    x_pos = np.arange(len(groups))
    labels = list(groups.keys())
    means = [groups[g]['mean'] for g in groups]
    colors = [groups[g]['color'] for g in groups]

    ax.bar(x_pos, means, width=0.5, color=colors,
           edgecolor=EDGE_COLOR, linewidth=0.6, alpha=0.88, zorder=3)

    rng = np.random.RandomState(42)
    marker_shapes = ['o', 's', '^', 'D']
    for i, g in enumerate(groups):
        ind = groups[g]['individuals']
        jitter = rng.uniform(-0.12, 0.12, size=len(ind))
        ax.scatter(i + jitter, ind, s=30, c=groups[g]['color'], zorder=5,
                   marker=marker_shapes[i],
                   clip_on=False, edgecolors='white', linewidth=0.5)

    for i, g in enumerate(groups):
        ax.text(i, means[i] + 0.035, f'n={groups[g]["n"]}',
                ha='center', fontsize=9, fontweight='bold', color=TEXT_COLOR)

    def star_bracket(ax, x1, x2, y, p_text):
        ax.plot([x1, x1, x2, x2], [y, y + 0.015, y + 0.015, y],
                lw=0.8, color=EDGE_COLOR)
        ax.text((x1 + x2) / 2, y + 0.02, p_text, ha='center',
                fontsize=9, fontweight='bold', color=TEXT_COLOR)

    star_bracket(ax, 0, 1, 0.81, 'sep. p=0.029\n(one-sided)')
    ax.text(1.0, 0.335, 'KW p=0.030 (all four groups)', ha='center',
            fontsize=8, fontstyle='italic', color='#444444')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Spearman ρ (CLIP vs MLLM)', fontsize=9)
    ax.set_ylim(0.3, 1.0)
    ax.text(0.02, 0.98,
            'Higher ρ = stronger CLIP–MLLM agreement\n→ shared encoder would predict CLIP > others',
            transform=ax.transAxes, fontsize=8, va='top', color='gray')
    style_axis(ax)

    fig.tight_layout(pad=0.5)
    save(fig, 'fig2_not_shared_encoder')
    print('  ✓ Figure 2 saved')


# ── Figure 3: CKA Heatmap ────────────────────────────────────────────────
def figS1_cka_heatmap():
    data = load_json('exp2_cka_all_encoders.json')
    cka = data['cka']
    encoders = list(cka.keys())
    n = len(encoders)

    mat = np.zeros((n, n))
    for i, e1 in enumerate(encoders):
        for j, e2 in enumerate(encoders):
            mat[i, j] = cka[e1][e2]

    labels_short = ['CLIP-L\n(OA)', 'CLIP-L\n(LAION)', 'CLIP-B\n(OA)', 'QwenViT', 'SigLIP', 'ResNet-50\n(ImageNet)', 'DINOv2']

    fig, ax = plt.subplots(figsize=(5.2, 4.6))

    im = ax.imshow(mat, vmin=0.7, vmax=1.0, cmap=HEATMAP_CMAP, aspect='equal')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels_short, fontsize=8.5)
    ax.set_yticklabels(labels_short, fontsize=8.5)

    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color=text_color_for(val, 0.7, 1.0))

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=8.5)
    cbar.set_label('CKA', fontsize=9, color=TEXT_COLOR)

    fig.tight_layout()
    save(fig, 'figS1_cka_heatmap')
    print('  ✓ Figure 3 saved')


# ── Figure 4: Class Geometry (full-page) ─────────────────────────────────
def fig3_class_geometry():
    from matplotlib.patches import Patch

    data = load_json('class_similarity_matrices.json')
    geom = load_json('class_geometry_analysis.json')
    mantel_results = load_json('mantel_test_results.json')

    class_names = data['class_names']
    matrices = data['matrices']
    encoders = list(matrices.keys())
    heat_order = ['CLIP ViT-L/14 (OpenAI)', 'CLIP ViT-L/14 (LAION)',
                  'CLIP ViT-B/32 (OpenAI)', 'SigLIP (ViT-SO400M)',
                  'Qwen2-VL Vision', 'ResNet-50 (ImageNet-1k)',
                  'DINOv2 ViT-B/14 (LVD-142M)']
    short_labels = [short_name(e) for e in heat_order]
    TICK_SHORT = {'On-stage int.': 'On-stage',
                  'blackboard-writing': 'writing'}
    tick_labels = [TICK_SHORT.get(c, c) for c in class_names]

    fig = plt.figure(figsize=(6.5, 5.5))
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1],
                           hspace=0.0, wspace=0.35)

    # ── Panel a: 7 similarity matrices in a 3×4 grid (left L-shape) ──
    heat_slots = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)]
    heat_axes = [fig.add_subplot(gs[r, c]) for (r, c) in heat_slots]

    vmin, vmax = 0.4, 1.0
    im = None
    for idx, (ax, enc, name) in enumerate(zip(heat_axes, heat_order,
                                              short_labels)):
        mat = np.clip(np.array(matrices[enc]), 0, 1.0)
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=HEATMAP_CMAP, aspect='equal')
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        # Symmetric matrices: y labels duplicate x labels, so drop y labels
        # everywhere; show x class names only on the last heatmap (DINOv2).
        ax.set_yticklabels([])
        if idx == 6:
            ax.set_xticklabels(tick_labels, rotation=45, ha='right',
                               fontsize=7.5)
        else:
            ax.set_xticklabels([])
        ax.set_title(name, fontsize=9.5, fontweight='bold', pad=4,
                     color=TEXT_COLOR)

    # Shared colorbar in the top-right grid cell (row 0, col 3)
    cax = fig.add_subplot(gs[0, 3])
    cb = plt.colorbar(im, cax=cax, orientation='vertical')
    # Align the colorbar axes exactly with the CLIP-B OA heatmap axes.
    ref = heat_axes[2].get_position()
    cax.set_position([cax.get_position().x0, ref.y0,
                      cax.get_position().width, ref.height])
    cax.tick_params(labelsize=8.5, colors=TEXT_COLOR, length=3)
    cax.set_title('Cosine\nsimilarity', fontsize=8.5, color=TEXT_COLOR, pad=4)

    # ── Panel b: 7x7 Mantel agreement matrix (rows 1-2, cols 2-3) ──
    # Each cell = Spearman rho between the 8x8 class-similarity matrices of
    # the two encoders shown above (upper triangles); structurally isomorphic
    # to panel a (class-level 8x8 -> encoder-level 7x7).
    mantel_scb5 = {frozenset((p['encoder_a'], p['encoder_b'])): p['mantel_rho']
                   for p in mantel_results['scb5_8_classes']}
    m_ax = fig.add_subplot(gs[1:3, 2:4])
    n7 = len(heat_order)
    mantel_mat = np.ones((n7, n7))
    for i in range(n7):
        for j in range(n7):
            if i != j:
                mantel_mat[i, j] = mantel_scb5.get(
                    frozenset((heat_order[i], heat_order[j])), np.nan)
    mask = np.tril(np.ones((n7, n7), dtype=bool), k=-1)
    mat_masked = np.ma.masked_where(mask, mantel_mat)
    m_ax.imshow(mat_masked, vmin=0.5, vmax=1.0, cmap=HEATMAP_CMAP, aspect='equal')
    m_ax.set_xticks(range(n7))
    m_ax.set_yticks(range(n7))
    m_ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
    m_ax.set_yticklabels([])
    for i in range(n7):
        for j in range(i, n7):
            val = mantel_mat[i, j]
            if np.isnan(val):
                continue
            m_ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                      fontsize=8, fontweight='bold',
                      color=text_color_for(val, 0.5, 1.0))
    m_ax.set_title('Mantel ρ between encoders', fontsize=10,
                   fontweight='bold', color=TEXT_COLOR, loc='center')
    style_axis(m_ax, grid_axis=None)

    fig.tight_layout(rect=[0.015, 0, 1, 0.99])
    save(fig, 'fig3_class_geometry')
    print('  ✓ Figure 3 saved (class geometry)')


# ── Figure 5: Prompt Variants ────────────────────────────────────────────
def fig4_prompt_variants():
    data3 = load_json('exp3_prompt_variants_v2.json')
    data5 = load_json('exp3_stand_screen_prompts.json')
    data = {**data3, **data5}
    classes = ['blackboard-writing', 'answer', 'blackBoard', 'stand', 'screen']

    fig, ax = plt.subplots(figsize=(6.9, 3.7))

    orig_accs = []
    best_accs = []
    variant_accs = []
    for cls in classes:
        items = data[cls]
        orig_accs.append(items[cls]['accuracy'])
        vals = [v['accuracy'] for v in items.values()]
        best_accs.append(max(vals))
        variant_accs.append(vals)

    x = np.arange(len(classes))
    width = 0.30

    rng = np.random.RandomState(42)
    for i, vals in enumerate(variant_accs):
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(i + jitter, vals, s=26, color=NEUTRAL_FILL, alpha=0.8,
                   edgecolors='white', linewidth=0.4, zorder=4,
                   label='Prompt variants' if i == 0 else None)

    ax.bar(x - width / 2, orig_accs, width,
           color=lighten(NEUTRAL_FILL, blend=0.10),
           edgecolor=EDGE_COLOR, linewidth=0.6, label='Original prompt', zorder=3)
    ax.bar(x + width / 2, best_accs, width,
           color=PRIMARY_BLUE,
           edgecolor=EDGE_COLOR, linewidth=0.6, label='Best of 5 variants', zorder=3)

    for i in range(len(classes)):
        ax.text(i - width / 2, orig_accs[i] + 1.5, f'{orig_accs[i]:.0f}',
                ha='center', fontsize=8, color=TEXT_COLOR)
        ax.text(i + width / 2, best_accs[i] + 1.5, f'{best_accs[i]:.0f}',
                ha='center', fontsize=8, fontweight='bold', color=TEXT_COLOR)

    for i in range(len(classes)):
        imp = best_accs[i] - orig_accs[i]
        ax.annotate(f'+{imp:.0f} pp', xy=(i + width / 2, best_accs[i] + 4.5),
                    xytext=(i + width / 2, best_accs[i] + 13),
                    ha='center', fontsize=8.5, fontweight='bold',
                    color=LINE_DARK,
                    arrowprops=dict(arrowstyle='->', color=EDGE_COLOR, lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=8.5)
    plt.setp(ax.get_xticklabels(), rotation=18, ha='right')
    ax.set_ylabel('CLIP Zero-Shot Accuracy (%)', fontsize=9)
    ax.set_title('OpenAI ViT-L/14', fontsize=8.5, color='#555555', loc='right')
    ax.set_ylim(0, 122)
    ax.legend(fontsize=8.5, loc='upper right', frameon=False)
    style_axis(ax)

    fig.tight_layout()
    save(fig, 'fig4_prompt_variants')
    print('  ✓ Figure 4 saved')


# ── Figure 6: Stanford40 Cross-Dataset ───────────────────────────────────
def fig5_stanford40():
    from scipy.cluster.hierarchy import linkage, leaves_list

    ext = RESULTS / 'stanford40_external'
    with open(ext / 'anchor_scores.json') as f:
        anchor_data = json.load(f)
    with open(ext / 'ollama_results.json') as f:
        ollama_data = json.load(f)

    per_class_anchor = anchor_data['per_class_acc']
    classes = sorted(per_class_anchor.keys())
    models = ollama_data['models']
    per_class_mllm = {c: np.mean([models[m]['per_class_acc'][c]['acc'] for m in models])
                      for c in classes}
    x_acc = np.array([per_class_anchor[c]['acc'] for c in classes])
    y_acc = np.array([per_class_mllm[c] for c in classes])
    rho_acc, _ = stats.spearmanr(x_acc, y_acc)

    with open(RESULTS / 'exp6_stanford40_geometry.json') as f:
        geo = json.load(f)
    clip_mat = np.array(geo['similarity_matrices']['CLIP_ViT_L_14_OpenAI'])
    qwen_mat = np.array(geo['similarity_matrices']['Qwen2_VL_Vision'])
    geom = geo['cross_encoder_comparison'][0]
    rho_geo, p_geo = geom['spearman_r'], geom['p_value']

    dist_clip = 1.0 - clip_mat
    Z = linkage(dist_clip[np.triu_indices_from(clip_mat, k=1)], method='average')
    order = leaves_list(Z)
    clip_ordered = clip_mat[order][:, order]
    qwen_ordered = qwen_mat[order][:, order]

    iu = np.triu_indices_from(clip_mat, k=1)
    clip_ut = clip_mat[iu]
    qwen_ut = qwen_mat[iu]

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.4))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # Panel a: scatter
    n = len(classes)
    x_sorted = np.sort(x_acc)
    slope, intercept, _, _, _ = stats.linregress(x_acc, y_acc)
    rng = np.random.RandomState(488)
    boot_preds = np.zeros((2000, n))
    for b in range(2000):
        idx = rng.choice(n, n, replace=True)
        s, i, _, _, _ = stats.linregress(x_acc[idx], y_acc[idx])
        boot_preds[b] = s * x_sorted + i
    ax_a.scatter(x_acc, y_acc, s=26, c=PRIMARY_BLUE, alpha=0.65,
                 edgecolors=EDGE_COLOR, linewidth=0.3, zorder=3)
    ax_a.fill_between(x_sorted, np.percentile(boot_preds, 2.5, axis=0),
                       np.percentile(boot_preds, 97.5, axis=0),
                       alpha=0.25, color=NEUTRAL_FILL)
    ax_a.plot(x_sorted, slope * x_sorted + intercept, color=LINE_DARK,
              linewidth=1.5, zorder=4)
    ax_a.set_xlabel('AnchorScore (%)', fontsize=9)
    ax_a.set_ylabel('MLLM Mean Accuracy (%)', fontsize=9)
    ax_a.set_xlim(40, 105)
    ax_a.set_ylim(70, 105)
    ax_a.text(0.95, 0.05, f'ρ = {rho_acc:.3f}\np < 0.001\nn = 40 classes',
              transform=ax_a.transAxes, fontsize=8.5, ha='right', va='bottom',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=EDGE_COLOR, alpha=0.85))
    ax_a.set_title('a  Phenomenon replicates',
                   fontsize=10, fontweight='bold', loc='left', color=TEXT_COLOR)
    style_axis(ax_a)

    # Panel b: geometry scatter
    ax_b.scatter(clip_ut, qwen_ut, s=9, c=NEUTRAL_FILL, alpha=0.55,
                 edgecolors='none', zorder=3)
    s2, i2, _, _, _ = stats.linregress(clip_ut, qwen_ut)
    xs2 = np.linspace(clip_ut.min(), clip_ut.max(), 100)
    ax_b.plot(xs2, s2 * xs2 + i2, color=PRIMARY_BLUE, linewidth=1.5, zorder=4)
    ax_b.set_xlabel('CLIP class similarity', fontsize=9)
    ax_b.set_ylabel('Qwen2-VL class similarity', fontsize=9)
    ax_b.text(0.95, 0.05, f'ρ = {rho_geo:.3f}\np < 10⁻⁴\n780 class pairs',
              transform=ax_b.transAxes, fontsize=8.5, ha='right', va='bottom',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=EDGE_COLOR, alpha=0.85))
    ax_b.set_title('b  Shared class geometry',
                   fontsize=10, fontweight='bold', loc='left', color=TEXT_COLOR)
    style_axis(ax_b)

    # Panel c & d: similarity matrices
    im_c = ax_c.imshow(clip_ordered, vmin=0.72, vmax=1.0, cmap=HEATMAP_CMAP,
                       aspect='equal')
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    ax_c.set_title('c  CLIP ViT-L/14 class similarity',
                   fontsize=10, fontweight='bold', loc='left', color=TEXT_COLOR)
    cb_c = plt.colorbar(im_c, ax=ax_c, shrink=0.75, pad=0.02)
    cb_c.ax.tick_params(labelsize=8.5, colors=TEXT_COLOR)
    cb_c.set_label('cosine sim', fontsize=9, color=TEXT_COLOR, labelpad=4)

    im_d = ax_d.imshow(qwen_ordered, vmin=0.72, vmax=1.0, cmap=HEATMAP_CMAP,
                       aspect='equal')
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    ax_d.set_title('d  Qwen2-VL class similarity (same order)',
                   fontsize=10, fontweight='bold', loc='left', color=TEXT_COLOR)
    cb_d = plt.colorbar(im_d, ax=ax_d, shrink=0.75, pad=0.02)
    cb_d.ax.tick_params(labelsize=8.5, colors=TEXT_COLOR)
    cb_d.set_label('cosine sim', fontsize=9, color=TEXT_COLOR, labelpad=4)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, 'fig5_stanford40')
    print(f'  ✓ Figure 5 saved (scatter ρ={rho_acc:.3f}, geometry ρ={rho_geo:.3f})')


def short_name(full):
    m = {
        'CLIP ViT-L/14 (OpenAI)': 'CLIP-L OA',
        'CLIP ViT-L/14 (LAION)': 'CLIP-L LAION',
        'CLIP ViT-B/32 (OpenAI)': 'CLIP-B OA',
        'Qwen2-VL Vision': 'QwenViT',
        'SigLIP (ViT-SO400M)': 'SigLIP',
        'ResNet-50 (ImageNet-1k)': 'ResNet-50',
        'DINOv2 ViT-B/14 (LVD-142M)': 'DINOv2',
    }
    return m.get(full, full)


# ── Run all ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating figures...')
    fig1_phenomenon_overview()
    fig2_not_shared_encoder()
    figS1_cka_heatmap()
    fig3_class_geometry()
    fig4_prompt_variants()
    fig5_stanford40()
    print(f'\nAll figures saved to {OUT}/')
    print('Files:', sorted(os.listdir(OUT)))
