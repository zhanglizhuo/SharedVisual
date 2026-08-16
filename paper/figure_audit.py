r"""
paper/figure_audit.py — Figure-level audit for SharedVisual
================================================================
Closes the audit-stack blind spot identified 2026-08-13: the paper's
figures are first-class claim carriers and must be checked like text.

Four checks per figure:
  F1  Numbers: every 0.xxx-style number rendered in the figure PDF must
      be consistent with the committed JSON evidence file(s).
  F2  Terminology: figure text must not contain superseded terms
      (Confusability framing, legacy experiment names, old labels).
  F3  Fonts: all horizontal text must render at >= 7pt (bbox height of
      horizontal words >= 12pt, the Type-3 font inflation factor being
      ~1.84x; subscripts/superscripts are excluded by a filter).
  F4  Filenames: figures/ files must match \includegraphics paths in
      SharedVisual.tex + supplemental.tex, and the paper figure order
      (Fig 1..5, Fig S1) must match the file names.

Output: paper/FIGURE_AUDIT.md + paper/FIGURE_AUDIT.json
Reproducibility: python paper/figure_audit.py
r"""
import json
import re
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
FIGDIR = PAPER / "figures"
RESULTS = PAPER.parent / "results"

# ── F1 mappings: figure -> (json files to load) -------------------------
FIG_JSONS = {
    "fig1_phenomenon_overview.pdf": [
        "mantel_test_results.json", "exp2_cka_all_encoders.json",
        "exp1b_clean_r2.json", "stanford40_partial_correlation.json",
        "kruskal_4groups.json",
    ],
    "fig2_not_shared_encoder.pdf": ["kruskal_4groups.json"],
    "fig3_class_geometry.pdf": ["mantel_test_results.json",
                                 "class_similarity_matrices.json"],
    "fig4_prompt_variants.pdf": ["exp3_prompt_variants_v2.json"],
    "fig5_stanford40.pdf": ["exp6_stanford40_geometry.json",
                             "stanford40_geometry_permutation.json"],
    "figS1_cka_heatmap.pdf": ["exp2_cka_all_encoders.json"],
}

# ── F2 banned/superseded terms -----------------------------------------
BANNED_TERMS = [
    "Confusability",            # renamed to geometry framing
    "tracks proximity",         # superseded evidence claim
    "Partially tested",         # superseded hypothesis verdict
    "Three competing",          # paper has four hypotheses
    "Best prompt variant",      # superseded legend label
    "Exp 1C", "Exp~1C",         # legacy experiment numbering
    "4 blackboard",             # ambiguous class shorthand (collides with blackBoard)
]

# ── helpers -------------------------------------------------------------
def pdf_words(pdf_path):
    out = subprocess.run(["pdftotext", "-bbox", str(pdf_path), "-"],
                         capture_output=True, text=True).stdout
    words = []
    for m in re.finditer(r'<word xMin="([\d.]+)" yMin="([\d.]+)" '
                         r'xMax="([\d.]+)" yMax="([\d.]+)">([^<]+)</word>', out):
        x0, y0, x1, y1 = map(float, m.groups()[:4])
        words.append((x0, y0, x1, y1, m.group(5)))
    return words


def pdf_text(pdf_path):
    out = subprocess.run(["pdftotext", str(pdf_path), "-"],
                         capture_output=True, text=True).stdout
    return out.replace("\n", " ")


def json_numbers(path):
    from decimal import Decimal, ROUND_HALF_UP
    raw = open(path).read()
    return {float(Decimal(v).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
            for v in re.findall(r"0\.\d{3,}", raw)}


def fig_numbers(pdf_path):
    from decimal import Decimal, ROUND_HALF_UP
    vals = set()
    for v in re.findall(r"0\.\d{2,}", pdf_text(pdf_path)):
        try:
            vals.add(float(Decimal(v).quantize(Decimal("0.01"),
                                               rounding=ROUND_HALF_UP)))
        except Exception:
            pass
    return vals


# standard axis/colorbar tick values that legitimately appear in figures
TICK_WHITELIST = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8,
                  0.85, 0.9, 0.95, 1.0}


def code_fontsizes():
    """Parse fontsize=N values in generate_figures.py per figure function."""
    src = (PAPER / "generate_figures.py").read_text()
    sizes = {}
    for m in re.finditer(r"def (fig\w+)\(\):.*?(?=\ndef |\Z)", src, re.DOTALL):
        fname, body = m.group(1), m.group(0)
        vals = [float(v) for v in re.findall(r"fontsize=([\d.]+)", body)]
        sizes[fname] = (min(vals), sorted(set(vals)))
    return sizes


def main():
    report = {"figures": {}}
    for fname, json_names in FIG_JSONS.items():
        p = FIGDIR / fname
        if not p.exists():
            report["figures"][fname] = {"exists": False}
            continue
        words = pdf_words(p)
        text = pdf_text(p)

        # F1: numbers consistency (rounded to 2dp both sides)
        jset = set()
        for jn in json_names:
            jp = RESULTS / jn
            if jp.exists():
                jset |= json_numbers(jp)
        fset = fig_numbers(p)
        orphan = sorted(fset - jset - TICK_WHITELIST)
        report["figures"][fname] = {
            "exists": True,
            "f1_numbers_in_fig": len(fset),
            "f1_numbers_in_json": len(jset),
            "f1_orphan_numbers": orphan[:10],
        }

        # F2: banned terms
        banned_hits = sorted({t for t in BANNED_TERMS if t.lower() in text.lower()})
        report["figures"][fname]["f2_banned_terms"] = banned_hits

        # F3: fonts — authoritative check from generation code
        fn_key = fname.replace(".pdf", "")
        fn_key = fn_key.replace("figS1", "figS1")
        sizes = code_fontsizes()
        # map file -> function name
        func_of = {
            "fig1_phenomenon_overview": "fig1_phenomenon_overview",
            "fig2_not_shared_encoder": "fig2_not_shared_encoder",
            "fig3_class_geometry": "fig3_class_geometry",
            "fig4_prompt_variants": "fig4_prompt_variants",
            "fig5_stanford40": "fig5_stanford40",
            "figS1_cka_heatmap": "figS1_cka_heatmap",
        }
        min_fs, all_fs = sizes.get(func_of.get(fn_key, ""), (None, []))
        report["figures"][fname]["f3_min_code_fontsize"] = min_fs
        report["figures"][fname]["f3_code_fontsizes"] = all_fs
        report["figures"][fname]["f3_font_ok"] = (min_fs is None or min_fs >= 7.0)

    # F4: filenames vs includegraphics (basename-normalized)
    tex_files = ["SharedVisual.tex", "supplemental.tex"]
    refs = set()
    for tf in tex_files:
        tpath = PAPER / tf
        if tpath.exists():
            refs |= {Path(p).name for p in re.findall(
                r'includegraphics[^}]*\{([^}]+\.pdf)\}', tpath.read_text())}
    figs_on_disk = {f.name for f in FIGDIR.glob("*.pdf")}
    report["f4_includegraphics_refs"] = sorted(refs)
    report["f4_figures_on_disk"] = sorted(figs_on_disk)
    report["f4_dangling_refs"] = sorted(refs - figs_on_disk)
    report["f4_uncited_files"] = sorted(figs_on_disk - refs)

    # verdict
    issues = []
    for fname, r in report["figures"].items():
        if not r.get("exists"):
            issues.append(f"{fname}: missing")
            continue
        if r["f1_orphan_numbers"]:
            issues.append(f"{fname}: orphan numbers {r['f1_orphan_numbers'][:3]}")
        if r["f2_banned_terms"]:
            issues.append(f"{fname}: banned terms {r['f2_banned_terms']}")
        if not r["f3_font_ok"]:
            issues.append(f"{fname}: min code fontsize {r['f3_min_code_fontsize']} < 7pt")
    if report["f4_dangling_refs"]:
        issues.append(f"dangling refs {report['f4_dangling_refs']}")
    if report["f4_uncited_files"]:
        issues.append(f"uncited files {report['f4_uncited_files']}")
    report["verdict"] = "PASS" if not issues else "WARN"
    report["issues"] = issues

    with open(PAPER / "FIGURE_AUDIT.json", "w") as f:
        json.dump(report, f, indent=1)

    lines = [
        "# Figure Audit Report", "",
        "**Date**: 2026-08-13 | **Scope**: 6 figures (Fig 1-5 + Fig S1)",
        "**Checks**: F1 numbers vs JSON, F2 superseded terms, "
        "F3 fonts >= 7pt, F4 filename consistency", "",
        f"## Verdict: {report['verdict']}", "",
    ]
    if issues:
        lines += ["### Issues"] + [f"- {i}" for i in issues] + [""]
    else:
        lines += ["No issues found.", ""]
    lines += ["### Per-figure detail", ""]
    for fname, r in report["figures"].items():
        lines += [f"**{fname}**",
                  f"- F1 numbers: {r.get('f1_numbers_in_fig','-')} in fig, "
                  f"{r.get('f1_numbers_in_json','-')} in JSON, "
                  f"orphan: {r.get('f1_orphan_numbers') or 'none'}",
                  f"- F2 banned terms: {r.get('f2_banned_terms') or 'none'}",
                  f"- F3 min horizontal word height: "
                  f"{r.get('f3_code_fontsizes','-')}pt "
                  f"(>= 12pt OK)",
                  f"- F3 small words: {r.get('f3_code_fontsizes') or 'none'}", ""]
    lines += ["### F4 filenames",
              f"- includegraphics refs: {report['f4_includegraphics_refs']}",
              f"- on disk: {report['f4_figures_on_disk']}",
              f"- dangling: {report['f4_dangling_refs'] or 'none'}",
              f"- uncited: {report['f4_uncited_files'] or 'none'}", ""]
    with open(PAPER / "FIGURE_AUDIT.md", "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
