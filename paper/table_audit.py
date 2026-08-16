r"""
paper/table_audit.py — Table-level audit for SharedVisual
================================================================
Tables are first-class claim carriers; check them like text and figures.

Four checks per table:
  T1  Numbers: every numeric value in the table must be consistent with
      the committed JSON evidence file(s) (half-up rounded to 2dp).
  T2  Terminology: table cells/captions must not contain superseded terms.
  T3  Caption hygiene: caption short ("what it is"), details in footnotes;
      flag captions over 45 words.
  T4  Cross-references: \label must be unique, \ref must resolve
      (compile-time check via the log's undefined references).

Output: paper/TABLE_AUDIT.md + paper/TABLE_AUDIT.json
Reproducibility: python paper/table_audit.py
"""
import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

PAPER = Path(__file__).resolve().parent
RESULTS = PAPER.parent / "results"

TABLE_JSONS = {
    "tab:patchlevel": ["exp1b_clean_r2.json", "exp1b_mlp.json",
                       "mantel_test_results.json"],
    "tab:mediation": ["mediation_bootstrap.json"],
    "tab:flowers102": ["flowers102_mantel.json"],
    "tab:flowers102b": ["flowers102_difficulty_multi.json"],
    "tab:mediation_controls": ["mediation_counterfactual_controls.json"],
    "tab:per_model_rho": ["kruskal_4groups.json"],
}

BANNED_TERMS = [
    "Confusability",
    "Step 1", "Step 2", "Step 3", "Step 4", "Step 5",
    "Exp 1C", "Exp~1C",
    "Best prompt variant",
    "Partially tested",
]

CAPTION_MAX_WORDS = 45


def tex_tables(tex_file):
    src = (PAPER / tex_file).read_text()
    tables = {}
    for m in re.finditer(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", src,
                         re.DOTALL):
        body = m.group(0)
        lab = re.search(r"\\label\{([^}]+)\}", body)
        cap = re.search(r"\\caption\{(.*?)\}", body, re.DOTALL)
        tables[lab.group(1)] = {
            "body": body,
            "caption": cap.group(1) if cap else "",
        }
    return tables


def json_floats(path):
    raw = open(path).read()
    return [float(v) for v in re.findall(r"[-+]?\d+\.\d{2,}", raw)]


def table_numbers(body):
    # exclude model names like LLaVA-1.5-7B via lookbehind
    return re.findall(r"(?<![\w.\-])(\d+\.\d{1,3})(?![\w.])", body)


def matches(value_str, jfloats):
    """True if any JSON float's absolute value rounds (half-up) to the
    table value at the table's displayed precision. Absolute values are
    compared because LaTeX renders negative signs as math symbols that
    the text extraction does not attach to the digits."""
    d = len(value_str.split(".")[1])
    q = Decimal(1).scaleb(-d)
    target = float(Decimal(value_str).quantize(q, rounding=ROUND_HALF_UP))
    for v in jfloats:
        try:
            rv = abs(float(Decimal(str(abs(v))).quantize(q,
                                                         rounding=ROUND_HALF_UP)))
        except Exception:
            continue
        if rv == target:
            return True
    return False


def main():
    tables = {}
    for tf in ["SharedVisual.tex", "supplemental.tex"]:
        tables.update(tex_tables(tf))

    report = {"tables": {}}
    for lab, jn_list in TABLE_JSONS.items():
        if lab not in tables:
            report["tables"][lab] = {"exists": False}
            continue
        body = tables[lab]["body"]
        cap = tables[lab]["caption"]
        jfloats = []
        for jn in jn_list:
            jp = RESULTS / jn
            if jp.exists():
                jfloats.extend(json_floats(jp))
        tvals = table_numbers(body)
        orphan = sorted({v for v in tvals if not matches(v, jfloats)},
                        key=lambda x: (len(x), x))
        report["tables"][lab] = {
            "exists": True,
            "t1_numbers_in_table": len(tvals),
            "t1_numbers_in_json": len(jfloats),
            "t1_orphan_numbers": orphan[:10],
            "t2_banned_terms": sorted({t for t in BANNED_TERMS
                                       if t.lower() in body.lower()}),
            "t3_caption_words": len(cap.split()),
        }

    # T4: unique labels + undefined refs in compile log
    all_tex = "".join((PAPER / tf).read_text()
                      for tf in ["SharedVisual.tex", "supplemental.tex"])
    labels = re.findall(r"\\label\{([^}]+)\}", all_tex)
    dupes = sorted({l for l in labels if labels.count(l) > 1})
    log = (PAPER / "SharedVisual.log")
    undef = 0
    if log.exists():
        undef = log.read_text(errors="ignore").count("undefined")
    report["t4_duplicate_labels"] = dupes
    report["t4_undefined_refs_in_log"] = undef

    issues = []
    for lab, r in report["tables"].items():
        if not r.get("exists"):
            issues.append(f"{lab}: missing table")
            continue
        if r["t1_orphan_numbers"]:
            issues.append(f"{lab}: orphan numbers {r['t1_orphan_numbers'][:3]}")
        if r["t2_banned_terms"]:
            issues.append(f"{lab}: banned terms {r['t2_banned_terms']}")
        if r["t3_caption_words"] > CAPTION_MAX_WORDS:
            issues.append(f"{lab}: caption {r['t3_caption_words']} words "
                          f"(> {CAPTION_MAX_WORDS})")
    if dupes:
        issues.append(f"duplicate labels {dupes}")
    if undef:
        issues.append(f"{undef} undefined refs in log")
    report["verdict"] = "PASS" if not issues else "WARN"
    report["issues"] = issues

    with open(PAPER / "TABLE_AUDIT.json", "w") as f:
        json.dump(report, f, indent=1)

    lines = [
        "# Table Audit Report", "",
        "**Date**: 2026-08-13 | **Scope**: 6 tables (4 main + 2 supplement)",
        "**Checks**: T1 numbers vs JSON, T2 superseded terms, "
        "T3 caption length, T4 labels/refs", "",
        f"## Verdict: {report['verdict']}", "",
    ]
    if issues:
        lines += ["### Issues"] + [f"- {i}" for i in issues] + [""]
    else:
        lines += ["No issues found.", ""]
    lines += ["### Per-table detail", ""]
    for lab, r in report["tables"].items():
        lines += [f"**{lab}**",
                  f"- T1 numbers: {r.get('t1_numbers_in_table','-')} in table, "
                  f"{r.get('t1_numbers_in_json','-')} in JSON, "
                  f"orphan: {r.get('t1_orphan_numbers') or 'none'}",
                  f"- T2 banned terms: {r.get('t2_banned_terms') or 'none'}",
                  f"- T3 caption words: {r.get('t3_caption_words','-')} "
                  f"(<= {CAPTION_MAX_WORDS} OK)", ""]
    lines += ["### T4 cross-references",
              f"- duplicate labels: {dupes or 'none'}",
              f"- undefined refs in compile log: {undef}", ""]
    with open(PAPER / "TABLE_AUDIT.md", "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
