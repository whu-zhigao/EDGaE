#!/usr/bin/env python3
"""Compare cloud re-run CSVs against the frozen V2/V3 reference set."""
import csv, math, re, sys
from pathlib import Path

FILES = {
    "icde2027-v2-trace-results.csv":      ["block","dataset","model","sampler","workers","control","seed"],
    "icde2027-v2-extra-results.csv":      ["block","dataset","model","sampler","workers","control","seed"],
    "icde-v3-mode-system-results.csv":    ["block","dataset","model","sampler","workers","control","seed"],
    "icde2027-v2-preprocess-results.csv": ["block","dataset","model","sampler","workers","control","seed"],
    "icde2027-v2-web-google-complete-results.csv": ["block","dataset","model","sampler","workers","control","seed"],
    "icde2027-v2-wiki-talk-complete-results.csv":  ["block","dataset","model","sampler","workers","control","seed"],
    "joint-quality-results.csv":          ["dataset","seed","instance","strategy","order"],
    "cache-shift-results.csv":            ["seed","shift_fraction"],
    "cost-aware-results.csv":             ["dataset","seed","instance","strategy"],
}
SKIP = {"run_id","measurement_utc","notes","cache_dir","runner_version","platform",
        "hardware_platform","measurement_source","measurement_valid"}
TIMING = re.compile(r"^(T_.+|.+_s$|.+_seconds?$|elapsed.*|duration.*|.*_time$)", re.I)

def load_indexed(path, cols):
    rows = list(csv.DictReader(open(path, newline="")))
    seen, out = {}, {}
    for r in rows:
        base = tuple(r.get(c,"") for c in cols)
        n = seen.get(base, 0); seen[base] = n + 1
        out[base + (n,)] = r
    return out

def numeric(v):
    if v is None or v == "": return None
    try: return float(v)
    except ValueError: return None

def compare(name, idcols, ref_path, cloud_path, subset):
    R = load_indexed(ref_path, idcols)
    C = load_indexed(cloud_path, idcols) if cloud_path.exists() else {}
    missing = sorted(set(R) - set(C)); extra = sorted(set(C) - set(R))
    pairs = sorted(set(R) & set(C))
    bad, timing_diff, col_hist = [], 0, {}
    for k in pairs:
        for col in R[k]:
            if col in SKIP or col not in C[k]: continue
            a, b = numeric(R[k][col]), numeric(C[k][col])
            if a is None and b is None: continue
            if (a is None) != (b is None):
                bad.append((k,col,R[k][col],C[k][col])); continue
            if math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9): continue
            if TIMING.match(col):
                timing_diff += 1; continue
            col_hist[col] = col_hist.get(col,0) + 1
            bad.append((k,col,R[k][col],C[k][col]))
    status = "OK" if not missing and not extra and not bad else "DIFF"
    if subset: status = "OK(subset)" if not extra and not bad else "DIFF"
    print(f"{name:46s} {status:11s} ref={len(R):4d} cloud={len(C):4d} "
          f"missing={len(missing):4d} extra={len(extra):3d} mismatch={len(bad):3d} timing_diff={timing_diff}")
    if col_hist: print(f"    mismatched columns: {col_hist}")
    for k,col,rv,cv in bad[:8]: print(f"    MISMATCH {k} :: {col}: ref={rv} cloud={cv}")
    for k in missing[:4]: print(f"    MISSING  {k}")
    for k in extra[:4]:   print(f"    EXTRA    {k}")
    return status in ("OK","OK(subset)")

def main():
    results, reference = Path(sys.argv[1]), Path(sys.argv[2])
    subset = "--subset" in sys.argv
    ok = True
    for name, idcols in FILES.items():
        rp = reference / name
        if not rp.exists():
            print(f"{name:46s} NO-REF     skipped"); continue
        ok &= compare(name, idcols, rp, results / name, subset)
    print("VERDICT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
