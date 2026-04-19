import argparse
import pysam
import pandas as pd

def load_bed(path):
    return pysam.TabixFile(path)

def in_bed(tb, chrom, pos1):
    start = pos1 - 1
    end = pos1
    try:
        for _ in tb.fetch(chrom, start, end):
            return 1
    except Exception:
        return 0
    return 0

def key_of(rec):
    if len(rec.alts) != 1:
        return None
    return (rec.contig, rec.pos, rec.ref, rec.alts[0])

def get_label_map(annot_vcf):
    labels = {}
    vf = pysam.VariantFile(annot_vcf)
    for rec in vf.fetch():
        if len(rec.alts or []) != 1:
            continue
        if "QUERY" not in rec.samples:
            continue
        bd = rec.samples["QUERY"].get("BD", None)
        if isinstance(bd, tuple):
            bd = bd[0]
        if bd in ("TP", "FP"):
            labels[key_of(rec)] = 1 if bd == "TP" else 0
    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dv_vcf", required=True)
    ap.add_argument("--annot_vcf", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--bed_hpoly", required=True)
    ap.add_argument("--bed_tandem", required=True)
    ap.add_argument("--bed_lowmap", required=True)
    ap.add_argument("--bed_segdup", required=True)
    ap.add_argument("--bed_alldiff", required=True)
    args = ap.parse_args()

    labels = get_label_map(args.annot_vcf)

    tb_h = load_bed(args.bed_hpoly)
    tb_t = load_bed(args.bed_tandem)
    tb_l = load_bed(args.bed_lowmap)
    tb_s = load_bed(args.bed_segdup)
    tb_a = load_bed(args.bed_alldiff)

    vf = pysam.VariantFile(args.dv_vcf)
    sample_name = list(vf.header.samples)[0]

    rows = []
    for rec in vf.fetch():
        if len(rec.alts or []) != 1:
            continue
        alt = rec.alts[0]

        # only INDEL
        if len(rec.ref) == len(alt):
            continue

        k = key_of(rec)
        if k not in labels:
            continue

        sm = rec.samples[sample_name]
        dp = sm.get("DP", 0) or 0
        gq = sm.get("GQ", 0) or 0
        ad = sm.get("AD", None)

        if ad is None or len(ad) < 2:
            ad_ref, ad_alt = 0, 0
        else:
            ad_ref, ad_alt = int(ad[0]), int(ad[1])

        denom = ad_ref + ad_alt
        vaf = float(ad_alt / denom) if denom > 0 else 0.0

        ref_len = len(rec.ref)
        alt_len = len(alt)
        indel_len = abs(ref_len - alt_len)

        rows.append({
            "sample": args.sample,
            "chrom": rec.contig,
            "pos": rec.pos,
            "ref": rec.ref,
            "alt": alt,
            "qual": float(rec.qual or 0.0),
            "dp": int(dp),
            "gq": int(gq),
            "ad_ref": ad_ref,
            "ad_alt": ad_alt,
            "vaf": vaf,
            "ref_len": ref_len,
            "alt_len": alt_len,
            "indel_len": indel_len,
            "is_ins": 1 if alt_len > ref_len else 0,
            "is_del": 1 if ref_len > alt_len else 0,
            "in_homopolymer": in_bed(tb_h, rec.contig, rec.pos),
            "in_tandem_repeat": in_bed(tb_t, rec.contig, rec.pos),
            "in_lowmap": in_bed(tb_l, rec.contig, rec.pos),
            "in_segdup": in_bed(tb_s, rec.contig, rec.pos),
            "in_alldifficult": in_bed(tb_a, rec.contig, rec.pos),
            "label": labels[k],
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()
