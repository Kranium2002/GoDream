"""Copy verbatim line ranges from pdftotext/HTML-to-text outputs into excerpts.txt.

Inputs: the folder of text files produced from the downloaded primary sources
(pdftotext 4.x on the PDFs; tag-stripped HTML for the snnTorch tutorials). Each excerpt is
written with its source file and 1-based line range so it can be re-checked. Lines are copied
unchanged (undecodable bytes shown as U+FFFD). Nothing here is project evidence.

Usage: python -I extract_excerpts.py <txt_dir> <out_file>
"""
import sys
from pathlib import Path

# (label, source document, file, first_line, last_line)
RANGES = [
    ("Horowitz ISSCC 2014 sec.5-6 text (DRAM, 45nm Fig.1.1.9, 70pJ/instr, cache 20pJ)",
     "doi:10.1109/ISSCC.2014.6757323", "horowitz_gwern.layout.txt", 115, 156),
    ("Horowitz ISSCC 2014 Fig.1.1.9 caption (table body is font-shifted; see horowitz_fig119_decoded.stdout.txt)",
     "doi:10.1109/ISSCC.2014.6757323", "horowitz_gwern.layout.txt", 315, 332),
    ("Han et al. NIPS 2015 Sec.1 + Figure 1 energy table",
     "arXiv:1506.02626v3", "han2015.layout.txt", 40, 74),
    ("Han et al. NIPS 2015 reference [7]", "arXiv:1506.02626v3", "han2015.layout.txt", 507, 507),
    ("DIET-SNN Sec.5 Energy efficiency (Table 3 caption, Eq.16-18)",
     "arXiv:2008.03658v3 / doi:10.1109/TNNLS.2021.3111897", "dietsnn.txt", 466, 572),
    ("DIET-SNN Sec.5 data-movement caveat", "arXiv:2008.03658v3", "dietsnn.txt", 577, 577),
    ("Spikformer App. C.2 Eq.21-24", "arXiv:2209.15425v2 (ICLR 2023)", "spikformer.layout.txt", 1072, 1112),
    ("Spike-driven Transformer Sec.4 energy text + Table 1", "arXiv:2307.01694v1 (NeurIPS 2023)", "sdt.layout.txt", 435, 470),
    ("Spike-driven Transformer Supp. S1 energy details", "arXiv:2307.01694v1", "sdt.layout.txt", 1098, 1131),
    ("Kundu et al. WACV 2021 Sec.4.3, Eq.11-13, Table 3-4",
     "doi:10.1109/WACV48630.2021.00400", "kundu_wacv.layout.txt", 444, 560),
    ("Kim et al. ICASSP 2022 Sec.3.4-4", "arXiv:2202.03133v2 / doi:10.1109/ICASSP43922.2022.9747906",
     "kim_rate_direct.txt", 170, 206),
    ("Dampfhoffer et al. TETCI abstract + intro", "doi:10.1109/TETCI.2022.3214509 (HAL cea-03852141 accepted version)",
     "dampfhoffer.txt", 20, 27),
    ("Dampfhoffer et al. scope", "doi:10.1109/TETCI.2022.3214509", "dampfhoffer.txt", 44, 44),
    ("Dampfhoffer et al. Eq.2, Eq.4 context, Eq.7, Table I", "doi:10.1109/TETCI.2022.3214509",
     "dampfhoffer.layout.txt", 186, 262),
    ("Dampfhoffer et al. Sec.II-E encoding-layer note and 1.38 bound", "doi:10.1109/TETCI.2022.3214509",
     "dampfhoffer.txt", 265, 269),
    ("Dampfhoffer et al. Table III target sparsity", "doi:10.1109/TETCI.2022.3214509",
     "dampfhoffer.layout.txt", 540, 575),
    ("Dampfhoffer et al. conclusion", "doi:10.1109/TETCI.2022.3214509", "dampfhoffer.txt", 414, 414),
    ("Davidson & Furber 2021 Eq.1-2 and result", "doi:10.3389/fnins.2021.651141", "davidson.txt", 126, 171),
    ("Lemaire et al. state of the art on metrics", "arXiv:2210.13107v1 / doi:10.1007/978-3-031-30105-6_48",
     "lemaire.txt", 36, 37),
    ("Lemaire et al. Sec.III-A ops incl. bias/reset/leak", "arXiv:2210.13107v1", "lemaire.txt", 63, 65),
    ("Lemaire et al. Eq.2 FC layer counts (layout)", "arXiv:2210.13107v1", "lemaire.layout.txt", None, None),
    ("Lemaire et al. constants + direct encoding", "arXiv:2210.13107v1", "lemaire.txt", 204, 212),
    ("Lemaire et al. results/memory dominance", "arXiv:2210.13107v1", "lemaire.txt", 292, 294),
    ("Yan et al. (preprint) abstract + conclusion", "arXiv:2409.08290v3", "yan2024.txt", 10, 10),
    ("Yan et al. (preprint) conclusion", "arXiv:2409.08290v3", "yan2024.txt", 1758, 1758),
    ("Eshraghian et al. Sec.3.1 input encoding", "arXiv:2109.12894v6 / doi:10.1109/JPROC.2023.3308088",
     "eshraghian.txt", 413, 414),
    ("Eshraghian et al. non-spiking data advice", "arXiv:2109.12894v6", "eshraghian.txt", 472, 472),
    ("Eshraghian et al. Sec.3.2 output decoding + Table 2 losses", "arXiv:2109.12894v6", "eshraghian.txt", 479, 534),
    ("Neftci et al. Sec.IV-B-1 surrogate derivatives", "arXiv:1901.09948v2 / doi:10.1109/MSP.2019.2931595",
     "neftci.txt", 521, 521),
    ("Wu et al. AAAI 2019 encoding/decoding", "arXiv:1809.05793v2 / doi:10.1609/aaai.v33i01.33011311",
     "wu2019.txt", 214, 275),
    ("Wu et al. AAAI 2019 simulation length", "arXiv:1809.05793v2", "wu2019.layout.txt", 380, 392),
    ("PopSAN Sec.2.2 population encoding/decoding", "arXiv:2010.09635v1 (CoRL 2020)", "popsan.txt", 58, 82),
    ("PopSAN population-size ablations", "arXiv:2010.09635v1", "popsan.txt", 161, 165),
    ("TabPFN-2.5 report distillation engine", "arXiv:2511.08667v2", "tabpfn25.txt", 31, 31),
    ("TabPFN-2.5 report distillation engine (Sec.3)", "arXiv:2511.08667v2", "tabpfn25.txt", 84, 84),
    ("TabPFN-2.5 report Sec.4.4", "arXiv:2511.08667v2", "tabpfn25.txt", 167, 168),
    ("TabPFN-2.5 report Sec.2.2 RL use", "arXiv:2511.08667v2", "tabpfn25.txt", 51, 51),
    ("GEM Sec.2 metrics Eq.2-4", "arXiv:1706.08840v6 (NIPS 2017)", "1706.08840.layout.txt", 136, 160),
    ("RWalk forgetting measure Eq.3", "arXiv:1801.10112v3 / doi:10.1007/978-3-030-01252-6_33",
     "1801.10112.layout.txt", 248, 270),
    ("snnTorch Regression Part I: membrane readout note",
     "https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_regression_1.html", "snn_tutorial_regression_1.txt", 120, 145),
    ("snnTorch Regression Part I: model definition", "same URL", "snn_tutorial_regression_1.txt", 205, 257),
    ("snnTorch Regression Part II: classification as regression",
     "https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_regression_2.html", "snn_tutorial_regression_2.txt", 94, 141),
    ("snnTorch Regression Part II: losses", "same URL", "snn_tutorial_regression_2.txt", 244, 310),
    ("snnTorch Population coding tutorial",
     "https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_pop.html", "snn_tutorial_pop.txt", 60, 196),
    ("snnTorch Tutorial 5: ATan default + membrane CE loss",
     "https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html", "snn_tutorial_5.txt", 170, 176),
    ("snnTorch Tutorial 5: loss loop", "same URL", "snn_tutorial_5.txt", 386, 420),
]


def main(txt_dir: str, out_file: str) -> int:
    root = Path(txt_dir)
    out = []
    for label, src, fname, a, b in RANGES:
        lines = (root / fname).read_text(encoding="utf-8", errors="replace").splitlines()
        if a is None:  # locate Lemaire Eq.2 by anchor
            a = next(i for i, l in enumerate(lines, 1) if "Fully-connected layers: The same" in l)
            b = a + 14
        out.append("=" * 100)
        out.append(f"[{label}] source={src} file={fname} lines={a}-{b}")
        out.append("-" * 100)
        out.extend(lines[a - 1:b])
    Path(out_file).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {len(RANGES)} excerpts to {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
