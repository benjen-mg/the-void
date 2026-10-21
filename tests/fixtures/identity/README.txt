Eight structures from CoRE MOF 2024's supporting-information subset, as
distributed in CoRE MOF Tools (CoREMOF/data/SI/CR.zip), unchanged. They
are the cases the round-62 survey of structure identity turned up
(docs/shared-index-design.md, Appendix A), used by
tests/test_index_identity.py:

  2020[Zn][dia]3[ASR]1.cif, 2020[Zn][qtz]3[ASR]1.cif  doi:10.1039/C9CC06849J
      dia and qtz Zn nets: same formula and local bonding, same key,
      different structures.
  0000[Co][nan]2[ASR]1.cif, 0000[Co][nan]3[ASR]10.cif  doi:10.1021/ja0c07257
      a 2D and a 3D Co structure with the same local bonding, which the
      key tells apart only by the component dimensionalities.
  2022[Cu][sql]2[ASR]2.cif, 2022[Cu][sql]2[ASR]4.cif  doi:10.1039/D1QI00349F
      two steps of one Cu sql series: the same structure, two
      determinations (densities 0.7344 and 0.7334 g/cm3).
  2020[Co][nia]3[ASR]1.cif, 2020[Co][nia]3[FSR]1.cif  doi:10.1021/jacs.9b12924
      CoRE's ASR and FSR copies of one structure: different bytes, the
      same determination.

G. Zhao, L. Brabson, S. Chheda, J. Huang, H. Kim, K. Liu et al., CoRE MOF DB:
a curated experimental metal-organic framework database with machine-learned
properties for integrated material-process screening, Matter 8, 102140 (2025).
Licence CC-BY-4.0.
