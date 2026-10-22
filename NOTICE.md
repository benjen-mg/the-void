# Third-party material

The Void's own code, documentation and artwork are licensed under
CC BY-NC-ND 4.0 (see LICENSE). The material below is not the author's
own. It is included under its own licence, which still applies to it:
you may use it under that licence, from its original source, whatever
LICENSE says.

## CoRE MOF 2024 (CC-BY-4.0)

G. Zhao, L. Brabson, S. Chheda, J. Huang, H. Kim, K. Liu et al., "CoRE MOF
DB: a curated experimental metal-organic framework database with
machine-learned properties for integrated material-process screening",
Matter 8, 102140 (2025). Licence: CC-BY-4.0,
https://creativecommons.org/licenses/by/4.0/

- `thevoid_core/data/coremof2024.json.gz`: a trimmed copy of the database's
  property file (some columns kept, the rest dropped;
  `tools/build_coremof_index.py` lists which).
- `thevoid_core/data/coremof2024_linkers.json.gz`: each entry's linkers as
  heavy-atom counts, worked out from the database's MOFid strings
  (`tools/build_coremof_linkers.py`).
- `thevoid_core/data/oms_calibration.json`: open-metal-site thresholds
  chosen to agree with the database's open-metal-site labels
  (`tools/calibrate_oms.py`).
- `thevoid_core/data/models/*`: property models trained on the database's
  values (`tools/train_bundled_models.py`); each `card.json` records its
  source and licence.
- `tests/fixtures/coremof`, `identity`, `oms`, `rods` and `open`: structure
  files from the database's supporting-information subset. Each folder's
  README.txt names the files and their DOIs, and says what, if anything,
  was changed.
- `tests/fixtures/mofxdb_page*.json` carry one of those files as a stand-in
  CIF.

## RCSR (Reticular Chemistry Structure Resource)

M. O'Keeffe, M. A. Peskov, S. J. Ramsden, O. M. Yaghi, "The Reticular
Chemistry Structure Resource (RCSR) database of, and symbols for,
crystal nets", Acc. Chem. Res. 41, 1782 (2008); http://rcsr.net

- `thevoid_core/data/rcsr_nets.json`: net symbols and their quotient graphs,
  built by `tools/build_topology_library.py`.

## Zeo++

T. F. Willems, C. H. Rycroft, M. Kazi, J. C. Meza, M. Haranczyk,
Microporous Mesoporous Mater. 149, 134 (2012).

- `tests/fixtures/zeopp_default_radii.json`: Zeo++'s default radius table,
  read back from Zeo++ to check the Void's copy of it.

## Python packages

The repository does not include the Python packages the Void depends on
(PySide6, PyVista, VTK, pymatgen, RDKit, NumPy, SciPy, scikit-learn and
others): pip installs them, each under its own licence.

The Windows installer does include them, with Python itself, unchanged.
Their licences are in THIRD-PARTY-LICENSES.txt next to the program. Qt
and PySide6 are used under the GNU LGPL v3: they are separate libraries
in the installed folder and can be replaced with other builds of the
same version.
