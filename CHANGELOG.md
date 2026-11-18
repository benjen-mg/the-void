# Changes

What changed in each published version of the Void, newest first. Every
version is on the [releases page](https://github.com/benjen-mg/the-void/releases),
with the Windows installer, and on Zenodo (doi:10.5281/zenodo.23578711).

Version numbers: the third number changes for a fix, the second when
something new is added, the first if results or the Python API change
in a way that could break existing work.

## 1.1.1 (19 November 2026)

**New licence: PolyForm Noncommercial 1.0.0**, replacing CC BY-NC-ND 4.0,
from this version on. Nothing else changed from 1.1.0. The Void stays free
for research, teaching and other non-commercial use, and commercial use
still needs permission. What changes: the licence is written for software,
includes a patent licence, says plainly that universities and public
research organisations may use it however they are funded, and allows
sharing modified versions for non-commercial purposes. Versions 1.0.0 to
1.1.0 stay under CC BY-NC-ND 4.0.

## 1.1.0 (15 November 2026)

**New: SHELX files and refinement details.** For people who solve and
refine structures in Olex2 or SHELXL and want to look at the framework
before a CIF exists.

- Opens SHELX `.res` and `.ins` files: File > Open, drag and drop,
  `the-void-analyze` and `thevoid.load`. Free variables, special
  positions, `PART` disorder, residues and riding hydrogens are read, so
  occupancies and cell contents are those of the refinement.
- Shows what a file reports about its refinement, from a SHELXL `.res`
  or from a CIF: R1, wR2, goodness of fit, data / restraints /
  parameters, the largest difference peak and hole, temperature and
  wavelength, and whether a solvent mask or twinning was used. It is in
  a Refinement box on the Structure tab, in the JSON and Markdown
  reports, and is `Framework.refinement` in Python.
- Checked against SHELXL: a `.res` and the CIF SHELXL wrote from the same
  refinement give the same sites, occupancies, disorder groups and
  refinement numbers.

The Void reads these files. It does not refine structures, and it does
not yet draw displacement ellipsoids.

## 1.0.1 (28 October 2026)

Not released on its own; included in 1.1.0.

- **Fixed:** every CIF the Void writes now has unique atom labels
  (Zn1, Zn2, ... C1, C2, ...), as the CIF standard requires. Before, the
  symmetry copies of an atom all kept the label of the atom they came
  from, and Olex2, SHELX and other programs rejected or renamed them.
  This applies to File > Export CIF, `Framework.write_cif`, batch CIFs
  and functionalised variants.

## 1.0.0 (21 October 2026)

The first public version: the desktop application, the command-line
tools and the `thevoid` Python package, with a Windows installer.
The [README](README.md) describes everything in it.
