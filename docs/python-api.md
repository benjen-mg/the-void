# The Void from Python

`import thevoid` gives you what the GUI and the command-line tools do,
from a script or a notebook. It runs the same code with the same
defaults, so a file gives the same numbers whichever way you run it.
Results use the names the JSON reports and summary CSV use, units
included: `lcd_A`, `pld_A`, `channel_asa_m2_per_g`.

Every example on this page is run by the test suite
(`tests/test_api_docs.py`) from the repository root, so they are kept
correct.

## One structure

```python
import thevoid

# A Zn MOF from CoRE MOF 2024's openly licensed set (CC-BY-4.0)
fw = thevoid.load("tests/fixtures/identity/2020[Zn][dia]3[ASR]1.cif")
print(fw)                         # Framework('2020[Zn][dia]3[ASR]1', Zn4H40C48N16O16, P2_12_12, 124 sites)

pores = fw.pores()                # probe radius 1.2 Å, as Analysis > Pore Analysis
print(round(pores.lcd_A, 3), round(pores.pld_A, 3))   # 5.11 4.078
assert pores.accessible_dimensionality == 3

co2 = fw.guest("CO2")             # kinetic diameter 3.05 Å
print(co2.status)                 # accessible
big = fw.guest("big", diameter=4.6)
print(big.status)                 # partially_accessible: fits the cavities, not the windows
assert big.status == "partially_accessible"

print(fw.topology().summary)      # dia 4-c
print(fw.partition().linker_formulas())   # {'C6H5N2O2': 8}
```

`load` takes a file path (CIF, or any format pymatgen reads), a
pymatgen `Structure`, or an ASE `Atoms`. A disordered CIF keeps its
disorder: every analysis drops the sites below 0.5 occupancy, as the
GUI does.

Every result has `to_dict()`, which returns the report's dictionary,
and prints its fields:

```python
sa = fw.surface_area("N2", samples_per_atom=300)
print(round(sa.channel_asa_m2_per_g), sa.pocket_surface_area_A2 > 0)   # 1115 True: a little is pocket
d = sa.to_dict()
assert d["probe"] == "N2" and d["samples_per_atom"] == 300
```

Probes and guests are named (`thevoid.PROBES`: N2, CO2, H2, H2O, CH4)
or given as a radius in Å, as with `fw.pores(1.655)`. To compute the
way CoRE MOF 2024 did, pass `radii="ccdc"`: Zeo++'s default radii.

```python
core_style = fw.pores(1.655, radii="ccdc")
print(round(core_style.pld_A, 2))   # 4.12; CoRE MOF 2024 lists 4.15
```

Results are cached by their arguments, so calling `fw.pores()` again
costs nothing.

## Everything at once, and reports

```python
report = fw.analyze(surface_samples_per_atom=300, guests=["CO2", "N2"], psd=True)
print(report.ok, sorted(report["results"]))
report.write("zn_dia.json")        # or .md; the same files the-void-analyze writes
row = report.row                   # its line of the batch summary
print(row["pld_A"], row["guest_CO2"])
```

`thevoid.analysis_options()` lists every option and its default; they
match the-void-analyze's. An option name that does not exist raises an
error; it is never silently ignored.

## Many structures

```python
import glob

files = sorted(glob.glob("tests/fixtures/coremof/*.cif"))[:3]
table = thevoid.analyze_batch(files, out="screen", jobs=2,
                              surface_samples_per_atom=200, oms=False, topology=False)
print(table)                       # BatchTable(3 structures, 0 with errors)
for row in table.rows:
    print(row["file"], row["pld_A"], row["channel_asa_m2_per_g"])
```

`out=` writes `summary.csv` and a report per structure, the same folder
`the-void-analyze --out` writes. File > Open Batch Results... and
`the-void-figures` can open it. Inputs can be paths, Frameworks,
Structures or Atoms, mixed. A structure that did not come from a file
is also written to `out` as a CIF, so every row can be opened and
checked.

With pandas installed:

```python
try:
    df = table.to_dataframe()
    print(df[["file", "pld_A", "channel_asa_m2_per_g"]].sort_values("pld_A"))
except ImportError as exc:        # pandas is optional
    print(exc)
```

## Figures

```python
fig = thevoid.figure(table, "scatter", "pld_A", "channel_asa_m2_per_g",
                     save="screen/figures/pld_vs_asa")
print(len(fig.points), fig.excluded)
```

The figure is the same as the-void-figures draws: 85 mm wide, and saved
as SVG, PDF and 600-dpi PNG with a JSON record beside them. In Jupyter
it shows inline.

## Functionalised variants

```python
cr = thevoid.load("tests/fixtures/coremof/2010[Cr][bpq]3[ASR]1.cif")
print(cr.functionalization_sites())          # orbits and patterns, as --list shows
variants = cr.functionalize(["NH2", "F"], per_linker=[1, 2])
for v in variants:
    print(v.name, v.status, v.closest_contact_A, v.ring_turn_deg)   # rings turned only if crowded
best = variants.accepted()[0].framework      # a Framework: analyse it directly
print(round(best.pores().pld_A, 2))
variants.write("cr_variants")                # CIFs, parent, functionalization.json, variants.csv
```

A variant that is tight or clashes with its rings as they are gets its
substituted rings turned about the linker axis by the smallest angle
that clears it, the same on every linker by symmetry
(`ring_turn=False` builds them flat).

To screen them:

```python
screen = thevoid.analyze_batch(sorted(glob.glob("cr_variants/*.cif")),
                               surface_samples_per_atom=200, oms=False, topology=False)
screen.join("cr_variants/variants.csv")      # adds the fn_ columns: group, pattern, check
print([(r["file"], r["fn_group"], r["pld_A"]) for r in screen.rows][:3])
```

## CoRE MOF 2024

```python
hits = thevoid.search_coremof(metals="Cu", pld_min=6, downloadable_only=True, limit=5)
print(len(hits), sorted(hits[0])[:4])

match = fw.coremof_matches()[0]
print(match)          # ('2020[Zn][dia]3[ASR]1', 'identified', 'CoRE MOF 2024 id 2020[Zn][dia]3[ASR]1')
```

`fw.compare_coremof()` sets the Void's values beside CoRE's, computed
CoRE's way. `thevoid.fetch_coremof("folder", pld_min=4, limit=20)`
downloads the openly licensed structures. Both need a few minutes, or
the network, so they are not run here.

## Property models

Four models ship with the Void, trained on CoRE MOF 2024: the water
Henry coefficient, and the thermal, solvent-removal and water
stabilities. They need scikit-learn (`pip install -e ".[ml]"`).

```python
print(sorted(thevoid.models()))   # the bundled models; each value is its card
predictions = cr.predict(samples_per_atom=300)
w = predictions["water_stability"]
print(f"{w.value:.2f} ({w.low:.2f}-{w.high:.2f}), in domain: {w.in_domain}")   # about 0.54 (0.24-0.84)
print(w.caveat)                   # CoRE's stabilities are themselves ML predictions
print(predictions.notes)          # what could not be computed (nothing, here)
```

Every prediction has a 90% interval (split-conformal: about 90% of
held-out CoRE structures fell inside theirs) and `in_domain`, which is
False for a structure unlike anything the model was trained on. The
descriptors are computed as CoRE MOF 2024 computes its own (CCDC radii,
1.655 Å probe), and those pore and surface results are cached on the
Framework like any other.

These are estimates, not measurements. Read `thevoid.models()[name]`
for what each was trained on and how well it did on held-out
structures (`test_metrics`). The README has the limits.

To train your own on a batch, for example on MOFX-DB uptakes:

```text
rows, info = thevoid.mofxdb_labels("hmof", "CO2", 0.1, out="labels.csv")
table = thevoid.analyze_batch(glob.glob("hmof/*.cif"), out="results", reference_csv="labels.csv")
model = thevoid.train("results", info["column"], unit="mol/kg")
model.save("co2_model")           # then fw.predict(["co2_model"]), or --predict co2_model
```

The same from the command line is `the-void-train mofxdb-labels` and
`the-void-train fit`. A model trained this way reads the batch's own
summary columns, and at prediction runs the same analyses with the
batch's options.

## A shared index

`thevoid.Index` is `the-void-index` from Python: a folder where a group
keeps every analysis it has run (docs/shared-index-design.md).

```python
index = thevoid.Index.create("lab-index", name="Docs Lab")
print(index.add(table, contributor="A. B."))   # AddSummary(3 added, ..., 3 withheld)
index.build()
print(len(index.table()), index.table().columns[-3:])
for match in index.find(cr):
    print(match.relation, match.file_name, match.the_void_version)   # same file ...
print(index.verify())                          # []
```

- **Adding.** `add` takes a results folder, a BatchTable written to
  one, or a Report together with its Framework (`fw.analyze()`, `fw`).
- **Structure files are withheld here** because nothing says these
  copies may be redistributed. Files fetched with `thevoid.fetch_coremof`
  carry their CC-BY-4.0 licence and would be stored. A file with the
  CSD's header never is. `licence=` and `licence_basis=` declare a
  licence for files nothing else decides.
- **`find`** tells the same file from the same determination (the same
  atoms in another file) and from the same structure (another
  determination of it).
- **Out of date entries.** `thevoid.train(index, "ref_uptake_CO2_0.1_298K")`
  trains on the index's current entries of one method.
  `the-void-index stale` lists entries whose numbers a later version of
  the Void computes differently, and `the-void-index refresh` analyses
  them again.

## ASE and pymatgen

```python
import ase.io

atoms = ase.io.read("tests/fixtures/coremof/2010[Cr][bpq]3[ASR]1.cif")
same = thevoid.load(atoms, name="Cr-BDC from ASE")
assert abs(same.pores().pld_A - cr.pores().pld_A) < 1e-6
back = cr.to_ase()                 # ordered structures only
```

## Stability

The names in `thevoid.__all__` are the public API, together with the
fields and methods documented on what they return. They change only
with a major version. A test (`tests/test_api.py`) pins them, so no change
can happen by accident. `thevoid_core`, the package behind the
application, is internal, and so is every result's `raw` attribute.
