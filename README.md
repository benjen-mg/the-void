# the Void

Crystallography and porous-materials desktop software, free for research
and other non-commercial use.
Load a crystal structure, look at it in 3D, measure it, and run the
geometric analyses that matter for MOFs and other framework materials:
pore size, accessible surface area, guest accessibility, open metal
sites, coordination environments, and simulated powder XRD.

**Version 1.0.0**, the first public release. The analyses below have been checked against
analytic cases and against real deposited structures (openly licensed
CoRE MOF 2024 structures, bundled as test fixtures; CALF-20, CCDC
2084733; and MUF-16(Mn), CCDC 1948905, driven through every analysis
and UI action during testing), but this has not yet been used in
anger by anybody. Treat
numbers it reports as you would any new tool's: worth checking against
something you already trust before you rely on them.

## Installing

Python 3.11 or newer.

```bash
pip install -r requirements.txt
python main.py
```

Or as a package, which also installs the `the-void` (desktop
application) and `the-void-analyze` (command line) commands, and the
`thevoid` Python API:

```bash
pip install -e .
the-void
```

The heavy dependencies are PySide6 (the interface), PyVista and VTK
(the 3D view), pymatgen (structure parsing and crystallography), and
RDKit (2D diagrams). Expect a few hundred megabytes and a couple of
minutes on a first install.

### Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

1412 tests, no GPU needed. On a headless machine set
`QT_QPA_PLATFORM=offscreen` first; Qt will otherwise fail to find a
display. Some tests need outside files and are skipped without them:

- Three run a real RASPA2 binary (`RASPA_SIMULATE=/path/to/simulate`).
- 58 check topology against the MOF and zeolite CIFs that ship with
  RASPA2's source (`RASPA_STRUCTURES=.../RASPA2-2.0.4/structures`).

CI provides both.

## What it does

**Opening structures** — CIF (including disordered files, with the
`_atom_site_disorder_group` column read where present), VASP
POSCAR/CONTCAR, and pymatgen JSON. Bonds are detected from covalent
radii with an adjustable tolerance, and contacts too short to be real
bonds are reported rather than silently drawn.

**3D view** — ball-and-stick, space-filling or wireframe; unit cell
box; hydrogens on or off; molecules completed across cell boundaries
rather than clipped; supercells up to 10×10×10; adjustable opacity and
background; per-element colour customisation; an element key (each
element's colour and its sites per cell) at the top of the Inspector's
Structure tab. Fragments can be selected and deleted by clicking, and
restored again. Panel sizes and the window's size are kept from one
session to the next.

**Measuring** — distances, angles, and torsion (dihedral) angles,
resolved across periodic boundaries so a measurement to an atom in the
next cell gives the real distance rather than the one through the
middle of the cell. Torsions follow the IUPAC sign convention
(-180° to 180°), the same as Mercury and Olex2.

**Analysis**

| | what you get |
|---|---|
| Pore analysis | largest cavity diameter, pore limiting diameter, void fraction, accessible and non-accessible volume, and the dimensionality of the accessible network (isolated pockets / 1D channels / 2D layers / 3D network) |
| Surface area | accessible and geometric surface area, per gram and per cm³, for N₂, CO₂, H₂, H₂O or a custom probe |
| Guest accessibility | whether a given guest can diffuse through the framework, can only sit in isolated cavities, or does not fit at all |
| Open metal sites | undercoordinated metal centres, by coordination number, the directional imbalance of the coordination sphere, and its widest empty cone (so square-planar sites with both axial positions empty count too); covers Zn, Cu, Co, Ni, Fe, Cr, Mn, Mg, Cd, V, Sc, Ti, Al, Ga, Zr and Hf, and the lanthanides, Y, Th–Pu, In and Ag, whose saturated coordination numbers are calibrated on CoRE MOF 2024's labels (`tools/calibrate_oms.py`, 90% agreement on a held-out half, against 80% for the other metals); names any other metal present as not checked |
| Coordination environment | neighbour list and bond lengths for a selected atom, via CrystalNN |
| MOF partitioning | organic linkers separated from metal-containing nodes: one unit cell's worth, each SBU and linker once and whole, infinite rod SBUs flagged (topology splits them at their metals) |
| Powder XRD | simulated pattern for eight lab radiation sources, as sticks or as a pseudo-Voigt profile (with the Kα₁/Kα₂ doublet for Kα sources), with peak merging and intensity filtering |
| 2D skeletal diagram | idealised line diagrams of the framework's linkers and nodes |
| Topology | the underlying net and its RCSR code (tbo, pcu, fcu, sod…), shown in the Structure panel and every report; interpenetration counted; see below |

Every analysis runs on a background thread, and its result can be
overlaid on the 3D view.

**Hand-off to simulation: RASPA2** — Simulate > Export RASPA2 GCMC Input
Set... (also under File > Export Structure) writes a folder RASPA2 runs
as-is: a P1 framework CIF, a local force field (UFF Lennard-Jones for
the framework, TraPPE CO₂, N₂ or CH₄, Lorentz–Berthelot), the adsorbate
definition, `simulation.input` for a GCMC run at your temperature and
pressures, and a README listing every assumption. It uses what the
Void already knows about the structure: minor disorder dropped as in
every void-space analysis; lattice solvent and free counter-ions (bonded
components that are not periodic) removed and listed; the number of
unit cells worked out for the cutoff; and pockets the adsorbate can fit
into but not reach blocked, using the Pore Analysis grid. Partial charges
are taken from the CIF's `_atom_site_charge` column when there is one;
otherwise the framework is exported uncharged, with a warning (for CO₂
and N₂ this usually underestimates uptake), or RASPA's charge
equilibration can be chosen, run as a quick first step on one unit cell.
It is a starting point for a simulation, not a result.

**Topology** — the Structure panel's Topology row names the framework's
underlying net by its RCSR code as soon as a structure is loaded
(worked out in the background), and every report carries it next to
the space group. The framework is reduced by the *single-node*
convention used for most codes quoted in MOF papers. The metal clusters,
with their bridging atoms and the carboxylate groups that hold them
together, are the nodes; a carboxylate counts as part of a cluster when
it chelates one metal or bridges metals within 4.5 Å of each other (a
paddlewheel's, not an oxalate's, which holds its metals ~5.5 Å apart
and is a linker). Linkers that join three or more clusters are nodes too, and
two-connected linkers are edges. The net is then compared with RCSR
reference nets by an exact isomorphism test, Systre's method, not by
coordination sequences alone. Zeolites are reduced to their T-atom nets.
It reproduces the published nets of 51 real frameworks, among them
IRMOF-1 (pcu), HKUST-1 (tbo), UiO-66 (fcu), MOF-177 (qom), the ZIF series
and a dozen zeolites, and it counts interpenetration (IRMOF-9: 2-fold pcu).
CALF-20, described in the literature as Zn-triazolate layers pillared by
oxalate, comes out as dmc (3,4)-c: 4-c Zn and 3-c triazolate, with the
oxalates as pillar edges.
An infinite SBU, such as the rods of MIL-53 and MOF-74 or an inorganic
layer, is split at its metals, the way CrystalNets does it and so the
way CoRE MOF 2024 names these nets. Each metal is a
vertex. Two metals are joined when they share an atom (a μ-O or μ-OH,
or a carboxylate oxygen bridging both); a carboxylate spanning them
syn-syn does not join them. MIL-53 and MIL-47 are bpq. On the 72 CoRE
SI rod frameworks whose net CoRE names from RCSR, the Void gives the
same net for 69. The *all-node* net, with every carboxylate its own
vertex (MIL-53: rna), is reported beside it; that one matches CoRE's
for 52 of 59, since it does not split azolates. The Void reports rather
than guesses when the net is unstable (single-node rht MOFs), or when
it is not in the library. 95 common nets are bundled; to match against
all 2,930 RCSR nets, point `THE_VOID_RCSR_ARCHIVE` (or `the-void-analyze
--rcsr-archive`) at a Systre `.arc` file.

**Pore size distribution** — Analysis > Pore Size Distribution...
gives the geometric PSD (Gelb & Gubbins, the definition PoreBlazer
uses): the distribution plotted with its cumulative curve, exportable
as CSV.

**Reports** — File > Export Analysis Report... (Ctrl+E) saves the
structure summary and every analysis run on it so far as Markdown or
JSON. Each report records the software and library versions, every
parameter, the methods used, and a SHA-256 of the input file, so a
number in a paper can be traced back to exactly how it was produced.
Help > About lists the methods and the references to cite.

**Reference databases: CoRE MOF 2024 and MOFX-DB** — Analysis >
Compare with CoRE MOF 2024... finds the loaded structure in the
bundled CoRE MOF 2024 index (17,202 computation-ready entries,
CC-BY-4.0, Zhao et al., Matter 2025). It looks the structure up by
name first: a CoRE id, a CSD refcode or the CIF's data block. Failing
that, it compares metals, cell contents, density and mean atomic mass,
and every match says how sure it is. CALF-20, for example, is found
without its name as CoRE's TASYAR. The database's LCD, PLD, surface
areas, topology and open metal sites are then set beside the Void's
own, with the differences flagged. The Void's side is computed the way
CoRE MOF computed its values (Zeo++'s default CCDC radii, a 1.655 Å
probe), so a difference reflects the method, not a convention. File >
Fetch Reference Structures... downloads structures into a folder ready
for batch analysis: CoRE MOF's 2,664 openly licensed ones, chosen by
PLD, LCD, metals, topology and so on (the CSD-derived rest are listed
by refcode), or any MOFX-DB database, hMOF included. Each folder gets
the database's values in reference.csv and its citation and licence in
README.txt.

**Predicted properties** — Analysis > Predict Properties... estimates
four properties with models trained on CoRE MOF 2024: the water Henry
coefficient at 298 K, the thermal decomposition temperature, and the
probabilities of staying intact on solvent removal and in water. Each
model is an ensemble of five gradient-boosted tree models on 44
descriptors: pore geometry and surface area (computed as CoRE computes
them), density, the metals, the linkers' heteroatom make-up, the net,
open metal sites. Every value comes with a 90% interval, which on
held-out structures held 90% of the true values, and a flag for
structures unlike the training set. On held-out CoRE structures,
split by framework:

| Model | Test MAE | Always-the-mean MAE | R² | Spearman |
|---|---|---|---|---|
| Water Henry coefficient (log10 mmol/g/Pa) | 1.09 | 2.17 | 0.73 | 0.73 |
| Thermal decomposition (°C) | 30.1 | 50.0 | 0.60 | 0.72 |
| Stable on solvent removal (probability) | 0.144 | 0.224 | 0.48 | 0.67 |
| Stable in water (probability) | 0.088 | 0.170 | 0.68 | 0.81 |

Read these as screening estimates. CoRE's three stabilities are
themselves machine-learned predictions, not measurements, so those
models are surrogates of surrogates. The water Henry coefficients are
CoRE's Widom-insertion simulations; that the value is water's is
inferred from the data (the file does not label it). There is no
bundled gas-uptake model: MOFX-DB could not be reached from where this
version was built. `the-void-train` trains one from hMOF isotherms in
a few commands (below). `the-void-analyze --predict` adds the
predictions to every report and `pred_` columns to the summary.
Needs scikit-learn: `pip install -e ".[ml]"`.

## Command line

`the-void-analyze` runs the same analyses, with the same code and
defaults, on any number of files without opening a window. You get a
JSON and a Markdown report per structure, plus one summary CSV with a
row per structure:

```bash
the-void-analyze MOFs/*.cif --out results --psd --guests CO2,N2,benzene=5.85 -j 4
the-void-analyze CALF-20.cif --surface-probe CO2 --xrd
the-void-analyze MOFs/*.cif --raspa --raspa-adsorbate CO2 --raspa-pressures 0.1,1,10
the-void-analyze MOFs/*.cif --compare-coremof
the-void-analyze --help

the-void-fetch coremof --out coremof --pld-min 3.3 --metals Cu --limit 50
the-void-fetch coremof --pld-min 6 --topology pcu --dry-run
the-void-fetch mofxdb --database hMOF --out hmof --pld-min 4 --limit 200
the-void-analyze hmof/*.cif --out results --reference-csv hmof/reference.csv

the-void-analyze coremof/*.cif --out results --compare-coremof --figures
the-void-figures results --scatter pld_A channel_asa_m2_per_g --colour topology
the-void-figures results --hist pld_A --filter "errors is empty" --width double

the-void-functionalize IRMOF-1.cif --list
the-void-functionalize IRMOF-1.cif --groups NH2,Br,CH3 --per-linker 1,2
the-void-analyze IRMOF-1_functionalized/*.cif --out screen \
    --reference-csv IRMOF-1_functionalized/variants.csv --figures
the-void-figures screen --scatter pld_A channel_asa_m2_per_g --colour fn_group

the-void-analyze new/*.cif --out screen --predict                  # every bundled model
the-void-train list                                                # what they are, how good
the-void-train mofxdb-labels hmof --adsorbate CO2 --pressure 0.1 --out labels.csv
the-void-analyze hmof/*.cif --out results --reference-csv labels.csv -j 4
the-void-train fit results --target ref_uptake_CO2_0.1_298K --unit mol/kg --out co2_model
the-void-analyze new/*.cif --out screen --predict co2_model,water_stability

the-void-index init MyLab-index --name "McKelvey Lab"
the-void-index add MyLab-index results --contributor "B. G."
the-void-index build MyLab-index          # then File > Open Batch Results... on MyLab-index/build
the-void-index find MyLab-index new.cif   # has anyone analysed this structure?
the-void-index verify MyLab-index
the-void-index stale MyLab-index          # numbers a later version computes differently
the-void-index refresh MyLab-index --contributor "B. G." -j 4   # analyse those again, 4 at once
the-void-train fit MyLab-index --target ref_uptake_CO2_0.1_298K --out co2_model
```

**A shared index** — `the-void-index` keeps every analysis a group has
run in one folder, best kept as a Git repository, with who ran it, with
which version and options (docs/shared-index-design.md).

- **Entries** are the reports, with local paths removed. Each is named
  by its content, so two people adding to their copies merge with no
  conflict.
- **Structure files are copied in only when their licence allows it:**
  CoRE MOF's SI files from `the-void-fetch`, or files you declare. A
  file with the CSD's header never is.
- **Which entries are the same structure** is worked out from the
  structure itself, not its name. A rewritten or reordered file is
  recognised as the same determination.
- **`build`** writes a results folder that the Batch Results window
  ranks, filters and opens.
- **`verify`** finds hand-edited entries, tampered files, and repeat
  analyses of the same structure that disagree.
- **In the application:**
  - File > Shared Index > Set Index Folder... chooses, or makes, the
    group's index. Every structure you open is then looked up in it,
    and the Structure panel's "Previously Analysed" row says whether it
    is there, as the same file, the same determination or the same
    structure. The tooltip lists who analysed it, with which version,
    and the PLD.
  - Add Current Structure... adds what you have run on it.
  - Batch Results > Add to Shared Index... adds the selected rows.
  - Open Index... builds the index and opens it in the Batch Results
    window.
- **From Python:** `thevoid.Index`.
- **Out of date numbers.** A table of which versions
  changed which numbers (`thevoid_core/data/method_changes.json`)
  marks the index's stale entries.
  - `the-void-index refresh` analyses them again with the same options,
    and keeps the old entries, marked superseded.
  - `the-void-train fit` trains on an index directly: its current
    entries of one method, with each structure's entries kept on one
    side of the test split.

`--compare-coremof` adds the CoRE MOF comparison to each report and
`coremof_` / `ccdc_` columns to the summary. `--reference-csv` joins
the values from a folder written by `the-void-fetch`, as `ref_`
columns.

If one file or one analysis fails, the rest still run; the failure is
recorded in that file's report and in the summary's `errors` column,
and the exit status is 1. `python -m thevoid_core.cli` works too.

**Ranking a batch** — File > Open Batch Results... (Ctrl+Shift+O)
opens a results folder in its own window. Click a column header to rank
by it, largest or smallest first. Rows with an empty cell (an analysis
that was skipped or failed) always go last, so they never top a
ranking. Add filter rows such as *PLD ≥ 3.3*, *Topology contains pcu*
or *Errors is empty*, search every column, show only the top N, and
export the view as CSV. Selecting a row shows its report; double-clicking
it opens the structure in the main window. Before loading, the file is
found again from the path the report recorded (even when that path was
relative, or the folder has moved) and checked against the report's
SHA-256. A file that has changed since the batch ran is only loaded if
you say so.

**Functionalised variants** — Build > Functionalise Linkers..., or
`the-void-functionalize`, makes hypothetical variants of a MOF: ring
hydrogens of its linkers replaced by -F, -Cl, -Br, -I, -CH3, -CF3, -OH,
-OCH3, -NH2, -NO2 or -CN (IRMOF-1 to IRMOF-3, say). Linkers related by
symmetry are substituted the same way. Patterns are counted once up to
symmetry: for terephthalate, one with one group per linker, ortho, meta
and para with two. Groups are placed rigidly from standard bond lengths
(Allen et al. 1987) on the ring bisector, and the framework is not
moved. A crowded group may bend a few degrees in the ring plane, as
real ones do; conjugated groups stay in the ring plane where there is
room, and rotors are turned for the most room. If that still leaves the
variant tight or clashing, its rings are turned about their linker axis
by the smallest angle, in 5° steps, that clears it, the
same on every linker by symmetry, and the turn is recorded. Every variant is
checked: a contact under 75% of the van der Waals sum rejects it
(hydrogen bonds excepted), one under 80% flags it "tight", and the
bond detector must see exactly the bonds each group was placed with.
The thresholds come from 160 real functionalised MOFs in CoRE MOF 2024
(1,706 groups): each real group, stripped to H and rebuilt here, lands
a median 0.08 Å from where it really is, and 158 of the 160 pass. The
two that fail are phenolates bonded to Co. The output folder holds a
P1 CIF per variant and the parent. It also holds
`functionalization.json`, recording every length, angle, threshold and
contact, and `variants.csv`, which `--reference-csv` joins into a batch
as `fn_` columns for ranking and figures.

**Figures for a paper** — Figure... in the Batch Results window, or
`the-void-figures` / `the-void-analyze --figures`, draws a scatter
(e.g. PLD against channel surface area, coloured by topology), a
histogram, or a parity plot of the Void against CoRE MOF 2024. The
figure is drawn from the rows as shown, filters and search applied, at
journal column width (85 or 175 mm) with 7 pt text, and saved as SVG,
PDF and 600-dpi PNG. The PDF embeds its fonts and the SVG keeps its
text editable. Beside each figure a JSON records every point, every row
left out and why, the filters, the style, and the SHA-256 of the
summary it came from. At most three categories get their own colour,
each with its own marker too, so the figure also reads in greyscale and
for colour-blind readers. The rest are grouped as "Other". A parity
plot uses only structures CoRE identified as the same, and shows the
comparison's tolerance band (0.15 Å for LCD and PLD, 5% for surface
area) with the median and mean absolute difference.
`--raspa` also writes a RASPA2 input folder per structure
(`<out>/<name>_RASPA_<adsorbate>/`) and records what it assumed in that
structure's report.

## From Python

`import thevoid` (installed with the package) gives a
script or a notebook the same analyses, with the same code and
defaults:

```python
import thevoid

fw = thevoid.load("CALF-20.cif")        # or a pymatgen Structure, or ASE Atoms
fw.pores().pld_A                        # 2.799
fw.guest("CO2").status                  # "partially_accessible"
fw.surface_area("N2").channel_asa_m2_per_g
fw.topology().summary                   # "dmc (3,4)-c"
fw.compare_coremof().flagged            # rows that differ from CoRE MOF 2024

table = thevoid.analyze_batch(paths, jobs=4, out="results", guests=["CO2"])
table.to_dataframe()                    # if pandas is installed
thevoid.figure(table, "scatter", "pld_A", "channel_asa_m2_per_g", save="fig")
fw.functionalize(["NH2", "Br"], per_linker=[1, 2]).write("variants")
fw.predict()["water_stability"]         # value, 90% interval, in_domain, caveat
index = thevoid.Index("MyLab-index")    # the shared index: add, build, table, find
```

Results are named as the reports and summary CSV name them, with
units: `lcd_A`, `channel_asa_m2_per_g`. The same file gives exactly
the numbers the-void-analyze reports. `docs/python-api.md` is the
guide, and every example in it is run by the tests. ASE and pandas are
optional: `pip install -e ".[api]"`. The names in `thevoid.__all__`
are the public API. `thevoid_core`, the package behind the
application, is internal.

## When something goes wrong

The application writes a rotating log to `~/.the_void/logs/`. Help >
Open Log Folder opens it. Help > Copy Diagnostics puts the version
information, the loaded file and the end of the log on the clipboard
for a bug report. An unexpected error now shows a message instead of
failing silently.

**Exporting** — the view as PNG, XRD peak lists and profile traces as
CSV, any results table (OMS sites, coordination neighbours, linkers and
nodes, measurements, XRD peaks) copied to the clipboard with Ctrl+C or
right-click > Copy All / Export CSV, and structures as CIF, extended XYZ, VASP POSCAR, Gaussian,
CP2K, or Quantum ESPRESSO input, or as a RASPA2 GCMC input set (above).

## Things worth knowing before you trust a number

These are properties of the methods, not defects, and they are
documented in the relevant functions' docstrings too.

- **LCD and PLD don't depend on the grid; the volumes do.** The
  largest cavity diameter is the exact maximum of the distance to the
  atom surfaces. The pore limiting diameter is found on the atoms'
  radical Voronoi network, the Zeo++ method. Near-critical windows and
  vertices are refined, and a refined window only counts
  once an actual path through it has been found and its clearance
  computed exactly. The grid's own percolating path sets a proven
  floor under the PLD. Both are exact on simple test lattices. On
  CALF-20 they give LCD 4.284 Å and PLD 2.799 Å; Zeo++'s high-accuracy
  mode, with the same radii, gives 4.304 Å and 2.825 Å. That mode
  slightly overstates the LCD: a brute-force maximisation confirms
  4.284. On 76 CoRE MOF 2024 structures, computed CoRE's way, the
  Void's PLD comes out a median 0.03 Å below CoRE's, and never more
  than 0.12 Å from it. Where atoms of very different sizes sit side by
  side, the radical network can miss a channel altogether; layered
  frameworks are the typical case. So whenever the radii differ, a
  second network is built with the larger atoms split into spheres of
  the smallest radius. That network follows the true channels, and its edges are
  valued exactly.
  Wherever the grid proves more than either network finds, the grid
  floor is reported. In both cases the result says which method it
  used. Void fraction, accessible volume and the 3D accessible
  points still come from the 0.3 Å grid.
- **Which void space is connected uses the Voronoi network too.** The
  grid alone cannot see a window only slightly wider than the probe
  (UiO-66 and N₂: all surface "pocket" on the grid alone). So the
  atoms' Voronoi network, whose edges run through window centres, is
  joined into the grid's percolation for accessible volume, channel
  ASA and guest accessibility. On nine frameworks (UiO-66, HKUST-1,
  IRMOF-1, MOF-205, MIL-88B, Zn2(BDC)2(dabco),
  ZIF-8, ZIF-71 and CALF-20) the channel/pocket split then matches
  Zeo++. In the last few hundredths of an ångström before a window
  closes, a few percent of the surface right at the window can still be
  counted as pocket.
- **Surface area is sampled.** Each atom gets 1000 evenly spread
  points in a random orientation. Different seeds give 1–1.5% scatter
  on CALF-20 and MUF-16, and more on small cells: 1456–1480 m²/g over
  three seeds for the 36-atom Cr-BDC. The mean is within 1% of Zeo++.
  The orientations are drawn in the cell's own frame, so
  the same crystal gives the same points however its axes lie in
  space; from a CIF or from ASE, the numbers are identical. They are
  drawn in a canonical order of the atoms, so a file
  that lists the same atoms in another order, or the same file
  rewritten, gives identical numbers too. Moving the origin or taking a
  supercell still changes them within the scatter above. The
  channel/pocket split follows Zeo++'s ASA/NASA definitions.
  "ASA per mass" includes pocket surface; "Channel ASA per mass" is
  the figure Zeo++ and most screening studies report.
- **Comparing with a database means using its convention.** CoRE MOF
  2024's values are Zeo++'s, with Zeo++'s default (CCDC) radii: H
  1.09 Å, Cu 1.40, Zn 1.39, and 2.00 Å for most metals. The Void's own
  analyses use about 2 Å for those metals and 1.20 Å for H, and a
  1.82 Å N₂ probe where CoRE uses 1.655 Å. The Compare window and
  `--compare-coremof` therefore recompute with CoRE's convention, and
  their rows say so. The `ref_` columns from `--reference-csv` are the
  database's numbers exactly as published: read them against the
  `ccdc_` columns, not against the Void's standard ones.
- **A functionalised variant is a starting geometry, not a relaxed
  structure.** Real MOFs relax around a crowded group: the rings of
  IRMOF-2 and MIL-47(V)-Br twist 19–26° against their carboxylates.
  The build moves only as much as it must. A ring turns only when the
  flat build is tight or clashes, and then by the smallest angle that
  clears it: 15° for IRMOF-1 + -Br, and for -Cl in the narrow channel of
  Cr-BDC, which the flat build rejects. That is less than real
  structures twist. A ring whose axis is not defined by just two
  outward bonds (the core of a tricarboxylate, or a phenolate bonded to
  a metal) is not turned. Relax a variant (DFT, a force field) before
  trusting its numbers closely.
  - **One ordering per pattern.** Each pattern is carried to every
    linker by one symmetry operation, which gives one ordered model.
    Other orderings of the same pattern can differ: IRMOF-1 with one
    -Br per BDC comes out with every Br in the small cavities (LCD
    14.45 Å, PLD 7.13 Å with its rings turned 15°). RASPA2's IRMOF-2
    model puts them in the large ones (12.89 Å, 8.40 Å). Void fraction
    and channel surface area agree within 2.5%.
  - **-NH2 matches.** IRMOF-1 with one -NH2 per BDC gives the same PLD
    as RASPA2's IRMOF-3 model (7.05 Å) and a channel ASA within 0.6%.
- **Open metal sites are a geometric test, not CoRE's definition.** A
  site is open when it has fewer neighbours than its metal's saturated
  number and its coordination sphere has a gap. On half of CoRE MOF
  2024's SI structures, held out, the Void agrees with CoRE's flag for
  80% of the long-established metals (Zn is the main difference:
  CoRE counts many four-coordinate Zn as open) and for 90% of the
  calibrated ones below. The saturated numbers of the
  lanthanides with Y (9), Th–Pu (7), In (7) and Ag (5) are fitted to
  CoRE's labels, not taken from textbooks; the fit and its scores are
  in `data/oms_calibration.json`. Metals with too few labelled
  structures (alkali and alkaline-earth metals, Pb, Bi, Mo, Pd, ...)
  are not checked, and reports say so.
- **The probe radii are one convention among several.** Kinetic
  diameters and Lennard-Jones σ differ by a few tenths of an ångström
  for the same gas. The values used are in `data/probe_radii.py`; use
  a custom probe to match a specific paper.
- **Disorder is handled by dropping minor components.** Every
  void-space analysis (pore, surface area, guest accessibility) first
  removes sites below 0.5 occupancy, because a distance field treats
  every atom as permanently present, and counting all alternatives of
  a disordered site at once can close pores that are really open. This
  matters: on CALF-20 it is the difference between a pore limiting
  diameter below 1 Å and 2.7 Å.
- **The RASPA2 export is geometry-informed, not physics-checked.** UFF
  is a generic force field and underbinds open metal sites; an
  uncharged framework misses CO₂'s and N₂'s electrostatics; pocket
  blocking uses hard-sphere radii, so a structure whose windows are
  narrower than the probe (CALF-20 and CO₂) is left unblocked, with a
  warning, rather than blocked shut. Every one of these is written into
  the folder's README.
- **Exports to POSCAR and Quantum ESPRESSO cannot store occupancies.**
  Exporting a disordered structure to either will offer you the same
  ordered approximation the analyses use, and tell you how many sites
  it drops. CIF keeps the occupancies.

## Project layout

```
thevoid_core/
  app/             application entry point
  ui/              windows, dialogs, panels, the 3D viewer widget
  services/        structure loading, bonding, and every analysis
    reference/     CoRE MOF 2024 index, comparison, fetching
    ml/            property models: descriptors, training, prediction
    index/         the shared index: identity, entries, licences, store
  data/            element radii, colours, metal classification,
                   RCSR nets, CoRE MOF 2024 index (CC-BY-4.0),
                   models/: the bundled property models
  exports/         XRD CSV writers
  cli.py           the-void-analyze batch tool
  fetch_cli.py     the-void-fetch
  figures_cli.py   the-void-figures
  functionalize_cli.py  the-void-functionalize
  train_cli.py     the-void-train
  index_cli.py     the-void-index
thevoid/           the public Python API: load, Framework,
                   analyze_batch, figure, search/fetch CoRE MOF
docs/python-api.md the API guide (its examples are tests)
docs/shared-index-design.md  design for a lab-shared index of analyses
tools/             build_topology_library.py, build_coremof_index.py,
                   build_coremof_linkers.py, train_bundled_models.py,
                   index_identity_survey.py, calibrate_oms.py
.github/workflows  CI: full test suite + CLI smoke test, Python 3.11/3.12
tests/             1412 tests, with openly licensed real-structure fixtures
```

## How to cite

If the Void, or numbers it produced, appear in a publication, please
cite it:

> B. M. Garmejani, the Void, version 1.0.0 (2026). Software, https://github.com/benjen-mg/the-void.

```bibtex
@software{the_void,
  author  = {Garmejani, Benyamin M.},
  title   = {the Void},
  version = {1.0.0},
  year    = {2026},
  url     = {https://github.com/benjen-mg/the-void},
  note    = {Licensed CC-BY-NC-ND-4.0}
}
```

Cite the version you used. GitHub's "Cite this repository" button reads
the same details from `CITATION.cff`, and Help > About the Void shows
them too. Please also cite the methods behind the numbers you report;
Help > About the Void lists them.

## Licence

The Void is licensed under the Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0 International licence
(CC BY-NC-ND 4.0); see `LICENSE`. In short: you may use it, and copy and
share it unchanged, for research, teaching and other non-commercial
purposes, provided you give credit (cite it as above). Commercial use
needs the author's permission. You may not distribute modified versions.
It comes with no warranty.

Material that is not the author's own keeps its own licence: see
`NOTICE.md`. The bundled CoRE MOF 2024 index (`data/coremof2024.json.gz`) is a
trimmed copy of the CoRE MOF database's property file, CC-BY-4.0:
G. Zhao et al., Matter 8, 102140 (2025). Its attribution is carried in
the file and shown wherever its values are.
