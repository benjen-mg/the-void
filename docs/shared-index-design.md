# A shared index of analysed structures — design

Status: design (round 62, roadmap 4b). Being implemented in steps
(section 12): step 1 is done in 0.9.3, step 2 in 0.9.4
(`thevoid_core/services/index/identity.py`) step 3 in 0.9.5
(`the-void-index`; section 7 describes what was built) step 4 in
0.9.6 (the GUI and `thevoid.Index`) and step 5 in 0.9.7 (stale entries,
refresh, training from an index). The owner left the open questions
to the designer (section 13 records what was decided). The measurements quoted
were made for this document on real structures. Appendix A says how,
and the scripts are described there so that they can be rerun.

## 1. What it is for

A research group that uses the Void analyses the same frameworks
again and again. Different people analyse them, with different versions
and options, and the numbers end up in folders that nobody else can
find. The shared index is one place where every analysis a group has
run is kept with its provenance. It lets anyone ask:

- "Has anyone in the group analysed this structure, with which version
  and options, and what did they get?"
- "Rank every Cu MOF we have analysed by PLD", across all members'
  batches.
- "Which of our numbers are stale because the Void changed since?"
- "Give me a training set for the-void-train", with every row computed
  the same way.

The owner chose the following (2026-09-29):

- **Lab first, public later.** The first version is for one group,
  shared through a folder or a Git repository, with no server. The
  format must allow a later public release, as CoRE MOF is released:
  one versioned, citable archive.
- **Results always; structures only when licensed.** Every entry
  carries its numbers and full provenance. The structure file itself is
  included only when its licence allows redistribution. Otherwise the
  entry carries a reference to it (a CSD refcode, a DOI), as CoRE MOF
  does for its CSD-derived structures.

It is not:

- a structure database (the CSD, CoRE MOF and MOFX-DB exist);
- a hosted service with accounts, for now (section 11 says what that
  would add);
- a place for results the Void did not compute. RASPA outputs and
  experimental isotherms can be joined as `ref_` columns, as they are
  now, but the index vouches only for its own analyses.

## 2. Builds on

Most of what the index needs exists already.

- **Every report records its provenance.**
  - `provenance`: the Void's and each library's version, the platform,
    `report_format_version`.
  - `input.sha256`: the input file's SHA-256.
  - `config`: every option.
  - `methods`: text for each method.
  - `errors`: what failed.
- **The summary CSV and the Batch Results window** already rank, filter
  and open structures from a folder of reports, and they check each
  file against its SHA-256 before opening it. An index can be shown in
  the same window if it is laid out as a results folder (section 7).
- **The bundled CoRE MOF 2024 index** is the precedent for the built
  table: `{source, columns, rows}` in one gzip file. It holds 1.5 MB for
  17,202 entries, with name- and composition-checked matching
  (`services/reference/coremof.py`).
- **`the-void-fetch`** records each folder's source, citation and
  licence in its README.
- **Model cards** (round 61) record the SHA-256 of the data a model was
  trained on.

Some things are missing, and this document designs them:

- a structure identity that survives rewriting a CIF;
- an entry format that does not leak local paths;
- per-file licence information;
- rules for merging entries from several people;
- rules for when entries go stale.

## 3. Identity: which entries are about the same structure?

This is the core problem. Four levels are needed: file, framework,
structure and determination. Each has a job the others cannot do.

### 3.1 File: `file_sha256` (exists)

This is the SHA-256 of the input bytes. It is exact and cheap, and it
is what the Batch Results window already checks. It is not a structure
identity: rewriting a CIF changes it without changing a single atom,
whether by pymatgen, Mercury or a text editor fixing a label.

### 3.2 Framework: `framework_key` (new; prototyped and measured)

`framework_key` is a hash of:

- the reduced formula of the ordered structure (sites under 0.5
  occupancy removed, as every analysis does);
- the histogram of Weisfeiler–Lehman colours of the periodic bond
  graph's quotient graph: 4 rounds, starting from the element, with
  counts divided by their greatest common divisor;
- the set of periodic dimensionalities (0–3) of its bonded components.

Why this construction:

- A supercell's quotient graph is a covering of the primitive cell's,
  and WL colours are unchanged under coverings. Dividing by the gcd
  therefore removes the choice of cell.
- Origin, orientation, atom order and CIF formatting do not enter at
  all.
- The dimensionalities are there because the quotient graph alone
  cannot tell a 2D layer from a 3D net with the same local bonding. The
  prototype found such a pair in CoRE (Appendix A).

It uses the Void's own bond detection (`CovalentBondDetector`) and its
component dimensionality (`mof_analysis.bonded_components`). These are
what every analysis already runs, and they take a median 0.11 s per
structure (p99 1.5 s).

Measured on all 2,664 CoRE MOF 2024 SI structures (Appendix A):

| Change to the structure | Key unchanged |
|---|---|
| Rewritten by pymatgen's CifWriter (new bytes, new SHA-256) | 2,664 / 2,664 |
| 2× supercell | 1,891 / 1,891 (all ≤ 300 atoms) |
| Rotated, origin shifted, atoms shuffled | 268 / 268 |
| Reduced to the primitive cell | 13 / 13 (those with a smaller one) |

What the key does not do is tell every two different structures
apart. Structures with the same composition, local bonding and
dimensionality share a key: polymorphs, isomers, and one framework at
two temperatures. So the key is the address of a group, not an
identity:

| All 2,664 CoRE SI files | Count |
|---|---|
| Distinct framework keys | 1,604 |
| Keys held by one file | 943 |
| Keys shared by 2 or more files | 661 (1,721 files) |
| ... that are one structure throughout (StructureMatcher) | 609 |
| ... that hold more than one structure | 52 (117 structures) |
| Distinct structures, key plus match | 1,669 |

Most shared keys are CoRE's ASR and FSR copies of one structure, or
one framework determined several times, and are one structure. The 52
mixed groups are what the key cannot see:

- Nets with the same local bonding: 2020[Zn][dia] and 2020[Zn][qtz].
- Other nets: 2018[Ce][csq] and [scu].
- A 2D and a 3D Co structure: 0000[Co][nan]2 and [nan]3.
- Two sod Cd polymorphs.

None of CoRE's ASR/FSR twin files is byte-identical, even where no
atom was removed. That is a second reason the file hash cannot be the
identity.

### 3.3 Structure and determination: two structure matches within the group

Within one framework_key group, pymatgen's StructureMatcher decides.
It runs at two tolerances, because two questions need answering:

- **Same structure** (`structure_id`), at ltol 0.2, stol 0.3, angle 5°:
  the same framework, whatever its state. This covers the same MOF
  solved at 100 K and 298 K, and each step of a breathing series. The
  Cu sql series from one paper is an example: 2022[Cu][sql]2[ASR]1–5
  and 7, six determinations spanning 0.661–0.766 g/cm³, are one
  structure at this tolerance. Its [ASR]6 is another.
- **Same determination** (`determination_id`), at ltol 0.02, stol 0.05,
  angle 1°: the same coordinates, to within rounding.
  - A pymatgen rewrite and a 2× supercell fit.
  - CoRE's ASR/FSR copies with nothing removed fit.
  - Two steps of the Cu sql series do not fit, even 2 and 4, which
    differ by 0.14% in density.
  - Neither do two published determinations of one V MOF (2020 and
    2017, densities 1.1103 and 1.1107 g/cm³).

StructureMatcher is too slow to search a whole index, which is why the
key comes first. Within a group it is cheap:

- The median group took 1.1 s. The largest, 36 Zn cag files, took
  4 min.
- The 661 groups needed 1,169 fits in all, comparing each file with
  one representative of each structure already found.
- Adding one entry compares it with one representative per structure
  in its group, usually one or two fits.
- The build caches every decision by the pair of file SHA-256s, so a
  rebuild only fits new pairs.

The ids are written as `<framework_key>.<n>` and
`<framework_key>.<n>.<m>`, numbered in entry_id order. The links are
recorded in the built table, never in the entries, so a better matcher
can regroup later without rewriting anything.

Density cannot stand in for either match (Appendix A):

- Within one structure, determinations differ by up to 15.9% (the
  Cu sql series), with a p95 of 1.9%.
- Between different structures sharing a key, the difference can be as
  small as 0.08%. 9 of 82 pairs differ by under 1%.

### 3.4 Names are aliases, never identity

CoRE ids, CSD refcodes, the CIF's data block and file names are
recorded as `aliases` for search. They are never used for identity: the
prototype found a CoRE SI file named `[Ho]` that contains Tb. The
existing CoRE matching (`find_matches`) supplies the CoRE ids.

## 4. What an entry is

**An entry is one analysis of one structure file, by one version of
the Void, with one configuration.** Entries are immutable. They are
never edited, only added or withdrawn, and one is named by the SHA-256
of its own canonical JSON (the `entry_id`). Content addressing gives
three things:

- The same analysis added twice is one file.
- Two people adding entries at the same time can never conflict. The
  prototype merged two Git clones that each added entries, one of them
  the same entry: `git pull` merged cleanly, with no conflict.
- An entry cannot change without its id changing, so a table that cites
  entry ids cites exactly those numbers.

### 4.1 Fields

The entry is the report the Void already writes, with local paths
removed and identity, licence and contributor added:

```text
entry_format_version   1
entry_id               sha256 of the canonical JSON of everything below
structure:
  file_sha256          as report.input.sha256
  file_name            base name only - never the local path
  framework_key        with key_version (the recipe above is key v1)
  formula, cell_formula, n_atoms, density_g_cm3, space_group, metals
  aliases              [{"kind": "coremof"|"csd"|"doi"|"data_block"|"name", "value": ...}]
  origin               "experimental" | "hypothetical" | "computed" (DFT-relaxed etc.)
  parent               entry_id or file_sha256 of the parent, for derived structures
                       (functionalised variants: the-void-functionalize's record)
  licence              {"spdx": "CC-BY-4.0" | null, "redistributable": true|false,
                        "basis": "fetched from CoRE MOF SI" | "declared by contributor"
                                 | "CSD header in file" | "unknown"}
analysis:
  software             report.provenance.software (the Void and every library)
  platform             report.provenance.platform
  config               report.config, unchanged
  method_key           sha256 of (the Void's version, config) - see 6
  results              report.results, unchanged
  errors               report.errors
  summary              the summary CSV row, so the table is built without re-deriving it
contributor:
  name                 as the contributor chooses; ORCID optional
  added                ISO date
  note                 free text ("SI of Smith 2025, as deposited")
```

What is deliberately left out of an entry:

- `input.file`, the absolute path. It is in every report today and
  gives away user names and folder layouts, for example
  `/home/<user>/...`.
- The generation time of the report: two runs of the same analysis
  should give the same entry.
- Anything about the viewer, such as camera or colours.

### 4.2 Sizes, measured

On the 150-structure batch of round 61:

- **Reports:** a median 7.9 KB, p90 11.9 KB, and 3.0 KB gzipped. Open
  metal sites (the per-site list) are 24% of the results, and the
  topology 10%.
- **CIFs:** a median 12.6 KB, and 2.8 KB gzipped.
- **A lab index of 10,000 analyses** is therefore about 80 MB of JSON,
  30 MB gzipped, plus the structures that may be stored. That is a
  comfortable size for a Git repository or a shared drive.
- **The built table** for 17,202 rows is about 1.5 MB gzipped, from the
  CoRE precedent.

## 5. Structures and licences

The rule is default-deny: a structure file is stored only if the entry
says it may be redistributed, and says why. The Void cannot read a
licence from a CIF. Neither CoRE's SI files nor the CSD's CIFs state
one, so the basis must come from where the file came from:

| Where the file came from | Licence recorded | Stored? |
|---|---|---|
| `the-void-fetch coremof` (the SI subset) | CC-BY-4.0, from the fetch folder | yes, with CoRE's attribution |
| `the-void-fetch mofxdb` | the database's stated licence | if that licence allows |
| A CIF with the CSD header ("This file contains crystal structure data downloaded from the Cambridge Structural Database", as CALF-20 in tests/fixtures has) | CSD-derived | **no**: refcode alias only |
| Built by the-void-functionalize | inherits from the parent; hypothetical | as the parent (see open question 3) |
| Anything else | unknown | **no**, unless the contributor declares a licence |

Every CSD-derived entry keeps its results and its refcode alias. Any
lab member with a CSD licence can fetch the file and check it against
`file_sha256`, with the same check the Batch Results window already
makes.

The results themselves are the lab's. At a public release they need a
licence of their own (open question 1).

One change was needed in the existing code. `the-void-fetch` wrote the
licence only into README.txt. Since 0.9.5 it also writes `ref_licence`
and `ref_licence_basis` columns, so each file's basis is
machine-readable. The name `ref_source` was already taken: CoRE's own
"SI"/"CSD" column. Folders fetched earlier are still recognised by that
column and the README's citation (`services/index/licence.py`).

## 6. Versions, and when numbers go stale

The Void's numbers change between versions, for good reasons: PLD in
rounds 55, 57 and 58, surface area in round 50, OMS coverage in 0.79. So an entry is only
comparable with another entry that has the same method: the same Void
version and the same config.

- **`method_key`** is the SHA-256 of (the Void's version, the config).
  The built table has one column per result per method_key family.
  Rows are never mixed silently; the table view defaults to the newest
  version.
- **Staleness.** A method changelog maps each version to the results
  it changed, for example "0.8.7: pld_A for
  layered frameworks". `the-void-index stale` lists entries whose
  numbers a later version changed. `the-void-index refresh` reruns them
  where the structure file is available, stored or on the member's own
  disk, found by its SHA-256. This never overwrites anything; it adds
  new entries.
- **Replicates.** Two entries with the same `determination_id` and the
  same method_key are replicates, and must agree. Entries that share
  only the `structure_id` are other determinations. The table shows
  them together, but they may legitimately differ: a breathing
  framework's PLD does.
  - **Same file and version:** the analyses are deterministic, so the
    results should be identical. A difference means a platform effect
    or a bug, and the index flags it. Round 60 showed parallel and
    serial runs give identical results.
  - **Different file, same determination:** the files differ in bytes
    but not in atoms. Measured on 12 CoRE structures:
    - LCD, PLD, void fraction and density were identical after
      rewriting. After shuffling the atom order they agreed to 1e-14,
      the order of floating-point sums.
    - Channel surface area was identical after rewriting, but moved
      0.08–3.5% (median 0.8%, 9 structures with surface; 500 samples
      per atom) when the atoms were only reordered. Each atom's sample
      rotation is drawn from one random stream in atom order
      (`analyze_surface_area`).
  - **Fixed in 0.9.3 (step 1).** The rotations are now drawn in a
    canonical order of the atoms: by element, then wrapped fractional
    coordinates rounded to 1e-4 (`surface_area.canonical_rotations`).
    - On 8 CoRE structures with surface, a shuffled or rewritten file
      gives exactly the same channel ASA. Before, only 0 of 8 shuffled
      files did.
    - Seed-to-seed scatter is unchanged, median 0.6% at 500 samples per
      atom, and 5-seed means moved by under 1%.
  - **What still differs.** A copy with a shifted origin, or a
    supercell, still samples differently, by up to 1.5% and 2.6% on
    those structures. Before 0.9.3, the shifted origin already moved
    channel ASA by up to 1.5%, because the channel/pocket grid is tied
    to the cell.
    - A rotation derived from each atom's environment was also tried.
      It made supercells exact, but it gave translation-equivalent
      atoms identical samples, which raised a small centred cell's
      seed scatter from 1.6% to 4.3%, so it was not adopted.
  - **So replicates are compared like this:**
    - exactly, when they come from the same file or the same atoms in
      any order;
    - within the CoRE-comparison tolerances (0.15 Å for LCD/PLD, 5% for
      surface area) when the determination match is the only link.

## 7. Layout and workflow for a lab (no server)

An index is a folder: a shared drive, or better, a Git repository,
which adds history, review and an audit trail at no cost.

```text
MyLab-index/
  INDEX.json               name, entry_format_version, maintainers, licence policy,
                           results licence (for a later release), key_version
  entries/ab/ab3f...json   one immutable entry per file, fanned out by the id's first byte
  structures/9c/9c1e...cif redistributable structures only, named by file_sha256
                           and stored as given (not gzipped), so the Batch Results
                           window can check and open them
  matches/4e/4e07...json   one structure-match decision per pair of files (0.9.5):
                           made where both files were at hand, so a member without
                           a CSD-derived file builds the same ids
  withdrawn/ab3f...json    a tombstone: entry_id, who, when, why
                           (e.g. "wrong structure", "licence withdrawn")
  build/                   generated, never committed: gitignored
    summary.csv            one row per current entry, the summary columns plus
                           framework_key, structure_id, determination_id,
                           method_key, contributor,
                           version, aliases, licence, stale
    <entry_id>.json        the entries again, flat, so that...
    index.json.gz          ...the Batch Results window and thevoid can open build/
                           as they open any results folder, and the CoRE-style table
                           serves search
```

The build is deterministic from `entries/`, `matches/` and
`withdrawn/`. It is safe to delete and rebuild, and it never needs
merging.

**As built in 0.9.5.**

- **The same analysis is added once.** An `analysis_id` covers the
  file and the analysis, but not the contributor, date or note. Adding
  the same results again is recognised and reported as "already
  present".
- **Two members who add the same analysis separately** write two entry
  files, with no conflict. The build shows them as one row, with the
  others in `also_added_by`.
- **A pair of files with no recorded decision** and a structure not at
  hand is treated as different, never merged on a guess. The build
  reports it (`BUILD.json` `undecided_pairs`). Giving the files to
  `build --structures DIR` decides the pair, and records the decision.
- **`add` compares each new file** with the others of its key. It
  looks for them in the index's stored structures, in `--structures`
  folders, and in the folders of the files being added.
- **Rows** are named `<file stem>~<entry id prefix>`, so one file
  analysed twice has two rows. Groups
(section 3.3) are formed in the build, in entry_id order, so every
member building the same entries gets the same structure and
determination ids.

The commands follow the existing tools. 0.9.5 has init, add, build,
find, verify and withdraw; search, stale and export are later steps
(section 12), and until then the Batch Results window searches build/:

```text
the-void-index init MyLab-index --name "McKelvey Lab"
the-void-index add MyLab-index results/ --contributor "B. Garmejani" --note "SI of ..."
    # every report in a the-void-analyze folder; files whose licence it cannot tell
    # are withheld and listed (--licence SPDX --licence-basis TEXT declares one)
the-void-index build MyLab-index
the-void-index search MyLab-index --metals Cu --pld-min 6        # like the-void-fetch filters
the-void-index find MyLab-index structure.cif                    # has anyone analysed this?
the-void-index stale MyLab-index
the-void-index verify MyLab-index                                # rebuild ids, check replicates
the-void-index export MyLab-index --out release/ --public        # section 11
```

They would appear in three other places:

- **The GUI:** "Add to Shared Index..." in the Batch Results window
  (the selected rows) and in File (the current structure's report). A
  "Previously analysed" line in the Structure panel when the loaded
  structure is found, using the same lookup as Compare with CoRE MOF
  2024.
- **Python:** `thevoid.Index(path)` with `.add(report_or_table)`,
  `.find(framework)`, `.table()` and `.search(...)`.
- **ML:** `thevoid.train(index, target, method_key=...)`. Void-v1
  models need rows computed alike, and the method_key is exactly that
  filter.

## 8. Merging, trust and mistakes (lab tier)

In a lab, members are trusted. The design protects against mistakes,
not malice:

- **Adding** is the only write. Two members never conflict (section 4).
- **A wrong entry is withdrawn, never deleted:** for example the wrong
  CIF, or an option set by mistake. A tombstone hides it from the
  build; Git history keeps the record.
- **`verify`** recomputes every `entry_id`, so a hand-edited entry
  shows up. It also checks every stored structure against its
  `file_sha256`, and reports replicates that disagree (section 6).
- **Review** is the Git host's pull requests, if the lab wants it.

## 9. Privacy

An entry contains a contributor name, the file's base name, and what
the report contains. Absolute paths are dropped (4.1). A contributor
may use initials. Unpublished structures of the lab's own are the case
that needs care, and are handled in two ways:

- Their CIFs are not stored unless the contributor declares a licence
  (default-deny).
- Their results can be marked `embargo_until`: kept in the index but
  left out of any public export before that date.

## 10. How it would be tested

In the style of the rest of the suite, one focused file per area:

- **Key invariance.** Real fixtures under rewriting, supercells,
  rotation, origin shift, atom order and the primitive cell. The known
  collision pairs from Appendix A must be separated by the
  StructureMatcher step.
- **Entry round trip.** A real report becomes an entry; its `entry_id`
  is stable across runs; no absolute path survives; the entry converts
  back to a report that the Batch Results loader reads.
- **Merging.** Two index copies that add entries independently,
  including the same entry, merge with no conflict, and build the same
  table in either order.
- **Licences.** The CSD header is detected; nothing without a
  redistributable basis is stored; a fetched CoRE folder's files are
  stored with CC-BY attribution.
- **Staleness and replicates.** A synthetic changelog marks the right
  entries stale; disagreeing replicates are reported.
- **The window.** The Batch Results window opens `build/` and verifies
  a stored structure's SHA-256 before opening it.

## 11. Public later: what changes

A public release is an export: one archive, versioned and deposited
with a DOI (for example on Zenodo), as CoRE MOF 2024 is. It contains:

- the built table, with columns and rows as the bundled CoRE index has;
- the entries;
- the redistributable structures;
- a README with the citation and licences.

It is not a live service. Once released, it can be bundled or fetched
as CoRE's index is (`the-void-fetch`), and matched against as CoRE's is
(Compare with...).

Moving from lab to public adds these steps:

- **Licence gating becomes strict:** entries with `redistributable:
  false` export results and aliases only, and embargoed entries are
  left out.
- **Contributors** are listed by name or ORCID only if they agree; the
  default is initials.
- **Signatures:** the release is signed (for example with minisign or
  Sigstore) so a copy can be checked against the published one.
  Signing entries individually is only needed for a hosted
  multi-contributor service.
- **A maintainer reproduction pass:** before release a maintainer
  reruns a random sample of entries from their stored structures. The
  release records the agreement rate, a "reproduced" check in the
  sense of section 6.

A hosted, writable community service (accounts, an API, moderation) is
a separate project. It would reuse the entry format and the identity
unchanged; nothing in the lab design has to be undone for it.

## 12. Plan, if the owner goes ahead

1. **Order-invariant surface sampling** (section 6), with its own
   test. Done in 0.9.3. It changes channel ASA only within sampling
   scatter.
2. **`services/index/identity.py`:** `framework_key` v1 and the
   grouping step, with the Appendix A measurements as tests. Done in
   0.9.4. `tools/index_identity_survey.py` repeats the survey over any
   folder.
3. **Entry format and `the-void-index init/add/build/find/verify`,** and
   `ref_licence`/`ref_licence_basis` columns in `the-void-fetch`. Done
   in 0.9.5, with `withdraw` as well.
4. **GUI:** "Add to Shared Index...", "Previously analysed", and
   opening `build/` in Batch Results. Python: `thevoid.Index`. Done in
   0.9.6. It is under File > Shared Index (Add Current Structure...,
   Open Index..., Set Index Folder...), Batch Results > Add to Shared
   Index..., and a "Previously Analysed" row in the Structure panel.
   A report from the Analysis menu has no config, so the index infers
   one from the settings its results record
   (`entries.inferred_config`). GUI analyses with different probes or
   samples then get different method_keys and are never compared as
   replicates.
5. **`stale`/`refresh`** with the method changelog, and `train` from an
   index. Done in 0.9.7.
   - **The changelog** is `thevoid_core/data/method_changes.json`. It
     has seven records, 0.8.5 to 0.9.3; a test checks
     that each names real columns and versions.
   - **An entry is stale** in a column when a newer version changed
     that column and the entry has a value there.
   - **An entry is superseded** when a newer version has analysed the
     same file with the same options.
   - **The build** has `stale_columns` and `superseded_by` columns.
   - **`refresh`** reruns current stale entries. It uses the same
     options, and the stored file or `--structures`. It adds new
     entries that inherit the old ones' licence, aliases and joined
     columns. Analysis-menu entries, whose options were inferred, are
     skipped. Since 0.9.8, `-j N` analyses N structures at once, in worker
     processes. The new entries are still added one at a time in the
     stale list's order, and the entry files are byte-identical to a
     one-job refresh. A stored copy whose SHA-256 no longer matches is
     skipped, not analysed.
   - **Training** (`the-void-train fit INDEX`, `thevoid.train(index,
     ...)`) uses current entries of one method_key, split by
     structure_id.
6. **`export --public`** and signing, when there is something to
   release.

Steps 1–3 are one round each.

## 13. Open questions, and what was decided

On 2026-09-29 the owner left these to the designer ("do as you think
suits best"). These defaults were taken, and each can be reversed
before the step that depends on it:

- **1.** CC-BY-4.0, as CoRE MOF.
- **2.** No group CSD licence is assumed, so CSD-derived structure
  files are never stored.
- **3.** Functionalised variants of CSD structures are treated as their
  parent: not stored.
- **4.** A private Git repository.
- **5.** the-void-analyze's defaults plus `--compare-coremof`.

The questions as posed:


1. **Results licence** for a public release: CC-BY-4.0, as CoRE's, is
   the obvious choice. CC0 would make the index easiest to reuse.
2. **Does the lab have a CSD licence** that all members hold? If so,
   CSD-derived structures could be stored in a lab-only index. It must
   then never be exported, and the tool would enforce that with a
   `csd_licensed_group` flag in INDEX.json. The default design assumes
   not.
3. **Hypothetical derivatives of CSD structures.** Is a functionalised
   variant of a CSD structure redistributable? Its coordinates are
   mostly the parent's. The design treats it as the parent (no), which
   is the cautious reading. The CCDC's terms would decide.
4. **Where the lab index lives:** a Git host (GitHub or GitLab,
   private), a university drive, or both. Git is recommended for its
   history, and 30–100 MB is well within any host's limits.
5. **Which analyses are "standard"** for the lab: a default config, so
   that most entries share one method_key and rank together. Proposed:
   the-void-analyze's defaults plus `--compare-coremof`.

## Appendix A. Measurements behind this design

All measurements were made on 2026-09-29 with the Void 0.9.1 (this
tree), pymatgen's StructureMatcher, and the 2,664 CoRE MOF 2024 SI
structures (CoRE MOF Tools' SI zips; CC-BY-4.0). The prototype scripts
were not shipped, and are summarised here. Since 0.9.4 the key and the
grouping are in `services/index/identity.py`, and
`tools/index_identity_survey.py` repeats the key, invariance and
grouping measurements with that code (the rerun is under "Rerun with
the shipped code" below).

**The key.** `framework_key` v1 is described in 3.2: 4 WL rounds on the
quotient graph of `CovalentBondDetector`'s bonds after
`major_occupancy_structure`, the gcd-normalised colour histogram, the
reduced formula and the set of component dimensionalities, then
SHA-256.

- A self-bond (an atom bonded to its own periodic image) counts twice
  in its neighbour list, once for each direction, as the two
  neighbours are distinct atoms in any supercell.
- The first prototype had no dimensionalities. It merged a 2D and a
  3D Co structure with identical local bonding (`0000[Co][nan]2[ASR]1`,
  `0000[Co][nan]3[ASR]10`: density 1.073 against 1.179), so they were
  added.

**Invariance.** Every structure was keyed as read, and then:

- after `CifWriter` → `Structure.from_str`;
- as a 1×2×1 supercell (structures ≤ 300 atoms);
- for 268 structures (the round-61 transfer and void-v1 sets), after
  a random rotation of the lattice, a random origin shift and a random
  permutation of the sites;
- for 13 of those, after `get_primitive_structure` (tolerance 0.01)
  found a smaller cell.

The table in 3.2 has the counts. There were no failures.

**Collisions.** Every group of two or more files sharing a key was
clustered with StructureMatcher (ltol 0.2, stol 0.3, angle_tol 5°,
primitive_cell, attempt_supercell, no volume scaling). Byte-identical
files would have been counted without a fit; there were none.

- 661 groups (1,721 files) were clustered.
- 609 are one structure.
- 52 hold 117 structures.
- 1,169 fits were needed.
- Density spread within a structure: median 0, p95 1.9%, maximum
  15.9% (2022[Cu][sql]2, six determinations). 14 of the 674 structures
  with more than one file spread over 3%, among them 2019/2017[Th][bcu] (0.904 against 0.795)
  and 2012[Zn][acs] (7.2%).
- Density difference between different structures in one group:
  minimum 0.08%, median 4.8%; 9 of 82 pairs under 1%.
- The tight matcher (ltol 0.02, stol 0.05, 1°) was checked on the
  pairs quoted in 3.3, and on a rewrite and a supercell of 2020[V][nan].

**Replicates.** The first 12 structures of the round-61 batch were
analysed four times: pore analysis and surface area at 500 samples per
atom, with topology and OMS off. The four runs were as read, again,
after CifWriter, and with shuffled atoms.

- LCD, PLD, void fraction and density were identical in the first
  three, and agreed to 1e-14 when shuffled.
- Channel ASA was identical in the first three. When shuffled it moved
  0.08, 0.23, 0.44, 0.63, 0.76, 0.99, 1.98, 2.10 and 3.51% for the 9
  structures with surface.

**Merging.** Two clones of a bare Git repository each added entry files
under `entries/<xx>/<id>.json`, and both added one identical file. The
second `git pull` merged with exit status 0, no conflict and all four
files present.

**Sizes.** These come from the 150 reports and CIFs of the round-61
void-v1 check, and the bundled CoRE index.

**Rerun with the shipped code (0.9.4).** The command was
`tools/index_identity_survey.py` on the same 2,664 files, with 2
processes.

- **Invariance.** The key was unchanged after a rewrite (2,664/2,664),
  a supercell (1,891/1,891) and rotation with origin shift and
  shuffling (2,664/2,664). The last of these covers every structure
  now, not 268. A key takes a median 0.038 s (p99 0.42 s).
- **Grouping.** 1,604 keys, 52 of the 661 shared keys split and 1,669
  structures: the prototype's numbers exactly.
- **Determinations.** With the tight match there are 1,797
  determinations.
  - 709 of them hold more than one file (1,576 files), which makes
    those files replicates.
  - 82 structures have more than one determination, up to 12.
- **ASR/FSR copies.** 584 of CoRE's ASR/FSR copy pairs share a key.
  578 of them are one structure and 563 one determination. The six
  pairs that did not match were not looked into.
- **Matching** took 2,477 fits: a median 1.1 s per group, the longest
  342 s.
