# Contributing

Thank you for helping to improve the Void. The most useful
contributions are reports: what went wrong, what is missing, and where a
number disagrees with another tool or with the literature.

## Reporting a bug

Open an issue with the **Bug report** form. Please include:

- what you did, what you expected, and what happened instead;
- the diagnostics from **Help > Copy Diagnostics** (the version of the
  Void and its packages, your system, and the end of the log);
- the structure file, if you may share it. Files from the Cambridge
  Structural Database (CSD) may not be shared publicly: give the CSD
  refcode or the CCDC number instead.

## Questioning a result

If a number looks wrong, say what you compared it with (another program,
a database such as CoRE MOF, or a paper), with the settings used on both
sides: probe radius, atomic radii, grid spacing. Many differences come
from different conventions rather than errors, and Help > About the
Void lists the methods the Void uses.

## Suggesting a feature

Open an issue with the **Feature request** form, and describe the
analysis or the workflow you need rather than only the solution you have
in mind.

## Code changes

The Void is licensed under CC BY-NC-ND 4.0 (see `LICENSE`), which does
not allow modified versions to be shared. For that reason pull requests
are not accepted. Please open an issue instead; if you have a fix in
mind, describe it there, and the maintainer will make the change.

## Running the tests

If you want to check a problem against the test suite:

```bash
pip install -r requirements-dev.txt
pip install -e .
pytest
```

On a machine without a display, set `QT_QPA_PLATFORM=offscreen` first.
