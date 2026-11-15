Real SHELXL files, from the test resources of ShelXFile by Daniel Kratzert
(https://github.com/dkratzert/ShelXFile, tests/resources), MIT licence:

    MIT License. Copyright (c) 2018 Daniel Kratzert. Permission is hereby
    granted, free of charge, to any person obtaining a copy of this software
    and associated documentation files (the "Software"), to deal in the
    Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions: The above
    copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
    "AS IS", WITHOUT WARRANTY OF ANY KIND.

  I-43d.res  a Ni complex in I-43d, refined with SHELXL: atoms on special
             positions (sof 10.33333, 10.25), three disorder parts with
             "PART -n 10.25", riding hydrogens. Unchanged.
  I-43d.cif  the CIF SHELXL wrote for the same refinement. Changed: the
             embedded reflection file (_shelx_hkl_file, 116,000 lines) was
             removed; nothing else.
  p21c.res   an Al/Ga alkoxide in P2(1)/c, refined with SHELXL-2018:
             residues (RESI), disorder parts tied to free variables
             ("PART 2 -31"), 945 parameters. Unchanged.

Used by tests/test_shelx.py and tests/test_refinement_info.py (version 1.1.0).
