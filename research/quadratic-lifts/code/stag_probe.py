import numpy as np
from nonsep_methods import stag_run, H
for x10 in [1.0, 2.0]:
    for h in [0.1, 0.05, 0.02, 0.01, 0.005]:
        n = int(1000/h)
        Hs, E, ok = stag_run(np.array([x10, 0.0]), h, n)
        H0 = Hs[0]
        print(f"x1_0={x10}, h={h:<6}: {'ok  ' if ok else 'FAIL'} steps={len(Hs)-1:7d}  max|H-H0|={np.nanmax(abs(Hs-H0)):.2e}  "
              f"discrete-E drift={np.nanmax(abs(E-E[0])):.1e}")
