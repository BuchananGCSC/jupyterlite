# Validation notebooks

`validation.ipynb` checks the exoplanet simulator's physics.

It runs `test_cases.json`, the same contract the JavaScript test suite uses,
against an independent Python implementation in `physics.py`. Agreement
means the two have not drifted apart. It does not prove either is right —
only the cases marked `"source": "published"` do that, because only those
compare against something outside the project.

The rest of the notebook is for the checks that are awkward in a browser:
habitable zone boundaries against Kopparapu et al. 2013, the ice-albedo
hysteresis loop, obliquity sweeps, and the day-night contrast on tidally
locked planets.

Everything runs in your browser. Nothing is sent anywhere, and changes you
make are stored locally, so edit freely — the copies in the repository are
the originals.

Source: https://github.com/BuchananGCSC/exoplanet-tools
