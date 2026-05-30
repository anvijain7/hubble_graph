# Hubble Research

This workspace contains a simple local Hubble constant analysis using the SH0ES Pantheon+ dataset.

## Files

- `hubble_fit.py`: loads the local dataset, filters to `z <= 0.1`, fits a local Hubble relation with `curve_fit`, and plots the fit.
- `Pantheon+SH0ES.dat`: the data file copied into the project workspace.
- `requirements.txt`: required Python packages.

## Run

```bash
python3 hubble_fit.py
```

The script saves the figure as `local_H0_fit.png` in the same folder.
