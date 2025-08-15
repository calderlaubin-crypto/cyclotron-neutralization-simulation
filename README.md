# Cyclotron-Maintained Space-Charge Neutralization PIC Simulation

This repository contains a 2D electrostatic Particle-in-Cell (PIC) simulation for studying **cyclotron-maintained space‑charge neutralization** in an electron-positron beam system.

## Features
- Relativistic electron and positron species
- Self-consistent electrostatic field solution via FFT-based Poisson solver
- Boris pusher for particle motion
- Gaussian smoothing for improved PIC stability
- Automatic tuning of magnetic field for a target Larmor radius
- Diagnostics: neutralization fraction, RMS field suppression, RMS charge

## Requirements
- Python 3.8+
- NumPy
- Matplotlib
- SciPy

Install dependencies with:
```bash
pip install numpy matplotlib scipy
```

## Running the Simulation
From the repository root, run:
```bash
python simulation.py
```

Results (plots and metadata) will be saved to the `plots/` directory.

## Output
- `final_rho.png`: Final net charge density map
- `final_E_mag.png`: Final electric field magnitude
- `neutralization_fraction.png`: Neutralization vs time
- `E_rms.png`: Field suppression vs time
- `rho_rms.png`: Net charge RMS vs time
- `run_meta.txt`: Metadata of the simulation parameters

## License
MIT License — You are free to use, modify, and distribute this code.
