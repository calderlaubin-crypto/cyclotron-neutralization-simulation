# PIC code for cyclotron-maintained space-charge neutralization (magnetized case, robust baseline)

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter

# -------------------- Physical constants --------------------
e  = 1.602176634e-19
me = 9.10938356e-31
e0 = 8.8541878128e-12
c  = 299792458.0
kB = 1.380649e-23

# -------------------- Tunable physics targets --------------------
# Domain and grid
Lx = 1.0e-3     # 1 mm square cross-section
Ly = 1.0e-3
Nx = 128
Ny = 128

# Beam/neutralizer densities
ne_e  = 6.9e16   # electron beam density
ne_p  = 6.9e16   # positron neutralizer density (ideal cancellation; can detune)

# Axial speeds
vz_e = 2.4e8
vz_p = 2.4e8

# Magnetic field (magnetized run)
Bz = 1.0  # Tesla

# Thermal spreads
Te_e_eV = 0.5
Te_p_eV = 0.5

# Particles per cell
ppc = 100

# Smoothing for charge deposition
rho_smooth_sigma = 0.8

# Number of plasma periods
N_plasma_periods = 50

# -------------------- Derived geometry & helpers --------------------
dx = Lx / Nx
dy = Ly / Ny
Lz = 1.0  # unit slab thickness
cell_area = dx * dy

# Convert temperatures to thermal speeds
Te_e = Te_e_eV * e
Te_p = Te_p_eV * e
vth_e = np.sqrt(kB*Te_e / me)
vth_p = np.sqrt(kB*Te_p / me)

# Plasma frequency
omega_pe = np.sqrt(ne_e * e**2 / (me * e0))
Tpe = 2.0 * np.pi / omega_pe

# Relativistic gamma
def gamma_from_v(v):
    return 1.0/np.sqrt(1.0 - (v/c)**2)
gamma_e = gamma_from_v(vz_e)

# Time step: stability limited
dt_stab = 0.2 / omega_pe
vmax_xy = 3*vth_e
cfl_dt = 0.25 * min(dx, dy) / (vmax_xy + 1e-16)
dt = min(dt_stab, cfl_dt)

steps = int(np.ceil(N_plasma_periods * Tpe / dt))

# -------------------- Diagnostics/metadata --------------------
if not os.path.exists('plots'):
    os.makedirs('plots')

meta = {
    'Lx': Lx, 'Ly': Ly, 'Nx': Nx, 'Ny': Ny,
    'dx': dx, 'dy': dy,
    'ne_e': ne_e, 'ne_p': ne_p,
    'vz_e': vz_e, 'vz_p': vz_p,
    'Te_e_eV': Te_e_eV, 'Te_p_eV': Te_p_eV,
    'Bz': Bz, 'omega_pe': omega_pe,
    'dt': dt, 'steps': steps,
}
with open('plots/run_meta.txt', 'w') as f:
    for k,v in meta.items():
        f.write(f"{k}: {v}\n")

# -------------------- Species initialization --------------------
def init_species(q, m, n0, vz_mean, vth):
    Np = ppc * Nx * Ny
    x = np.random.rand(Np) * Lx
    y = np.random.rand(Np) * Ly
    vx = np.random.normal(0.0, vth/np.sqrt(2), size=Np)
    vy = np.random.normal(0.0, vth/np.sqrt(2), size=Np)
    vz = np.ones(Np) * vz_mean
    w = (n0 * cell_area * Lz) / ppc
    return {'x': x, 'y': y, 'vx': vx, 'vy': vy, 'vz': vz,
            'q': q, 'm': m, 'Np': Np, 'w': w}

Electrons = init_species(-e, me, ne_e, vz_e, vth_e)
Positrons = init_species(+e, me, ne_p, vz_p, vth_p)
species_list = [Electrons, Positrons]

# -------------------- Grid ops --------------------
kx = np.fft.fftfreq(Nx, d=dx) * 2*np.pi
ky = np.fft.fftfreq(Ny, d=dy) * 2*np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K2 = KX**2 + KY**2
K2[0,0] = 1.0

def deposit_rho(species_list):
    rho = np.zeros((Nx, Ny))
    for sp in species_list:
        q = sp['q']; w = sp['w']
        gx = sp['x'] / dx
        gy = sp['y'] / dy
        i0 = np.floor(gx).astype(int) % Nx
        j0 = np.floor(gy).astype(int) % Ny
        tx = gx - i0
        ty = gy - j0
        i1 = (i0 + 1) % Nx
        j1 = (j0 + 1) % Ny
        w00 = (1-tx)*(1-ty)
        w10 = tx*(1-ty)
        w01 = (1-tx)*ty
        w11 = tx*ty
        charge = q * w
        np.add.at(rho, (i0,j0), charge * w00)
        np.add.at(rho, (i1,j0), charge * w10)
        np.add.at(rho, (i0,j1), charge * w01)
        np.add.at(rho, (i1,j1), charge * w11)
    rho /= (cell_area * Lz)
    if rho_smooth_sigma > 0:
        rho = gaussian_filter(rho, sigma=rho_smooth_sigma)
    rho -= rho.mean()
    return rho

def solve_poisson(rho):
    rho_k = np.fft.fft2(rho)
    phi_k = -rho_k / (e0 * K2)
    phi_k[0,0] = 0.0
    Ex = np.fft.ifft2(1j*KX*phi_k).real
    Ey = np.fft.ifft2(1j*KY*phi_k).real
    return -Ex, -Ey

def gather_field(sp, Ex, Ey):
    gx = sp['x'] / dx
    gy = sp['y'] / dy
    i0 = np.floor(gx).astype(int) % Nx
    j0 = np.floor(gy).astype(int) % Ny
    tx = gx - i0
    ty = gy - j0
    i1 = (i0 + 1) % Nx
    j1 = (j0 + 1) % Ny
    Exp = (Ex[i0,j0]*(1-tx)*(1-ty) + Ex[i1,j0]*tx*(1-ty) +
           Ex[i0,j1]*(1-tx)*ty + Ex[i1,j1]*tx*ty)
    Eyp = (Ey[i0,j0]*(1-tx)*(1-ty) + Ey[i1,j0]*tx*(1-ty) +
           Ey[i0,j1]*(1-tx)*ty + Ey[i1,j1]*tx*ty)
    return Exp, Eyp

def boris_push(sp, Ex, Ey):
    """ Full Boris pusher including Bz """
    q = sp['q']; m = sp['m']
    vx, vy, vz = sp['vx'], sp['vy'], sp['vz']
    Exp, Eyp = gather_field(sp, Ex, Ey)

    # Half electric kick
    vxm = vx + (q * Exp / m) * (0.5 * dt)
    vym = vy + (q * Eyp / m) * (0.5 * dt)
    vzm = vz  # no Ez in this 2D setup

    # Magnetic rotation
    t = (q * Bz / m) * (0.5 * dt)
    tx, ty, tz = 0, 0, t
    s = 2*t / (1 + t**2)
    vxp = vxm + vym * tz
    vyp = vym - vxm * tz
    vxm2 = vxm + vyp * s
    vym2 = vym - vxp * s

    # Drift
    sp['x'] = (sp['x'] + vxm2 * dt) % Lx
    sp['y'] = (sp['y'] + vym2 * dt) % Ly

    # Final half electric kick
    sp['vx'] = vxm2 + (q * Exp / m) * (0.5 * dt)
    sp['vy'] = vym2 + (q * Eyp / m) * (0.5 * dt)
    sp['vz'] = vzm

# -------------------- Robust baseline for neutralization --------------------
M_baseline = 30
rhos = np.zeros((M_baseline, Nx, Ny))
per_sample_rms = np.zeros(M_baseline)

for i in range(M_baseline):
    sp_e_sample = init_species(-e, me, ne_e, vz_e, vth_e)
    r = deposit_rho([sp_e_sample])
    rhos[i] = r
    per_sample_rms[i] = np.sqrt(np.mean(r**2))

rho_e_only_gridavg = rhos.mean(axis=0)
rms_gridavg = np.sqrt(np.mean(rho_e_only_gridavg**2))
median_sample_rms = np.median(per_sample_rms)
abs_floor = 1e-12
den_scalar = max(rms_gridavg, median_sample_rms, abs_floor)

print(f"Baseline samples M={M_baseline}: RMS(single sample)={per_sample_rms[0]:.3e}, "
      f"median RMS={median_sample_rms:.3e}, grid RMS={rms_gridavg:.3e}")
print(f"Chosen denominator den_scalar={den_scalar:.3e}")

def neutralization_fraction(rho_net, den=den_scalar):
    num = np.sqrt(np.mean(rho_net**2))
    den_safe = max(den, 1e-16)
    return 1.0 - (num / den_safe)

# -------------------- Electron-only field baseline --------------------
sp_e_baseline = init_species(-e, me, ne_e, vz_e, vth_e)
rho_e_baseline = deposit_rho([sp_e_baseline])
Ex_e, Ey_e = solve_poisson(rho_e_baseline)
Erms_e_baseline = np.sqrt(np.mean(Ex_e**2 + Ey_e**2))
print(f"Electron-only Erms baseline={Erms_e_baseline:.3e}")

# -------------------- Main loop --------------------
eta_hist, Erms_hist, rho_rms_hist, eta_field_hist = [], [], [], []
snap_every = max(1, steps//200)

for it in range(steps):
    rho = deposit_rho(species_list)
    Ex, Ey = solve_poisson(rho)
    for sp in species_list:
        boris_push(sp, Ex, Ey)
    if it % snap_every == 0 or it == steps-1:
        rho_net = deposit_rho(species_list)
        Ex_d, Ey_d = solve_poisson(rho_net)
        Erms = np.sqrt(np.mean(Ex_d**2 + Ey_d**2))
        eta = neutralization_fraction(rho_net)
        eta_field = 1.0 - (Erms / (Erms_e_baseline + 1e-30))
        eta_hist.append(eta)
        Erms_hist.append(Erms)
        rho_rms_hist.append(np.sqrt(np.mean(rho_net**2)))
        eta_field_hist.append(eta_field)
        print(f"Step {it}/{steps} — η_rho={eta:.4f}, η_field={eta_field:.4f}, E_rms={Erms:.3e}")

# -------------------- Final diagnostics --------------------
rho_final = deposit_rho(species_list)
Ex_f, Ey_f = solve_poisson(rho_final)
E_mag = np.sqrt(Ex_f**2 + Ey_f**2)

plt.figure(figsize=(6,5))
plt.imshow(rho_final.T, origin='lower', extent=[0,Lx,0,Ly])
plt.colorbar(label='ρ (C/m³)')
plt.title('Final net charge density')
plt.tight_layout(); plt.savefig('plots/final_rho_mag.png', dpi=300)

plt.figure(figsize=(6,5))
plt.imshow(E_mag.T, origin='lower', extent=[0,Lx,0,Ly])
plt.colorbar(label='|E| (V/m)')
plt.title('Final |E| map')
plt.tight_layout(); plt.savefig('plots/final_E_mag.png', dpi=300)

plt.figure(figsize=(6,4))
plt.plot(eta_hist); plt.grid(alpha=0.3)
plt.xlabel('Snapshot'); plt.ylabel('η_ρ')
plt.title('Neutralization vs time (charge-based)')
plt.tight_layout(); plt.savefig('plots/neutralization_fraction_mag.png', dpi=300)

plt.figure(figsize=(6,4))
plt.plot(eta_field_hist); plt.grid(alpha=0.3)
plt.xlabel('Snapshot'); plt.ylabel('η_E')
plt.title('Neutralization vs time (field-based)')
plt.tight_layout(); plt.savefig('plots/neutralization_field_mag.png', dpi=300)

plt.figure(figsize=(6,4))
plt.plot(Erms_hist); plt.grid(alpha=0.3)
plt.xlabel('Snapshot'); plt.ylabel('E_rms (V/m)')
plt.title('Field suppression vs time')
plt.tight_layout(); plt.savefig('plots/E_rms_mag.png', dpi=300)

plt.figure(figsize=(6,4))
plt.plot(rho_rms_hist); plt.grid(alpha=0.3)
plt.xlabel('Snapshot'); plt.ylabel('RMS(ρ) (C/m³)')
plt.title('Net charge RMS vs time')
plt.tight_layout(); plt.savefig('plots/rho_rms_mag.png', dpi=300)

print("Magnetized run complete. Plots saved to ./plots/")
