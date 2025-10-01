# Full PIC script with magnetized/unmagnetized toggle, robust baselines,

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
np.random.seed(12345678)
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

# Beam/neutralizer densities (per m^3). a 2D slab with unit thickness in z is assumed.
ne_e  = 6.9e16   # electron beam density
ne_p  = 6.9e16   # positron neutralizer density (set equal for ideal cancellation; you can detune)

# Axial speeds (mono-energetic here; can add spreads)
vz_e = 2.4e8
vz_p = 2.4e8

# Magnetic field: toggle here. Set to 0.0 for unmagnetized, >0 for magnetized.
Bz = 2.0   # Tesla Set Bz = 0.0 for unmag case.

# Thermal spreads to set a finite Debye length (this improves PIC stability)
Te_e_eV = 0.5
Te_p_eV = 0.5

# Particles per cell
ppc = 100

# Smoothing for charge to reduce grid noise (σ in cells)
rho_smooth_sigma = 0.8

# Number of plasma or cyclotron periods to simulate (meaning depends on Bz)
N_periods = 30     # if Bz>0 interpreted as cyclotron periods; if Bz==0 we switch to plasma periods below

# -------------------- Derived geometry & helpers --------------------
dx = Lx / Nx
dy = Ly / Ny
Lz = 1.0  # unit thickness for 2D -> 3D scaling
cell_area = dx * dy
volume = Lx * Ly * Lz

# Convert temperatures to thermal speeds
Te_e = Te_e_eV * e
Te_p = Te_p_eV * e
vth_e = np.sqrt(kB*Te_e / me)
vth_p = np.sqrt(kB*Te_p / me)

# Compute ω_pe for electrons (dominant)
omega_pe = np.sqrt(ne_e * e**2 / (me * e0))

# Compute γ from axial speed (ignore v_perp for initial estimate)
def gamma_from_v(v):
   return 1.0/np.sqrt(1.0 - (v/c)**2)

gamma_e = gamma_from_v(vz_e)

# If Bz==0 we'll use plasma periods; otherwise cyclotron periods
rL_target = 4.0 * dx  # intended gyroradius resolution

# Solve for v_perp required by rL_target given Bz (if Bz>0)
if Bz > 0.0:
    vperp_e = rL_target * e * Bz / (gamma_e * me)
    # ensure vperp is reasonable
    if vperp_e >= 0.2*c:
        # cap with warning and reduce rL_target effectively
        print("Warning: computed v_perp >= 0.2c; capping vperp to 0.2c and recomputing effective rL.")
        vperp_e = 0.2*c
        rL_target = (gamma_e * me * vperp_e) / (e * Bz)
else:
    vperp_e = 0.0

# Cyclotron frequency (if Bz>0)
omega_c = e * Bz / (gamma_e * me) if Bz > 0 else 0.0
Tc = 2.0 * np.pi / omega_c if Bz > 0 else None

# -------------------- Time step and steps (unified for Bz=0 and Bz>0) --------------------
# Candidate dt from plasma frequency
dt_plasma = 0.2 / omega_pe

# Candidate dt from cyclotron (if Bz=0, omega_c=0 so this is huge -> irrelevant)
dt_cyclotron = 0.2 / max(omega_c, 1e-30)

# Candidate dt from particle CFL
vmax_xy_candidate = max(3.0*vth_e, vperp_e, 1e-20)
cfl_dt = 0.25 * min(dx, dy) / vmax_xy_candidate

# Choose common dt that works for both cases
dt = min(dt_plasma, dt_cyclotron, cfl_dt)

# Common total physical duration: interpret N_periods consistently
if Bz > 0:
    T_period = 2.0 * np.pi / omega_c   # cyclotron period
else:
    T_period = 2.0 * np.pi / omega_pe  # plasma period
T_total = N_periods * T_period

# Force both runs to cover same duration with same number of steps
steps = int(np.ceil(T_total / dt))
dt = T_total / steps   # adjust so steps*dt == T_total exactly

print(f"dt={dt:.3e}, steps={steps}, T_total={T_total:.3e}, "
      f"dt_candidates: plasma={dt_plasma:.3e}, cyclo={dt_cyclotron:.3e}, cfl={cfl_dt:.3e}")

# -------------------- Diagnostics/metadata --------------------
if not os.path.exists('plots'):
   os.makedirs('plots')
if not os.path.exists('dumps'):
   os.makedirs('dumps')

meta = {
   'Lx': Lx, 'Ly': Ly, 'Nx': Nx, 'Ny': Ny,
   'dx': dx, 'dy': dy,
   'ne_e': ne_e, 'ne_p': ne_p,
   'vz_e': vz_e, 'vz_p': vz_p,
   'Te_e_eV': Te_e_eV, 'Te_p_eV': Te_p_eV,
   'Bz': Bz, 'omega_c': omega_c, 'omega_pe': omega_pe,
   'vperp_e': vperp_e, 'rL_target': rL_target,
   'dt': dt, 'steps': steps, 'Tc': Tc, 'N_periods': N_periods,
}

with open('plots/run_meta.txt', 'w') as f:
   for k,v in meta.items(): f.write(f"{k}: {v}\n")

# -------------------- Species initialization --------------------

def init_species(q, m, n0, vz_mean, vperp, vth):
   Np = ppc * Nx * Ny
   x = np.random.rand(Np) * Lx
   y = np.random.rand(Np) * Ly
   # for magnetized case initialize ring with random angle + thermal spread
   if vperp > 0:
       angles = 2*np.pi*np.random.rand(Np)
       vx = vperp * np.cos(angles) + np.random.normal(0.0, vth/np.sqrt(2), size=Np)
       vy = vperp * np.sin(angles) + np.random.normal(0.0, vth/np.sqrt(2), size=Np)
   else:
       vx = np.random.normal(0.0, vth / np.sqrt(2), size=Np)
       vy = np.random.normal(0.0, vth / np.sqrt(2), size=Np)
   vz = np.ones(Np) * vz_mean
   w = (n0 * cell_area * Lz) / ppc
   return {'x': x, 'y': y, 'vx': vx, 'vy': vy, 'vz': vz,
           'q': q, 'm': m, 'Np': Np, 'w': w}

Electrons = init_species(-e, me, ne_e, vz_e, vperp_e, vth_e)
Positrons = init_species(+e, me, ne_p, vz_p, vperp_e, vth_p)
species_list = [Electrons, Positrons]

# -------------------- Grid ops --------------------

kx = np.fft.fftfreq(Nx, d=dx) * 2*np.pi
ky = np.fft.fftfreq(Ny, d=dy) * 2*np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K2 = KX**2 + KY**2
K2[0,0] = 1.0   # avoid dividing by zero in spectral solve

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
       np.add.at(rho, (i0, j0), charge * w00)
       np.add.at(rho, (i1, j0), charge * w10)
       np.add.at(rho, (i0, j1), charge * w01)
       np.add.at(rho, (i1, j1), charge * w11)
   rho /= (cell_area * Lz)
   if rho_smooth_sigma > 0:
       rho = gaussian_filter(rho, sigma=rho_smooth_sigma)
   # remove mean to enforce zero box-average (consistent with periodic solver)
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
   Exp = (Ex[i0,j0]*(1-tx)*(1-ty) + Ex[i1,j0]*tx*(1-ty) + Ex[i0,j1]*(1-tx)*ty + Ex[i1,j1]*tx*ty)
   Eyp = (Ey[i0,j0]*(1-tx)*(1-ty) + Ey[i1,j0]*tx*(1-ty) + Ey[i0,j1]*(1-tx)*ty + Ey[i1,j1]*tx*ty)
   return Exp, Eyp

def boris_push(sp, Ex, Ey):
   """Boris mover including Bz rotation (works for Bz=0 as well)."""
   q = sp['q']; m = sp['m']
   vx, vy = sp['vx'], sp['vy']
   Exp, Eyp = gather_field(sp, Ex, Ey)
   # half electric kick
   vxm = vx + (q*Exp/m)*(0.5*dt)
   vym = vy + (q*Eyp/m)*(0.5*dt)
   # magnetic rotation around z (handles Bz=0)
   t = (q*Bz/m)*(0.5*dt)
   s = 2*t/(1+t**2)
   # rotation formula (z-only)
   vxp = vxm + vym * t
   vyp = vym - vxm * t
   vxn = vxp + vyp * s
   vyn = vyp - vxp * s
   # final half electric kick
   sp['vx'] = vxn + (q*Exp/m)*(0.5*dt)
   sp['vy'] = vyn + (q*Eyp/m)*(0.5*dt)
   # advance positions
   sp['x'] = (sp['x'] + sp['vx']*dt) % Lx
   sp['y'] = (sp['y'] + sp['vy']*dt) % Ly

def rho_from_single(sp):
   return deposit_rho([sp])

# -------------------- Neutralization denominators (robust) --------------------
# Build an averaged baseline from M independent electron-only realizations.
M_baseline = 30
rhos = np.zeros((M_baseline, Nx, Ny))
per_sample_rms = np.zeros(M_baseline)

for i in range(M_baseline):
    sp_e_sample = init_species(-e, me, ne_e, vz_e, vperp_e, vth_e)  # fresh sample
    rhos[i] = deposit_rho([sp_e_sample])
    per_sample_rms[i] = np.sqrt(np.mean(rhos[i]**2))

rho_e_only_gridavg = rhos.mean(axis=0)
rms_gridavg = np.sqrt(np.mean(rho_e_only_gridavg**2))
median_sample_rms = np.median(per_sample_rms)
abs_floor = 1e-12
den_scalar = max(rms_gridavg, median_sample_rms, abs_floor)

print(f"Baseline samples M={M_baseline}: RMS(single sample) = {per_sample_rms[0]:.3e}, RMS(averaged baseline) = {rms_gridavg:.3e}")
print(f"Chosen denominator den_scalar = {den_scalar:.3e}")
print(f"Baseline avg field min/max: {rho_e_only_gridavg.min():.3e} / {rho_e_only_gridavg.max():.3e}")

# Save baseline dumps for reproducibility
np.savez('dumps/electron_baseline_samples.npz', rhos=rhos, per_sample_rms=per_sample_rms, rho_gridavg=rho_e_only_gridavg)

# -------------------- Field-based baseline (Option A) --------------------
sp_e_baseline = init_species(-e, me, ne_e, vz_e, vperp_e, vth_e)
rho_e_baseline = deposit_rho([sp_e_baseline])
Ex_e, Ey_e = solve_poisson(rho_e_baseline)
Erms_e_baseline = np.sqrt(np.mean(Ex_e**2 + Ey_e**2))
print(f"Electron-only Erms baseline = {Erms_e_baseline:.3e}")

# -------------------- Neutralization / overlap functions --------------------
def neutralization_fraction_charge(rho_net, den=den_scalar):
    """Charge-based neutralization fraction using L2 norm ratio.
       No clipping: returns raw eta = 1 - ratio.
    """
    num = np.sqrt(np.mean(rho_net**2))
    den_safe = max(den, 1e-16)
    ratio = num / den_safe
    eta = 1.0 - ratio
    return eta, ratio

def overlap_O(rho_b, rho_e):
    """Compute overlap metric O"""
    num = -np.sum(rho_b * rho_e) * (dx*dy)
    den = np.sqrt(np.sum(rho_b**2)*(dx*dy) * np.sum(rho_e**2)*(dx*dy)) + 1e-30
    return num / den

# -------------------- Energy helpers --------------------
def compute_kinetic_energy(species_list):
    KE = 0.0
    for sp in species_list:
        v2 = sp['vx']**2 + sp['vy']**2 + sp['vz']**2
        # sp['w'] already contains volume-per-macro
        KE += np.sum(0.5 * sp['m'] * v2 * sp['w'])
    return KE  # Joules

def compute_field_energy(Ex, Ey):
    FE = 0.5 * e0 * np.sum(Ex**2 + Ey**2) * (dx*dy*Lz)
    return FE

# -------------------- Main loop (time advance + diagnostics) --------------------
eta_hist = []
eta_raw_hist = []
eta_field_hist = []
Erms_hist = []
rho_rms_hist = []
overlap_hist = []
KE_hist = []
FE_hist = []
TE_hist = []
dE_rel_hist = []

snap_every = max(1, steps//200)

# initial energies
rho0 = deposit_rho(species_list)
Ex0, Ey0 = solve_poisson(rho0)
KE0 = compute_kinetic_energy(species_list)
FE0 = compute_field_energy(Ex0, Ey0)
TE0 = KE0 + FE0

print(f"Initial KE={KE0:.6e} J, FE={FE0:.6e} J, TE={TE0:.6e} J")

# save an initial species-resolved snapshot
rho_b_init = deposit_rho([Positrons])   # positive species
rho_e_init = deposit_rho([Electrons])   # electron species
np.savez('dumps/initial_species_rho.npz', rho_b=rho_b_init, rho_e=rho_e_init)

for it in range(steps):
   rho = deposit_rho(species_list)
   Ex, Ey = solve_poisson(rho)

   # advance particles
   for sp in species_list:
       boris_push(sp, Ex, Ey)

   # diagnostics snapshot
   if it % snap_every == 0 or it == steps-1:
       rho_net = deposit_rho(species_list)
       # species-resolved deposits (for overlap metric)
       rho_b = deposit_rho([Positrons])
       rho_e = deposit_rho([Electrons])

       eta, ratio = neutralization_fraction_charge(rho_net)
       Erms = np.sqrt(np.mean(Ex ** 2 + Ey ** 2))
       # overlap
       O = overlap_O(rho_b, rho_e)

       # energies
       KE = compute_kinetic_energy(species_list)
       FE = compute_field_energy(Ex, Ey)
       TE = KE + FE
       dE_rel = (TE - TE0) / (TE0 + 1e-30)

       # store histories
       eta_hist.append(eta)
       eta_raw_hist.append(ratio)
       Erms_hist.append(Erms)
       rho_rms_hist.append(np.sqrt(np.mean(rho_net**2)))
       overlap_hist.append(O)
       KE_hist.append(KE)
       FE_hist.append(FE)
       TE_hist.append(TE)
       dE_rel_hist.append(dE_rel)

       # species-resolved dump every snapshot (optional — can be heavy)
       snap_idx = len(eta_hist)-1
       np.savez(f'dumps/snap_{snap_idx:04d}.npz',
                rho_net=rho_net, rho_b=rho_b, rho_e=rho_e,
                Ex=Ex, Ey=Ey, KE=KE, FE=FE, TE=TE, step=it)

       print(f"Step {it}/{steps} — eta_charge={eta:.4f}, E_rms={Erms:.3e}, "
             f"O={O:.3f}, KE={KE:.3e}, FE={FE:.3e}, TE={TE:.3e}, dE_rel={dE_rel:.2e}")

# -------------------- Final diagnostics/plots --------------------
rho_final = deposit_rho(species_list)
Ex_f, Ey_f = solve_poisson(rho_final)
E_mag = np.sqrt(Ex_f**2 + Ey_f**2)

# save final species-resolved fields
rho_b_final = deposit_rho([Positrons])
rho_e_final = deposit_rho([Electrons])
np.savez('dumps/final_species_rho_mag.npz', rho_b=rho_b_final, rho_e=rho_e_final, rho_net=rho_final, Ex=Ex_f, Ey=Ey_f)

plt.figure(figsize=(6,5))
plt.imshow(rho_final.T, origin='lower', extent=[0,Lx,0,Ly])
plt.colorbar(label='ρ (C/m³)')
plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.title('Final net charge density')
plt.tight_layout(); plt.savefig('plots/final_rho_mag.png', dpi=300)

plt.figure(figsize=(6,5))
plt.imshow(E_mag.T, origin='lower', extent=[0,Lx,0,Ly])
plt.colorbar(label='|E| (V/m)')
plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.title('Final |E| map')
plt.tight_layout(); plt.savefig('plots/final_E_mag_mag.png', dpi=300)

plt.figure(figsize=(6,4))
plt.plot(eta_hist, label='eta_charge')
plt.xlabel('Snapshot'); plt.ylabel('Neutralization metric')
plt.title('Neutralization vs time')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('plots/neutralization_fraction_mag.png', dpi=300)

plt.figure(figsize=(6,4))
plt.plot(Erms_hist)
plt.xlabel('Snapshot'); plt.ylabel('E_rms (V/m)')
plt.title('Field suppression vs time')
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('plots/E_rms_mag.png', dpi=300)

plt.figure(figsize=(6,4))
plt.plot(rho_rms_hist)
plt.xlabel('Snapshot'); plt.ylabel('RMS(ρ) (C/m³)')
plt.title('Net charge RMS vs time')
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('plots/rho_rms_mag.png', dpi=300)

plt.figure(figsize=(6,4))
plt.plot(overlap_hist)
plt.xlabel('Snapshot'); plt.ylabel('Overlap O')
plt.title('Species overlap vs time')
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('plots/overlap_O_mag.png', dpi=300)

plt.figure(figsize=(6,4))
plt.plot(KE_hist, label='KE'); plt.plot(FE_hist, label='FE'); plt.plot(TE_hist, label='TE')
plt.legend(); plt.grid(True, alpha=0.3)
plt.xlabel('Snapshot'); plt.ylabel('Energy (J)')
plt.title('Energy history')
plt.tight_layout(); plt.savefig('plots/energy_history_mag.png', dpi=300)

plt.figure(figsize=(6,4))
plt.plot(dE_rel_hist)
plt.xlabel('Snapshot'); plt.ylabel('relative dE')
plt.title('Relative energy drift (TE-TE0)/TE0')
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('plots/energy_drift_mag.png', dpi=300)

# Save timeseries arrays to a npz for post-processing
np.savez('dumps/timeseries.npz',
         eta_hist=np.array(eta_hist), eta_raw_hist=np.array(eta_raw_hist),
         Erms_hist=np.array(Erms_hist), rho_rms_hist=np.array(rho_rms_hist), overlap_hist=np.array(overlap_hist),
         KE_hist=np.array(KE_hist), FE_hist=np.array(FE_hist), TE_hist=np.array(TE_hist),
         dE_rel_hist=np.array(dE_rel_hist))

print('Run complete. Plots saved to ./plots and metadata to plots/run_meta.txt; dumps in ./dumps')
