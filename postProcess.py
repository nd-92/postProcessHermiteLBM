import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

# --- 1. Read ----------------------------------------------------------

cutLocation = 24.0
mesh = pv.read("postProcess/U_100000.vts")
nx, ny, nz = mesh.dimensions
print(f"dims: {nx} x {ny} x {nz}  ({nx*ny*nz/1e6:.1f}M points)")

# --- 2. Coordinate arrays --------------------------------------------
# VTK stores structured-grid points x-fastest. Reshape with C-order to
# (nz, ny, nx, ...) and transpose to (nx, ny, nz, ...) to get an array
# indexed [i, j, k] = value at grid point (i, j, k).
pts3 = np.asarray(mesh.points).reshape(nz, ny, nx, 3).transpose(2, 1, 0, 3)
x = pts3[:, 0, 0, 0].copy()  # (nx,)
y = pts3[0, :, 0, 1].copy()  # (ny,)
z = pts3[0, 0, :, 2].copy()  # (nz,)
del pts3

print(f"x range: {x[0]:.3f} .. {x[-1]:.3f}")
print(f"y range: {y[0]:.3f} .. {y[-1]:.3f}")
print(f"z range: {z[0]:.3f} .. {z[-1]:.3f}")
# Expected: 0..48, 0..48, 0..208. If these are wrong, the storage
# ordering is different — report the values and we can adjust.

# --- 3. Locate the slice at x = 24 -----------------------------------
i0 = int(np.argmin(np.abs(x - cutLocation)))
print(f"slice index i0 = {i0}  (x = {x[i0]})")

i_lo = max(i0 - 1, 0)
i_hi = min(i0 + 2, nx)
sl = slice(i_lo, i_hi)


# --- 4. Load only the slab of each velocity component -----------------
# Scalar fields share the same x-fastest ordering; Fortran-order reshape
# makes the first axis vary fastest, matching VTK. Then slice on x.
def load_slab(name):
    flat = np.asarray(mesh.point_data[name])
    arr3 = flat.reshape(nx, ny, nz, order="F")
    return np.ascontiguousarray(arr3[sl, :, :])


Ux = load_slab("U_x")
Uy = load_slab("U_y")
Uz = load_slab("U_z")
print(f"slab shape: {Ux.shape}")

x_s = x[sl]

# --- 5. Vorticity on the slab ----------------------------------------
omega_x = np.gradient(Uz, y, axis=1) - np.gradient(Uy, z, axis=2)
omega_y = np.gradient(Ux, z, axis=2) - np.gradient(Uz, x_s, axis=0)
omega_z = np.gradient(Uy, x_s, axis=0) - np.gradient(Ux, y, axis=1)

omega_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)

# --- 6. Extract the cut plane -------------------------------------
k0 = i0 - i_lo
W2d = omega_mag[k0, :, :]  # (ny, nz)
W2d_max = np.nanmax(W2d)
W2d_norm = W2d / W2d_max
# print(f"|omega| range on plane: {np.nanmin(W2d):.4g} .. {np.nanmax(W2d):.4g}")

# --- 7. Plot (z horizontal, y vertical) ------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(
    W2d_norm,
    origin="lower",
    extent=[z.min(), z.max(), y.min(), y.max()],
    aspect="equal",
    cmap="Greys",
    interpolation="lanczos",
    vmin=0.0,
    vmax=1.0,
)
ax.set_axis_off()
fig.savefig("VorticityMagnitude.png", dpi=600, bbox_inches="tight", pad_inches=0)
