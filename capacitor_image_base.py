import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_masks(gnd_path, v1_path):
    gnd_mask = np.array(Image.open(gnd_path).convert('L')) < 128  # black = True
    v1_mask = np.array(Image.open(v1_path).convert('L')) < 128
    shape = gnd_mask.shape
    V = np.zeros(shape)

    V[v1_mask] = 1.0   # 1V electrode
    V[gnd_mask] = 0.0  # GND

    mask_fixed = v1_mask | gnd_mask
    return V, mask_fixed

def solve_laplace(V, mask_fixed, max_iter=100000, tol=5e-6):
    for i in range(max_iter):
        V_old = V.copy()
        # solving laplace 2D using finite difference methods (FDM)
        # 5-point stencil update
        V[1:-1, 1:-1] = 0.25 * (V_old[1:-1, :-2] + V_old[1:-1, 2:] +
                                V_old[:-2, 1:-1] + V_old[2:, 1:-1])
        V[mask_fixed] = V_old[mask_fixed]
        if np.max(np.abs(V - V_old)) < tol:
            print('Total iter: ', i)
            break
    return V

def compute_electric_field(V):
    # Calculate electrifield from gradient of voltage
    Ey, Ex = np.gradient(-V, pixel_size)  # Now E is in V/m
    return Ex, Ey



# Choose this for normal capacitor
# path1 = r"E:\Singapore\git_hub\git_capacitance_numerical_jpg\example_2\001.jpg"
# path2 =r"E:\Singapore\git_hub\git_capacitance_numerical_jpg\example_2\002.jpg"

# Choose this for off-center capacitor
path1 = r"E:\Singapore\git_hub\git_capacitance_numerical_jpg\example_2\off_001.jpg"
path2 =r"E:\Singapore\git_hub\git_capacitance_numerical_jpg\example_2\off_002.jpg"

# Load masks from images
V, mask_fixed = load_masks(path1,path2)
# Solve Laplace’s equation
V_sol = solve_laplace(V, mask_fixed)


pixel_size = 5e-3  # e.g., 1 mm per pixel → 1e-3 meters

# Compute electric field
Ex, Ey = compute_electric_field(V_sol)

# Plot results
gnd_mask = np.array(Image.open(path1).convert('L')) < 128  # black = True
v1_mask = np.array(Image.open(path2).convert('L')) < 128  # black = True

plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.imshow(V_sol, cmap='inferno')
plt.title("Potential V")
plt.colorbar()

# plt.subplot(1, 2, 2)
# plt.streamplot(np.arange(V.shape[1]), np.arange(V.shape[0]), Ex, Ey, color='blue', density=1.5)
# plt.title("Electric Field Lines")
# plt.tight_layout()
# plt.show()

plt.subplot(1, 2, 2)
plt.streamplot(np.arange(V.shape[1]), np.arange(V.shape[0]), Ex, Ey, color='blue', density=2)
# Overlay GND mask with a semi-transparent red color
plt.imshow(gnd_mask, cmap='Blues', alpha=0.4, origin='upper')
plt.imshow(v1_mask, cmap='Reds', alpha=0.4, origin='upper')
plt.title("Electric Field Lines")

plt.tight_layout()
plt.savefig("cyl_capacitor_off.pdf", dpi=300, bbox_inches="tight")
plt.show()

# Define constants
epsilon_0 = 8.854e-12  # F/m
V0 = 1.0  # applied voltage (1V from your mask)

# Compute |E|^2
E_mag_squared = Ex**2 + Ey**2

# Total energy U = (1/2) * ε₀ * ∫ E² dA
L=1
dA = pixel_size**2
U = 0.5 * epsilon_0 * np.sum(E_mag_squared) * dA*L

# Capacitance C = 2U / V0^2
C = 2 * U / V0**2

# print(f"Numerical capacitance: {C:.3e} F")



# --- Geometry parameters (estimate from image) ---
a_pixels = 50/2   # radius of inner rod in pixels (adjust to match your image)
b_pixels = 150/2  # radius of outer cylinder in pixels

a = a_pixels * pixel_size
b = b_pixels * pixel_size
L = 200e-3 # Length along the axis (z), use image height

# --- Analytic Capacitance ---
epsilon_0 = 8.854e-12
epsilon_r = 1.0
C_analytic = (2 * np.pi * epsilon_0 * epsilon_r * L) / np.log(b / a)

print(f"Analytic capacitance (coaxial): {C_analytic:.3e} F")
print(f"Numerical capacitance (image-based): {C:.3e} F")
print(f"Relative error: {abs(C - C_analytic) / C_analytic * 100:.2f}%")



# --- Geometry parameters (estimate from image) off center---
a_pixels = 50/2   # radius of inner rod in pixels (adjust to match your image)
b_pixels = 150/2  # radius of outer cylinder in pixels
d_pixels = 30

a = a_pixels * pixel_size
b = b_pixels * pixel_size
d = d_pixels * pixel_size
L = 1 # Length along the axis (z), use image height

# --- Analytic Capacitance ---
epsilon_0 = 8.854e-12
epsilon_r = 1.0
C_analytic = (2 * np.pi * epsilon_0 * epsilon_r * L) / np.arccosh((a**2+b**2-d**2)/(2*a*b))

print(f"Analytic capacitance (coaxial): {C_analytic:.3e} F")
print(f"Numerical capacitance (image-based): {C:.3e} F")
print(f"Relative error: {abs(C - C_analytic) / C_analytic * 100:.2f}%")