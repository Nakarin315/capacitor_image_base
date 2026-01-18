# Image-Based Numerical Capacitance Solver

This repository implements a **2D finite-difference Laplace solver** to compute electric potential, electric fields, and capacitance directly from **binary electrode mask images**. Electrode geometries are defined using black-and-white images, enabling rapid electrostatic simulations without CAD tools.

The capacitance is extracted using the electric field energy method and compared against analytical solutions for validation.

---

## Features

- Image-based electrode definition (binary masks)
- 2D Laplace solver using finite difference method (5-point stencil)
- Electric field computation from numerical gradients
- Capacitance calculation via energy integration
- Visualization of electric potential and field lines
- Analytical validation for coaxial and off-center geometries
- Supports arbitrary 2D electrode layouts

---

## Requirements

- Python 3.x  
- NumPy  
- Matplotlib  
- Pillow (PIL)

Install dependencies with:

```bash
pip install numpy matplotlib pillow
