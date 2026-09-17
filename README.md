# Physics-Informed Neural Networks (PINNs) for Transient Diffusivity in Reservoir Engineering

## Overview

This repository contains the numerical models and documentation for applying Physics-Informed Neural Networks (PINNs) to solve the 1D Cartesian and 2D cylindrical radial transient diffusivity equations. By transforming the traditional partial differential equations (PDEs) governing fluid flow in porous media into a loss function optimization problem, this project provides a continuous spatiotemporal solution framework that circumvents the limitations of discrete numerical mesh grids.

Using an advanced simultaneous inverse PDE formulation, the model successfully estimates missing reservoir physics—specifically, average permeability ($k$) and distance to sealing boundaries ($x_e$)—directly from noisy production flowrate history.

## Key Capabilities & Technical Architecture

* **Simultaneous Inverse Problem Resolution:** Unlike traditional sequential solvers, this architecture utilizes custom trainable variables ($\eta_k$ for permeability and $\eta_l$ for boundary distance) to resolve inverse parameters simultaneously, drastically reducing training time and converging on true reservoir properties (e.g., 120 md permeability and 500 m boundary distance) even with noisy production data.


* **Conservative PINNs (CPINNs) for Stiff PDEs:** To address the severe non-linearity and violent pressure gradients at the near-wellbore region in the 2D radial case, the architecture utilizes domain decomposition. The reservoir cell is divided into subdomains connected by a 50m interface, enforcing both pressure continuity and pressure flux (derivative) conservation to satisfy the law of conservation of mass.


* **Automatic Differentiation for Flowrate Extraction:** Production flowrate histories are not manually discrete; they are dynamically calculated backward through the computational graph from the pressure output to the distance input using the exact chain rule via automatic differentiation.


* **Equation Scaling:** All governing PDEs are transformed into dimensionless space ($\tilde{x}$ or $\tilde{r}$) and dimensionless time ($\tilde{t}$) forms to strictly bind inputs and outputs between $[0, 1]$, satisfying the fundamental scaling requirements for efficient neural network gradient descent.



## Mathematical Formulation

### 1D Cartesian Diffusivity

The dimensionless PDE for 1D Cartesian equation scaling in PINNs is defined as:


$$\frac{\partial^{2}\tilde{P}}{\partial\tilde{x}^{2}}=\frac{1}{t_{D_{max}}}\frac{\partial\tilde{P}}{\partial\tilde{t}}$$

To solve the simultaneous inverse problem, the PDE is modified using permeability and length factors:


$$\frac{\partial\tilde{P}}{\partial\tilde{t}}=\frac{\partial^{2}P}{\partial\tilde{x}^{2}}\frac{\eta_{k}}{\eta_{l}^{2}}(\frac{k_{max}t_{max}}{\phi\mu c_{t}x_{e,max}^{2}})$$

### 2D Cylindrical Radial Diffusivity

The dimensionless radial PDE utilized for the CPINN domain decomposition is defined as:


$$\frac{1}{t_{D,max}}\frac{\partial\overline{P}}{\partial\overline{t}}=\frac{1}{\overline{r}}\frac{\partial\overline{P}}{\partial\overline{r}}+\frac{\partial^{2}\overline{P}}{\partial\overline{r}^{2}}$$

## Simulated Cases & Results

### Section 1: 1D Cartesian Cases

* **Case 1.1 (Forward):** Well controlled by constant bottom-hole pressure (BHP) at $x_w = 0$ with a single no-flow boundary at $x_e = 500m$. The PINN successfully learned the PDE to map the pressure transient across the reservoir.


* **Case 1.2 (Forward):** Well placed inside the domain between two no-flow boundaries ($-500m$ and $750m$). Solved using subdomains connected at the well interface.


* **Inverse Problem:** Injected with noisy synthetic flowrate data. Over multiple runs to account for neural network stochasticity, the model predicted permeability with a low error margin (predicting 116.58 md – 125.90 md against a true 120 md) and boundary distance with exceptional accuracy (predicting 498.03 m – 499.14 m against a true 500 m).



### Section 2: 2D Radial Cases (Domain Decomposition)

* **Case 2.1 (Constant BHP):** Solved using CPINNs to enforce pressure and flux continuity at a 50m interface, capturing the steep pressure drop near the wellbore.


* **Case 2.2 (Constant Rate):** Accurately modeled violent pressure drops at late times (60 to 80 days) caused by the small cross-sectional area near the wellbore, with subdomains effectively separating the near-wellbore and farther reservoir regions.



## Tech Stack & Implementation Notes

* **Core Framework:** Python, utilizing PyTorch and DeepXDE for robust neural network architecture and auto-differentiation graph compilation.
* **Data Handling:** NumPy and SciPy for tensor manipulation and mathematical validation.
* **Visualization:** Matplotlib for plotting multi-trace pressure transient episodes, flowrate histories, and inverse parameter convergence dashboards.

## Author & References

* **Documentation & Custom Inverse Formulation:** Igbereyivwe Oghenetejiri Derek.


* **Original Methodology:** Badawi, D., & Gildin, E. (2023). *Physics-Informed Neural Network for the Transient Diffusivity Equation in Reservoir Engineering*.
