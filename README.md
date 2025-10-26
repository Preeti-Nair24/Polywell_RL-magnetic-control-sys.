# Polywell_RL-magnetic-control-sys.
**What This Simulation Does**

This is a complete end-to-end simulation of a Polywell fusion reactor where reinforcement learning (RL) in MATLAB controls the magnetic field by adjusting coil currents in ANSYS Maxwell, which then affects plasma confinement, leading to fusion reactions, producing neutrons, and generating power from heat absorbed through by liquid hydrogen introduced around the Polywell geometry.

**Polywell Fusion Concept**

What is Polywell?

A Polywell is a type of fusion reactor that uses 6 circular magnetic coils arranged at the faces of a cube. These coils create a magnetic field configuration called "magnetic cusps" that confines plasma.

Why This Design?

-   **6 coils** create a spherical magnetic "bottle"
-   Plasma particles spiral along magnetic field lines
-   At the **center**: Magnetic field is WEAK → plasma confined
-   At the **edges**: Magnetic field is STRONG → particles reflect back
-   **Cusps** (where coils meet): Particles can escape if field is wrong

**Workflow structure**
+-----------------------------------+
|        Input Parameters           |
|  (Geometry, Fuel, Coils, LH₂)     |
+-----------------------------------+
               ↓
+-----------------------------------+
|    Ansys Maxwell: Magnetic Fields |
|  - Magnetostatic/Transient Solvers|
|  - Geometry: Cubic polywell, coils|
|  - Excitations: Coil currents     |
|  - MHD: Plasma as conductive fluid|
|  - Output: B-fields, plasma data  |
+-----------------------------------+
               ↓ (B-field, plasma data)
               |
+-----------------------------------+
| MATLAB: RL for Confinement Control|
|  - RL Toolbox: PPO Agent          |
|  - Simulink: Plasma response      |
|  - Actions: Coil currents         |
|  - Observations: Density, well    |
|  - Reward: Fusion rate - losses   |
|  - Output: Optimized currents     |
+-----------------------------------+
               ↕ (Feedback: Currents to Maxwell)
               |
+-----------------------------------+
|    LH₂ Heat Capture (Fluent/MATLAB)|
|  - Fluent: CFD for LH₂ flow       |
|  - MATLAB: Lumped thermal model   |
|  - Input: Fusion heat flux        |
|  - Output: Heat, LH₂ temp data    |
+-----------------------------------+
               ↓
+-----------------------------------+
|   Power Calculation & Visualization|
|  - Calculate: Thermal/electrical   |
|  - Plot: Heat, LH₂ temp vs. time   |
|  - Output: Data table, plots       |
+-----------------------------------+
               ↓
+-----------------------------------+
|            Results                |
|  - Data: Heat, temp over time     |
|  - Visuals: Dual-axis plots       |
+-----------------------------------+

**The Big Picture**

Start with 6 magnetic coils

↓

RL adjusts their currents

↓

ANSYS Maxwell calculates magnetic field

↓

Plasma particles get confined (or escape)

↓

Confined plasma undergoes fusion

↓

Fusion produces neutrons and energy

↓

Liquid hydrogen absorbs heat

↓

Temperature of the hydrogen used to calculate overall power generated




-   Continues until the RL agent achieves optimal performance.

