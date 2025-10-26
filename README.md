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
![Alt text](misc/workflow.png)

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

