# Polywell_RL-magnetic-control-sys.
What This Simulation Does
This is a complete end-to-end simulation of a Polywell fusion reactor where reinforcement learning (RL) in MATLAB controls the magnetic field by adjusting coil currents in ANSYS Maxwell, which then affects plasma confinement, leading to fusion reactions, producing neutrons, and generating power through a liquid hydrogen coolant system.
The Big Picture
You start with 6 magnetic coils
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
System generates electricity and thrust

🔬 The Polywell Fusion Concept
What is a Polywell?
A Polywell is a type of fusion reactor that uses 6 circular magnetic coils arranged at the faces of a cube. These coils create a magnetic field configuration called "magnetic cusps" that confines plasma.
