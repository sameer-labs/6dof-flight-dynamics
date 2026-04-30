# 6-DOF Longitudinal Flight Dynamics Model

A Python simulation of fixed-wing aircraft flight dynamics, built from scratch during my first year of Aerospace Engineering at City, University of London. I hadn't covered flight mechanics in lectures yet — so naturally I decided to build a full rigid body dynamics model. Perfectly normal behaviour.

This is not an Airbus simulator. But the physics is real, the aerodynamic data is real, and it genuinely took a while to get working.

---

## What it models

- 12-state rigid body equations of motion (full 6-DOF: 3 translational + 3 rotational)
- Longitudinal forces: lift, drag, thrust, gravity — all resolved in the body frame (NED convention)
- Pitching moment with Cmq pitch damping derivative
- Lateral-directional moments: roll (dihedral effect + roll damping) and yaw (weathercock stability + yaw damping)
- Real NACA 2412 aerofoil polar data from XFoil (RE = 1,000,000) via numpy interpolation
- Full 3-2-1 Euler angle kinematics and Direction Cosine Matrix (DCM) position equations
- Gyroscopic coupling terms in the rotational dynamics
- scipy RK45 integration with configurable timestep
- Structured validation against Cessna 172 published cruise data

---

## File structure

| File | What it does |
| --- | --- |
| `flight_dynamics.py` | Core equations of motion, forces, moments, simulation runner |
| `aerocoefficients.py` | NACA 2412 CL/CD/Cm lookup via numpy interpolation |
| `plot_results.py` | Clean matplotlib visualisation module |
| `data/naca2412_polar.csv` | Aerofoil polar data from [airfoiltools.com](http://airfoiltools.com) |
| `plots/` | Output plots (generated on run) |

---

## How to run

```bash
git clone https://github.com/YOUR_USERNAME/6dof-flight-dynamics
cd 6dof-flight-dynamics
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python flight_dynamics.py
```

This runs three tests and saves plots to the `plots/` folder:

1. Trim response (straight and level flight)
2. Lateral perturbation (v = 2 m/s sideslip)
3. Pitch instability demonstration

---

## Aircraft parameters

Generic light aircraft with NACA 2412 wing section.

| Parameter | Value |
| --- | --- |
| Mass | 1,000 kg |
| Wing reference area | 16 m² |
| Wingspan | 10 m |
| Mean chord | 1.5 m |
| Cruise speed | 64 m/s |
| Pitch inertia Iyy | 4,000 kg·m² |
| Air density | 1.225 kg/m³ (sea level, fixed) |

---

## Validation against Cessna 172

Cruise conditions validated against published Cessna 172 figures (Jane's / POH public data):

| Metric | Value | Expected | Result |
| --- | --- | --- | --- |
| CL | 0.2442 | 0.4–0.5 | WARN* |
| CD total | 0.0307 | 0.03–0.04 | PASS |
| L/D | 7.96 | 10–12 | WARN* |
| Lift L | 9,802 N | ≈ mg | PASS |
| Drag D | 1,232 N | ≈ T | PASS |
| T/D | 0.999 | 0.95–1.05 | PASS |

*CL and L/D warn because this model is lighter than the Cessna 172 (1,000 kg vs 1,111 kg). At the same cruise speed, less lift is needed, so the aircraft operates at a lower angle of attack where CL ≈ 0.24. CL_required = mg/(q̄S) = 0.244 — the model is internally consistent. The warnings reflect a different operating point, not a physics error.

---

## Simulation results

### Trim response

The trim test shows the phugoid mode: a long-period altitude/speed oscillation where the aircraft trades potential and kinetic energy. Period ≈ 25–30s. This is a real flight dynamics phenomenon, not a bug. It does not damp because there is no horizontal stabiliser model — see known limitations.

Lateral states (roll, heading, sideslip) remain at zero throughout, confirming correct longitudinal-lateral decoupling.

### Pitch instability demonstration

A second scenario with increased thrust (T = 2,000 N) and initial pitch θ = 5° demonstrates pitch divergence — the aircraft pitches down without a restoring tail moment. This is the correct physical behaviour of an aircraft without a horizontal stabiliser, and neatly illustrates why tail surfaces exist.

---

## What's working

- Full 12-state rigid body EOM with Coriolis and gyroscopic coupling
- Real NACA 2412 aerodynamic data via numpy interpolation
- Pitch damping (Cmq term)
- Roll and yaw stability derivatives
- Full DCM position equations
- scipy RK45 integration completing successfully
- Cessna 172 cruise validation (L/W = 0.999, T/D = 0.999)
- Clean modular code structure

---

## Known limitations

- **No horizontal stabiliser model** — the phugoid oscillation never damps. A tail model would fix this.
- **No ISA atmosphere** — air density fixed at sea level. Lift slightly overestimated at altitude.
- **Constant thrust** — no engine model or throttle response.
- **No stall model** — NACA 2412 polar extrapolated beyond ~15°. High-alpha behaviour is not physically meaningful.
- **Simplified lateral derivatives** — stability derivatives are literature estimates, not computed from geometry.
- **X-Plane SITL integration** — planned but cut due to time constraints (first-year exams). Future work.

---

## What I'd do next

- Add a horizontal stabiliser pitching moment to damp the phugoid
- Implement ISA standard atmosphere for altitude-varying density
- Closed-loop pitch controller to hold level flight
- X-Plane Connect (NASA XPC) integration for SITL validation

---

## References

- NACA 2412 polar data: [airfoiltools.com](http://airfoiltools.com)
- Anderson, J.D. — *Introduction to Flight*
- Nelson, R.C. — *Flight Stability and Automatic Control*
- scipy.integrate.solve_ivp documentation
- Cessna 172 reference: Jane's All the World's Aircraft / Cessna 172 POH (public figures)

---

## Author

Sameer Ahmed — First Year Aerospace Engineering, City, University of London (2025–26)

Built this before covering flight mechanics in lectures. Would not recommend, but also would not un-recommend.