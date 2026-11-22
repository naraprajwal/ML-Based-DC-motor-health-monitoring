# Safe & Reversible DC Motor Fault Induction Guide (College Electrical Machines Lab)

> **Safety Rule:** Never perform destructive or irreversible modifications on a college Electrical Machines Lab DC motor (typically 220 V DC, 1–5 HP). All testing must be performed under faculty/lab-technician supervision with emergency stop access, thermal monitoring, and isolated instrumentation.

Note that **#4 (Bad Bearings)** and **#6 (Motor Bearing Failure)** represent early and severe stages of the same mechanical failure mode and are grouped together below.

---

## Tier A: Directly Inducible & 100% Reversible on the Lab Motor

### 1. Overload of Motor Windings
- **Physical Mechanism:** Sustained mechanical load torque exceeding rated torque causes elevated armature current ($I_a$), increased copper losses ($I_a^2 R_a$), speed droop, and thermal rise.
- **Safe Lab Induction Method:**
  1. Use the lab's existing **brake drum (Prony brake)** or **coupled DC generator + resistive lamp/rheostat load bank**.
  2. Gradually increase load to **110%–130% of rated armature current** for short, controlled bursts (2–5 minutes).
  3. Continuously monitor casing/commutator temperature and terminate immediately if casing temperature exceeds **55–60 °C** (well below typical Class B/F insulation limits).
- **Observable Signatures:** High $I_a$, reduced RPM, positive temperature slope ($dT/dt > 0$).

### 2. Excessive Vibration (Unbalance, Misalignment, Structural Looseness)
- **Physical Mechanism:** Mass asymmetry on the rotor/coupling, angular/parallel shaft offset, or loose base mounting produces strong $1\times$ RPM and $2\times$ RPM vibration harmonics.
- **Safe Lab Induction Methods:**
  - **Rotor Unbalance:** Secure a small known asymmetric mass (e.g., nut/bolt clamped tightly with a stainless hose clamp) onto the coupling or brake-drum rim behind a protective safety guard.
  - **Soft-Foot / Base Looseness:** Slightly loosen (½ turn, secured with a locknut) one mounting bolt on the motor bedplate, or insert a thin rubber/shim washer under one foot.
  - **Coupling Misalignment:** Insert a 0.5–1.0 mm shim under one side of the motor mount to introduce slight controlled offset at the flexible coupling (only with technician approval).
- **Observable Signatures:** Strong $1\times$ and $2\times$ shaft-frequency peaks in accelerometer FFT; elevated RMS vibration without abnormal electrical current draw.

---

## Tier B: Safe Proxies or Spare-Rig / Removable-Part Testing

### 3. Dirty Commutators / Brush Wear
- **Do NOT** apply oil, grease, dust, or abrasives to a healthy college lab commutator.
- **Safe Electrical Proxy (Main Lab Motor):**
  - Insert a **high-wattage, low-ohm power rheostat (0.2–1.0 Ω)** in series with the armature circuit to mimic elevated brush-commutator contact resistance ($R_b$).
  - Optionally toggle a small parallel resistor or vary contact slightly to introduce intermittent contact-resistance ripple.
- **Direct Mechanical Test (Spare/Older Motor Only):**
  - Slightly reduce brush spring tension or temporarily swap in a worn/chipped spare carbon brush to generate mild sparking and high-frequency commutation ripple.
- **Observable Signatures:** Elevated $V_a/\text{RPM}$ voltage drop, high-frequency current ripple ($dI_a/dt$ RMS), localized commutator heating.

### 4 & 6. Bad Bearings / Motor Bearing Failure
- **Do NOT** inject grit/sand or dry-run the primary lab motor's bearings.
- **Recommended Safe Lab Methods:**
  - **Removable Bearing Swap / Dedicated Spare Motor:** Use a low-cost spare brushed DC motor (e.g., 24 V/48 V or fractional-HP motor) or swap a removable end-bell bearing (e.g., 6204/6205) with a pre-damaged (EDM-notched or grease-depleted) test bearing kept strictly for experiments.
  - **Friction-Only Proxy (Main Lab Motor):** Apply a light, constant drag on the brake drum to emulate increased viscous friction ($B$). Note: this reproduces speed/current friction effects, while true periodic impact vibrations (BPFO/BPFI harmonics and high kurtosis) require a physically notched bearing on a test rig.
- **Observable Signatures:** High vibration **kurtosis** and **crest factor**, elevated energy in the bearing resonance band (130–190+ Hz), increased friction current, and localized bearing end-bell heating.

---

## Tier C: NEVER Induce Directly (Use Safe External Proxies & Simulation)

### 5. Degraded Winding Insulation Breakdown (Inter-Turn / Ground Fault)
- **Why Never Direct:** Scraping, overheating, or shorting windings permanently destroys the motor and poses severe electrical shock and fire hazards.
- **Safe Lab Proxies:**
  - **Field-Weakening / Partial Field-Short Proxy:** Use the lab's **field rheostat** to reduce field excitation current, mimicking the reduced effective flux ($K_e\Phi$) and resulting speed/current shift of a partial field inter-turn fault.
  - **Offline Insulation Verification:** Measure baseline winding-to-frame insulation resistance safely with the lab's **500 V DC Megger** (motor completely disconnected and unpowered).
- **Observable Signatures (Simulated):** Drop in effective $R_a$, $L_a$, and $K_e\Phi$, elevated current draw, rapid localized thermal rise.

### 7 (List #2). Loose Windings
- **Why Never Direct:** Cannot be induced reversibly on varnish-impregnated lab windings.
- **Safe Validation Approach:**
  - Model in physics simulation (electromagnetic force vibration at $100/200\text{ Hz}$ proportional to $I_a^2$ + minor $R_a$ jitter).
  - To validate hardware detection of electromagnetic-frequency chatter, temporarily attach an external small **vibration exciter / coin motor** driven at 100 Hz / 200 Hz to the motor casing.
