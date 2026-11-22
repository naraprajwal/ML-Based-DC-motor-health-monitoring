# Proposed Bill of Materials (BOM) — 220 V Lab DC Motor Health Monitoring System

## Design Basis & Safety Notes
- **Target Machine:** College Electrical Machines Lab Separately Excited / Shunt DC Motor (165–220 V DC, ~10–30 A).
- **Key Upgrade over Hobbyist IoT Prototypes:** Unlike 12 V hobby setups (which use non-isolated voltage dividers, low-current ACS712/721, and digital SW-420 vibration switches), a 220 V lab DC motor requires:
  1. **Galvanically isolated DC voltage and current transducers** (≥ 2 kV isolation).
  2. **True 3-axis MEMS accelerometer** capable of ≥ 500 Hz sampling for FFT and kurtosis/bearing analysis.
  3. **Battery-powered wireless DAQ (Raspberry Pi Pico W)** so the laptop is never physically wired to the high-voltage motor circuit.

---

## Core Data Acquisition (DAQ) BOM

| # | Category | Recommended Component / Spec | Qty | Purpose | Est. Cost (INR) |
| :-: | :--- | :--- | :-: | :--- | :---: |
| 1 | **Wireless MCU** | **Raspberry Pi Pico W** *(or ESP32-DevKitC)* | 1 | Sensor acquisition (SPI/I2C/GPIO) and Wi-Fi telemetry streaming to laptop | ₹650 |
| 2 | **Isolated DC Voltage Transducer** | **LEM LV 25-P** *or* **0–300 V DC Isolated Voltage Transmitter** (0–3.3 V / 0–5 V output, ≥ 2 kV isolation) | 1–2 | Safe isolated measurement of Armature Voltage $V_a$ (and optional Field Voltage $V_f$). *Note: Do NOT use ZMPT101B (AC-only).* | ₹2,000 – ₹4,500 |
| 3 | **Isolated DC Current Transducer** | **ACS758LCB-050B** (50 A Hall-effect, 4.8 kV isolation) *or* **LEM LA 55-P** | 1–2 | Isolated Armature Current $I_a$ with high bandwidth to capture brush-commutation ripple | ₹800 – ₹2,500 |
| 4 | **External ADC** | **MCP3008** (10-bit SPI, 8-ch) *or* **ADS1115** (16-bit I2C, 4-ch) | 1 | Clean multi-channel analog acquisition *(MCP3008 SPI recommended for 500 Hz+ simultaneous $V_a, I_a$ sampling)* | ₹350 |
| 5 | **3-Axis Vibration Sensor** | **ADXL345** (SPI, up to 3200 Hz ODR) *or* **MPU6050** | 1 | Captures $1\times$/$2\times$ RPM harmonics, electromagnetic chatter, and bearing impulses; mount rigidly on drive-end bearing housing | ₹300 |
| 6 | **Non-Contact Temp Sensor** | **MLX90614** (I2C IR thermometer) | 1 | Non-contact monitoring of motor casing / commutator surface temperature | ₹650 |
| 7 | **Contact Temp Sensor** | **DS18B20** (waterproof probe) *or* **MAX6675 + K-Type Thermocouple** | 1 | Direct contact temperature measurement on the bearing end-bell | ₹250 |
| 8 | **Shaft Speed (RPM) Sensor** | **TCRT5000 / FC-03 IR Optical Tachometer** + reflective tape *(or A3144 Hall sensor + magnet)* | 1 | Non-contact RPM measurement for speed-droop & synchronous harmonic tracking | ₹150 |
| 9 | **Isolated Power Supply** | **5 V USB Power Bank (10,000 mAh)** | 1 | Fully floats the Pico W DAQ from mains earth — prevents ground loops and protects electronics | ₹900 |
| 10 | **Enclosure, Wiring & Protection** | ABS IP65 enclosure, 600 V CAT III shrouded 4 mm banana leads, 1 A fast-blow inline fuse on voltage tap, magnetic/clamp mount for accelerometer, perfboard/headers | 1 set | Electrical safety, neat assembly, and repeatable mechanical mounting | ₹1,200 |

**Estimated Core DAQ Total:** **₹7,250 – ₹11,450**

---

## Optional Dedicated Benchtop Fault-Test Rig (For Destructive/Bearing Tests)

| # | Component | Spec | Qty | Purpose | Est. Cost (INR) |
| :-: | :--- | :--- | :-: | :--- | :---: |
| 11 | **Spare Low-Voltage DC Motor** | 24 V / 48 V Brushed DC Motor (100–250 W) | 1 | Safe testbed for bearing swaps, worn brushes, and unbalance without risking the 220 V lab motor | ₹1,800 |
| 12 | **Spare Test Bearings** | 6204 / 6205 Deep-Groove Ball Bearings (1 healthy + 1 pre-notched/grease-depleted) | 2 | Reproducible bearing fault vibration testing | ₹300 |
| 13 | **Armature Series Proxy Resistor** | 1 Ω, 100 W aluminum-housed wirewound resistor | 1 | Safe external proxy for dirty commutator / elevated brush contact resistance | ₹400 |

**Estimated Optional Test Rig Total:** **₹2,500**
