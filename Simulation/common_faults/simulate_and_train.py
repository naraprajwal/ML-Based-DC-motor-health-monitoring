"""
DC Motor Common Faults Simulation & Machine Learning Classification Pipeline
===========================================================================
Models a 220V separately excited laboratory DC motor across 7 operating states:
  0: Healthy
  1: Winding Overload
  2: Loose Windings
  3: Dirty Commutator / Brush Wear
  4: Bearing Fault (Bad Bearings -> Bearing Failure)
  5: Degraded Winding Insulation (Early Inter-Turn Short Proxy)
  6: Excessive Vibration (Imbalance / Misalignment / Loose Mount)

Generates time-series signals (Armature Voltage Va, Armature Current Ia,
Shaft Speed RPM, 3-axis/combined Vibration, Casing Temperature), extracts
time + frequency domain features per window, and trains:
  - Binary Health Classifier (Healthy vs Faulty)
  - Multi-Class Fault Classifier (7 classes)

Uses grouped train/test split by simulation run ID to prevent window leakage.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Reproducibility
RNG = np.random.default_rng(42)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Nominal 220V Lab DC Motor Parameters (Separately Excited)
NOMINAL = {
    "Ra": 1.20,          # Armature resistance (Ohm)
    "La": 0.025,         # Armature inductance (H)
    "Ke_nom": 1.25,      # Back-EMF / Torque constant at nominal field (V*s/rad)
    "J": 0.040,          # Rotor inertia (kg*m^2)
    "B": 0.008,          # Viscous friction (N*m*s/rad)
    "C_th": 220.0,       # Thermal capacitance (J/K) - scaled for short-window trend
    "R_th": 0.45,        # Thermal resistance to ambient (K/W)
    "T_amb": 27.0,       # Ambient temperature (deg C)
    "Vf_nom": 220.0,     # Nominal field voltage (V)
}

FAULT_CLASSES = [
    "Healthy",
    "Winding_Overload",
    "Loose_Windings",
    "Dirty_Commutator",
    "Bearing_Fault",
    "Insulation_Degradation",
    "Excessive_Vibration",
]


def simulate_motor_run(fault_name: str, rng: np.random.Generator, fs: float = 500.0, duration: float = 4.0):
    """
    Simulate one run of a separately excited DC motor using RK2/Euler integration
    with fault-specific electrical, thermal, and vibration physics.
    """
    n_steps = int(fs * duration)
    dt = 1.0 / fs
    t = np.arange(n_steps) * dt

    # Randomized normal operating point
    Va_base = rng.uniform(185.0, 220.0)
    Vf = rng.uniform(195.0, 220.0)
    TL_base = rng.uniform(8.0, 18.0)
    T_amb = NOMINAL["T_amb"] + rng.uniform(-2.0, 3.0)

    # Start from nominal parameters with small unit-to-unit manufacturing tolerance
    Ra = NOMINAL["Ra"] * rng.uniform(0.97, 1.03)
    La = NOMINAL["La"] * rng.uniform(0.97, 1.03)
    flux_ratio = (Vf / NOMINAL["Vf_nom"]) * rng.uniform(0.98, 1.02)
    K = NOMINAL["Ke_nom"] * flux_ratio
    J = NOMINAL["J"]
    B = NOMINAL["B"] * rng.uniform(0.95, 1.05)

    # Fault severity parameters
    sev = rng.uniform(0.35, 1.0) if fault_name != "Healthy" else 0.0

    # Fault-specific parameter modifications
    extra_copper_loss_factor = 1.0
    bearing_heat_W = 0.0
    brush_contact_R = 0.0
    spark_prob = 0.0
    imbalance_amp = rng.uniform(0.03, 0.07)   # baseline normal vibration (g)
    misalign_amp = rng.uniform(0.01, 0.03)
    bearing_imp_amp = 0.0
    loose_winding_vib_amp = 0.0
    torque_ripple_1x = 0.0
    winding_jitter_R = 0.0

    if fault_name == "Winding_Overload":
        # Sustained overload: 1.4x to 2.1x nominal load torque
        TL_base = TL_base * (1.40 + 0.70 * sev)
        extra_copper_loss_factor = 1.15 + 0.25 * sev

    elif fault_name == "Loose_Windings":
        # Loose armature/field coils: small intermittent resistance/inductance jitter
        # + electromagnetic force vibration proportional to Ia^2 at 2x supply/slot freq
        winding_jitter_R = (0.04 + 0.08 * sev) * Ra
        loose_winding_vib_amp = 0.22 + 0.45 * sev

    elif fault_name == "Dirty_Commutator":
        # Oxidized/dirty commutator & worn brushes:
        # higher contact resistance + periodic bar-pass ripple + arcing current spikes
        brush_contact_R = 0.25 + 0.75 * sev  # Ohm
        spark_prob = 0.015 + 0.045 * sev
        extra_copper_loss_factor = 1.05 + 0.15 * sev

    elif fault_name == "Bearing_Fault":
        # Bad bearings -> bearing failure:
        # increased friction + periodic BPFO impact train + bearing-endbell heating
        B = B * (1.8 + 2.8 * sev)
        bearing_imp_amp = 0.45 + 1.10 * sev
        bearing_heat_W = 25.0 + 75.0 * sev

    elif fault_name == "Insulation_Degradation":
        # Early inter-turn short / degraded winding insulation:
        # effective resistance & inductance drop, effective K drops slightly,
        # localized circulating current adds rapid heating + 2x electrical ripple
        short_frac = 0.08 + 0.18 * sev
        Ra = Ra * (1.0 - short_frac)
        La = La * (1.0 - 1.2 * short_frac)
        K = K * (1.0 - 0.65 * short_frac)
        extra_copper_loss_factor = 1.45 + 0.95 * sev  # localized hot-spot loss

    elif fault_name == "Excessive_Vibration":
        # Rotor unbalance / shaft misalignment / loose base bolts:
        # strong 1x and 2x RPM harmonics + small 1x torque ripple
        imbalance_amp = 0.40 + 0.90 * sev
        misalign_amp = 0.25 + 0.60 * sev
        torque_ripple_1x = 0.4 + 1.0 * sev

    # Steady-state initialization (motor already running at t=0 for realistic monitoring)
    Ra_eff0 = Ra + brush_contact_R
    # Solve steady state: K*Ia = TL + B*w, Va = Ra*Ia + K*w
    # => w = (K*Va - Ra*TL) / (K^2 + Ra*B)
    w = max(5.0, (K * Va_base - Ra_eff0 * TL_base) / (K**2 + Ra_eff0 * B))
    Ia = (TL_base + B * w) / K
    theta = rng.uniform(0.0, 2.0 * np.pi)
    # Initial temperature slightly warmed up depending on operating condition
    T_curr = T_amb + rng.uniform(6.0, 14.0) + (8.0 * sev if fault_name in ("Winding_Overload", "Insulation_Degradation") else 0.0)

    # Pre-allocate arrays
    Va_arr = np.empty(n_steps)
    Ia_arr = np.empty(n_steps)
    rpm_arr = np.empty(n_steps)
    temp_arr = np.empty(n_steps)
    vib_arr = np.empty(n_steps)

    # Supply ripple phase (e.g., 100 Hz rectifier ripple for 50 Hz mains)
    phase_ripple = rng.uniform(0, 2 * np.pi)
    n_comm_bars = 16  # typical lab DC motor commutator segments
    bpfo_mult = 3.58  # Ball Pass Frequency Outer race multiplier (typical 6205 bearing)

    # Sub-stepping for numerical stability of electrical equation (La/Ra ~ 0.02s)
    sub_steps = 4
    dt_sub = dt / sub_steps

    for i in range(n_steps):
        ti = t[i]
        # Supply voltage with realistic 100 Hz rectifier ripple (~1.2% peak)
        Va_inst = Va_base * (1.0 + 0.012 * np.sin(2.0 * np.pi * 100.0 * ti + phase_ripple))

        # Copper temperature coefficient (0.00393 / K)
        Ra_temp = Ra * (1.0 + 0.00393 * (T_curr - 25.0))

        # Commutator / brush dynamic resistance & arcing
        comm_mod = 0.0
        if fault_name == "Dirty_Commutator":
            comm_mod = brush_contact_R * (1.0 + 0.35 * np.sin(n_comm_bars * theta))
            if rng.random() < spark_prob:
                comm_mod += rng.uniform(0.4, 1.4) * sev

        # Loose winding micro-jitter in resistance
        jitter_R = rng.normal(0.0, winding_jitter_R) if winding_jitter_R > 0 else 0.0
        Ra_total = max(0.2, Ra_temp + comm_mod + jitter_R)

        # Load torque with optional 1x misalignment ripple
        TL_inst = TL_base + torque_ripple_1x * np.sin(theta)

        # Integrate electromechanical ODEs
        for _ in range(sub_steps):
            Eb = K * w
            dIa = (Va_inst - Ra_total * Ia - Eb) / La
            Te = K * Ia
            dw = (Te - TL_inst - B * w) / J
            Ia = Ia + dIa * dt_sub
            w = max(0.1, w + dw * dt_sub)
            theta = theta + w * dt_sub

        # Thermal dynamics (1st-order lumped model)
        P_loss = extra_copper_loss_factor * (Ia**2) * Ra_total + B * (w**2) + bearing_heat_W
        dT = (P_loss - (T_curr - T_amb) / NOMINAL["R_th"]) / NOMINAL["C_th"]
        T_curr = T_curr + dT * dt

        # Shaft frequency in Hz
        f_shaft = w / (2.0 * np.pi)

        # Synthesize physically grounded vibration signal (in g)
        # 1. Baseline + 1x imbalance + 2x misalignment
        vib = (
            imbalance_amp * np.sin(theta)
            + misalign_amp * np.sin(2.0 * theta + 0.5)
            + rng.normal(0.0, 0.035)
        )

        # 2. Bearing fault: sharp periodic impulses at BPFO * f_shaft (high kurtosis!)
        if bearing_imp_amp > 0:
            # Periodic impulse train modeled via narrow raised-cosine / Gaussian pulse
            bpfo_phase = (bpfo_mult * theta) % (2.0 * np.pi)
            pulse = np.exp(-0.5 * ((bpfo_phase - np.pi) / 0.18) ** 2)
            # Modulated by high-frequency structural resonance (~165 Hz)
            vib += bearing_imp_amp * pulse * np.sin(2.0 * np.pi * 165.0 * ti) * rng.uniform(0.8, 1.2)

        # 3. Loose windings: electromagnetic force chatter (~100 Hz & 200 Hz, scales with Ia^2)
        if loose_winding_vib_amp > 0:
            em_scale = np.clip((Ia / 15.0) ** 2, 0.3, 2.5)
            vib += loose_winding_vib_amp * em_scale * (
                0.65 * np.sin(2.0 * np.pi * 100.0 * ti + 0.2)
                + 0.35 * np.sin(2.0 * np.pi * 200.0 * ti + 1.1)
            ) * rng.uniform(0.85, 1.15)

        # 4. Dirty commutator: mild high-freq brush chatter
        if fault_name == "Dirty_Commutator":
            vib += (0.06 + 0.12 * sev) * np.sin(n_comm_bars * theta)

        # 5. Insulation degradation: mild 100 Hz electromagnetic asymmetry hum
        if fault_name == "Insulation_Degradation":
            vib += (0.05 + 0.10 * sev) * np.sin(2.0 * np.pi * 100.0 * ti)

        # Add realistic sensor measurement noise
        Va_arr[i] = Va_inst + rng.normal(0.0, 0.35)
        Ia_arr[i] = Ia + rng.normal(0.0, 0.08)
        rpm_arr[i] = (w * 60.0 / (2.0 * np.pi)) + rng.normal(0.0, 1.5)
        temp_arr[i] = T_curr + rng.normal(0.0, 0.05)
        vib_arr[i] = vib

    return {
        "t": t,
        "Va": Va_arr,
        "Ia": Ia_arr,
        "rpm": rpm_arr,
        "temp": temp_arr,
        "vib": vib_arr,
        "fs": fs,
        "severity": sev,
    }


def extract_window_features(sig: dict, start_idx: int, end_idx: int) -> dict:
    """
    Extract time-domain, physics-derived, and frequency-domain (FFT) features
    from a signal window.
    """
    fs = sig["fs"]
    Va = sig["Va"][start_idx:end_idx]
    Ia = sig["Ia"][start_idx:end_idx]
    rpm = sig["rpm"][start_idx:end_idx]
    temp = sig["temp"][start_idx:end_idx]
    vib = sig["vib"][start_idx:end_idx]
    n = len(Va)
    t_win = np.arange(n) / fs

    feats = {}

    # 1. Electrical & Mechanical Time-Domain Stats
    for name, arr in [("Va", Va), ("Ia", Ia), ("rpm", rpm), ("temp", temp), ("vib", vib)]:
        feats[f"{name}_mean"] = float(np.mean(arr))
        feats[f"{name}_std"] = float(np.std(arr))
        feats[f"{name}_rms"] = float(np.sqrt(np.mean(arr**2)))
        feats[f"{name}_ptp"] = float(np.ptp(arr))

    # Temperature slope (deg C / s) — critical for Overload & Insulation Breakdown
    feats["temp_slope"] = float(np.polyfit(t_win, temp, 1)[0])

    # Physics-informed ratios (help normalize across varying operating points!)
    mean_Ia = max(0.1, feats["Ia_mean"])
    mean_rpm = max(1.0, feats["rpm_mean"])
    feats["power_in_W"] = float(np.mean(Va * Ia))
    feats["Va_over_rpm"] = float(feats["Va_mean"] / mean_rpm)
    feats["Ia_over_Va"] = float(mean_Ia / max(1.0, feats["Va_mean"]))
    feats["Ia_diff_rms"] = float(np.sqrt(np.mean(np.diff(Ia) ** 2)))  # catches brush sparking!
    feats["Ia_kurtosis"] = float(kurtosis(Ia))

    # Vibration higher-order stats (critical for Bearing Impulses!)
    vib_rms = max(1e-6, feats["vib_rms"])
    feats["vib_kurtosis"] = float(kurtosis(vib))
    feats["vib_skew"] = float(skew(vib))
    feats["vib_crest_factor"] = float(np.max(np.abs(vib)) / vib_rms)

    # 2. Frequency-Domain (FFT) Features for Vibration & Current
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    vib_fft = np.abs(np.fft.rfft(vib - np.mean(vib))) / n
    Ia_fft = np.abs(np.fft.rfft(Ia - np.mean(Ia))) / n

    f_shaft = mean_rpm / 60.0  # fundamental shaft frequency (Hz), typically ~20-30 Hz

    def band_energy(spec, f_low, f_high):
        mask = (freqs >= f_low) & (freqs <= f_high)
        if not np.any(mask):
            return 0.0
        return float(np.sqrt(np.sum(spec[mask] ** 2)))

    # Vibration spectral bands
    feats["vib_fft_1x_rpm"] = band_energy(vib_fft, max(1.0, f_shaft - 3.0), f_shaft + 3.0)
    feats["vib_fft_2x_rpm"] = band_energy(vib_fft, max(1.0, 2 * f_shaft - 3.5), 2 * f_shaft + 3.5)
    feats["vib_fft_100hz"] = band_energy(vib_fft, 95.0, 105.0)
    feats["vib_fft_200hz"] = band_energy(vib_fft, 195.0, 205.0)
    feats["vib_fft_bearing_hf"] = band_energy(vib_fft, 130.0, 190.0)  # bearing resonance band
    feats["vib_fft_total"] = float(np.sqrt(np.sum(vib_fft**2)))

    # Spectral ratios (normalize energy distribution)
    tot_v = max(1e-6, feats["vib_fft_total"])
    feats["vib_ratio_1x2x"] = (feats["vib_fft_1x_rpm"] + feats["vib_fft_2x_rpm"]) / tot_v
    feats["vib_ratio_100_200"] = (feats["vib_fft_100hz"] + feats["vib_fft_200hz"]) / tot_v
    feats["vib_ratio_bearing"] = feats["vib_fft_bearing_hf"] / tot_v

    # Current spectral bands (catches commutator bar-pass & sparking HF)
    feats["Ia_fft_100hz"] = band_energy(Ia_fft, 95.0, 105.0)
    feats["Ia_fft_hf"] = band_energy(Ia_fft, 120.0, 240.0)

    return feats


def generate_dataset(runs_per_class: int = 120):
    rows = []
    sample_signals = {}
    run_id = 0

    for class_idx, fault_name in enumerate(FAULT_CLASSES):
        for r in range(runs_per_class):
            sig = simulate_motor_run(fault_name, RNG, fs=500.0, duration=4.0)
            if r == 0:
                sample_signals[fault_name] = sig

            # Split 4-second run into two 2-second windows (1000 samples each)
            for w_idx, (s, e) in enumerate([(0, 1000), (1000, 2000)]):
                feats = extract_window_features(sig, s, e)
                feats["run_id"] = run_id
                feats["window_idx"] = w_idx
                feats["fault_class"] = fault_name
                feats["fault_label"] = class_idx
                feats["is_faulty"] = 0 if fault_name == "Healthy" else 1
                feats["severity"] = sig["severity"]
                rows.append(feats)
            run_id += 1

    df = pd.DataFrame(rows)
    return df, sample_signals


def plot_sample_signals(sample_signals: dict):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    selected = ["Healthy", "Winding_Overload", "Dirty_Commutator", "Bearing_Fault", "Insulation_Degradation", "Excessive_Vibration", "Loose_Windings"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))

    # 1. Armature Current (first 0.5s zoomed)
    for name, col in zip(selected, colors):
        sig = sample_signals[name]
        mask = sig["t"] <= 0.5
        axes[0].plot(sig["t"][mask], sig["Ia"][mask], label=name, color=col, alpha=0.85, linewidth=1.1)
    axes[0].set_title("Simulated Armature Current Ia (0 - 0.5 s zoom)")
    axes[0].set_ylabel("Current (A)")
    axes[0].set_xlabel("Time (s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=4, fontsize=8, loc="upper right")

    # 2. Vibration FFT Spectrum (0 - 250 Hz)
    for name, col in zip(selected, colors):
        sig = sample_signals[name]
        vib = sig["vib"][:1000]
        freqs = np.fft.rfftfreq(len(vib), d=1.0 / sig["fs"])
        spec = np.abs(np.fft.rfft(vib - np.mean(vib))) / len(vib)
        axes[1].plot(freqs, spec, label=name, color=col, alpha=0.85, linewidth=1.1)
    axes[1].set_title("Vibration Frequency Spectrum (FFT) Across Fault Modes")
    axes[1].set_ylabel("Amplitude (g)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_xlim(0, 240)
    axes[1].grid(True, alpha=0.3)

    # 3. Temperature Rise (4s window trend)
    for name, col in zip(selected, colors):
        sig = sample_signals[name]
        axes[2].plot(sig["t"], sig["temp"] - sig["temp"][0], label=name, color=col, alpha=0.9, linewidth=1.3)
    axes[2].set_title("Relative Casing Temperature Rise (Delta T over 4 s)")
    axes[2].set_ylabel("Delta Temp (deg C)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "simulated_signals_comparison.png"), dpi=160)
    plt.close()


def plot_confusion_matrix(cm, labels, title, filename):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True Fault Class",
        xlabel="Predicted Fault Class",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=160)
    plt.close()


def plot_feature_importance(model, feature_cols, filename):
    importances = model.feature_importances_
    idx = np.argsort(importances)[-15:]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(idx)), importances[idx], color="#2b6cb0")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_cols[i] for i in idx])
    ax.set_xlabel("Feature Importance (Extra Trees)")
    ax.set_title("Top 15 Most Discriminative Features for DC Motor Fault Diagnosis")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=160)
    plt.close()


def main():
    print("Generating physics-based DC motor fault dataset (7 classes)...")
    df, sample_signals = generate_dataset(runs_per_class=140)
    df.to_csv(os.path.join(OUT_DIR, "dataset_common_faults.csv"), index=False)

    meta_cols = ["run_id", "window_idx", "fault_class", "fault_label", "is_faulty", "severity"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].values
    y_bin = df["is_faulty"].values
    y_multi = df["fault_label"].values
    groups = df["run_id"].values

    # Grouped 75/25 split by run_id (no leakage between windows of the same run)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_multi, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    yb_train, yb_test = y_bin[train_idx], y_bin[test_idx]
    ym_train, ym_test = y_multi[train_idx], y_multi[test_idx]

    # 1. Binary Health Classifier
    bin_clf = ExtraTreesClassifier(n_estimators=250, random_state=42, class_weight="balanced", n_jobs=-1)
    bin_clf.fit(X_train, yb_train)
    yb_pred = bin_clf.predict(X_test)

    bin_acc = accuracy_score(yb_test, yb_pred)
    bin_bal_acc = balanced_accuracy_score(yb_test, yb_pred)
    bin_f1 = f1_score(yb_test, yb_pred, average="macro")

    # 2. Multi-Class Fault Classifier
    multi_clf = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    multi_clf.fit(X_train, ym_train)
    ym_pred = multi_clf.predict(X_test)

    multi_acc = accuracy_score(ym_test, ym_pred)
    multi_bal_acc = balanced_accuracy_score(ym_test, ym_pred)
    multi_f1 = f1_score(ym_test, ym_pred, average="macro")

    report_bin = classification_report(yb_test, yb_pred, target_names=["Healthy", "Faulty"], digits=4)
    report_multi = classification_report(ym_test, ym_pred, target_names=FAULT_CLASSES, digits=4)

    summary = []
    summary.append("=== DATASET SUMMARY ===")
    summary.append(f"Total simulation runs : {df['run_id'].nunique()}")
    summary.append(f"Total windows         : {len(df)} (2s windows @ 500 Hz)")
    summary.append(f"Extracted features    : {len(feature_cols)}")
    summary.append(f"Train / Test windows  : {len(train_idx)} / {len(test_idx)} (grouped by run_id)")
    summary.append("")
    summary.append("=== BINARY HEALTH CLASSIFICATION (Extra Trees) ===")
    summary.append(f"Accuracy          : {bin_acc*100:.2f}%")
    summary.append(f"Balanced Accuracy : {bin_bal_acc*100:.2f}%")
    summary.append(f"Macro F1-Score    : {bin_f1:.4f}")
    summary.append(report_bin)
    summary.append("=== 7-CLASS FAULT IDENTIFICATION (Extra Trees) ===")
    summary.append(f"Accuracy          : {multi_acc*100:.2f}%")
    summary.append(f"Balanced Accuracy : {multi_bal_acc*100:.2f}%")
    summary.append(f"Macro F1-Score    : {multi_f1:.4f}")
    summary.append(report_multi)

    summary_text = "\n".join(summary)
    print(summary_text)
    with open(os.path.join(OUT_DIR, "metrics_summary.txt"), "w") as f:
        f.write(summary_text)

    # Generate plots
    plot_sample_signals(sample_signals)
    cm_multi = confusion_matrix(ym_test, ym_pred)
    plot_confusion_matrix(
        cm_multi,
        FAULT_CLASSES,
        f"7-Class DC Motor Fault Confusion Matrix (Acc = {multi_acc*100:.2f}%)",
        "confusion_matrix_multiclass.png",
    )
    plot_feature_importance(multi_clf, feature_cols, "feature_importance.png")
    print(f"All outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
