"""
DC Motor Randomized Multi-Fault Simulation & Multi-Label ML Pipeline
====================================================================
Simulates a 220V separately excited laboratory DC motor under:
  - Healthy operation (~30% of runs)
  - Single-fault operation (~35% of runs)
  - Simultaneous 2-fault combinations (~25% of runs)
  - Simultaneous 3-fault combinations (~10% of runs)

The 6 common DC motor faults modeled simultaneously are:
  1. Winding_Overload
  2. Loose_Windings
  3. Dirty_Commutator (and brush wear/sparking)
  4. Bearing_Fault (bad bearings -> bearing failure)
  5. Insulation_Degradation (early inter-turn short proxy)
  6. Excessive_Vibration (unbalance / misalignment / loose mount)

Trains & evaluates (using grouped 75/25 split by run_id):
  - Binary Health Classifier (Extra Trees)
  - Multi-Label Fault Identifier (MultiOutputClassifier with Extra Trees)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
)

RNG = np.random.default_rng(42)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

NOMINAL = {
    "Ra": 1.20,          # Armature resistance (Ohm)
    "La": 0.025,         # Armature inductance (H)
    "Ke_nom": 1.25,      # Back-EMF / Torque constant at nominal field (V*s/rad)
    "J": 0.040,          # Rotor inertia (kg*m^2)
    "B": 0.008,          # Viscous friction (N*m*s/rad)
    "C_th": 220.0,       # Thermal capacitance (J/K)
    "R_th": 0.45,        # Thermal resistance to ambient (K/W)
    "T_amb": 27.0,       # Ambient temperature (deg C)
    "Vf_nom": 220.0,     # Nominal field voltage (V)
}

FAULT_NAMES = [
    "Winding_Overload",
    "Loose_Windings",
    "Dirty_Commutator",
    "Bearing_Fault",
    "Insulation_Degradation",
    "Excessive_Vibration",
]


def simulate_multifault_run(active_faults: dict, rng: np.random.Generator, fs: float = 500.0, duration: float = 4.0):
    """
    Simulate one run of a separately excited DC motor with an arbitrary combination
    of simultaneous faults specified in `active_faults` = {fault_name: severity in [0.25, 1.0]}.
    """
    n_steps = int(fs * duration)
    dt = 1.0 / fs
    t = np.arange(n_steps) * dt

    # Randomized normal operating point
    Va_base = rng.uniform(180.0, 220.0)
    Vf = rng.uniform(190.0, 220.0)
    TL_base = rng.uniform(8.0, 18.0)
    T_amb = NOMINAL["T_amb"] + rng.uniform(-2.0, 3.0)

    # Unit-to-unit baseline tolerance
    Ra = NOMINAL["Ra"] * rng.uniform(0.96, 1.04)
    La = NOMINAL["La"] * rng.uniform(0.96, 1.04)
    flux_ratio = (Vf / NOMINAL["Vf_nom"]) * rng.uniform(0.98, 1.02)
    K = NOMINAL["Ke_nom"] * flux_ratio
    J = NOMINAL["J"]
    B = NOMINAL["B"] * rng.uniform(0.94, 1.06)

    # Accumulators for simultaneous fault effects
    extra_copper_loss_factor = 1.0
    bearing_heat_W = 0.0
    brush_contact_R = 0.0
    spark_prob = 0.0
    imbalance_amp = rng.uniform(0.03, 0.07)
    misalign_amp = rng.uniform(0.01, 0.03)
    bearing_imp_amp = 0.0
    loose_winding_vib_amp = 0.0
    torque_ripple_1x = 0.0
    winding_jitter_R = 0.0
    comm_vib_amp = 0.0
    insul_vib_amp = 0.0
    initial_temp_boost = 0.0

    # Apply all active faults simultaneously
    if "Winding_Overload" in active_faults:
        sev = active_faults["Winding_Overload"]
        TL_base = TL_base * (1.35 + 0.70 * sev)
        extra_copper_loss_factor += 0.15 + 0.25 * sev
        initial_temp_boost += 6.0 * sev

    if "Loose_Windings" in active_faults:
        sev = active_faults["Loose_Windings"]
        winding_jitter_R += (0.04 + 0.08 * sev) * Ra
        loose_winding_vib_amp += 0.20 + 0.45 * sev

    if "Dirty_Commutator" in active_faults:
        sev = active_faults["Dirty_Commutator"]
        brush_contact_R += 0.22 + 0.75 * sev
        spark_prob += 0.015 + 0.045 * sev
        extra_copper_loss_factor += 0.05 + 0.15 * sev
        comm_vib_amp += 0.05 + 0.12 * sev

    if "Bearing_Fault" in active_faults:
        sev = active_faults["Bearing_Fault"]
        B = B * (1.7 + 2.8 * sev)
        bearing_imp_amp += 0.40 + 1.10 * sev
        bearing_heat_W += 25.0 + 75.0 * sev
        initial_temp_boost += 3.0 * sev

    if "Insulation_Degradation" in active_faults:
        sev = active_faults["Insulation_Degradation"]
        short_frac = 0.08 + 0.18 * sev
        Ra = Ra * (1.0 - short_frac)
        La = La * (1.0 - 1.2 * short_frac)
        K = K * (1.0 - 0.65 * short_frac)
        extra_copper_loss_factor += 0.45 + 0.90 * sev
        insul_vib_amp += 0.05 + 0.10 * sev
        initial_temp_boost += 7.0 * sev

    if "Excessive_Vibration" in active_faults:
        sev = active_faults["Excessive_Vibration"]
        imbalance_amp += 0.35 + 0.85 * sev
        misalign_amp += 0.22 + 0.55 * sev
        torque_ripple_1x += 0.4 + 0.9 * sev

    # Steady-state initialization
    Ra_eff0 = Ra + brush_contact_R
    w = max(5.0, (K * Va_base - Ra_eff0 * TL_base) / (K**2 + Ra_eff0 * B))
    Ia = (TL_base + B * w) / K
    theta = rng.uniform(0.0, 2.0 * np.pi)
    T_curr = T_amb + rng.uniform(6.0, 14.0) + initial_temp_boost

    Va_arr = np.empty(n_steps)
    Ia_arr = np.empty(n_steps)
    rpm_arr = np.empty(n_steps)
    temp_arr = np.empty(n_steps)
    vib_arr = np.empty(n_steps)

    phase_ripple = rng.uniform(0, 2 * np.pi)
    n_comm_bars = 16
    bpfo_mult = 3.58
    sub_steps = 4
    dt_sub = dt / sub_steps

    for i in range(n_steps):
        ti = t[i]
        Va_inst = Va_base * (1.0 + 0.012 * np.sin(2.0 * np.pi * 100.0 * ti + phase_ripple))
        Ra_temp = Ra * (1.0 + 0.00393 * (T_curr - 25.0))

        comm_mod = 0.0
        if brush_contact_R > 0.0:
            comm_mod = brush_contact_R * (1.0 + 0.35 * np.sin(n_comm_bars * theta))
            if rng.random() < spark_prob:
                comm_mod += rng.uniform(0.35, 1.35) * active_faults["Dirty_Commutator"]

        jitter_R = rng.normal(0.0, winding_jitter_R) if winding_jitter_R > 0.0 else 0.0
        Ra_total = max(0.2, Ra_temp + comm_mod + jitter_R)
        TL_inst = TL_base + torque_ripple_1x * np.sin(theta)

        for _ in range(sub_steps):
            Eb = K * w
            dIa = (Va_inst - Ra_total * Ia - Eb) / La
            Te = K * Ia
            dw = (Te - TL_inst - B * w) / J
            Ia = Ia + dIa * dt_sub
            w = max(0.1, w + dw * dt_sub)
            theta = theta + w * dt_sub

        P_loss = extra_copper_loss_factor * (Ia**2) * Ra_total + B * (w**2) + bearing_heat_W
        dT = (P_loss - (T_curr - T_amb) / NOMINAL["R_th"]) / NOMINAL["C_th"]
        T_curr = T_curr + dT * dt

        # Vibration synthesis (superposition of all active mechanical & EM sources)
        vib = (
            imbalance_amp * np.sin(theta)
            + misalign_amp * np.sin(2.0 * theta + 0.5)
            + rng.normal(0.0, 0.04)
        )

        if bearing_imp_amp > 0.0:
            bpfo_phase = (bpfo_mult * theta) % (2.0 * np.pi)
            pulse = np.exp(-0.5 * ((bpfo_phase - np.pi) / 0.18) ** 2)
            vib += bearing_imp_amp * pulse * np.sin(2.0 * np.pi * 165.0 * ti) * rng.uniform(0.8, 1.2)

        if loose_winding_vib_amp > 0.0:
            em_scale = np.clip((Ia / 15.0) ** 2, 0.3, 2.5)
            vib += loose_winding_vib_amp * em_scale * (
                0.65 * np.sin(2.0 * np.pi * 100.0 * ti + 0.2)
                + 0.35 * np.sin(2.0 * np.pi * 200.0 * ti + 1.1)
            ) * rng.uniform(0.85, 1.15)

        if comm_vib_amp > 0.0:
            vib += comm_vib_amp * np.sin(n_comm_bars * theta)

        if insul_vib_amp > 0.0:
            vib += insul_vib_amp * np.sin(2.0 * np.pi * 100.0 * ti)

        # Sensor measurements with noise
        Va_arr[i] = Va_inst + rng.normal(0.0, 0.40)
        Ia_arr[i] = Ia + rng.normal(0.0, 0.10)
        rpm_arr[i] = (w * 60.0 / (2.0 * np.pi)) + rng.normal(0.0, 1.8)
        temp_arr[i] = T_curr + rng.normal(0.0, 0.06)
        vib_arr[i] = vib

    return {
        "t": t,
        "Va": Va_arr,
        "Ia": Ia_arr,
        "rpm": rpm_arr,
        "temp": temp_arr,
        "vib": vib_arr,
        "fs": fs,
    }


def extract_window_features(sig: dict, start_idx: int, end_idx: int) -> dict:
    fs = sig["fs"]
    Va = sig["Va"][start_idx:end_idx]
    Ia = sig["Ia"][start_idx:end_idx]
    rpm = sig["rpm"][start_idx:end_idx]
    temp = sig["temp"][start_idx:end_idx]
    vib = sig["vib"][start_idx:end_idx]
    n = len(Va)
    t_win = np.arange(n) / fs

    feats = {}
    for name, arr in [("Va", Va), ("Ia", Ia), ("rpm", rpm), ("temp", temp), ("vib", vib)]:
        feats[f"{name}_mean"] = float(np.mean(arr))
        feats[f"{name}_std"] = float(np.std(arr))
        feats[f"{name}_rms"] = float(np.sqrt(np.mean(arr**2)))
        feats[f"{name}_ptp"] = float(np.ptp(arr))

    feats["temp_slope"] = float(np.polyfit(t_win, temp, 1)[0])

    mean_Ia = max(0.1, feats["Ia_mean"])
    mean_rpm = max(1.0, feats["rpm_mean"])
    feats["power_in_W"] = float(np.mean(Va * Ia))
    feats["Va_over_rpm"] = float(feats["Va_mean"] / mean_rpm)
    feats["Ia_over_Va"] = float(mean_Ia / max(1.0, feats["Va_mean"]))
    feats["Ia_diff_rms"] = float(np.sqrt(np.mean(np.diff(Ia) ** 2)))
    feats["Ia_kurtosis"] = float(kurtosis(Ia))

    vib_rms = max(1e-6, feats["vib_rms"])
    feats["vib_kurtosis"] = float(kurtosis(vib))
    feats["vib_skew"] = float(skew(vib))
    feats["vib_crest_factor"] = float(np.max(np.abs(vib)) / vib_rms)

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    vib_fft = np.abs(np.fft.rfft(vib - np.mean(vib))) / n
    Ia_fft = np.abs(np.fft.rfft(Ia - np.mean(Ia))) / n
    f_shaft = mean_rpm / 60.0

    def band_energy(spec, f_low, f_high):
        mask = (freqs >= f_low) & (freqs <= f_high)
        return float(np.sqrt(np.sum(spec[mask] ** 2))) if np.any(mask) else 0.0

    feats["vib_fft_1x_rpm"] = band_energy(vib_fft, max(1.0, f_shaft - 3.0), f_shaft + 3.0)
    feats["vib_fft_2x_rpm"] = band_energy(vib_fft, max(1.0, 2 * f_shaft - 3.5), 2 * f_shaft + 3.5)
    feats["vib_fft_100hz"] = band_energy(vib_fft, 95.0, 105.0)
    feats["vib_fft_200hz"] = band_energy(vib_fft, 195.0, 205.0)
    feats["vib_fft_bearing_hf"] = band_energy(vib_fft, 130.0, 190.0)
    feats["vib_fft_total"] = float(np.sqrt(np.sum(vib_fft**2)))

    tot_v = max(1e-6, feats["vib_fft_total"])
    feats["vib_ratio_1x2x"] = (feats["vib_fft_1x_rpm"] + feats["vib_fft_2x_rpm"]) / tot_v
    feats["vib_ratio_100_200"] = (feats["vib_fft_100hz"] + feats["vib_fft_200hz"]) / tot_v
    feats["vib_ratio_bearing"] = feats["vib_fft_bearing_hf"] / tot_v

    feats["Ia_fft_100hz"] = band_energy(Ia_fft, 95.0, 105.0)
    feats["Ia_fft_hf"] = band_energy(Ia_fft, 120.0, 240.0)

    return feats


def generate_multifault_dataset(n_runs: int = 1200):
    rows = []
    examples = {}

    # Distribution: 30% Healthy (0), 35% Single (1), 25% Double (2), 10% Triple (3)
    fault_counts = RNG.choice([0, 1, 2, 3], size=n_runs, p=[0.30, 0.35, 0.25, 0.10])

    for run_id, k_faults in enumerate(fault_counts):
        active_faults = {}
        if k_faults > 0:
            chosen = RNG.choice(FAULT_NAMES, size=k_faults, replace=False)
            for fname in chosen:
                active_faults[fname] = float(RNG.uniform(0.25, 1.0))

        sig = simulate_multifault_run(active_faults, RNG, fs=500.0, duration=4.0)

        # Save illustrative examples for plotting
        if k_faults == 0 and "Healthy" not in examples:
            examples["Healthy"] = sig
        elif k_faults == 1 and "Single Fault" not in examples:
            f0 = list(active_faults.keys())[0]
            examples[f"Single ({f0})"] = sig
        elif k_faults == 2 and "2-Fault Combo" not in examples:
            label = " + ".join(active_faults.keys())
            examples[f"2-Fault ({label})"] = sig
        elif k_faults == 3 and "3-Fault Combo" not in examples:
            label = " + ".join(active_faults.keys())
            examples[f"3-Fault ({label})"] = sig

        for w_idx, (s, e) in enumerate([(0, 1000), (1000, 2000)]):
            feats = extract_window_features(sig, s, e)
            feats["run_id"] = run_id
            feats["window_idx"] = w_idx
            feats["num_faults"] = int(k_faults)
            feats["is_faulty"] = 0 if k_faults == 0 else 1
            for fname in FAULT_NAMES:
                feats[f"label_{fname}"] = 1 if fname in active_faults else 0
            rows.append(feats)

    return pd.DataFrame(rows), examples


def plot_binary_cm(cm, acc, filename):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    labels = ["Healthy", "Abnormal"]
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=labels, yticklabels=labels,
        title=f"Multi-Fault Binary Health Classification\n(Acc = {acc*100:.2f}%)",
        ylabel="Actual", xlabel="Predicted",
    )
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=160)
    plt.close()


def plot_multilabel_bars(metrics_rows, filename):
    names = [r["Fault"] for r in metrics_rows]
    prec = [r["Precision"] for r in metrics_rows]
    rec = [r["Recall"] for r in metrics_rows]
    f1s = [r["F1"] for r in metrics_rows]

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(x - w, prec, width=w, label="Precision", color="#4299e1")
    ax.bar(x, rec, width=w, label="Recall", color="#48bb78")
    ax.bar(x + w, f1s, width=w, label="F1-Score", color="#2b6cb0")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Multi-Label Fault Identification Performance (Simultaneous Fault Combinations)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    for i, v in enumerate(f1s):
        ax.text(i + w, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=160)
    plt.close()


def plot_example_signals(examples, filename):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5))
    colors = ["#2b6cb0", "#38a169", "#dd6b20", "#e53e3e"]

    for (label, sig), col in zip(examples.items(), colors):
        mask = sig["t"] <= 0.5
        axes[0].plot(sig["t"][mask], sig["Ia"][mask], label=label, color=col, alpha=0.88, linewidth=1.1)

        vib = sig["vib"][:1000]
        freqs = np.fft.rfftfreq(len(vib), d=1.0 / sig["fs"])
        spec = np.abs(np.fft.rfft(vib - np.mean(vib))) / len(vib)
        axes[1].plot(freqs, spec, label=label, color=col, alpha=0.88, linewidth=1.1)

        axes[2].plot(sig["t"], sig["temp"] - sig["temp"][0], label=label, color=col, alpha=0.9, linewidth=1.3)

    axes[0].set_title("Armature Current Ia (0 - 0.5 s zoom): Healthy vs Simultaneous Multi-Fault Runs")
    axes[0].set_ylabel("Current (A)")
    axes[0].set_xlabel("Time (s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Vibration FFT Spectrum (0 - 240 Hz)")
    axes[1].set_ylabel("Amplitude (g)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_xlim(0, 240)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Relative Casing Temperature Rise (Delta T over 4 s)")
    axes[2].set_ylabel("Delta Temp (deg C)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=160)
    plt.close()


def main():
    print("Generating randomized multi-fault DC motor dataset (1200 runs / 2400 windows)...")
    df, examples = generate_multifault_dataset(n_runs=1200)
    df.to_csv(os.path.join(OUT_DIR, "dataset_multifault.csv"), index=False)

    label_cols = [f"label_{f}" for f in FAULT_NAMES]
    meta_cols = ["run_id", "window_idx", "num_faults", "is_faulty"] + label_cols
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].values
    y_bin = df["is_faulty"].values
    Y_multi = df[label_cols].values
    groups = df["run_id"].values

    # Grouped 75/25 split by run_id
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_bin, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    yb_train, yb_test = y_bin[train_idx], y_bin[test_idx]
    Ym_train, Ym_test = Y_multi[train_idx], Y_multi[test_idx]

    # 1. Binary Health Classifier
    bin_clf = ExtraTreesClassifier(n_estimators=250, random_state=42, class_weight="balanced", n_jobs=-1)
    bin_clf.fit(X_train, yb_train)
    yb_pred = bin_clf.predict(X_test)

    bin_acc = accuracy_score(yb_test, yb_pred)
    bin_bal_acc = balanced_accuracy_score(yb_test, yb_pred)
    bin_f1 = f1_score(yb_test, yb_pred, average="macro")

    # 2. Multi-Label Fault Classifier (MultiOutput Extra Trees)
    base_et = ExtraTreesClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
    multi_clf = MultiOutputClassifier(base_et)
    multi_clf.fit(X_train, Ym_train)
    Ym_pred = multi_clf.predict(X_test)

    exact_match = accuracy_score(Ym_test, Ym_pred)  # subset accuracy (all 6 labels simultaneously exact)
    h_loss = hamming_loss(Ym_test, Ym_pred)
    macro_f1 = f1_score(Ym_test, Ym_pred, average="macro")

    metrics_rows = []
    for idx, fname in enumerate(FAULT_NAMES):
        yt = Ym_test[:, idx]
        yp = Ym_pred[:, idx]
        metrics_rows.append({
            "Fault": fname,
            "Precision": precision_score(yt, yp),
            "Recall": recall_score(yt, yp),
            "F1": f1_score(yt, yp),
            "Accuracy": accuracy_score(yt, yp),
            "Support": int(np.sum(yt)),
        })

    # Breakdown by number of simultaneous faults in test set
    test_n_faults = df.iloc[test_idx]["num_faults"].values
    exact_by_k = {}
    for k in [0, 1, 2, 3]:
        m = test_n_faults == k
        if np.any(m):
            exact_by_k[k] = (int(np.sum(m)), accuracy_score(Ym_test[m], Ym_pred[m]))

    lines = []
    lines.append("=== RANDOMIZED MULTI-FAULT DATASET SUMMARY ===")
    lines.append(f"Total simulation runs : {df['run_id'].nunique()}")
    lines.append(f"Total windows         : {len(df)} (2s windows @ 500 Hz)")
    lines.append(f"Extracted features    : {len(feature_cols)}")
    lines.append(f"Train / Test windows  : {len(train_idx)} / {len(test_idx)} (grouped by run_id)")
    counts = df.groupby("num_faults")["run_id"].nunique().to_dict()
    lines.append(f"Runs by fault count   : {counts}")
    lines.append("")
    lines.append("=== BINARY HEALTH CLASSIFICATION (Healthy vs Abnormal) ===")
    lines.append(f"Accuracy          : {bin_acc*100:.2f}%")
    lines.append(f"Balanced Accuracy : {bin_bal_acc*100:.2f}%")
    lines.append(f"Macro F1-Score    : {bin_f1:.4f}")
    lines.append(classification_report(yb_test, yb_pred, target_names=["Healthy", "Abnormal"], digits=4))
    lines.append("=== MULTI-LABEL FAULT IDENTIFICATION (6 Simultaneous Faults) ===")
    lines.append(f"Macro F1-Score         : {macro_f1:.4f}")
    lines.append(f"Exact-Match Accuracy   : {exact_match*100:.2f}% (all 6 fault flags simultaneously correct)")
    lines.append(f"Hamming Accuracy       : {(1.0 - h_loss)*100:.2f}% (per-label average accuracy)")
    lines.append("")
    lines.append(f"{'Fault Type':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Accuracy':>10} {'Support':>9}")
    lines.append("-" * 78)
    for r in metrics_rows:
        lines.append(
            f"{r['Fault']:<25} {r['Precision']:>10.4f} {r['Recall']:>10.4f} "
            f"{r['F1']:>10.4f} {r['Accuracy']:>10.4f} {r['Support']:>9d}"
        )
    lines.append("")
    lines.append("=== EXACT-MATCH ACCURACY BY NUMBER OF SIMULTANEOUS FAULTS (TEST SET) ===")
    for k, (cnt, acc_k) in exact_by_k.items():
        lines.append(f"  {k} simultaneous fault(s) ({cnt:>3d} windows): {acc_k*100:.2f}% exact match")

    summary_text = "\n".join(lines)
    print(summary_text)
    with open(os.path.join(OUT_DIR, "metrics_multifault_summary.txt"), "w") as f:
        f.write(summary_text)

    plot_binary_cm(confusion_matrix(yb_test, yb_pred), bin_acc, "confusion_matrix_multifault_binary.png")
    plot_multilabel_bars(metrics_rows, "multilabel_f1_per_fault.png")
    plot_example_signals(examples, "simulated_multifault_signals.png")
    print(f"All multi-fault outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
