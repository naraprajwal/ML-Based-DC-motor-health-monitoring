
# ================================================================
# DC MOTOR HEALTH MONITORING - DATASET D
# Physics-based simulation + combined faults + ML
#
# Motor nameplate:
# 220 V, 3.7 kW, 19 A, 1500 RPM
#
# Assumption:
# Separately excited brushed DC motor.
# Confirm topology and actual parameters with laboratory TA.
#
# pip install numpy pandas scipy matplotlib scikit-learn
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ================================================================
# 1. MOTOR PARAMETERS
# ================================================================

P = {
    "Va_rated": 220.0,
    "Vf_rated": 220.0,

    "Ia_rated": 19.0,
    "If_rated": 1.0,

    "rpm_rated": 1500.0,
    "power_rated": 3700.0,

    # Initial estimated parameters
    "Ra": 1.25,
    "La": 0.08,

    "Rf": 220.0,
    "Lf": 8.0,

    "Ke": 1.30,
    "Kt": 1.30,

    "J": 0.08,
    "B": 0.012,

    "TL_nom": 22.8,

    # Thermal parameters
    "alpha_copper": 0.00393,
    "Rth": 0.08,
    "Cth": 5000.0,

    "ambient": 25.0
}


# ================================================================
# 2. FAULT LABELS
# ================================================================

FAULTS = [
    "armature_resistance",
    "friction",
    "field_weakening",
    "overload"
]


# ================================================================
# 3. MOTOR SIMULATION
# ================================================================

def simulate_motor(run_id, dt=0.01, duration=6.0):

    rng = np.random.default_rng(1000 + run_id)

    time = np.arange(0, duration, dt)

    # ------------------------------------------------------------
    # Randomized operating point
    # ------------------------------------------------------------

    Va_base = rng.uniform(165, 220)
    Vf_base = rng.uniform(185, 220)

    load_base = rng.uniform(10, 25)

    Ra = P["Ra"] * rng.uniform(0.92, 1.08)
    La = P["La"] * rng.uniform(0.90, 1.10)

    Rf = P["Rf"] * rng.uniform(0.95, 1.05)
    Lf = P["Lf"] * rng.uniform(0.90, 1.10)

    J = P["J"] * rng.uniform(0.90, 1.10)
    B = P["B"] * rng.uniform(0.85, 1.15)

    TL = load_base

    field_scale = 1.0

    # ------------------------------------------------------------
    # Random fault combinations
    # ------------------------------------------------------------

    active_faults = {
        f: 0 for f in FAULTS
    }

    # 42% healthy, 58% abnormal
    if rng.random() < 0.58:

        number_of_faults = rng.choice(
            [1, 2, 3, 4],
            p=[0.36, 0.34, 0.22, 0.08]
        )

        selected_faults = rng.choice(
            FAULTS,
            size=number_of_faults,
            replace=False
        )

        for fault in selected_faults:
            active_faults[fault] = 1

    # ------------------------------------------------------------
    # Fault severity
    # ------------------------------------------------------------

    if active_faults["armature_resistance"]:

        Ra *= rng.uniform(1.15, 1.50)

    if active_faults["friction"]:

        B *= rng.uniform(1.8, 3.5)

    if active_faults["field_weakening"]:

        field_scale *= rng.uniform(0.65, 0.90)

    if active_faults["overload"]:

        TL *= rng.uniform(1.25, 1.70)

    Vf_base *= field_scale

    # ------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------

    # State:
    # x[0] = Ia
    # x[1] = If
    # x[2] = omega
    # x[3] = temperature

    x0 = [
        0.0,
        0.0,
        0.0,
        P["ambient"]
    ]

    # ------------------------------------------------------------
    # Differential equations
    # ------------------------------------------------------------

    def motor_equations(t, x):

        Ia, If, omega, temperature = x

        # Temperature-dependent armature resistance
        Ra_T = Ra * (
            1 + P["alpha_copper"] *
            (temperature - P["ambient"])
        )

        # Supply disturbances
        Va = (
            Va_base
            + 1.2 * np.sin(2 * np.pi * 0.7 * t)
            + 0.45 * np.sin(2 * np.pi * 7 * t)
        )

        Vf = (
            Vf_base
            + 0.35 * np.sin(2 * np.pi * 0.4 * t)
        )

        # Magnetic saturation approximation
        phi = np.tanh(
            max(If, 0.0) / P["If_rated"]
        )

        # Electrical armature dynamics
        dIa_dt = (
            Va
            - Ra_T * Ia
            - P["Ke"] * phi * omega
        ) / La

        # Electrical field dynamics
        dIf_dt = (
            Vf - Rf * If
        ) / Lf

        # Electromagnetic torque
        Te = P["Kt"] * phi * Ia

        # Mechanical dynamics
        dOmega_dt = (
            Te - B * omega - TL
        ) / J

        # Thermal dynamics
        copper_loss = Ia**2 * Ra_T

        cooling_loss = (
            temperature - P["ambient"]
        ) / P["Rth"]

        dTemperature_dt = (
            copper_loss - cooling_loss
        ) / P["Cth"]

        return [
            dIa_dt,
            dIf_dt,
            dOmega_dt,
            dTemperature_dt
        ]

    # ------------------------------------------------------------
    # Numerical integration
    # ------------------------------------------------------------

    solution = solve_ivp(
        motor_equations,
        [0, duration],
        x0,
        t_eval=time,
        method="RK45",
        max_step=dt,
        rtol=1e-5,
        atol=1e-7
    )

    Ia = solution.y[0]
    If = solution.y[1]

    omega = solution.y[2]
    temperature = solution.y[3]

    RPM = omega * 60 / (2 * np.pi)

    # ------------------------------------------------------------
    # Sensor measurements + noise
    # ------------------------------------------------------------

    Ia_measured = Ia + rng.normal(
        0, 0.12, len(time)
    )

    If_measured = If + rng.normal(
        0, 0.015, len(time)
    )

    RPM_measured = RPM + rng.normal(
        0, 3.0, len(time)
    )

    temperature_measured = temperature + rng.normal(
        0, 0.25, len(time)
    )

    Va_measured = (
        Va_base
        + 1.2 * np.sin(2 * np.pi * 0.7 * time)
        + 0.45 * np.sin(2 * np.pi * 7 * time)
        + rng.normal(0, 0.3, len(time))
    )

    Vf_measured = (
        Vf_base
        + 0.35 * np.sin(2 * np.pi * 0.4 * time)
        + rng.normal(0, 0.2, len(time))
    )

    # Illustrative vibration proxy ONLY.
    # Not a physically validated vibration model.

    acceleration = (
        P["Kt"] * np.tanh(np.maximum(If, 0))
        * Ia - B * omega - TL
    ) / J

    vibration_proxy = (
        0.04 * np.abs(acceleration)
        + rng.normal(0, 0.08, len(time))
    )

    # ------------------------------------------------------------
    # Output dataframe
    # ------------------------------------------------------------

    df = pd.DataFrame({

        "run_id": run_id,

        "time_s": time,

        "Va": Va_measured,
        "Vf": Vf_measured,

        "Ia": Ia_measured,
        "If": If_measured,

        "rpm": RPM_measured,

        "temperature": temperature_measured,

        "vibration_proxy": vibration_proxy,

        "Ia_true": Ia,
        "If_true": If,

        "rpm_true": RPM,
        "temperature_true": temperature,

        "health_label": int(
            any(active_faults.values())
        )
    })

    for fault in FAULTS:

        df["fault_" + fault] = active_faults[fault]

    metadata = {
        "run_id": run_id,
        "Va_base": Va_base,
        "Vf_base": Vf_base,
        "load_Nm": TL,
        "Ra": Ra,
        "B": B,
        "J": J,
        "health_label": int(
            any(active_faults.values())
        )
    }

    for fault in FAULTS:

        metadata["fault_" + fault] = active_faults[fault]

    return df, metadata


# ================================================================
# 4. FEATURE EXTRACTION
# ================================================================

SIGNALS = [
    "Ia",
    "If",
    "rpm",
    "Va",
    "Vf",
    "temperature",
    "vibration_proxy"
]


def extract_features(window):

    features = {}

    for signal in SIGNALS:

        x = window[signal].to_numpy()

        features[signal + "_mean"] = np.mean(x)

        features[signal + "_std"] = np.std(x)

        features[signal + "_rms"] = np.sqrt(
            np.mean(x**2)
        )

        features[signal + "_min"] = np.min(x)

        features[signal + "_max"] = np.max(x)

    # Temporal features
    for signal in ["Ia", "If", "rpm"]:

        x = window[signal].to_numpy()

        features[signal + "_diff_rms"] = np.sqrt(
            np.mean(np.diff(x)**2)
        )

        features[signal + "_slope"] = np.polyfit(
            np.arange(len(x)), x, 1
        )[0]

    return features


# ================================================================
# 5. GENERATE DATASET D
# ================================================================

N_RUNS = 1200

DT = 0.01

DURATION = 6.0

WINDOW_SIZE = 300

all_windows = []

all_metadata = []

example_traces = {}

print("Generating simulated motor runs...")

for run_id in range(N_RUNS):

    trace, metadata = simulate_motor(
        run_id,
        dt=DT,
        duration=DURATION
    )

    all_metadata.append(metadata)

    if run_id < 10:
        example_traces[run_id] = trace

    # Each run has two windows of 3 seconds
    for window_id in range(2):

        start = window_id * WINDOW_SIZE

        end = start + WINDOW_SIZE

        window = trace.iloc[start:end]

        features = extract_features(window)

        features["run_id"] = run_id

        features["window_id"] = window_id

        features["health_label"] = metadata["health_label"]

        for fault in FAULTS:

            features["fault_" + fault] = metadata[
                "fault_" + fault
            ]

        all_windows.append(features)


dataset = pd.DataFrame(all_windows)

metadata_df = pd.DataFrame(all_metadata)

dataset.to_csv(
    "dataset_D_windows.csv",
    index=False
)

metadata_df.to_csv(
    "dataset_D_run_metadata.csv",
    index=False
)

print("\nDataset generated!")

print("Shape:", dataset.shape)

print("\nHealth class distribution:")

print(dataset["health_label"].value_counts())


# ================================================================
# 6. TRAIN-TEST SPLIT
# ================================================================

TARGETS = [
    "health_label"
] + [
    "fault_" + f for f in FAULTS
]

FEATURES = [
    c for c in dataset.columns
    if c not in TARGETS + ["run_id", "window_id"]
]

X = dataset[FEATURES]

y = dataset[TARGETS]

groups = dataset["run_id"]

# IMPORTANT:
# Split by entire motor simulation runs.
# Prevents windows from the same simulation entering
# both training and test datasets.

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y["health_label"], groups)
)

X_train = X.iloc[train_idx]

X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]

y_test = y.iloc[test_idx]


# ================================================================
# 7. BINARY HEALTH CLASSIFICATION
# ================================================================

binary_model = make_pipeline(

    StandardScaler(),

    ExtraTreesClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
)

binary_model.fit(
    X_train,
    y_train["health_label"]
)

binary_pred = binary_model.predict(X_test)

accuracy = accuracy_score(
    y_test["health_label"],
    binary_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test["health_label"],
    binary_pred
)

print("\n==============================")
print("BINARY HEALTH CLASSIFICATION")
print("==============================")

print("Accuracy:", round(accuracy, 4))

print(
    "Balanced accuracy:",
    round(balanced_accuracy, 4)
)

print(
    classification_report(
        y_test["health_label"],
        binary_pred,
        target_names=["Healthy", "Abnormal"],
        digits=4
    )
)


# ================================================================
# 8. MULTI-LABEL FAULT IDENTIFICATION
# ================================================================

fault_targets = [
    "fault_" + f for f in FAULTS
]

multi_model = MultiOutputClassifier(

    ExtraTreesClassifier(
        n_estimators=180,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
)

multi_model.fit(
    X_train,
    y_train[fault_targets]
)

multi_pred = multi_model.predict(X_test)

macro_f1 = f1_score(
    y_test[fault_targets],
    multi_pred,
    average="macro",
    zero_division=0
)

print("\n==============================")
print("MULTI-LABEL FAULT IDENTIFICATION")
print("==============================")

print("Macro F1:", round(macro_f1, 4))

print("\nIndividual fault reports:")

for i, fault in enumerate(FAULTS):

    print("\n", fault)

    print(
        classification_report(
            y_test["fault_" + fault],
            multi_pred[:, i],
            digits=4,
            zero_division=0
        )
    )


# ================================================================
# 9. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    y_test["health_label"],
    binary_pred,
    labels=[0, 1]
)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.xticks(
    [0, 1],
    ["Healthy", "Abnormal"]
)

plt.yticks(
    [0, 1],
    ["Healthy", "Abnormal"]
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title(
    f"Health Classification\n"
    f"Accuracy = {accuracy:.2%}"
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "dataset_D_confusion_matrix.png",
    dpi=160
)

plt.show()


# ================================================================
# 10. PLOT EXAMPLE SIMULATED MOTOR RESPONSES
# ================================================================

healthy_run = None

combined_fault_run = None

for run_id, trace in example_traces.items():

    health = trace["health_label"].iloc[0]

    if health == 0 and healthy_run is None:

        healthy_run = trace

    if health == 1 and combined_fault_run is None:

        faults_active = sum(
            trace["fault_" + f].iloc[0]
            for f in FAULTS
        )

        if faults_active >= 2:

            combined_fault_run = trace


if healthy_run is not None and combined_fault_run is not None:

    fig, axes = plt.subplots(
        3, 1,
        figsize=(10, 8),
        sharex=True
    )

    axes[0].plot(
        healthy_run["time_s"],
        healthy_run["rpm"],
        label="Healthy"
    )

    axes[0].plot(
        combined_fault_run["time_s"],
        combined_fault_run["rpm"],
        label="Combined faults"
    )

    axes[1].plot(
        healthy_run["time_s"],
        healthy_run["Ia"],
        label="Healthy"
    )

    axes[1].plot(
        combined_fault_run["time_s"],
        combined_fault_run["Ia"],
        label="Combined faults"
    )

    axes[2].plot(
        healthy_run["time_s"],
        healthy_run["temperature"],
        label="Healthy"
    )

    axes[2].plot(
        combined_fault_run["time_s"],
        combined_fault_run["temperature"],
        label="Combined faults"
    )

    axes[0].set_ylabel("RPM")

    axes[1].set_ylabel("Armature current (A)")

    axes[2].set_ylabel("Temperature (°C)")

    axes[2].set_xlabel("Time (s)")

    for ax in axes:

        ax.grid(True)

        ax.legend()

    plt.suptitle(
        "Simulated DC Motor Responses"
    )

    plt.tight_layout()

    plt.savefig(
        "dataset_D_motor_responses.png",
        dpi=160
    )

    plt.show()


# ================================================================
# 11. SAVE METRICS
# ================================================================

with open(
    "dataset_D_metrics.txt",
    "w"
) as f:

    f.write(
        "DC MOTOR DATASET D - INITIAL SIMULATION RESULTS\n\n"
    )

    f.write(
        f"Runs: {N_RUNS}\n"
    )

    f.write(
        f"Windows: {len(dataset)}\n"
    )

    f.write(
        f"Binary accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Balanced accuracy: {balanced_accuracy:.4f}\n"
    )

    f.write(
        f"Multi-label macro F1: {macro_f1:.4f}\n\n"
    )

    f.write(
        classification_report(
            y_test["health_label"],
            binary_pred,
            digits=4
        )
    )

    f.write(
        "\n\nConfusion matrix:\n"
    )

    f.write(
        str(cm)
    )

print("\nAll results saved successfully!")

print("Files created:")
print("1. dataset_D_windows.csv")
print("2. dataset_D_run_metadata.csv")
print("3. dataset_D_metrics.txt")
print("4. dataset_D_confusion_matrix.png")
print("5. dataset_D_motor_responses.png")
