"""Section 3: Compare NILM models, including a causal Seq2Point CNN."""

import os
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import FIG_DIR, PROC_DIR, SEED
from refit_analysis import (
    WASHING_MACHINE_ACTIVE_THRESHOLD_W,
    load_house_one_minute,
)
from washing_machine_mapping import MODEL_TEST_HOUSE, MODEL_TRAIN_HOUSES

WINDOW_LENGTH = 31
MAX_TRAIN_SAMPLES_PER_HOUSE = 50_000
NEGATIVE_TO_POSITIVE_RATIO = 10
EPOCHS = 25
BATCH_SIZE = 512
THRESHOLD_CANDIDATES = np.arange(0.10, 1.00, 0.01)
MODEL_NAMES = ("Always off", "Random Forest", "Two-stage LightGBM", "Seq2Point CNN")


class Seq2PointCNN(nn.Module):
    """Map a 31-minute aggregate window to center-minute activity and power."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 3, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.activity_head = nn.Linear(64, 1)
        self.power_head = nn.Linear(64, 1)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(sequence)
        return self.activity_head(encoded).squeeze(1), self.power_head(encoded).squeeze(1)


def make_tabular_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    aggregate = data["Aggregate"]
    features = pd.DataFrame(index=data.index)
    features["aggregate_now"] = aggregate
    for lag in (1, 5, 15, 30):
        features[f"aggregate_lag_{lag}"] = aggregate.shift(lag)
    for window in (5, 15, 30):
        features[f"aggregate_mean_prev_{window}"] = (
            aggregate.shift(1).rolling(window=window, min_periods=window).mean()
        )

    minute_of_day = data.index.hour * 60 + data.index.minute
    features["hour_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    features["hour_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    features["weekday_sin"] = np.sin(2 * np.pi * data.index.dayofweek / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * data.index.dayofweek / 7)
    target = data["WashingMachine"]
    valid = features.notna().all(axis=1) & target.notna()
    return features.loc[valid], target.loc[valid]


def prepare_house(
    house: int,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    data = load_house_one_minute(house)
    features, target = make_tabular_features(data)
    row_positions = data.index.get_indexer(features.index)
    aggregate = data["Aggregate"].to_numpy(dtype=np.float32) / 1000.0
    windows = np.lib.stride_tricks.sliding_window_view(
        aggregate, WINDOW_LENGTH
    )[row_positions - WINDOW_LENGTH + 1]
    if not np.isfinite(windows).all():
        raise ValueError(
            f"House {house}: feature validation admitted an incomplete "
            "Seq2Point window."
        )
    return features, target, np.ascontiguousarray(windows[:, np.newaxis, :])


def select_training_rows(
    target: pd.Series, random_generator: np.random.Generator
) -> np.ndarray:
    values = target.to_numpy(dtype=np.float32)
    positive = np.flatnonzero(values >= WASHING_MACHINE_ACTIVE_THRESHOLD_W)
    negative = np.flatnonzero(values < WASHING_MACHINE_ACTIVE_THRESHOLD_W)
    if positive.size == 0:
        raise ValueError("A training household has no active washing-machine rows.")

    positive_limit = MAX_TRAIN_SAMPLES_PER_HOUSE // (
        NEGATIVE_TO_POSITIVE_RATIO + 1
    )
    if positive.size > positive_limit:
        positive = random_generator.choice(
            positive, size=positive_limit, replace=False
        )
    negative_limit = min(
        negative.size,
        positive.size * NEGATIVE_TO_POSITIVE_RATIO,
        MAX_TRAIN_SAMPLES_PER_HOUSE - positive.size,
    )
    negative = random_generator.choice(
        negative, size=negative_limit, replace=False
    )
    return np.sort(np.concatenate((positive, negative)))


def fit_random_forest(
    features_by_house: list[pd.DataFrame],
    targets_by_house: list[pd.Series],
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        max_features=0.8,
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(
        pd.concat(features_by_house, ignore_index=True),
        pd.concat(targets_by_house, ignore_index=True),
    )
    return model


def fit_lightgbm(
    features_by_house: list[pd.DataFrame],
    targets_by_house: list[pd.Series],
) -> tuple[lgb.LGBMClassifier, lgb.LGBMRegressor]:
    features = pd.concat(features_by_house, ignore_index=True)
    target = pd.concat(targets_by_house, ignore_index=True)
    active = target >= WASHING_MACHINE_ACTIVE_THRESHOLD_W

    activity_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        colsample_bytree=0.8,
        subsample=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=SEED,
        verbosity=-1,
    )
    power_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        colsample_bytree=0.8,
        subsample=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=SEED,
        verbosity=-1,
    )
    activity_model.fit(features, active.astype("int8"))
    power_model.fit(features.loc[active], target.loc[active])
    return activity_model, power_model


def fit_seq2point(
    windows_by_house: list[np.ndarray],
    targets_by_house: list[pd.Series],
) -> Seq2PointCNN:
    features = np.concatenate(windows_by_house, axis=0)
    target = np.concatenate(
        [series.to_numpy(dtype=np.float32) / 1000.0 for series in targets_by_house]
    )
    activity = target >= WASHING_MACHINE_ACTIVE_THRESHOLD_W / 1000.0

    torch.manual_seed(SEED)
    model = Seq2PointCNN()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    activity_loss = nn.BCEWithLogitsLoss()
    power_loss = nn.SmoothL1Loss()
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(activity.astype(np.float32)),
        torch.from_numpy(target),
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for sequence, active_target, power_target in loader:
            optimizer.zero_grad(set_to_none=True)
            activity_logits, power_prediction = model(sequence)
            loss = activity_loss(activity_logits, active_target)
            active_batch = active_target.bool()
            if active_batch.any():
                loss = loss + power_loss(
                    power_prediction[active_batch],
                    power_target[active_batch],
                )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(sequence)

        print(
            f"Seq2Point epoch {epoch + 1:02d}/{EPOCHS}: "
            f"balanced-sample loss={total_loss / len(dataset):.5f}"
        )

    return model


def predict_seq2point(
    model: Seq2PointCNN, windows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    powers = []
    with torch.inference_mode():
        for start in range(0, len(windows), 4096):
            batch = torch.from_numpy(windows[start : start + 4096])
            logits, power = model(batch)
            probabilities.append(torch.sigmoid(logits).numpy())
            powers.append(power.numpy())
    return np.concatenate(probabilities), np.maximum(
        np.concatenate(powers), 0.0
    ) * 1000.0


def calibrate_threshold(
    model_name: str,
    probabilities: np.ndarray,
    conditional_power: np.ndarray,
    target: pd.Series,
) -> tuple[float, pd.DataFrame]:
    actual = target.to_numpy(dtype=np.float32)
    actual_active = actual >= WASHING_MACHINE_ACTIVE_THRESHOLD_W
    records = []
    for threshold in THRESHOLD_CANDIDATES:
        predicted_active = (
            (probabilities >= threshold)
            & (conditional_power >= WASHING_MACHINE_ACTIVE_THRESHOLD_W)
        )
        prediction = np.where(predicted_active, conditional_power, 0.0)
        records.append(
            {
                "model": model_name,
                "probability_threshold": float(threshold),
                "validation_active_precision": precision_score(
                    actual_active, predicted_active, zero_division=0
                ),
                "validation_active_recall": recall_score(
                    actual_active, predicted_active, zero_division=0
                ),
                "validation_active_f1": f1_score(
                    actual_active, predicted_active, zero_division=0
                ),
                "validation_mae_w": mean_absolute_error(actual, prediction),
                "validation_rmse_w": np.sqrt(
                    mean_squared_error(actual, prediction)
                ),
            }
        )
    results = pd.DataFrame(records)
    best = results.sort_values(
        ["validation_active_f1", "validation_mae_w"],
        ascending=[False, True],
    ).iloc[0]
    return float(best["probability_threshold"]), results


def compute_metrics(
    target: pd.Series, prediction: np.ndarray, model_name: str
) -> dict[str, float | str | int]:
    actual = target.to_numpy(dtype=np.float32)
    predicted = np.asarray(prediction, dtype=np.float32)
    active = actual >= WASHING_MACHINE_ACTIVE_THRESHOLD_W
    predicted_active = predicted >= WASHING_MACHINE_ACTIVE_THRESHOLD_W
    actual_energy = float(actual.sum() / 60_000.0)
    predicted_energy = float(predicted.sum() / 60_000.0)
    return {
        "model": model_name,
        "samples": len(actual),
        "mae_w": mean_absolute_error(actual, predicted),
        "rmse_w": np.sqrt(mean_squared_error(actual, predicted)),
        "r2": r2_score(actual, predicted),
        "active_precision": precision_score(active, predicted_active, zero_division=0),
        "active_recall": recall_score(active, predicted_active, zero_division=0),
        "active_f1": f1_score(active, predicted_active, zero_division=0),
        "actual_energy_kwh": actual_energy,
        "predicted_energy_kwh": predicted_energy,
        "energy_error_percent": (
            100 * (predicted_energy - actual_energy) / actual_energy
            if actual_energy > 0
            else float("nan")
        ),
    }


def plot_predictions(
    target: pd.Series, predictions: dict[str, np.ndarray]
) -> Path:
    center = target.index[0]
    start = center - pd.Timedelta(hours=2)
    end = start + pd.Timedelta(hours=24)
    observed = target.loc[start:end]

    fig, axis = plt.subplots(figsize=(14, 5))
    axis.plot(
        observed.index,
        observed,
        label="Actual washing-machine power",
        linewidth=1.1,
        color="black",
    )
    for name, values in predictions.items():
        series = pd.Series(values, index=target.index).loc[start:end]
        axis.plot(series.index, series, label=name, linewidth=0.9, alpha=0.8)
    axis.set_title(
        f"House {MODEL_TEST_HOUSE}: actual vs predicted washing-machine power"
    )
    axis.set_ylabel("Power (W)")
    axis.set_xlabel("Local timestamp")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    output = FIG_DIR / "section3_actual_vs_predicted.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def fit_models(
    houses: tuple[int, ...],
    random_generator: np.random.Generator,
) -> tuple[
    RandomForestRegressor,
    lgb.LGBMClassifier,
    lgb.LGBMRegressor,
    Seq2PointCNN,
]:
    tabular_features = []
    sampled_targets = []
    sampled_windows = []
    for house in houses:
        features, target, windows = prepare_house(house)
        selected = select_training_rows(target, random_generator)
        positive_count = int(
            (target.iloc[selected] >= WASHING_MACHINE_ACTIVE_THRESHOLD_W).sum()
        )
        print(
            f"House {house}: {len(features):,} valid rows; "
            f"training on {len(selected):,} stratified windows "
            f"({positive_count:,} active)."
        )
        tabular_features.append(features.iloc[selected])
        sampled_targets.append(target.iloc[selected])
        sampled_windows.append(windows[selected])

    forest = fit_random_forest(tabular_features, sampled_targets)
    activity_model, power_model = fit_lightgbm(tabular_features, sampled_targets)
    seq2point = fit_seq2point(sampled_windows, sampled_targets)
    return forest, activity_model, power_model, seq2point


def run() -> None:
    torch.set_num_threads(max(1, min(os.cpu_count() or 1, 8)))
    generator = np.random.default_rng(SEED)
    validation_house = MODEL_TRAIN_HOUSES[-1]
    calibration_houses = MODEL_TRAIN_HOUSES[:-1]

    print("Fitting threshold-calibration models on training-side households.")
    _, calibration_lgbm_classifier, calibration_lgbm_regressor, calibration_cnn = (
        fit_models(calibration_houses, generator)
    )
    validation_features, validation_target, validation_windows = prepare_house(
        validation_house
    )
    lgbm_validation_prob = calibration_lgbm_classifier.predict_proba(
        validation_features
    )[:, 1]
    lgbm_validation_power = np.maximum(
        calibration_lgbm_regressor.predict(validation_features), 0.0
    )
    lgbm_threshold, lgbm_calibration = calibrate_threshold(
        "Two-stage LightGBM",
        lgbm_validation_prob,
        lgbm_validation_power,
        validation_target,
    )
    cnn_validation_prob, cnn_validation_power = predict_seq2point(
        calibration_cnn, validation_windows
    )
    cnn_threshold, cnn_calibration = calibrate_threshold(
        "Seq2Point CNN",
        cnn_validation_prob,
        cnn_validation_power,
        validation_target,
    )
    del calibration_lgbm_classifier, calibration_lgbm_regressor, calibration_cnn

    print("\nRefitting final models on all training households.")
    forest, lgbm_classifier, lgbm_regressor, seq2point = fit_models(
        MODEL_TRAIN_HOUSES, generator
    )
    test_features, test_target, test_windows = prepare_house(MODEL_TEST_HOUSE)
    lgbm_probabilities = lgbm_classifier.predict_proba(test_features)[:, 1]
    lgbm_power = np.maximum(lgbm_regressor.predict(test_features), 0.0)
    cnn_probabilities, cnn_power = predict_seq2point(seq2point, test_windows)

    forest_prediction = np.maximum(forest.predict(test_features), 0.0)
    lgbm_prediction = np.where(
        lgbm_probabilities >= lgbm_threshold, lgbm_power, 0.0
    )
    cnn_prediction = np.where(cnn_probabilities >= cnn_threshold, cnn_power, 0.0)
    predictions = {
        "Random Forest": forest_prediction,
        "Two-stage LightGBM": lgbm_prediction,
        "Seq2Point CNN": cnn_prediction,
    }
    results = pd.DataFrame(
        [
            compute_metrics(test_target, np.zeros(len(test_target)), "Always off"),
            *[
                compute_metrics(test_target, prediction, name)
                for name, prediction in predictions.items()
            ],
        ]
    )
    calibration = pd.concat(
        [lgbm_calibration, cnn_calibration], ignore_index=True
    )

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = PROC_DIR / "section3_model_metrics.csv"
    calibration_path = PROC_DIR / "section3_threshold_validation.csv"
    results.to_csv(metrics_path, index=False)
    calibration.to_csv(calibration_path, index=False)

    predictions_path = PROC_DIR / "section3_test_predictions.csv"
    pd.DataFrame(
        {
            "Time": test_target.index,
            "Aggregate": test_features["aggregate_now"].to_numpy(),
            "ActualWashingMachine": test_target.to_numpy(),
            **predictions,
        }
    ).to_csv(predictions_path, index=False)
    figure = plot_predictions(test_target, predictions)

    print(
        f"\nTraining households: {', '.join(map(str, MODEL_TRAIN_HOUSES))}; "
        f"threshold validation household: {validation_house}; "
        f"held-out evaluation household: {MODEL_TEST_HOUSE}."
    )
    print(
        "Seq2Point uses a 31-minute trailing aggregate window to predict the "
        "target at its final minute; all models are scored on the same valid minutes."
    )
    print(
        f"Selected thresholds: LightGBM={lgbm_threshold:.2f}, "
        f"Seq2Point={cnn_threshold:.2f}."
    )
    print(results.to_string(index=False))
    print(f"\nMetrics: {metrics_path}")
    print(f"Threshold validation: {calibration_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Evaluation plot: {figure}")


if __name__ == "__main__":
    run()
