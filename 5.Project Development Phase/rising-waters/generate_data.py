"""Generate synthetic historical weather data for flood prediction training."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "flood_data.csv"
RANDOM_STATE = 42


def generate_flood_dataset(n_samples: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    annual_rainfall = rng.uniform(800, 3500, n_samples)
    monsoon_rainfall = annual_rainfall * rng.uniform(0.55, 0.85, n_samples)
    pre_monsoon_rainfall = annual_rainfall * rng.uniform(0.05, 0.20, n_samples)
    post_monsoon_rainfall = annual_rainfall - monsoon_rainfall - pre_monsoon_rainfall
    post_monsoon_rainfall = np.clip(post_monsoon_rainfall, 50, 1200)
    cloud_visibility = rng.uniform(0.5, 15, n_samples)

    flood_score = (
        0.35 * (annual_rainfall / 3500)
        + 0.40 * (monsoon_rainfall / 2800)
        + 0.15 * (pre_monsoon_rainfall / 600)
        + 0.25 * ((15 - cloud_visibility) / 15)
    )
    noise = rng.normal(0, 0.03, n_samples)
    flood = (flood_score + noise >= 0.56).astype(int)

    return pd.DataFrame(
        {
            "annual_rainfall_mm": np.round(annual_rainfall, 1),
            "monsoon_rainfall_mm": np.round(monsoon_rainfall, 1),
            "pre_monsoon_rainfall_mm": np.round(pre_monsoon_rainfall, 1),
            "post_monsoon_rainfall_mm": np.round(post_monsoon_rainfall, 1),
            "cloud_visibility_km": np.round(cloud_visibility, 2),
            "flood": flood,
        }
    )


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_flood_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df)} records to {DATA_PATH}")
    print(f"Flood events: {df['flood'].sum()} ({df['flood'].mean():.1%})")


if __name__ == "__main__":
    main()
