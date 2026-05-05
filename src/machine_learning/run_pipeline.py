import subprocess
import argparse
import sys

"""
python3 run_pipeline.py \
  --input parquet_data.parquet \
  --date_frac 0.5
"""


def run_step(cmd, name):
    print(f"\n=== Running: {name} ===")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ Failed at step: {name}")
        sys.exit(result.returncode)

    print(f"✅ Completed: {name}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--features", default="features.parquet")
    parser.add_argument("--predictions", default="predictions.parquet")
    parser.add_argument("--date_frac", type=float, default=1.0)

    args = parser.parse_args()

    # Step 1: Feature Engineering
    run_step(
        [
            "python3",
            "deep_feature_engineering.py",
            "--input", args.input,
            "--output", args.features,
            "--date_frac", str(args.date_frac),
        ],
        "Feature Engineering"
    )

    # Step 2: Model Training
    run_step(
        [
            "python3",
            "deep_train_model.py",
            "--input", args.features,
        ],
        "Model Training"
    )

    # Step 3: Validation
    run_step(
        [
            "python3",
            "train_valid.py",
            "--input", args.predictions,
        ],
        "Validation"
    )

    # Step 4: Signals
    run_step(
        [
	    "python3",
	    "generate_signals.py",
	    "--input", args.predictions,
        ],
        "Signal Generation"
    )

    print("\n🎉 Pipeline completed successfully.")


if __name__ == "__main__":
    main()
