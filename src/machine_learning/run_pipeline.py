import subprocess
import argparse
import sys
import os


def run_step(cmd, name, expected_outputs=None):
    print(f"\n=== Running: {name} ===")
    print("Command:", " ".join(cmd))

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ Failed at step: {name}")
        sys.exit(result.returncode)

    # Validate expected output files
    if expected_outputs:
        missing = [f for f in expected_outputs if not os.path.exists(f)]
        if missing:
            print(f"\n❌ Step '{name}' completed but missing outputs: {missing}")
            sys.exit(1)

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
        "Feature Engineering",
        expected_outputs=[args.features],
    )

    # Step 2: Model Training (should CREATE predictions)
    run_step(
        [
            "python3",
            "deep_train_model.py",
            "--input", args.features,
            "--pred_out", args.predictions,  # <-- IMPORTANT FIX
        ],
        "Model Training",
        expected_outputs=[args.predictions],
    )

    # Step 3: Validation
    run_step(
        [
            "python3",
            "train_valid.py",
            "--input", args.predictions,
        ],
        "Validation",
    )

    # Step 4: Signals
    run_step(
        [
            "python3",
            "generate_signals.py",
            "--input", args.predictions,
        ],
        "Signal Generation",
    )

    print("\n🎉 Pipeline completed successfully.")


if __name__ == "__main__":
    main()
