"""
demo.py — Interactive menu for the Chest X-Ray Pneumonia Detection project.

Usage:
    python demo.py

This is the single entry point for the entire project.
Choose what you want to do from the menu and the script handles the rest.
"""

import os
import sys


# ── Formatting helpers ────────────────────────────────────────────────────────

def clear():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def header():
    """Print the project banner."""
    print("=" * 58)
    print("  🫁  Chest X-Ray Pneumonia Detection — AI Demo")
    print("=" * 58)
    print("  Built with PyTorch + DenseNet121 + MLflow")
    print("  Classes: NORMAL  |  PNEUMONIA")
    print("=" * 58)
    print()


def section(title: str):
    """Print a section divider with a title."""
    print()
    print("─" * 58)
    print(f"  {title}")
    print("─" * 58)


def success(msg: str):
    print(f"\n  ✓  {msg}")


def error(msg: str):
    print(f"\n  ✗  {msg}")


def info(msg: str):
    print(f"  →  {msg}")


def pause():
    """Wait for the user to press Enter before going back to the menu."""
    print()
    input("  Press Enter to return to the menu...")


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def check_dataset() -> bool:
    """Return True if the data/ folder exists with at least a train/ subfolder."""
    return os.path.isdir(os.path.join("data", "train"))


def check_model() -> bool:
    """Return True if a trained model checkpoint exists."""
    return os.path.isfile("best_model.pth")


def print_status():
    """Print a one-line status for dataset and model."""
    ds  = "✓ Found" if check_dataset() else "✗ Not found (run Option 0)"
    mdl = "✓ Found" if check_model()   else "✗ Not found (run Option 1)"
    print(f"  Dataset : {ds}")
    print(f"  Model   : {mdl}")
    print()


# ── Menu options ──────────────────────────────────────────────────────────────

def option_download():
    """Option 0 — Download the Kaggle dataset."""
    section("Option 0 — Download Dataset")
    print("  This will download the chest-xray-pneumonia dataset from Kaggle.")
    print()
    print("  Before running, make sure you have:")
    print("    1. A Kaggle account  →  https://www.kaggle.com")
    print("    2. kaggle.json placed at ~/.kaggle/kaggle.json")
    print()
    confirm = input("  Continue? (y/n): ").strip().lower()
    if confirm == "y":
        import download_dataset
        download_dataset.download()
    else:
        info("Cancelled.")
    pause()


def option_train():
    """Option 1 — Train the model."""
    section("Option 1 — Train Model")

    if not check_dataset():
        error("Dataset not found. Please run Option 0 first.")
        pause()
        return

    print("  This will train a ResNet18 model on your chest X-ray data.")
    print("  Training saves the best model to: best_model.pth")
    print()

    # Let the user adjust epochs
    epoch_input = input("  How many epochs? (default: 10, press Enter to keep): ").strip()
    if epoch_input.isdigit():
        import train as train_module
        train_module.NUM_EPOCHS = int(epoch_input)

    print()
    info("Starting training...")
    import train as train_module
    train_module.train()
    pause()


def option_evaluate():
    """Option 2 — Evaluate model on the test set."""
    section("Option 2 — Evaluate Model")

    if not check_model():
        error("No trained model found. Please run Option 1 first.")
        pause()
        return

    info("Running evaluation on the test set...")
    print()
    import evaluate
    evaluate.evaluate()
    pause()


def option_predict():
    """Option 3 — Predict a single image."""
    section("Option 3 — Predict Single Image")

    if not check_model():
        error("No trained model found. Please run Option 1 first.")
        pause()
        return

    print("  Enter the path to a chest X-ray image.")
    print("  Example: data/test/PNEUMONIA/person1_virus_1.jpeg")
    print()

    image_path = input("  Image path: ").strip()

    if not image_path:
        error("No path entered.")
        pause()
        return

    if not os.path.isfile(image_path):
        error(f"File not found: {image_path}")
        pause()
        return

    print()
    info("Running prediction...")
    from predict import predict
    label, confidence = predict(image_path)

    print()
    print("  " + "=" * 40)
    print(f"  Prediction : {label}")
    print(f"  Confidence : {confidence * 100:.1f}%")
    print("  " + "=" * 40)

    if label == "PNEUMONIA":
        print("\n  ⚠️  Pneumonia indicators detected.")
    else:
        print("\n  ✓  No pneumonia indicators detected.")

    pause()


def option_visualize():
    """Option 4 — Visualize test results as a grid."""
    section("Option 4 — Visualize Test Results")

    if not check_model():
        error("No trained model found. Please run Option 1 first.")
        pause()
        return

    if not check_dataset():
        error("Dataset not found. Please run Option 0 first.")
        pause()
        return

    print("  Displays a grid of test images with predictions and confidence scores.")
    print()

    # Number of images
    num_input  = input("  How many images to show? (default: 12): ").strip()
    num_images = int(num_input) if num_input.isdigit() else 12

    # Only wrong?
    wrong_input = input("  Show only wrong predictions? (y/n, default: n): ").strip().lower()
    only_wrong  = wrong_input == "y"

    print()
    info("Generating grid...")
    import visualize_results
    visualize_results.visualize(num_images=num_images, only_wrong=only_wrong)
    pause()


def option_gradcam_single():
    """Option 5a — Grad-CAM heatmap for a single image."""
    section("Option 5a — Grad-CAM: Single Image")

    if not check_model():
        error("No trained model found. Please run Option 1 first.")
        pause()
        return

    print("  Shows where the model focused when making its prediction.")
    print("  Example: data/test/PNEUMONIA/person1_virus_1.jpeg")
    print()

    image_path = input("  Image path: ").strip()

    if not image_path:
        error("No path entered.")
        pause()
        return

    if not os.path.isfile(image_path):
        error(f"File not found: {image_path}")
        pause()
        return

    info("Generating Grad-CAM heatmap...")
    import gradcam
    gradcam.run_gradcam(image_path)
    pause()


def option_export():
    """Option 7 — Export model to TorchScript for deployment."""
    section("Option 7 — Export Model to TorchScript")

    if not check_model():
        error("No trained model found. Please run Option 1 first.")
        pause()
        return

    print("  TorchScript converts your model into a portable format")
    print("  that can run without Python — great for FastAPI or C++ apps.")
    print()
    print("  Output file: densenet121_pneumonia_prod.pt")
    print()
    confirm = input("  Export model? (y/n): ").strip().lower()
    if confirm == "y":
        import export as export_module
        export_module.export()
        success("Model exported to densenet121_pneumonia_prod.pt")
    else:
        info("Cancelled.")
    pause()


def option_report():
    """Option 6 — Generate a PDF summary report."""
    section("Option 6 — Generate PDF Report")

    if not check_model():
        error("No trained model found. Please run Option 1 first.")
        pause()
        return

    if not check_dataset():
        error("Dataset not found. Please run Option 0 first.")
        pause()
        return

    print("  This will generate a full PDF report including:")
    print("    • Overall and per-class accuracy")
    print("    • Confusion matrix")
    print("    • Grad-CAM heatmap samples")
    print("    • Plain-English performance summary")
    print()
    print("  Output file: xray_report.pdf")
    print()
    confirm = input("  Generate report? (y/n): ").strip().lower()
    if confirm == "y":
        import report
        report.generate_report()
        success("Report saved as xray_report.pdf")
    else:
        info("Cancelled.")
    pause()


def option_gradcam_compare():
    """Option 5b — Grad-CAM comparison across multiple images."""
    section("Option 5b — Grad-CAM: Compare NORMAL vs PNEUMONIA")

    if not check_model():
        error("No trained model found. Please run Option 1 first.")
        pause()
        return

    if not check_dataset():
        error("Dataset not found. Please run Option 0 first.")
        pause()
        return

    print("  Auto-picks images from the test folder and shows side-by-side heatmaps.")
    print()

    n_input       = input("  Images per class (default: 2): ").strip()
    num_per_class = int(n_input) if n_input.isdigit() else 2

    info("Generating comparison grid...")
    import compare_gradcam
    compare_gradcam.compare_gradcam(num_per_class=num_per_class)
    pause()


# ── Main menu loop ────────────────────────────────────────────────────────────

MENU_OPTIONS = [
    ("0", "Download dataset from Kaggle",          option_download),
    ("1", "Train the model",                        option_train),
    ("2", "Evaluate model (accuracy + matrix)",     option_evaluate),
    ("3", "Predict a single X-ray image",           option_predict),
    ("4", "Visualize test results (image grid)",    option_visualize),
    ("5a", "Grad-CAM heatmap — single image",       option_gradcam_single),
    ("5b", "Grad-CAM comparison — NORMAL vs PNEUMONIA", option_gradcam_compare),
    ("6", "Generate PDF report",                    option_report),
    ("7", "Export model to TorchScript (for deployment)", option_export),
    ("q", "Quit",                                   None),
]


def print_menu():
    """Print the main menu."""
    print_status()
    print("  What would you like to do?\n")
    for key, label, _ in MENU_OPTIONS:
        print(f"    [{key}]  {label}")
    print()


def main():
    while True:
        clear()
        header()
        print_menu()

        choice = input("  Enter your choice: ").strip().lower()
        print()

        if choice == "q":
            clear()
            print()
            print("  Goodbye! Happy learning. 🫁")
            print()
            sys.exit(0)

        # Find and run the matching option
        matched = False
        for key, label, fn in MENU_OPTIONS:
            if choice == key.lower() and fn is not None:
                fn()
                matched = True
                break

        if not matched:
            print(f"  Invalid choice: '{choice}'. Please enter a number from the menu.")
            pause()


if __name__ == "__main__":
    main()
