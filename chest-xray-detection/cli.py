"""
cli.py — Entry points for the installed command-line tools.

After running `pip install -e .` in this folder, every function here
becomes a shell command you can run from anywhere:

    xray-download      download the Kaggle dataset
    xray-train         train the model
    xray-evaluate      evaluate accuracy, confusion matrix, ROC curve
    xray-predict       predict a single X-ray image
    xray-visualize     show a grid of test predictions
    xray-gradcam       Grad-CAM heatmap for a single image
    xray-compare       compare Grad-CAM heatmaps side-by-side
    xray-report        generate a PDF performance report
    xray-export        export model to TorchScript
    xray-serve         start the FastAPI server
    xray-test-api      validate the running API against test images
    xray-demo          interactive menu (runs all of the above)

You do not need to edit this file. Change settings in config.py.
"""

import sys


def download():
    """xray-download — download the chest-xray-pneumonia dataset from Kaggle."""
    import download_dataset
    download_dataset.download()


def train():
    """xray-train — fine-tune DenseNet121 on the training set."""
    import train as train_module
    train_module.train()


def evaluate():
    """xray-evaluate — run evaluation on the test set and show metrics."""
    import evaluate as eval_module
    eval_module.evaluate()


def predict():
    """
    xray-predict <image_path> — predict NORMAL or PNEUMONIA for one image.

    Example:
        xray-predict data/test/PNEUMONIA/person1_virus_1.jpeg
    """
    import predict as predict_module
    predict_module.main()


def visualize():
    """
    xray-visualize [--num N] [--only-wrong] — show a grid of test predictions.

    Examples:
        xray-visualize
        xray-visualize --num 16
        xray-visualize --only-wrong
    """
    import argparse
    import visualize_results

    parser = argparse.ArgumentParser(description="Visualise test predictions.")
    parser.add_argument("--num",        type=int,           default=12,
                        help="Number of images to display (default: 12)")
    parser.add_argument("--only-wrong", action="store_true",
                        help="Show only incorrect predictions")
    args = parser.parse_args()
    visualize_results.visualize(num_images=args.num, only_wrong=args.only_wrong)


def gradcam():
    """
    xray-gradcam <image_path> — generate a Grad-CAM heatmap for one image.

    Example:
        xray-gradcam data/test/PNEUMONIA/person1_virus_1.jpeg
    """
    import gradcam as gradcam_module
    if len(sys.argv) < 2:
        print("Usage: xray-gradcam <path_to_xray_image>")
        sys.exit(1)
    gradcam_module.run_gradcam(image_path=sys.argv[1])


def compare():
    """
    xray-compare [--num-per-class N] [image ...] — compare Grad-CAM heatmaps.

    Examples:
        xray-compare
        xray-compare --num-per-class 3
        xray-compare img1.jpg img2.jpg img3.jpg
    """
    import argparse
    import compare_gradcam

    parser = argparse.ArgumentParser(description="Compare Grad-CAM heatmaps.")
    parser.add_argument("images",         nargs="*",
                        help="Optional: specific image paths")
    parser.add_argument("--num-per-class", type=int, default=2,
                        help="Images per class when auto-picking (default: 2)")
    args = parser.parse_args()
    compare_gradcam.compare_gradcam(
        image_paths=args.images if args.images else None,
        num_per_class=args.num_per_class,
    )


def report():
    """xray-report — generate a PDF summary report (xray_report.pdf)."""
    import report as report_module
    report_module.generate_report()


def export():
    """xray-export — export the trained model to TorchScript (.pt file)."""
    import export as export_module
    export_module.export()


def serve():
    """
    xray-serve [--host HOST] [--port PORT] — start the FastAPI server.

    Examples:
        xray-serve
        xray-serve --port 9000
    """
    import argparse
    import uvicorn
    from config import API_HOST, API_PORT

    parser = argparse.ArgumentParser(description="Start the X-ray prediction API.")
    parser.add_argument("--host",   default=API_HOST, help=f"Host (default: {API_HOST})")
    parser.add_argument("--port",   type=int, default=API_PORT, help=f"Port (default: {API_PORT})")
    parser.add_argument("--reload", action="store_true",
                        help="Auto-reload on code changes (development only)")
    args = parser.parse_args()

    print(f"Starting API server at http://{args.host}:{args.port}")
    print(f"API docs: http://localhost:{args.port}/docs")
    uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload)


def test_api():
    """
    xray-test-api [--url URL] [--num N] — validate the running API.

    Examples:
        xray-test-api
        xray-test-api --num 20
        xray-test-api --url http://my-server.com
    """
    import argparse
    import test_api as test_module
    from config import API_URL, NUM_TEST_IMAGES

    parser = argparse.ArgumentParser(description="Test the API against test images.")
    parser.add_argument("--url", default=API_URL,
                        help=f"Server URL (default: {API_URL})")
    parser.add_argument("--num", type=int, default=NUM_TEST_IMAGES,
                        help=f"Images per class (default: {NUM_TEST_IMAGES})")
    args = parser.parse_args()
    test_module.run_tests(base_url=args.url, num_per_class=args.num)


def verify():
    """xray-verify — check dataset structure and confirm all paths are correct."""
    import verify_dataset
    success = verify_dataset.verify()
    import sys
    sys.exit(0 if success else 1)


def demo():
    """xray-demo — launch the interactive command-line menu."""
    import demo as demo_module
    demo_module.main()
