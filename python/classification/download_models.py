import os
import sys
import urllib.request

# -----------------------------------------------------------------------------
# Configuration: Model filenames and download URLs
# Filenames match the USERCONFIG.py registry from the previous setup.
# -----------------------------------------------------------------------------
MODELS_MAP = {
    # The model lives under tflite/ in that repo; without the subdirectory this
    # URL 404s and the default model silently fails to download.
    "wastenet_model.tflite": "https://github.com/KrisnaSantosa15/wastenet-garbage-classifier/raw/main/tflite/model.tflite",
    "eco_ai_model.tflite": "https://github.com/Eraly-ml/eco_AI/raw/main/model.tflite",
    "recyclable_waste_model.tflite": "https://huggingface.co/Jyoti0815/recyclable-image-detection/resolve/main/waste_model.tflite"
}


def _progress_bar(block_num, block_size, total_size):
    """Callback function to display download progress in terminal."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, (downloaded * 100) / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r Progress: [{percent:6.2f}%] ({mb_downloaded:.2f} MB / {mb_total:.2f} MB)")
    else:
        sys.stdout.write(f"\r Downloaded: {downloaded / (1024 * 1024):.2f} MB")
    sys.stdout.flush()


_DEFAULT_TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def download_models(target_folder: str = _DEFAULT_TARGET):
    """
    Downloads pretrained quantized models to the specified folder.

    Args:
        target_folder (str): Directory path where models will be saved.
    """
    # Create target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)
    abs_path = os.path.abspath(target_folder)
    print(f"Destination Path: {abs_path}\n" + "=" * 60)

    # Set custom User-Agent to avoid HTTP 403 Forbidden errors from remote hosts
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Arduino-UNO-Q-Downloader)')]
    urllib.request.install_opener(opener)

    for filename, url in MODELS_MAP.items():
        destination_file = os.path.join(target_folder, filename)

        # Skip if already downloaded
        if os.path.exists(destination_file):
            print(f"[SKIP] '{filename}' already exists.")
            continue

        print(f"[DOWNLOADING] '{filename}'")
        print(f" Source URL: {url}")

        try:
            urllib.request.urlretrieve(url, destination_file, reporthook=_progress_bar)
            print(f"\n[SUCCESS] Saved to: {destination_file}\n" + "-" * 60)
        except Exception as e:
            print(f"\n[ERROR] Failed to download '{filename}': {e}\n" + "-" * 60)

    print("\nAll model downloads completed.")


if __name__ == "__main__":
    # Specify your target directory here (e.g., '/home/arduino/models' or './models')
    TARGET_DIRECTORY = _DEFAULT_TARGET

    # Allow optional command line argument: python3 download_models.py /path/to/folder
    if len(sys.argv) > 1:
        TARGET_DIRECTORY = sys.argv[1]

    download_models(target_folder=TARGET_DIRECTORY)