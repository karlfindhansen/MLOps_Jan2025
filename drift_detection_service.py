import os
import random
import cv2
from fastapi import FastAPI, UploadFile
from zipfile import ZipFile
import shutil

app = FastAPI()

# Directory paths
REFERENCE_DIR = "data"
DRIFTED_DIR = "drifted_data/drifted_data"


@app.post("/simulate_drift/")
async def simulate_drift(uploaded_file: UploadFile):
    """
    Simulate drift on the uploaded dataset.
    """
    # Save uploaded file
    uploaded_file_path = "uploaded_dataset.zip"
    with open(uploaded_file_path, "wb") as f:
        shutil.copyfileobj(uploaded_file.file, f)

    # Extract the dataset
    with ZipFile(uploaded_file_path, "r") as zip_ref:
        zip_ref.extractall(DRIFTED_DIR)

    # Simulate drift: randomly remove images and apply transformations
    simulate_extinction(DRIFTED_DIR)
    apply_transformations(DRIFTED_DIR)

    return {"message": f"Drift simulated. Modified dataset is available at {DRIFTED_DIR}"}


def simulate_extinction(data_dir):
    """
    Simulate extinction by removing images of certain species, accounting for nested 'data/data/' structure.
    """
    # Check for nested "data" directory
    nested_data_path = os.path.join(data_dir, "data")
    if os.path.exists(nested_data_path) and os.path.isdir(nested_data_path):
        # If nested 'data' exists, update the data_dir to point to it
        data_dir = nested_data_path

    # List of species to remove
    species_to_remove = ["001.Black_footed_Albatross", "002.Laysan_Albatross"]

    # Iterate over species to remove their directories
    for species in species_to_remove:
        species_path = os.path.join(data_dir, species)
        if os.path.exists(species_path):
            shutil.rmtree(species_path)
            print(f"Removed species: {species}")
        else:
            print(f"Species not found: {species}")


def apply_transformations(data_dir):
    """
    Apply transformations like flipping, blurring, or brightness changes to simulate drift.
    """
    for species_dir in os.listdir(data_dir):
        species_path = os.path.join(data_dir, species_dir)
        if os.path.isdir(species_path):
            for image_file in os.listdir(species_path):
                image_path = os.path.join(species_path, image_file)

                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    continue

                # Randomly apply transformations
                if random.random() < 0.5:
                    # Flip image horizontally
                    image = cv2.flip(image, 1)

                if random.random() < 0.5:
                    # Add Gaussian blur
                    image = cv2.GaussianBlur(image, (5, 5), 0)

                if random.random() < 0.5:
                    # Adjust brightness
                    image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)

                # Save the transformed image
                cv2.imwrite(image_path, image)
    print("Applied transformations to dataset.")


@app.post("/detect_drift/")
async def detect_drift(uploaded_reference: UploadFile, uploaded_current: UploadFile):
    """
    Detect drift by comparing the structure and image files of two datasets.
    Handles nested 'data/data' structure.
    """
    # Define paths for extracted directories
    reference_extracted_path = "data"
    current_extracted_path = "drifted_data"

    # Ensure directories are clean before extraction
    if os.path.exists(reference_extracted_path):
        shutil.rmtree(reference_extracted_path)
    if os.path.exists(current_extracted_path):
        shutil.rmtree(current_extracted_path)

    # Save and extract the reference dataset
    reference_zip_path = "reference_dataset.zip"
    with open(reference_zip_path, "wb") as f:
        shutil.copyfileobj(uploaded_reference.file, f)

    with ZipFile(reference_zip_path, "r") as zip_ref:
        zip_ref.extractall(reference_extracted_path)

    # Save and extract the current dataset
    current_zip_path = "current_dataset.zip"
    with open(current_zip_path, "wb") as f:
        shutil.copyfileobj(uploaded_current.file, f)

    with ZipFile(current_zip_path, "r") as zip_ref:
        zip_ref.extractall(current_extracted_path)

    # Resolve nested 'data/data' directories
    reference_resolved_path = resolve_nested_data_dir(reference_extracted_path)
    current_resolved_path = resolve_nested_data_dir(current_extracted_path)

    # Perform drift comparison
    drift_report = compare_datasets(reference_resolved_path, current_resolved_path)

    return {"drift_report": drift_report}


def resolve_nested_data_dir(base_dir):
    """
    Resolves nested directory structures to locate the folder containing dataset classes.
    Handles cases like `data/data` or `data_drifted/drifted_data/data`.
    """
    while True:
        # List all items in the directory
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

        # Check if subdirectories contain class folders (e.g., "001.Black_footed_Albatross")
        if any(subdir.startswith("001.") for subdir in subdirs):
            break

        # If no class folders found, assume it's a nested directory and go deeper
        if len(subdirs) == 1:
            base_dir = os.path.join(base_dir, subdirs[0])
        else:
            break  # Prevent infinite loop if the structure is unexpected

    return base_dir


def compare_datasets(reference_dir, current_dir):
    """
    Compare two datasets to detect drift.
    """
    drift_report = {}

    # Compare number of classes
    reference_classes = set(os.listdir(reference_dir))
    current_classes = set(os.listdir(current_dir))
    print(current_dir)
    print(reference_dir)
    drift_report["missing_classes"] = list(reference_classes - current_classes)
    drift_report["new_classes"] = list(current_classes - reference_classes)

    # Compare number of images per class
    for class_name in reference_classes.intersection(current_classes):
        ref_count = len(os.listdir(os.path.join(reference_dir, class_name)))
        curr_count = len(os.listdir(os.path.join(current_dir, class_name)))
        drift_report[f"{class_name}_image_count_difference"] = ref_count - curr_count

    return drift_report


@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Drift Detection API!"}
