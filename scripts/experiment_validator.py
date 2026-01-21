#!/usr/bin/env python
"""Experiment script for tuning poster validation thresholds.

This script processes images from a test directory and outputs detailed
validation results to help determine optimal threshold values.

It compares actual validation results against ground truth expectations
stored in datasets/eval/validator_ground_truth.json.

Usage:
    python scripts/experiment_validator.py
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to import conference_reader package
sys.path.insert(0, str(Path(__file__).parent.parent))

from conference_reader.image_loader import ImageLoader
from conference_reader.extraction import DocumentExtractor, ValidImageConfig


TEST_IMAGE_DIRECTORY = "/data/neurips/invalid_poster_images/"
GROUND_TRUTH_FILE = "datasets/eval/validator_ground_truth.json"

# EXPERIMENT CONFIGURATION - Adjust these values to test different thresholds


VALIDATION_CONFIG = ValidImageConfig(
    min_text_length=900,  # Try: 500, 800, 900, 1000
    require_heading=True,  # Keep enabled
    min_heading_count=2,  # Try: 1, 2, 3 (posters have sections)
    min_children_count=40,  # Try: 25, 40, 60 (structural complexity)
)


def load_ground_truth() -> dict[str, bool]:
    """Load ground truth expectations from JSON file.

    Returns:
        Dictionary mapping filename to expected valid status (True=valid, False=invalid)
        Returns empty dict if file not found or has parsing errors.
    """
    try:
        with open(GROUND_TRUTH_FILE, "r") as f:
            data = json.load(f)
        # Filter out null values (not yet labeled) and metadata fields (starting with _)
        return {k: v for k, v in data.items() if v is not None and not k.startswith("_")}
    except FileNotFoundError:
        print(f"Warning: Ground truth file not found: {GROUND_TRUTH_FILE}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in ground truth file: {e}")
        return {}


def main():
    """Run validation experiment on test images."""
    print("=" * 80)
    print("POSTER VALIDATION EXPERIMENT")
    print("=" * 80)
    print(f"\nTest Directory: {TEST_IMAGE_DIRECTORY}")

    # Load ground truth expectations
    ground_truth = load_ground_truth()
    if ground_truth:
        labeled_count = len(ground_truth)
        print(f"Ground Truth File: {GROUND_TRUTH_FILE} ({labeled_count} labeled)")
    else:
        print(f"Ground Truth File: {GROUND_TRUTH_FILE} (no labels yet)")

    print(f"\nValidation Configuration:")
    print(f"  min_text_length:     {VALIDATION_CONFIG.min_text_length}")
    print(f"  require_heading:     {VALIDATION_CONFIG.require_heading}")
    print(f"  min_heading_count:   {VALIDATION_CONFIG.min_heading_count}")
    print(f"  min_children_count:  {VALIDATION_CONFIG.min_children_count}")
    print("\n" + "=" * 80 + "\n\n")

    # Load images
    try:
        loader = ImageLoader(directory=TEST_IMAGE_DIRECTORY)
        image_paths = loader.get_image_paths()
        print(f"Found {len(image_paths)} image(s)\n")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading images: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract with validation
    extractor = DocumentExtractor(valid_image_config=VALIDATION_CONFIG)
    documents = extractor.extract_batch(image_paths)

    # Track accuracy metrics
    correct_predictions = 0
    incorrect_predictions = []

    # Print results for each document
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {'-'*76}")
        print(f"   Filename:        {doc.filename}")
        print(f"   Success:         {doc.success}")
        print(f"   Error Message:   {doc.error_message}")

        # Check against ground truth if available
        if doc.filename in ground_truth:
            expected_valid = ground_truth[doc.filename]
            actual_valid = doc.success
            match = expected_valid == actual_valid

            if match:
                status = "✓ CORRECT"
                correct_predictions += 1
            else:
                status = "✗ WRONG"
                incorrect_predictions.append({
                    'filename': doc.filename,
                    'expected': expected_valid,
                    'actual': actual_valid,
                    'error_message': doc.error_message
                })

            exp_label = "valid" if expected_valid else "invalid"
            act_label = "valid" if actual_valid else "invalid"
            print(f"   Expected:        {exp_label}")
            print(f"   Actual:          {act_label}")
            print(f"   Result:          {status}")

        # Show first 20 chars of extracted text
        preview = doc.extracted_text[:20] if doc.extracted_text else "[EMPTY]"
        preview = preview.replace("\n", "\\n")  # Show newlines visibly
        print(f"   Text Preview:    {preview}...")

        # Show quality metadata if available
        if doc.quality_grade:
            print(f"   Quality Grade:   {doc.quality_grade}")
        if doc.quality_score is not None:
            print(f"   Quality Score:   {doc.quality_score:.3f}")
        if doc.low_quality_grade:
            print(f"   Low Qual Grade:  {doc.low_quality_grade}")
        if doc.low_score is not None:
            print(f"   Low Score:       {doc.low_score:.3f}")
        if doc.ocr_score is not None:
            print(f"   OCR Score:       {doc.ocr_score:.3f}")
        if doc.layout_score is not None:
            print(f"   Layout Score:    {doc.layout_score:.3f}")

        print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    successful = [d for d in documents if d.success]
    failed = [d for d in documents if not d.success]

    print(f"Total images:      {len(documents)}")
    print(f"Valid posters:     {len(successful)}")
    print(f"Invalid/Failed:    {len(failed)}")

    if failed:
        print(f"\nFailed images:")
        for doc in failed:
            print(f"  - {doc.filename}")
            print(f"    Reason: {doc.error_message}")

    # Print accuracy report if ground truth labels exist
    if ground_truth:
        print("\n" + "=" * 80)
        print("ACCURACY REPORT")
        print("=" * 80)

        total_labeled = len(ground_truth)
        accuracy = (correct_predictions / total_labeled * 100) if total_labeled > 0 else 0

        # Calculate confusion matrix values
        true_positives = 0   # Expected valid, got valid
        true_negatives = 0   # Expected invalid, got invalid
        false_positives = 0  # Expected invalid, got valid (false correct)
        false_negatives = 0  # Expected valid, got invalid (false error)

        false_correct_files = []  # False positives
        false_error_files = []    # False negatives

        for doc in documents:
            if doc.filename in ground_truth:
                expected = ground_truth[doc.filename]
                actual = doc.success

                if expected and actual:
                    true_positives += 1
                elif not expected and not actual:
                    true_negatives += 1
                elif not expected and actual:
                    false_positives += 1
                    false_correct_files.append(doc.filename)
                elif expected and not actual:
                    false_negatives += 1
                    false_error_files.append(doc.filename)

        print(f"\nConfusion Matrix:")
        print(f"                    Predicted Valid    Predicted Invalid")
        print(f"  Actually Valid    {true_positives:6d}             {false_negatives:6d}")
        print(f"  Actually Invalid  {false_positives:6d}             {true_negatives:6d}")

        print(f"\nMetrics:")
        print(f"Total labeled:         {total_labeled}")
        print(f"Correct predictions:   {correct_predictions}")
        print(f"Incorrect predictions: {len(incorrect_predictions)}")
        print(f"Accuracy:              {accuracy:.1f}%")

        if incorrect_predictions:
            print(f"\nMisclassification Details:")

            if false_correct_files:
                print(f"\n  False Correct (Expected INVALID, Got VALID):")
                for filename in false_correct_files:
                    print(f"    - {filename}")

            if false_error_files:
                print(f"\n  False Error (Expected VALID, Got INVALID):")
                for filename in false_error_files:
                    # Find the error message
                    error_msg = next(
                        (item['error_message'] for item in incorrect_predictions
                         if item['filename'] == filename),
                        ""
                    )
                    print(f"    - {filename}")
                    if error_msg:
                        print(f"      Reason: {error_msg}")
        else:
            print(f"\n✓ Perfect accuracy! All labeled images classified correctly.")


if __name__ == "__main__":
    main()
