import json
import logging
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_sft_samples(grouped_data: List[Dict]) -> List[Dict]:
    """
    Convert grouped data into SFT samples.

    Input: origin_text (paragraph text) + errors for that paragraph
    Output: a list of SFT samples (instruction + output JSON)
    """
    sft_samples: List[Dict] = []
    sample_id = 0

    for group in grouped_data:
        modified_text = group.get("origin_text", "")
        errors = group.get("errors", [])

        if not modified_text or not errors:
            logger.warning("Skipping empty group")
            continue

        sample_id += 1
        error_count = len(errors)

        # Build SFT instruction prompt.
        input_text = f"""### Task: Academic Paper Error Detection ###

You are an expert academic paper reviewer. Your task is to carefully analyze the following text from a research paper and identify ALL potential scientific errors, methodological flaws, or integrity issues.

**Important**: The text below may contain multiple errors. You must detect and report each error independently.

---

### Text to Review ###
{modified_text}

---

### Instructions ###
Analyze the text above and identify all errors. For each error found, provide:
1. **error_location**: The exact erroneous content that needs correction (the problematic text segment)
2. **error_type**: Type of error (choose from: evidence_data_integrity | method_logic_consistency | experimental_design_protocol | claim_interpretation_distortion | reference_background_fabrication | ethical_integrity_omission | rhetorical_presentation_manipulation | context_misalignment_incoherence)
3. **error_explanation**: Detailed explanation of why this is an error and its impact on academic integrity

**Output Format**: Return a JSON array containing all detected errors. If multiple errors exist, list them all.

Example output structure:
[
  {{
    "error_location": "specific problematic text segment",
    "error_type": "method_logic_consistency",
    "error_explanation": "detailed explanation..."
  }},
  {{
    "error_location": "another problematic segment",
    "error_type": "rhetorical_presentation_manipulation",
    "error_explanation": "another explanation..."
  }}
]"""

        # Build SFT output JSON.
        output_errors = []
        for error in errors:
            output_errors.append(
                {
                    # NOTE: The upstream data uses `replacement` as the erroneous segment to locate.
                    # If your schema differs, adjust this mapping accordingly.
                    "error_location": error.get("replacement", ""),
                    "error_type": error.get("corruption_type", ""),
                    "error_explanation": error.get("error_explanation", ""),
                }
            )

        output_text = json.dumps(output_errors, ensure_ascii=False, indent=2)

        sft_samples.append(
            {
                "id": f"sample_{sample_id}",
                "instruction": input_text,
                "output": output_text,
                "metadata": {
                    "error_count": error_count,
                    "modified_text_length": len(modified_text),
                    "error_types": [e.get("corruption_type") for e in errors],
                    "source": group.get("source", {}),
                },
            }
        )

        if sample_id % 100 == 0:
            logger.info("Generated %d SFT samples", sample_id)

    logger.info("Total SFT samples generated: %d", len(sft_samples))
    return sft_samples


def convert_grouped_to_sft(input_file: str, output_file: str) -> None:
    """Load grouped JSON and write SFT training data JSON."""
    try:
        logger.info("Loading grouped data: %s", input_file)
        with open(input_file, "r", encoding="utf-8") as f:
            grouped_data = json.load(f)

        logger.info("Loaded %d groups", len(grouped_data))

        sft_samples = build_sft_samples(grouped_data)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(sft_samples, f, ensure_ascii=False, indent=2)

        logger.info("SFT data saved to: %s", output_file)

        total_samples = len(sft_samples)
        if total_samples == 0:
            logger.warning("No samples produced; skipping stats.")
            return

        total_errors = sum(sample["metadata"]["error_count"] for sample in sft_samples)
        multi_error_samples = sum(
            1 for sample in sft_samples if sample["metadata"]["error_count"] > 1
        )

        error_type_counts: Dict[str, int] = {}
        for sample in sft_samples:
            for etype in sample["metadata"]["error_types"]:
                error_type_counts[etype] = error_type_counts.get(etype, 0) + 1

        logger.info("=== Summary ===")
        logger.info("Total samples: %d", total_samples)
        logger.info("Total errors: %d", total_errors)
        logger.info("Avg errors/sample: %.2f", total_errors / total_samples)
        logger.info(
            "Multi-error samples: %d (%.1f%%)",
            multi_error_samples,
            (multi_error_samples / total_samples) * 100.0,
        )

        logger.info("Error type distribution:")
        for etype, count in sorted(
            error_type_counts.items(), key=lambda x: x[1], reverse=True
        ):
            logger.info("  %s: %d", etype, count)

    except Exception as e:
        logger.error("Conversion failed: %s", str(e))
        raise


if __name__ == "__main__":
    input_file = ""
    output_file = ""
    convert_grouped_to_sft(input_file, output_file)
