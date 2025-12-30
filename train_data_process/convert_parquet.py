import json
import os
from typing import List, Dict, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def load_json_data(json_path: str) -> List[Dict[str, Any]]:

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    raise ValueError("JSON content must be an object or an array of objects.")


def extract_text_to_review(instruction: str) -> str:

    marker = "### Text to Review ###"
    if marker not in instruction:
        return ""

    parts = instruction.split(marker, 1)
    tail = parts[1]
    return tail.split("---", 1)[0].strip()


def parse_ground_truth(output_text: str) -> List[Dict[str, Any]]:

    try:
        gt = json.loads(output_text.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in output: {e}") from e

    if not isinstance(gt, list):
        raise ValueError("Output must be a JSON array.")

    required = {"error_location", "error_type", "error_explanation"}
    for i, err in enumerate(gt):
        if not isinstance(err, dict):
            raise ValueError(f"Each error entry must be an object; got {type(err)} at index {i}.")
        missing = required - set(err.keys())
        if missing:
            raise ValueError(f"Error entry missing fields {sorted(missing)} at index {i}.")

    return gt


def process_single_sample(sample: Dict[str, Any]) -> Dict[str, Any]:

    for field in ("instruction", "output"):
        if field not in sample:
            raise KeyError(f"Missing required field: {field}")

    instruction = str(sample["instruction"]).strip()
    text_to_review = extract_text_to_review(instruction)
    ground_truth = parse_ground_truth(str(sample["output"]))

    return {
        "instruction": instruction,
        "input": text_to_review,
        "ground_truth": ground_truth,
    }


def json_to_verl_parquet(
    json_input_path: str,
    parquet_output_path: str,
    overwrite: bool = False,
    failed_log_path: str = "failed_samples.json",
) -> None:

    if os.path.exists(parquet_output_path) and not overwrite:
        raise FileExistsError(
            f"Output already exists: {parquet_output_path}. Set overwrite=True to replace it."
        )

    print(f"Loading JSON: {json_input_path}")
    original_data = load_json_data(json_input_path)
    print(f"Loaded samples: {len(original_data)}")

    processed_samples: List[Dict[str, Any]] = []
    failed_samples: List[Dict[str, Any]] = []

    for idx, sample in enumerate(original_data, start=1):
        try:
            processed_samples.append(process_single_sample(sample))
        except Exception as e:
            failed_samples.append({"index": idx, "reason": str(e)})
            print(f"Skipping invalid sample (index {idx}): {e}")

    print("\nProcessing summary:")
    print(f"  Valid samples: {len(processed_samples)}")
    print(f"  Invalid samples: {len(failed_samples)}")

    if failed_samples:
        with open(failed_log_path, "w", encoding="utf-8") as f:
            json.dump(failed_samples, f, ensure_ascii=False, indent=2)
        print(f"  Failed sample details saved to: {failed_log_path}")

    if not processed_samples:
        raise ValueError("No valid samples to convert.")

    df = pd.DataFrame(processed_samples)
    table = pa.Table.from_pandas(df, preserve_index=False)

    if parquet_output_path.endswith(".parquet"):
        pq.write_table(table, parquet_output_path, compression="snappy")
    else:
        os.makedirs(parquet_output_path, exist_ok=True)
        pq.write_to_dataset(table, root_path=parquet_output_path, compression="snappy")

    print(f"\nSaved Parquet to: {parquet_output_path}")
    print(f"Columns: {list(df.columns)}")
    print("Preview (first sample):")
    print(f"  instruction length: {len(df.iloc[0]['instruction'])} chars")
    print(f"  input length: {len(df.iloc[0]['input'])} chars")
    print(f"  ground_truth size: {len(df.iloc[0]['ground_truth'])} errors")


if __name__ == "__main__":
    JSON_INPUT_PATH = (
        ""
    )
    PARQUET_OUTPUT_PATH = (
        ""
    )
    OVERWRITE = True

    json_to_verl_parquet(
        json_input_path=JSON_INPUT_PATH,
        parquet_output_path=PARQUET_OUTPUT_PATH,
        overwrite=OVERWRITE,
    )

