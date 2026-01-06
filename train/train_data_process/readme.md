# Training Data Processing Scripts

This directory contains a collection of scripts for processing training data for academic paper error detection. These scripts convert raw synthetic data into formats required by different training frameworks.

## Script Overview

### 1. `extracted_data.py` - Data Extraction Script

Extracts edit information from synthetic data files and groups them by original text paragraphs.

**Features:**
- Traverses paper folders to find JSON files containing `synth`
- Extracts edit records (edits) from each synthetic file
- Finds corresponding original paragraph text based on context information in `audit_log`
- Groups all error edits from the same paragraph together
- Filters out edits that require cross-section checking (`needs_cross_section=True`)

**Usage:**
```python
# Set the following variables in the script:
root_folder = ""  # Root directory path containing paper folders
output_file = ""  # Output JSON file path

process_paper_folders(root_folder, output_file)
```

**Output Format:**
```json
[
  {
    "origin_text": "Original paragraph text",
    "errors": [
      {
        "id": edit_id,
        "corruption_type": "error_type",
        "target_find": "...",
        "error_explanation": "error explanation",
        "replacement": "replacement content"
      }
    ],
    "source": {
      "paper_folder": "paper_folder_name",
      "file_name": "file_name"
    }
  }
]
```

---

### 2. `convert_sft_format.py` - SFT Format Conversion Script

Converts grouped data into Supervised Fine-Tuning (SFT) format training samples.

**Features:**
- Generates training samples with complete instruction prompts for each group
- Builds standardized error detection task prompts
- Formats error information as JSON output
- Generates statistics (sample count, error count, error type distribution, etc.)

**Usage:**
```python
# Set the following variables in the script:
input_file = ""  # Path to grouped data JSON file (output of extracted_data.py)
output_file = ""  # Output path for SFT format data

convert_grouped_to_sft(input_file, output_file)
```

**Output Format:**
```json
[
  {
    "id": "sample_1",
    "instruction": "Complete task instruction prompt...",
    "output": "[{\"error_location\": \"...\", \"error_type\": \"...\", \"error_explanation\": \"...\"}]",
    "metadata": {
      "error_count": number_of_errors,
      "modified_text_length": text_length,
      "error_types": ["list_of_error_types"],
      "source": {...}
    }
  }
]
```

**Error Types:**
- `evidence_data_integrity` - Evidence data integrity
- `method_logic_consistency` - Method logic consistency
- `experimental_design_protocol` - Experimental design protocol
- `claim_interpretation_distortion` - Claim interpretation distortion
- `reference_background_fabrication` - Reference background fabrication
- `ethical_integrity_omission` - Ethical integrity omission
- `rhetorical_presentation_manipulation` - Rhetorical presentation manipulation
- `context_misalignment_incoherence` - Context misalignment incoherence

---

### 3. `convert_alpaca.py` - Alpaca Format Conversion Script

Converts SFT format data into Alpaca format required by the LLaMA-Factory framework.

**Features:**
- Reads SFT format JSON data
- Converts to LLaMA-Factory Alpaca format (instruction/input/output structure)
- Uses SFT instruction as Alpaca instruction
- Uses SFT output as Alpaca output
- Sets input field to empty string

**Usage:**
```python
# Set the following variables in the script:
input_file = ""  # Path to SFT format JSON file (output of convert_sft_format.py)
output_file = ""  # Output path for Alpaca format

convert_to_llamafactory_format(input_file, output_file)
```

**Output Format:**
```json
[
  {
    "instruction": "Task instruction...",
    "input": "",
    "output": "JSON formatted error list"
  }
]
```

---

### 4. `convert_parquet.py` - Parquet Format Conversion Script

Converts SFT format JSON data to Parquet format for training frameworks like VERL.

**Features:**
- Extracts instruction and output from SFT JSON
- Extracts "Text to Review" section from instruction as input field
- Parses JSON format ground truth from output
- Validates data format and completeness
- Saves processed data as Parquet file (with Snappy compression)
- Logs failed samples to a log file

**Usage:**
```python
# Set the following variables in the script:
JSON_INPUT_PATH = ""  # Path to SFT format JSON file
PARQUET_OUTPUT_PATH = ""  # Parquet output file path
OVERWRITE = True  # Whether to overwrite existing output file

json_to_verl_parquet(
    json_input_path=JSON_INPUT_PATH,
    parquet_output_path=PARQUET_OUTPUT_PATH,
    overwrite=OVERWRITE,
)
```

**Output Format:**
The Parquet file contains the following columns:
- `instruction`: Complete instruction text
- `input`: Text to review extracted from instruction
- `ground_truth`: JSON array containing error information list

**Data Validation:**
- Checks required fields (instruction, output)
- Validates that output is a valid JSON array
- Validates that each error entry contains required fields: `error_location`, `error_type`, `error_explanation`
- Failed samples are logged to `failed_samples.json` file

---

## Data Processing Pipeline

Recommended data processing workflow:

```
Raw synthetic data folder
    ↓
[1] extracted_data.py
    ↓
Grouped JSON data
    ↓
[2] convert_sft_format.py
    ↓
SFT format JSON data
    ↓
    ├─→ [3] convert_alpaca.py → LLaMA-Factory format
    └─→ [4] convert_parquet.py → VERL format (Parquet)
```

## Dependencies

```bash
pip install pandas pyarrow
```

## Notes

1. **Data Path Configuration**: Each script requires setting input and output paths in the `if __name__ == "__main__":` section
2. **File Overwrite**: `convert_parquet.py` will overwrite existing output files by default, controlled by the `overwrite` parameter
3. **Error Handling**: All scripts include logging and error handling mechanisms
4. **Encoding Format**: All JSON files use UTF-8 encoding, supporting Chinese characters
5. **Data Validation**: `convert_parquet.py` performs strict data validation, and invalid samples will be skipped and logged

