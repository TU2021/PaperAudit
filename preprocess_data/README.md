# Paper Preprocessing Pipeline

This directory contains scripts for downloading, parsing, and preprocessing academic papers from OpenReview to generate synthetic corrupted papers for training corruption detection models. 

The pipeline works as follows: 

(1) **Download papers** from OpenReview API, including PDFs, reviews, and metadata; 

(2) **Filter papers** by page count to remove overly long papers (optional); 

(3) **Parse PDFs** into structured JSON format using LlamaParse and LLM-based OCR correction, extracting text blocks and images; 

(4) **Add section labels** to parsed papers by splitting text based on section boundaries and labeling each block with its section (Abstract, Introduction, Method, Experiments, etc.); 

(5) **Post-clean** the papers by removing headers, footers, and checklist sections; 

(6) **Generate synthetic corruptions** using LLM to create realistic corrupted versions of papers across 8 corruption categories (evidence/data manipulation, method logic flaws, experimental design issues, etc.). The final output consists of both original clean papers and their corrupted versions, which can be used to train and evaluate corruption detection models.

## Pipeline Overview

The preprocessing pipeline consists of 6 main stages, executed in the following order:

```
1. Download Papers          → 2. Filter PDFs (Optional)  → 3. Parse PDFs
     ↓                            ↓                           ↓
download_openreview.py    filter_overlong_pdf.py      parse_paper.py
     ↓                            ↓                           ↓
4. Add Section Labels      → 5. Post-Clean            → 6. Generate Corruptions
     ↓                            ↓                           ↓
add_section.py             filter_parse.py            synth_corruptions_for_detector.py
```

## Scripts Description

### 1. `download_openreview.py` - Download Papers from OpenReview

**Purpose**: Downloads papers, reviews, and metadata from OpenReview for a specified conference.

**Input**: None (downloads from OpenReview API)

**Output**: 
- `{out_dir}/{CONFERENCE}_{YEAR}_{TYPE}/{paper_id}-{paper_title}/`
  - `paper.pdf` - Paper PDF file
  - `reviews.json` - Reviews and discussion threads
  - `metadata.json` - Paper metadata
- `summary.json` - Summary of all downloaded papers

**Required Arguments**:
- `--conference, -c`: Conference name (e.g., ICLR.cc, NeurIPS.cc, ICML.cc)
- `--year, -y`: Conference year (e.g., 2024, 2025)
- `--type, -t`: Paper type (e.g., oral, poster, spotlight)
- `--username, -u`: OpenReview username/email
- `--password, -p`: OpenReview password

**Example**:
```bash
python download_openreview.py -c ICLR.cc -y 2025 -t oral -u your_email@example.com -p your_password
```

---

### 2. `filter_overlong_pdf.py` - Filter PDFs by Page Count (Optional)

**Purpose**: Counts PDF pages and filters papers by page count. Useful for removing overly long papers.

**Input**: Directory containing paper subdirectories (each with `paper.pdf`)

**Output**:
- Filtered papers copied to destination directory
- `pdf_page_report.csv` - CSV report with page counts
- `pdf_page_report.json` - JSON report with summary and histogram

**Required Arguments**:
- `--root`: Root directory containing paper subdirectories

**Optional Arguments**:
- `--max-pages`: Threshold - copy papers with pages < threshold to destination
- `--dest`: Destination directory for filtered papers
- `--dry-run`: Show what would be copied without actually copying

**Example**:
```bash
# Count pages only
python filter_overlong_pdf.py --root /path/to/papers

# Filter papers with less than 30 pages
python filter_overlong_pdf.py --root /path/to/papers --max-pages 30 --dest /path/to/filtered
```

---

### 3. `parse_paper.py` - Parse PDFs to JSON

**Purpose**: Batch parses PDF files using LlamaParse and LLM-based OCR correction. Converts PDFs into structured JSON format with text and images.

**Input**: Directory containing paper subdirectories (each with `paper.pdf`)

**Output**: 
- `paper_parse_origin.json` - Parsed paper with content blocks (text and images)

**Required Arguments**:
- `--root-dir`: Root directory containing paper subdirectories
- `--model`: Model name for OCR correction and image classification

**Required Environment Variables**:
- `LLAMA_API_KEY`: LlamaParse API key
- `OPENAI_API_KEY`: OpenAI API key

**Example**:
```bash
python parse_paper.py --root-dir /path/to/papers --model gpt-5-2025-08-07
```

**Features**:
- OCR correction using LLM
- Image classification (manuscript figures vs. artifacts)
- Parallel processing with resume capability
- Exponential backoff retry

---

### 4. `add_section.py` - Add Section Labels

**Purpose**: Adds section labels to parsed papers. Splits text blocks by section boundaries and labels each block with its section (Abstract, Introduction, Method, etc.).

**Input**: Directory containing `paper_parse_origin.json` files

**Output**: 
- `paper_parse_add_section.json` - Paper with section labels added
- `preseg_*.json` - Intermediate presegmented JSON (optional, if `--save-preseg` is used)

**Required Arguments**:
- `--root-dir`: Root directory, recursively searches for input JSON files
- `--model`: LLM model name for text splitting and section labeling

**Required Environment Variables**:
- `OPENAI_API_KEY`: OpenAI API key

**Example**:
```bash
python add_section.py --root-dir /path/to/papers --model gpt-5-2025-08-07
```

**Features**:
- Text splitting based on section boundaries
- Section labeling (Abstract, Introduction, Method, Experiments, etc.)
- Concurrent execution with resume capability

---

### 5. `filter_parse.py` - Post-Clean Sections

**Purpose**: Performs rule-based post-processing to clean up parsed papers. Removes headers, footers, and checklist sections.

**Input**: Directory containing `paper_parse_add_section.json` files

**Output**: 
- `paper_final.json` - Cleaned paper ready for corruption generation

**Required Arguments**:
- `--root-dir`: Root directory, recursively searches for input JSON files

**Optional Arguments**:
- `--strip-header-footer`: Remove headers and footers (keeps first occurrence)
- `--drop-checklist`: Remove Checklist sections
- `--input-name`: Input JSON filename (default: `paper_parse_add_section.json`)
- `--output-name`: Output JSON filename (default: `paper_final.json`)

**Example**:
```bash
# Basic usage
python filter_parse.py --root-dir /path/to/papers

# With header/footer stripping and checklist removal
python filter_parse.py --root-dir /path/to/papers --strip-header-footer --drop-checklist
```

**Features**:
- Header/footer detection and removal (keeps first occurrence)
- Checklist section removal
- Rebuilds section labels after cleaning

---

### 6. `synth_corruptions_for_detector.py` - Generate Synthetic Corruptions

**Purpose**: Generates synthetic corrupted versions of papers for training/evaluating corruption detection models. Applies realistic corruptions across 8 categories.

**Input**: Directory containing `paper_final.json` files

**Output**: 
- `paper_synth_{model}.json` - Corrupted paper with audit log

**Required Arguments**:
- `--root-dir`: Root directory containing paper subdirectories (each with `paper_final.json`)
- `--model`: LLM model name for generating corruptions

**Required Environment Variables**:
- `OPENAI_API_KEY`: OpenAI API key

**Example**:
```bash
# Basic usage
python synth_corruptions_for_detector.py --root-dir /path/to/papers --model gpt-5-2025-08-07

# Regenerate files with low success rate
python synth_corruptions_for_detector.py --root-dir /path/to/papers --model gpt-5-2025-08-07 --overwrite-apply 10
```

**Features**:
- 8 corruption types (evidence/data integrity, method logic, experimental design, etc.)
- Multimodal support (text + images)
- Concurrent processing with resume capability
- Configurable edit counts and parameters

**Corruption Types**:
1. Evidence/Data Integrity
2. Method/Logic Consistency
3. Experimental Design/Protocol
4. Claim/Interpretation Distortion
5. Reference/Background Fabrication
6. Ethical/Integrity Omission
7. Rhetorical/Presentation Manipulation
8. Context Misalignment/Incoherence

---

## Complete Workflow Example

Here's a complete example workflow from downloading papers to generating corruptions:

```bash
# Step 1: Download papers from OpenReview
python download_openreview.py \
  -c ICLR.cc -y 2025 -t oral \
  -u your_email@example.com -p your_password \
  -o ./downloads

# Step 2 (Optional): Filter papers by page count
python filter_overlong_pdf.py \
  --root ./downloads/ICLR_2025_oral \
  --max-pages 30 \
  --dest ./downloads/ICLR_2025_oral_filtered

# Step 3: Parse PDFs to JSON
export LLAMA_API_KEY="your_llama_key"
export OPENAI_API_KEY="your_openai_key"
python parse_paper.py \
  --root-dir ./downloads/ICLR_2025_oral \
  --model gpt-5-2025-08-07 \
  -j 10

# Step 4: Add section labels
python add_section.py \
  --root-dir ./downloads/ICLR_2025_oral \
  --model gpt-5-2025-08-07 \
  -j 20

# Step 5: Post-clean (remove headers/footers, checklist)
python filter_parse.py \
  --root-dir ./downloads/ICLR_2025_oral \
  --strip-header-footer \
  --drop-checklist

# Step 6: Generate synthetic corruptions
python synth_corruptions_for_detector.py \
  --root-dir ./downloads/ICLR_2025_oral \
  --model gpt-5-2025-08-07 \
  -j 10
```

## File Structure After Processing

After running the complete pipeline, each paper directory will contain:

```
{paper_id}-{paper_title}/
├── paper.pdf                          # Original PDF (from step 1)
├── reviews.json                       # Reviews (from step 1)
├── metadata.json                      # Metadata (from step 1)
├── images/                            # Extracted images (from step 3)
│   ├── image_001.png
│   └── ...
├── paper_parse_origin.json            # Parsed paper (from step 3)
├── paper_parse_add_section.json       # With section labels (from step 4)
├── paper_final.json                   # Cleaned paper (from step 5)
└── paper_synth_{model}.json          # Corrupted version (from step 6)
```

## Notes

- All scripts support **resume capability** - they skip files that have already been processed
- Most scripts support **concurrent processing** via `-j` or `--concurrency` flag
- Scripts use `.inprogress` files to prevent duplicate processing
- Check individual script help messages for detailed parameter descriptions: `python <script>.py --help`

## Dependencies

- Python 3.8+
- Required packages: `openreview`, `tqdm`, `litellm`, `openai`, `pillow`, `pypdf` (or `PyPDF2`)
- Optional: `opencv-python` (for image template matching)
- API keys: `LLAMA_API_KEY`, `OPENAI_API_KEY`

