import os
import json
import logging
from typing import List, Dict, Optional
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_origin_text_by_context(edit_id: int, synth_data: Dict) -> Optional[str]:
    """
    Find the original paragraph text for a given edit_id based on audit_log context.

    Note: context.content_index must be incremented by 1 to match paper.content[index].
    Returns None if required fields are missing or context is empty.
    """
    audit_log = synth_data.get("audit_log", {})
    apply_results = audit_log.get("apply_results", [])

    target_result = next((r for r in apply_results if r.get("id") == edit_id), None)
    if not target_result:
        return None

    context = target_result.get("context")
    if not context:
        return None

    content_index = context.get("content_index")
    if content_index is None:
        return None

    target_index = content_index + 1

    paper = synth_data.get("paper", {})
    content_array = paper.get("content", [])
    if not isinstance(content_array, list):
        return None

    for content_item in content_array:
        if content_item.get("index") == target_index:
            return content_item.get("text")

    return None


def process_single_synth_file(file_path: str) -> List[Dict]:
    """Process one synth JSON file and group extracted edits by origin_text."""
    if not os.path.exists(file_path):
        logger.warning("File not found: %s", file_path)
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            synth_data = json.load(f)
    except Exception as e:
        logger.error("Failed to read JSON %s: %s", file_path, e)
        return []

    if not synth_data.get("synthetic_for_detector", False):
        logger.warning("Not a detector synth file, skipped: %s", file_path)
        return []

    audit_log = synth_data.get("audit_log", {})
    edits = audit_log.get("edits", [])
    if not isinstance(edits, list):
        return []

    grouped_data: Dict[str, List[Dict]] = defaultdict(list)

    for edit in edits:
        edit_id = edit.get("id")
        if edit_id is None:
            continue

        if edit.get("needs_cross_section", True):
            continue

        origin_text = find_origin_text_by_context(edit_id, synth_data)
        if not origin_text:
            continue

        grouped_data[origin_text].append(
            {
                "id": edit_id,
                "corruption_type": edit.get("corruption_type"),
                "target_find": edit.get("target_find"),
                "error_explanation": edit.get("error_explanation"),
                "replacement": edit.get("replacement"),
            }
        )

    return [{"origin_text": k, "errors": v} for k, v in grouped_data.items()]


def process_paper_folders(root_folder: str, output_file: str) -> None:
    """
    Traverse:
      root_folder/
        paper_folder_1/
          *synth*.json
        paper_folder_2/
          *synth*.json

    Aggregate results into a single JSON file.
    """
    if not os.path.isdir(root_folder):
        logger.error("Root directory does not exist: %s", root_folder)
        return

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_files = 0
    success_files = 0
    all_data: List[Dict] = []

    logger.info("Start processing root folder: %s", root_folder)

    for paper_folder_name in os.listdir(root_folder):
        paper_folder_path = os.path.join(root_folder, paper_folder_name)
        if not os.path.isdir(paper_folder_path):
            continue

        logger.info("Entering paper folder: %s", paper_folder_name)

        for file_name in os.listdir(paper_folder_path):
            if not file_name.endswith(".json") or "synth" not in file_name.lower():
                continue

            total_files += 1
            file_path = os.path.join(paper_folder_path, file_name)
            logger.info("Processing file: %s", file_path)

            try:
                result = process_single_synth_file(file_path)
            except Exception as e:
                logger.error("Failed to process %s: %s", file_path, e)
                continue

            if not result:
                continue

            for item in result:
                item["source"] = {
                    "paper_folder": paper_folder_name,
                    "file_name": file_name,
                }
                all_data.append(item)

            success_files += 1

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    logger.info("============================================================")
    logger.info("Done.")
    logger.info("Total synth files scanned: %d", total_files)
    logger.info("Files with extracted data: %d", success_files)
    logger.info("Total extracted groups: %d", len(all_data))
    logger.info("Saved to: %s", output_file)


if __name__ == "__main__":
    root_folder = ""
    output_file = ""
    process_paper_folders(root_folder, output_file)
