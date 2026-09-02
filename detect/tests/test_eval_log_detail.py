import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


DETECT_DIR = Path(__file__).resolve().parents[1]

# eval_log_detail only needs these JSON helpers.  Stub them so this focused unit
# test remains runnable without installing API client dependencies.
utils = types.ModuleType("utils")
utils.load_json = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
utils.save_json = lambda obj, path: Path(path).write_text(json.dumps(obj), encoding="utf-8")
sys.modules["utils"] = utils

spec = importlib.util.spec_from_file_location("eval_log_detail", DETECT_DIR / "eval_log_detail.py")
eval_log_detail = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(eval_log_detail)
extract_gt_items = eval_log_detail.extract_gt_items
extract_matched_flags = eval_log_detail.extract_matched_flags


def applied_error(error_type):
    return {
        "applied": True,
        "corruption_type": error_type,
        "difficulty": "easy",
        "location": "Introduction",
    }


class EvalLogDetailTest(unittest.TestCase):
    def test_nested_apply_results_exclude_unapplied_edits(self):
        synth_obj = {
            "audit_log": {
                "apply_results": [applied_error("applied"), {**applied_error("skipped"), "applied": False}],
                "edits": [applied_error("applied"), applied_error("skipped")],
            }
        }

        self.assertEqual([item["corruption_type"] for item in extract_gt_items(synth_obj)], ["applied"])

    def test_empty_apply_results_do_not_fall_back_to_proposed_edits(self):
        synth_obj = {
            "audit_log": {
                "apply_results": [],
                "edits": [applied_error("proposed-only")],
            }
        }

        self.assertEqual(extract_gt_items(synth_obj), [])

    def test_match_flags_follow_match_order_for_one_based_indices(self):
        eval_obj = {
            "matches": [
                {"gt_index": 1, "matched": True},
                {"gt_index": 2, "matched": False},
                {"gt_index": 3, "matched": True},
            ]
        }

        self.assertEqual(extract_matched_flags(eval_obj, 3), [True, False, True])


if __name__ == "__main__":
    unittest.main()
