import importlib.util
import sys
import types
import unittest
from pathlib import Path


DETECT_DIR = Path(__file__).resolve().parents[1]

prompts = types.ModuleType("prompts")
prompts.PromptTemplates = object
sys.modules["prompts"] = prompts

utils = types.ModuleType("utils")
utils.call_llm_chat_with_empty_retries = lambda *args, **kwargs: None
utils.call_web_search_via_tool = lambda *args, **kwargs: None
utils.extract_json_from_text = lambda text: text
utils.load_json = lambda path: None
utils.save_json = lambda obj, path: None
sys.modules["utils"] = utils

spec = importlib.util.spec_from_file_location("agents", DETECT_DIR / "agents.py")
agents = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["agents"] = agents
spec.loader.exec_module(agents)
slice_json_for_task_with_outline = agents.slice_json_for_task_with_outline


class SectionSlicingTest(unittest.TestCase):
    def test_collects_all_disjoint_ranges_for_repeated_section_title(self):
        blocks = [
            {"type": "text", "text": "x", "content_index": index}
            for index in range(1, 24)
        ]
        outline = [
            {"title": "Preliminaries", "start_index": 6, "end_index": 14},
            {"title": "Method", "start_index": 15, "end_index": 15},
            {"title": "Preliminaries", "start_index": 16, "end_index": 16},
            {"title": "Method", "start_index": 17, "end_index": 22},
        ]

        sliced = slice_json_for_task_with_outline(blocks, "Method", outline, max_chars=100)

        self.assertEqual([block["content_index"] for block in sliced], [15, 17, 18, 19, 20, 21, 22])


if __name__ == "__main__":
    unittest.main()
