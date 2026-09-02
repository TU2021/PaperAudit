import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mas_reference_helper import (
    _build_references_text,
    _extract_citation_keys_from_section_blocks,
    _match_reference_entries,
)


class ReferenceHelperTest(unittest.TestCase):
    def test_keeps_author_year_suffixes_and_unicode_surnames(self):
        blocks = [{
            "type": "text",
            "text": "(Liu et al., 2023a; Liu et al. 2023b; Veličković et al., 2021)",
        }]

        author_year, numeric = _extract_citation_keys_from_section_blocks(blocks)

        self.assertEqual(numeric, [])
        self.assertEqual(author_year, [("liu", "2023a"), ("liu", "2023b"), ("velickovic", "2021")])

    def test_matches_first_author_and_exact_year_suffix(self):
        ref_text = "\n\n".join([
            "Guo, Y., Liu, J., and Du, J. Unrelated paper. 2023.",
            "Liu, H. Correct a paper. 2023a.",
            "Liu, J. Correct b paper. 2023b.",
        ])

        matched = _match_reference_entries(ref_text, [("liu", "2023a"), ("liu", "2023b")], [])

        self.assertEqual(matched, [
            "Liu, H. Correct a paper. 2023a.",
            "Liu, J. Correct b paper. 2023b.",
        ])

    def test_reference_blocks_remain_separate_entries(self):
        ref_text = _build_references_text([
            {"type": "text", "text": "Hamilton, W. GraphSAGE. 2017."},
            {"type": "text", "text": "Hassani, K. MVGRL. 2020."},
        ])

        self.assertEqual(
            _match_reference_entries(ref_text, [("hamilton", "2017")], []),
            ["Hamilton, W. GraphSAGE. 2017."],
        )


if __name__ == "__main__":
    unittest.main()
