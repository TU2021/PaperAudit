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
    def test_single_author_and_narrative_citations(self):
        keys, _ = _extract_citation_keys_from_section_blocks([{
            "type": "text", "text": "Malladi et al. (2023); Dayi & Chen (2024); (Olson, 1965); (Meta, 2024)",
        }])
        self.assertEqual(keys, [("malladi", "2023"), ("dayi", "2024"), ("olson", "1965"), ("meta", "2024")])

    def test_natural_order_names_and_organization(self):
        entries = ["Nicholas Carlini and David Wagner. Title. 2024.",
                   "J. R. Smith & Jane Doe. Another title. 2023.",
                   "Meta. Llama 3.2. 2024."]
        self.assertEqual(_match_reference_entries("\n\n".join(entries),
                         [("carlini", "2024"), ("smith", "2023"), ("meta", "2024")], []), entries)

    def test_never_falls_back_to_coauthor(self):
        for entry, key in [
            ("Guo, Y., Liu, J. Unrelated paper. 2023.", ("liu", "2023")),
            ("John Smith and David Wagner. Unrelated paper. 2024.", ("wagner", "2024")),
            ("John Smith. Understanding Wagner. 2024.", ("wagner", "2024")),
        ]:
            with self.subTest(entry=entry):
                self.assertEqual(_match_reference_entries(entry, [key], []), [])

    def test_wrapped_author_list_is_not_a_new_entry(self):
        entry = "Guo, Y.,\nLiu, J., and Smith, A. A complete paper. 2023."
        self.assertEqual(_match_reference_entries(entry, [("guo", "2023")], []), [entry])
        self.assertEqual(_match_reference_entries(entry, [("liu", "2023")], []), [])

    def test_adjacent_entries_split_without_blank_line(self):
        first = "Guo, Y. First paper. 2023."
        second = "Liu, J. Second paper. 2024."
        self.assertEqual(_match_reference_entries(first + "\n" + second,
                         [("guo", "2023"), ("liu", "2024")], []), [first, second])

    def test_author_year_and_numeric_match_deduplicated(self):
        entry = "[1] Meta. Llama 3.2. 2024."
        self.assertEqual(_match_reference_entries(entry, [("meta", "2024")], ["1"]), [entry])

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
