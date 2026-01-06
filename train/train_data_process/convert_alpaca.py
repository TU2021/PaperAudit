import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def convert_to_llamafactory_format(input_file: str, output_file: str) -> None:
    """
    Convert SFT JSON data into LLaMA-Factory Alpaca format:

    [
      {
        "instruction": "...",
        "input": "",
        "output": "..."
      }
    ]
    """
    try:
        logger.info("Loading SFT data: %s", input_file)
        with open(input_file, "r", encoding="utf-8") as f:
            sft_data = json.load(f)

        logger.info("Loaded %d samples", len(sft_data))

        llamafactory_data = [
            {
                "instruction": sample["instruction"],
                "input": "",
                "output": sample["output"],
            }
            for sample in sft_data
        ]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(llamafactory_data, f, ensure_ascii=False, indent=2)

        logger.info("LLaMA-Factory data saved to: %s", output_file)
        logger.info("Total samples: %d", len(llamafactory_data))

    except Exception as e:
        logger.error("Conversion failed: %s", str(e))
        raise


if __name__ == "__main__":
    input_file = ""
    output_file = ""
    convert_to_llamafactory_format(input_file, output_file)
