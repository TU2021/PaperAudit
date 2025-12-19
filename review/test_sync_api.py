"""
Example usage of the new synchronous review_paper API.

This demonstrates how to use the S1Agent's synchronous method
to review a paper and get structured results.
"""

from agents import S1Agent
import json
import glob
from dotenv import load_dotenv; load_dotenv()

def test_sync_api(agent: S1Agent, pdf_file: str):
    
    # Query for the review
    query = "Review the paper."

    try:
        # Call the synchronous review method
        result = agent.review_paper(pdf_file, query)
        return result
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Review failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    agent = S1Agent()

    pdf_files = sorted(glob.glob("data/*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/ directory.")

    for pdf_file in pdf_files:
        result = test_sync_api(agent, pdf_file)
        print(f"\nResults for {pdf_file}:")
        print(json.dumps(result, indent=2))
        break
        print("\n" + "=" * 80 + "\n")
