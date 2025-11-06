"""
Test script to fetch and analyze the AgentNet dataset structure from Hugging Face.
This will help us understand what data structure we need to replicate for SN24 revamp.
"""

from huggingface_hub import hf_hub_download, list_repo_files
import json
from pprint import pprint


def analyze_agentnet_dataset():
    """Fetch and analyze the AgentNet dataset structure."""

    print("=" * 80)
    print("FETCHING AGENTNET DATASET FROM HUGGING FACE")
    print("=" * 80)

    try:
        # First, list all files in the repo to find JSONL files
        print("\n[1/6] Listing files in xlangai/AgentNet repository...")
        repo_files = list_repo_files("xlangai/AgentNet", repo_type="dataset")
        repo_files = list(repo_files)

        print(f"\nFound {len(repo_files)} total files in repository")
        print("\nAll files:")
        for f in repo_files[:20]:
            print(f"  - {f}")
        if len(repo_files) > 20:
            print(f"  ... and {len(repo_files) - 20} more")

        jsonl_files = [f for f in repo_files if f.endswith(".jsonl")]

        print(f"\nFound {len(jsonl_files)} JSONL files:")
        for f in jsonl_files:
            print(f"  - {f}")

        # Download and read first JSONL file (Ubuntu 5k dataset)
        if jsonl_files:
            # Let's start with the smaller Ubuntu dataset
            target_file = "agentnet_ubuntu_5k.jsonl"
            print(f"\n[2/6] Downloading {target_file}...")
            jsonl_path = hf_hub_download(
                repo_id="xlangai/AgentNet", filename=target_file, repo_type="dataset"
            )

            print(f"\n[3/6] Reading JSONL file...")
            examples = []
            with open(jsonl_path, "r") as f:
                for i, line in enumerate(f):
                    if i >= 3:  # Only read first 3 examples
                        break
                    examples.append(json.loads(line))
                    print(f"   - Loaded example {i + 1}")

            print(f"\n[4/6] Analyzing dataset structure...")
            print("\n" + "=" * 80)
            print("DATASET SCHEMA AND FIELDS")
            print("=" * 80)

            if examples:
                first_example = examples[0]
                print(f"\nTotal fields: {len(first_example.keys())}")
                print(f"\nField names: {sorted(first_example.keys())}")

                print("\n" + "=" * 80)
                print("COLUMN DETAILS")
                print("=" * 80)

                for key in sorted(first_example.keys()):
                    value = first_example[key]
                    print(f"\n  • {key}")
                    print(f"    Type: {type(value).__name__}")

                    # Show sample value
                    if isinstance(value, str):
                        preview = value[:200] if len(value) > 200 else value
                        print(f"    Sample: {preview}")
                        if len(value) > 200:
                            print(f"    (truncated from {len(value)} chars)")
                    elif isinstance(value, list):
                        print(f"    Length: {len(value)} items")
                        if len(value) > 0:
                            print(f"    First item type: {type(value[0]).__name__}")
                            print(f"    First item sample: {str(value[0])[:200]}")
                    elif isinstance(value, dict):
                        print(f"    Keys: {list(value.keys())}")
                    elif isinstance(value, (int, float, bool)):
                        print(f"    Value: {value}")
                    else:
                        print(f"    Sample: {str(value)[:200]}")

                print("\n" + "=" * 80)
                print("DETAILED EXAMPLES (First 2 entries)")
                print("=" * 80)

                for idx in range(min(2, len(examples))):
                    print(f"\n{'=' * 40}")
                    print(f"EXAMPLE {idx + 1}")
                    print(f"{'=' * 40}")

                    example = examples[idx]
                    for key in sorted(example.keys()):
                        print(f"\n[{key}]")
                        value = example[key]

                        if value is None:
                            print("  (null)")
                        elif isinstance(value, str):
                            if len(value) > 1000:
                                print(f"{value[:1000]}...")
                                print(f"[truncated, total length: {len(value)}]")
                            else:
                                print(value)
                        elif isinstance(value, list):
                            print(f"List with {len(value)} items:")
                            if len(value) <= 3:
                                pprint(value, indent=4)
                            else:
                                print("First 3 items:")
                                pprint(value[:3], indent=4)
                                print(f"... and {len(value) - 3} more items")
                        elif isinstance(value, dict):
                            pprint(value, indent=4)
                        else:
                            print(value)

                # Analyze trajectory structure if present
                print("\n" + "=" * 80)
                print("TRAJECTORY STRUCTURE ANALYSIS")
                print("=" * 80)

                if "traj" in first_example:
                    print("\n'traj' field found! Analyzing structure...")
                    first_traj = first_example["traj"]

                    if isinstance(first_traj, list) and len(first_traj) > 0:
                        print(f"\nTrajectory is a list with {len(first_traj)} steps")
                        print("\nFirst step structure:")
                        pprint(first_traj[0], indent=2)

                        # Check if all steps have same keys
                        if isinstance(first_traj[0], dict):
                            print(f"\nStep keys: {list(first_traj[0].keys())}")

                            # Analyze the 'value' field if present
                            if "value" in first_traj[0] and isinstance(
                                first_traj[0]["value"], dict
                            ):
                                print(
                                    f"\nValue object keys: {list(first_traj[0]['value'].keys())}"
                                )

                print("\n" + "=" * 80)
                print("ANALYSIS COMPLETE")
                print("=" * 80)
                print(
                    "\nThis structure will guide our SN24 revamp to collect similar data."
                )

        else:
            print("\n❌ No JSONL files found in the repository")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    analyze_agentnet_dataset()
