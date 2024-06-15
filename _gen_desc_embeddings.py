import os
import json
from typing import List
import huggingface_hub
import random
import time
import ulid
from datasets import load_dataset
from tempfile import TemporaryDirectory
from openai import OpenAI
from _desc_mlp import check_desc_embedding_against_MLP

HF_DATASET = "omegalabsinc/omega-multimodal"
DATA_FILES_PREFIX = "default/train/"
MAX_FILES = 3
MAX_DESCRIPTIONS_TO_TEST = 1000
CACHE_FILE = "desc_embeddings_recent.json"
MIN_AGE = 48 * 60 * 60  # 48 hours

openAIClient = OpenAI()

def pull_and_cache_descriptions() -> List[str]:
    # Get the list of files in the dataset repository
    omega_ds_files = huggingface_hub.repo_info(repo_id=HF_DATASET, repo_type="dataset").siblings
    
    # Filter files that match the DATA_FILES_PREFIX
    matching_files = [
        f.rfilename
        for f in omega_ds_files
        if f.rfilename.startswith(DATA_FILES_PREFIX)
    ]
    #print("matching_files:", matching_files)
    
    # Randomly sample up to MAX_FILES from the matching files
    sampled_files = random.sample(matching_files, min(MAX_FILES, len(matching_files)))
    #print("sampled_files:", sampled_files)
    
    # Load the dataset using the sampled files
    desc_embeds = []
    with TemporaryDirectory() as temp_dir:
        omega_dataset = load_dataset(HF_DATASET, data_files=sampled_files, cache_dir=temp_dir)["train"]
        for entry in omega_dataset:
            desc_embed = []
            if "description" in entry and "description_embed" in entry:
                desc_embed.append(entry["description"])
                desc_embed.append(entry["description_embed"])
                desc_embeds.append(desc_embed)
    
    # Cache the descriptions to a local file
    with open(CACHE_FILE, "w") as f:
        json.dump(desc_embeds, f)
    
    return desc_embeds

def get_timestamp_from_filename(filename: str):
    return ulid.from_str(os.path.splitext(filename.split("/")[-1])[0]).timestamp().timestamp

def pull_and_cache_recent_descriptions() -> List[str]:
    # Get the list of files in the dataset repository
    omega_ds_files = huggingface_hub.repo_info(repo_id=HF_DATASET, repo_type="dataset").siblings
    
    # Filter files that match the DATA_FILES_PREFIX
    recent_files = [
        f.rfilename
        for f in omega_ds_files if
        f.rfilename.startswith(DATA_FILES_PREFIX) and 
        time.time() - get_timestamp_from_filename(f.rfilename) < MIN_AGE
    ][:MAX_FILES]
    
    # Randomly sample up to MAX_FILES from the matching files
    sampled_files = random.sample(recent_files, min(MAX_FILES, len(recent_files)))
    
    # Load the dataset using the sampled files
    desc_embeds = []
    with TemporaryDirectory() as temp_dir:
        omega_dataset = load_dataset(HF_DATASET, data_files=sampled_files, cache_dir=temp_dir)["train"]
        for entry in omega_dataset:
            desc_embed = []
            if "description" in entry and "description_embed" in entry:
                desc_embed.append(entry["description"])
                desc_embed.append(entry["description_embed"])
                desc_embeds.append(desc_embed)
    
    # Cache the descriptions to a local file
    with open(CACHE_FILE, "w") as f:
        json.dump(desc_embeds, f)
    
    return desc_embeds

def load_cached_descriptions() -> List[str]:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            descriptions = json.load(f)
            # Ensure we don't try to sample more than the available descriptions
            num_to_sample = min(MAX_DESCRIPTIONS_TO_TEST, len(descriptions))
            return random.sample(descriptions, num_to_sample)
    return []
        
def analyze_description(description: str) -> str:
    examples = """
    Description: "A beautiful sunset over the mountains."
    Analysis: VALID

    Description: "programming generate example video image description"
    Analysis: RANDOM

    Description: "Learn how to investigate the effect of varying temperature on rate of photosynthesis"
    Analysis: VALID

    Description: "live string video image core social videos"
    Analysis: RANDOM
    """

    prompt = f"I need your assistance determining if descriptions of videos are valid or not. In some cases, people put arbitrary (but high scoring) words together that seem random, and in other cases the descriptions are accurate. Here are some examples:\n{examples}\n Based on that, determine if the following is valid or random, and return a single token \"VALID\" or \"RANDOM\" based on your analysis: {description}"
    
    response = openAIClient.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        n=1,
        temperature=0.5,
        max_tokens=1
    )
    return response.choices[0].message.content.strip("\"").strip("'")
        
def main():
    #pull_and_cache_descriptions()
    #pull_and_cache_recent_descriptions()

    desc_embeds = load_cached_descriptions()
    print("total desc_embeds:", len(desc_embeds))
    if desc_embeds is None:
        print("No matching files found.")
        return
    
    # run analysis on each description with MLP model
    count = 1
    for desc_embed in desc_embeds:
        print(f"Description: {desc_embed[0]}")
        result = check_desc_embedding_against_MLP(desc_embed[1])
        print("")

        #if count == 10:
            #break
        count += 1
    
    """
    # run analysis on each description by calling ChatGPT4
    count = 1
    for desc_embed in desc_embeds:
        result = analyze_description(desc_embed[0])
        print(f"Description: {desc_embed[0]}\nAnalysis: {result}\n")
        if result == "VALID":
            label = 1
        elif result == "RANDOM":
            label = 0
        else:
            label = 1
        
        desc_embed.append(label)

        #if count == 10:
            #break
        count += 1

    # Rewrite the JSON file with the updated data
    with open('desc_embeddings_scored.json', 'w') as f:
        json.dump(desc_embeds, f, indent=4)
    """

if __name__ == "__main__":
    main()