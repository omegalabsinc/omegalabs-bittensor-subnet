import os
import json
from typing import List
import huggingface_hub
import random
import time
import ulid
from datasets import load_dataset
from tempfile import TemporaryDirectory
import openai
from openai import OpenAI
#from _desc_mlp import check_desc_embedding_against_MLP
from omega.imagebind_desc_mlp import is_desc_embedding_valid

HF_DATASET = "omegalabsinc/omega-multimodal"
DATA_FILES_PREFIX = "default/train/"
MAX_FILES = 3
MAX_DESCRIPTIONS_TO_TEST = 10000
CACHE_FILE = "desc_embeddings_recent.json"
#CACHE_FILE = "desc_embeddings_scored.json"
#CACHE_FILE = "desc_embeddings_openai_scored.json"
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
            print("total descriptions on file:", len(descriptions))
            num_to_sample = min(MAX_DESCRIPTIONS_TO_TEST, len(descriptions))
            return random.sample(descriptions, num_to_sample)
    return []
        
def analyze_description(description: str) -> str:
    examples = """
    Description: "a close up of a video game with a tank and some vehicles in the snow and buildings in the background and a sky a close up of a screen shot of some characters in the game, with different colors and textures on them, including blue"
    Analysis: 5

    Description: "complaints learn approximately professional hire announced temporarily list thumbnail title video image"
    Analysis: 1

    Description: "Improving Student Motivation to Encourage Self-Regulated Learners"
    Analysis: 3

    Description: "Victoria 3 Dev Diary #57&58 - The Journey So Far & Interest Revisions Ð¾ *) screenshot !@ thumbnail ;-) ê³ ~" < so ft ver to $. factual coming .# dra "[ the"
    Analysis: 2

    Description: "A beautiful sunset over the mountains."
    Analysis: 4

    Description: "accommodations indoor rica post resort view outdoor photos inspiring pics added internal childrens another name jungle adventure exterior travel"
    Analysis: 1

    Description: "cartoon of a man holding a pen and writing on a piece of paper with the words 'i am sorry, you are not going to get paid' drawing of a tall building with trees and clouds in the background, with a ruler on top of it, next to a tree"
    Analysis: 5

    Description: "Commonly Mispronounced English Daily English Words | How To Pronounce Correctly? Nysha #shorts a woman is talking about subscribing to youtube"
    Analysis: 4

    Description: "live string video image core social videos"
    Analysis: 1

    Description: "Recent Trends on Renewable enery, Smart grid and Electric Vehicle Technology"
    Analysis: 3

    Description: "Understanding Jungian Archetypesarchives archi archival thoven archers"
    Analysis: 2

    Description: "6 Innovations That Will Change HVAC Forever | #hvac #hvaclife #heatingservices #coolingsystemchanger improves heating newest innovators"
    Analysis: 4
    """

    prompt = f"I need your assistance determining the quality of video descriptions. In some cases, people put arbitrary (but high scoring) words together that seem random, and in other cases the descriptions are coherent and accurate. Here are some examples:\n{examples}\n Based on that, determine a score where 5 is the best valid description and 1 is a terrible/random description. If a description reads poorly it should be rated a 1 or 2. Return a single integer token between 1 and 5 based on your analysis: {description}"
    
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

def generate_openai_desc_embeds(description: str) -> str:
    response = openAIClient.embeddings.create(
        input=description,
        model="text-embedding-3-small",
        dimensions=1024
    )
    embedding = response.data[0].embedding
    return embedding
        
def main():
    #pull_and_cache_descriptions()
    #pull_and_cache_recent_descriptions()

    desc_embeds = load_cached_descriptions()
    print("total desc_embeds:", len(desc_embeds))
    if desc_embeds is None:
        print("No matching files found.")
        return
    
    """
    # generate OpenAI embeddings for each description and save to local json
    count = 1
    openai_desc_embeds = []
    for desc_embed in desc_embeds:
        embedding = generate_openai_desc_embeds(desc_embed[0])
        #print(f"Description: {desc_embed[0]}\nOpen AI Embedding: {embedding}\n")
        print(f"OpenAI embedding generated for description {count}.")
        openai_desc_embed = []
        openai_desc_embed.append(desc_embed[0])
        openai_desc_embed.append(embedding)
        openai_desc_embed.append(desc_embed[2])

        openai_desc_embeds.append(openai_desc_embed)

        #if count == 1:
            #break
        count += 1

    # Rewrite the JSON file with the updated data
    with open('desc_embeddings_openai_scored.json', 'w') as f:
        json.dump(openai_desc_embeds, f, indent=4)
    """
    
    # run analysis on each description with MLP model
    count = 1
    one_count = 0
    two_count = 0
    three_count = 0
    four_count = 0
    five_count = 0
    for desc_embed in desc_embeds:
        print(f"Description: {desc_embed[0]}")
        #result = check_desc_embedding_against_MLP(desc_embed[1])
        result = is_desc_embedding_valid(desc_embed[1])
        if result == 1:
            one_count += 1
        elif result == 2:
            two_count += 1
        elif result == 3:
            three_count += 1
        elif result == 4:
            four_count += 1
        elif result == 5:
            five_count += 1
        print("")

        #if count == 10:
            #break
        count += 1

    print(f"Total scored 1: {one_count}")
    print(f"Total scored 2: {two_count}")
    print(f"Total scored 3: {three_count}")
    print(f"Total scored 4: {four_count}")
    print(f"Total scored 5: {five_count}")
    
    """
    # run analysis on each description by calling ChatGPT4
    count = 1
    one_count = 0
    two_count = 0
    three_count = 0
    four_count = 0
    five_count = 0
    for desc_embed in desc_embeds:
        result = analyze_description(desc_embed[0])
        print(f"Description: {desc_embed[0]}\nAnalysis: {result}\n")
        if result == "1":
            one_count += 1
        elif result == "2":
            two_count += 1
        elif result == "3":
            three_count += 1
        elif result == "4":
            four_count += 1
        elif result == "5":
            five_count += 1
        
        desc_embed.append(result)

        #if count == 10:
            #break
        count += 1

    print(f"Total scored 1: {one_count}")
    print(f"Total scored 2: {two_count}")
    print(f"Total scored 3: {three_count}")
    print(f"Total scored 4: {four_count}")
    print(f"Total scored 5: {five_count}")

    # Rewrite the JSON file with the updated data
    with open('desc_embeddings_scored.json', 'w') as f:
        json.dump(desc_embeds, f, indent=4)
    """

if __name__ == "__main__":
    main()