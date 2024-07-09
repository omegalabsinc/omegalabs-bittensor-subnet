import json
import random

MAX_DESCRIPTIONS_TO_SCORE = 1000

def prompt_for_score(description):
    while True:
        try:
            score = int(input(f"Description: {description}\nPlease score this description from 1 to 5 (5 being the best, 1 being the worst): "))
            if score in range(1, 6):
                return score
            else:
                print("Invalid input. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

def main():
    # Load the JSON data
    with open('desc_embeddings_recent.json', 'r') as f:
        descriptions = json.load(f)
        # Ensure we don't try to sample more than the available descriptions
        num_to_sample = min(MAX_DESCRIPTIONS_TO_SCORE, len(descriptions))
        desc_embeds = random.sample(descriptions, num_to_sample)

    # Loop through each description and prompt for a score
    for desc_embed in desc_embeds:
        description = desc_embed[0]
        score = prompt_for_score(description)
        desc_embed.append(score)

    # Write the updated data to a new JSON file
    with open('desc_embeddings_scored.json', 'w') as f:
        json.dump(desc_embeds, f, indent=4)

if __name__ == "__main__":
    main()