import bittensor as bt

from openai import OpenAI
import torch
from transformers import pipeline


def get_llm_prompt(query: str) -> str:
    return f"Take the given query `{query}` and augment it to be more detailed. For example, add specific names, types, embellishments, richness. Do not make it longer than 12 words."


class AbstractAugment:
    def __init__(self, **kwargs):
        pass

    def __call__(self, query: str) -> str:
        try:
            new_query = self.augment_query(query)
            bt.logging.info(f"Augmented query: '{query}' -> '{new_query}'")
        except Exception as e:
            print(f"Error augmenting query: {e}")
            return query
        
    def augment_query(self, query: str) -> str:
        raise NotImplementedError


class NoAugment(AbstractAugment):
    def __init__(self, **kwargs):
        bt.logging.info("Running no query augmentation")

    def augment_query(self, query: str) -> str:
        return query


class LocalLLMAugment(AbstractAugment):
    def __init__(self, **kwargs):
        self.device = kwargs.get("device")
        if self.device == "cpu":
            raise ValueError("Cannot run Local LLM on CPU")
        model_name = "teknium/OpenHermes-2.5-Mistral-7B"
        self.pipe = pipeline("text-generation", model=model_name, device=self.device, torch_dtype=torch.float16, pad_token_id=32000)
        bt.logging.info(f"Running query augmentation with local LLM {model_name} (thanks Nous!)")

    def augment_query(self, query: str) -> str:
        prompt = f"""<|im_start|>system
        You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>
        <|im_start|>user
        {get_llm_prompt(query)}<|im_end|>
        <|im_start|>assistant
        Detailed query: """
        new_query = self.pipe(prompt, max_new_tokens=64)[0]["generated_text"][len(prompt):].strip().strip("\"").strip("'")
        return new_query


class OpenAIAugment(AbstractAugment):
    def __init__(self, **kwargs):
        self.client = OpenAI()
        bt.logging.info("Running query augmentation with OpenAI GPT-4")

    def augment_query(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "user",
                    "content": get_llm_prompt(query)
                }
            ],
            temperature=0.9,
            max_tokens=64,
            top_p=1,
        )
        return response.choices[0].message.content.strip("\"").strip("'")
