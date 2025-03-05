from typing import Iterable, Optional, Type, Union
from pydantic import BaseModel
from openai.resources.beta.chat.completions import ChatCompletionMessageParam
from openai import AsyncOpenAI
from validator_api.scoring.query_deepseek import query_deepseek
import os
import json
import asyncio

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
)


async def query_openai(
    messages: Iterable[ChatCompletionMessageParam],
    output_model: Optional[Type[BaseModel]] = None,
    retries: int = 3,
) -> Union[BaseModel, dict]:
    """
    Query the OpenAI o1 model with retries.

    Args:
        messages: An iterable of chat completion messages following the OpenAI format.
            Each message should have 'role' and 'content' fields.
        output_model: Optional Pydantic BaseModel class to validate and parse the response.
        retries: Number of times to retry on failure. Default is 3.
            Uses exponential backoff between retries.

    Returns:
        Union[BaseModel, dict]: If output_model is provided, returns an instance of that model.
            Otherwise, returns the parsed JSON response as a dictionary.

    Raises:
        Exception: If all retry attempts fail
    """
    for attempt in range(retries):
        try:
            response = await openai_client.beta.chat.completions.parse(
                model="o1-2024-12-17",
                messages=messages,
                response_format=output_model,
            )
            if not response.choices[0].message.content:
                raise Exception("Empty response from API")

            parsed_data = json.loads(response.choices[0].message.content)

            if output_model is not None:
                return output_model.model_validate(parsed_data)
            return parsed_data

        except Exception as e:
            if attempt < retries - 1:
                sleep_time = 2**attempt
                print(
                    f"OpenAI attempt {attempt + 1} failed: {str(e)}. Retrying in {sleep_time} seconds..."
                )
                await asyncio.sleep(sleep_time)
                continue
            raise e


async def query_llm(
    messages: Iterable[ChatCompletionMessageParam],
    output_model: Optional[Type[BaseModel]] = None,
    retries: int = 3,
) -> Union[BaseModel, dict]:
    """
    Query LLM models with fallback behavior. Tries DeepSeek first, falls back to OpenAI if DeepSeek fails.

    Args:
        messages: An iterable of chat completion messages following the OpenAI format.
            Each message should have 'role' and 'content' fields.
        output_model: Optional Pydantic BaseModel class to validate and parse the response.
        retries: Number of times to retry each model before falling back. Default is 3.

    Returns:
        Union[BaseModel, dict]: If output_model is provided, returns an instance of that model.
            Otherwise, returns the parsed JSON response as a dictionary.

    Raises:
        Exception: If both DeepSeek and OpenAI attempts fail after all retries
    """
    try:
        return await query_deepseek(messages, output_model, retries)
    except Exception as e:
        if not openai_client:
            raise e

        print(f"DeepSeek failed, falling back to OpenAI: {str(e)}")
        return await query_openai(messages, output_model, retries)
