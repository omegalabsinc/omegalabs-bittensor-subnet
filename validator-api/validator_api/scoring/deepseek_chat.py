"""
provides deepseek chat via chutes api
"""

import httpx
import json
import traceback
import os
import asyncio
from typing import Iterable, Type, Optional, Union
from pydantic import BaseModel
from openai.resources.beta.chat.completions import ChatCompletionMessageParam

CHUTES_API_TOKEN = os.getenv("CHUTES_API_TOKEN")

async def query_deepseek(
    messages: Iterable[ChatCompletionMessageParam],
    output_model: Optional[Type[BaseModel]] = None,
    retries: int = 3
) -> Union[BaseModel, dict]:
    """
    Query the DeepSeek chat model via the Chutes API with streaming (non-streaming appears to be broken).
    
    This function sends a chat completion request to DeepSeek, processes the streamed response,
    and optionally validates it against a provided Pydantic model.
    Your prompt must have the "json" keyword somewhere and an example; reference:
    https://api-docs.deepseek.com/guides/json_mode

    Args:
        messages: An iterable of chat completion messages following the OpenAI format.
            Each message should have 'role' and 'content' fields.
        output_model: Optional Pydantic BaseModel class to validate and parse the response.
            If provided, the response will be validated against this model.
            If not provided, the raw parsed JSON dict will be returned.
        retries: Number of times to retry on failure. Default is 3.
            Uses exponential backoff between retries (1s, 2s, 4s).

    Returns:
        Union[BaseModel, dict]: If output_model is provided, returns an instance of that model.
            Otherwise, returns the parsed JSON response as a dictionary.

    Raises:
        ValueError: If no content is received or if JSON parsing/validation fails
        TimeoutError: If the request times out while reading the stream
        httpx.HTTPError: If an HTTP error occurs during the request
        Exception: For any other unexpected errors after all retries are exhausted

    Notes:
        - The function automatically handles cleanup of streaming responses
        - Malformed response chunks are logged but skipped
        - Response content is cleaned of markdown formatting and <think> tags
        - JSON responses are expected and enforced via the API's response_format parameter
    """
    last_exception = None
    
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                headers = {
                    "Authorization": f"Bearer {CHUTES_API_TOKEN}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "deepseek-ai/DeepSeek-R1",
                    "messages": messages,
                    "stream": True,
                    "max_tokens": 1000,
                    "temperature": 0.5,
                    "response_format": {
                        "type": "json_object"
                    }
                }
                
                async with client.stream(
                    "POST",
                    "https://chutes-deepseek-ai-deepseek-r1.chutes.ai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120.0
                ) as response:
                    response.raise_for_status()
                    content = ""
                    
                    try:
                        async for line in response.aiter_lines():
                            if line.strip():
                                # Remove "data: " prefix and handle SSE format
                                if line.startswith("data: "):
                                    line = line[6:]
                                if line == "[DONE]":
                                    continue
                                    
                                try:
                                    chunk = json.loads(line)
                                    if delta_content := chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                                        content += delta_content
                                except json.JSONDecodeError as e:
                                    print(f"Failed to parse chunk: {e}")
                                    continue
                                except IndexError:
                                    print("Received malformed response chunk from Chutes API call")
                                    continue
                        
                        if not content:  # Check if we got any content
                            raise ValueError("No content received from API")

                        content = content.replace('```json', '').replace('```', '').strip()
                        
                        if '</think>' in content:
                            # get the content after the </think> tag
                            content = content.split('</think>')[-1].strip()
                        
                        # Parse JSON and optionally validate against output model
                        try:
                            parsed_data = json.loads(content)
                            # print(f"Parsed data: {parsed_data}")
                            if output_model is not None:
                                return output_model.model_validate(parsed_data)
                            return parsed_data
                        except json.JSONDecodeError:
                            raise ValueError(f"Failed to parse response as JSON: {content}")
                        
                    except httpx.ReadTimeout:
                        raise TimeoutError("Request timed out while reading the stream")
                    finally:
                        await response.aclose()
                        
        except Exception as e:
            last_exception = e
            if attempt < retries - 1:
                sleep_time = 2 ** attempt
                print(f"Chutes API call attempt {attempt + 1} failed with error: {str(e)}. Retrying in {sleep_time} seconds...")
                await asyncio.sleep(sleep_time)
            continue
            
    # If we get here, all retries failed
    print(f"All {retries} attempts failed in deepseek_model")
    raise last_exception