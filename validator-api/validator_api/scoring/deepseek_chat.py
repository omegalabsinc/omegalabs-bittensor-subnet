"""
provides deepseek chat via chutes api
"""

import httpx
import json
import traceback
import os
from typing import Iterable
from openai.resources.beta.chat.completions import ChatCompletionMessageParam

CHUTES_API_TOKEN = os.getenv("CHUTES_API_TOKEN")

async def query_deepseek(messages: Iterable[ChatCompletionMessageParam]) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:  # Increase default timeout to 60 seconds
        try:
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
                    
                    if not content:  # Check if we got any content
                        raise ValueError("No content received from API")
                    return content
                    
                except httpx.ReadTimeout:
                    raise TimeoutError("Request timed out while reading the stream")
                    
        except httpx.HTTPError as e:
            print(f"HTTP error occurred: {str(e)}")
            traceback.print_exc()
            raise
        except Exception as e:
            print(f"Unexpected error in deepseek_model: {str(e)}")
            raise