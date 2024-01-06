import asyncio
import logging
import os
from typing import Any

import aiolimiter
import openai
import openai.error
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio
import random
from time import sleep
import time

async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
):
    async with limiter:
        for _ in range(10):
            try:
                return await openai.ChatCompletion.acreate(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.error.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 20 seconds."
                )
                await asyncio.sleep(10)
            except asyncio.exceptions.TimeoutError or openai.error.Timeout or asyncio.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except openai.error.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                await asyncio.sleep(10)
            except Exception as e:
                logging.warning(e)
                await asyncio.sleep(10)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
    messages,
    engine_name: str,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    requests_per_minute: int = 100,
):
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        prompt_template: Prompt template to use.
        model_config: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    session = ClientSession()
    openai.aiosession.set(session)
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=engine_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in messages
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    await session.close()
    
    outputs = []
    for response in responses:
        outputs.append(response["choices"][0]["message"]["content"])
    return outputs

def generate_from_openai_chat_completion_single(
    messages,
    engine_name: str,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 1.0
):
    for _ in range(5):
        try:
            response = openai.ChatCompletion.create(
                            model=engine_name,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                        )
        except openai.error.RateLimitError:
            logging.warning(
                "OpenAI API rate limit exceeded. Sleeping for 20 seconds."
            )
            time.sleep(10)
        except asyncio.exceptions.TimeoutError or openai.error.Timeout or asyncio.TimeoutError:
            logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
            time.sleep(10)
        except openai.error.APIError as e:
            logging.warning(f"OpenAI API error: {e[:20]}")
            time.sleep(10)
        except Exception as e:
            logging.warning(e[:20])
            time.sleep(10)

    return response["choices"][0]["message"]["content"]