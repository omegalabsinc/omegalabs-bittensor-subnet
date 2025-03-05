import asyncio
import functools

RETRIES = 3
DELAY_SECS = 2


def run_async(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


async def run_with_retries(func, *args, **kwargs):
    """func can be sync or async, since we await the output if it's a coroutine"""
    for i in range(0, RETRIES):
        try:
            output = func(*args, **kwargs)
            if asyncio.iscoroutine(output):
                return await output
            else:
                return output
        except:
            if i == RETRIES - 1:
                raise
            await asyncio.sleep(DELAY_SECS)
    raise RuntimeError("Should never happen")
