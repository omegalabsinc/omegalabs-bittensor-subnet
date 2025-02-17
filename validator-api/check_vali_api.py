import aiohttp
import asyncio
import datetime
import os

API_URL = "https://validator.api.omega-labs.ai"
NUM_REQUESTS = 5
SAVE_DIR = "api_logs"

async def check_validator_api():
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}") as response:
            return await response.text()

async def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(SAVE_DIR, exist_ok=True)
    output_file = f"{SAVE_DIR}/validator_api_check_{timestamp}.txt"
    start_time = asyncio.get_event_loop().time()
    tasks = [check_validator_api() for _ in range(NUM_REQUESTS)]
    results = await asyncio.gather(*tasks)
    end_time = asyncio.get_event_loop().time()

    with open(output_file, 'w') as f:
        for i, result in enumerate(results, 1):
            f.write(f"Request {i}:\n{result}\n\n")
        f.write(f"Total time taken: {end_time - start_time:.2f} seconds")

    print(f"Results saved to {output_file}")

asyncio.run(main())
