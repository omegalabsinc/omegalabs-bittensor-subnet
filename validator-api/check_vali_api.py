import aiohttp
import asyncio
import datetime
import os
import time
import matplotlib.pyplot as plt

API_URL = "https://validator.api.omega-labs.ai"
# API_URL = "http://localhost:8001"
API_URL = "https://sn24-api.omegatron.ai"
NUM_REQUESTS = 1000
SAVE_DIR = "api_logs"

async def check_validator_api():
    start = time.time()
    async with aiohttp.ClientSession() as session:
        # async with session.get(f"{API_URL}/api/focus/get_list") as response:
        async with session.get(f"{API_URL}") as response:
            return await response.text(), time.time() - start

async def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(SAVE_DIR, exist_ok=True)
    output_file = f"{SAVE_DIR}/validator_api_check_{timestamp}.txt"
    start_time = asyncio.get_event_loop().time()
    tasks = [check_validator_api() for _ in range(NUM_REQUESTS)]
    results = await asyncio.gather(*tasks)
    end_time = asyncio.get_event_loop().time()

    # Extract durations for histogram
    durations = [result[1] for result in results]
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=50, edgecolor='black')
    plt.title('Distribution of API Request Durations')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.savefig(f"{SAVE_DIR}/duration_histogram_{timestamp}.png")
    plt.close()

    with open(output_file, 'w') as f:
        f.write(f"API URL: {API_URL}\n")
        f.write(f"Number of requests: {NUM_REQUESTS}\n")
        f.write(f"Total time taken: {end_time - start_time:.2f} seconds\n")
        f.write(f"Average time per request: {(end_time - start_time) / NUM_REQUESTS:.2f} seconds\n")
        f.write(f"Max response time: {max(result[1] for result in results):.2f} seconds\n\n\n\n")
        for i, result in enumerate(results, 1):
            f.write(f"Request {i}:\n{result[0]}\nTime taken: {result[1]:.2f} seconds\n\n")

    print(f"Results saved to {output_file}")
    print(f"Histogram saved to {SAVE_DIR}/duration_histogram_{timestamp}.png")

asyncio.run(main())
