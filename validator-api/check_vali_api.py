import aiohttp
import asyncio
import datetime
import os
import matplotlib.pyplot as plt
import sys

# API_URL = "https://validator.api.omega-labs.ai"
# API_URL = "http://localhost:8001"
API_URL = "https://sn24-api.omegatron.ai"
NUM_REQUESTS = 200
SAVE_DIR = "api_logs"
SECONDS_DELAY = 0.5
TIMEOUT_SECONDS = 10.0

if len(sys.argv) > 1:
    API_URL = sys.argv[1]


async def check_validator_api(idx: int):
    await asyncio.sleep(idx * SECONDS_DELAY)
    start = asyncio.get_event_loop().time()
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{API_URL}", timeout=TIMEOUT_SECONDS) as response:
                end = asyncio.get_event_loop().time()
                duration = end - start
                print(f"Done {idx} in {duration:.2f} seconds, got {response.status}")
                return await response.text(), duration, response.status
        except asyncio.TimeoutError:
            print(f"Request {idx} timed out after {TIMEOUT_SECONDS} seconds")
            return "Timeout", TIMEOUT_SECONDS, None


async def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(SAVE_DIR, exist_ok=True)
    output_file = f"{SAVE_DIR}/validator_api_check_{timestamp}.txt"
    tasks = [check_validator_api(idx) for idx in range(NUM_REQUESTS)]
    results = await asyncio.gather(*tasks)

    # Extract durations for histogram
    durations = [result[1] for result in results]

    total_time = sum(durations)
    timeout_count = sum(
        1 for message, duration, status in results if message == "Timeout"
    )
    error_521_count = sum(1 for message, duration, status in results if status == 521)

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=50, edgecolor="black")
    plt.title(
        f"Distribution of API Request Durations ({API_URL}, {timeout_count} timeouts, {error_521_count} 521 errors)"
    )
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Frequency")
    plt.savefig(f"{SAVE_DIR}/duration_histogram_{timestamp}.png")
    plt.close()

    with open(output_file, "w") as f:
        f.write(f"API URL: {API_URL}\n")
        f.write(f"Number of requests: {NUM_REQUESTS}\n")
        f.write(f"Total time taken: {total_time:.2f} seconds\n")
        f.write(f"Average time per request: {total_time / NUM_REQUESTS:.2f} seconds\n")
        f.write(f"Max response time: {max(durations):.2f} seconds\n")
        f.write(f"Number of timeouts: {timeout_count}\n")
        f.write(f"Number of 521 errors: {error_521_count}\n\n\n")
        for i, (message, duration, status) in enumerate(results, 1):
            f.write(f"Request {i}:\n{message}\nTime taken: {duration:.2f} seconds\n")
            if status:
                f.write(f"Status code: {status}\n")
            f.write("\n")

    print(f"API_URL: {API_URL}")
    print(f"Results saved to {output_file}")
    print(f"Histogram saved to {SAVE_DIR}/duration_histogram_{timestamp}.png")
    print(f"Total number of timeouts: {timeout_count}")
    print(f"Total number of 521 errors: {error_521_count}")


asyncio.run(main())
