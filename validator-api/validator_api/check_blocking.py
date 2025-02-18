import asyncio
from contextlib import asynccontextmanager


@asynccontextmanager
async def detect_blocking(name: str):
    # this prints a message if an operation/endpoint is blocking for too long
    loop = asyncio.get_running_loop()
    start_time = loop.time()
    last_yield = start_time
    yielded = True
    monitoring_task = None

    def check_yield():
        nonlocal last_yield, yielded
        current = loop.time()
        if current - last_yield > 0.1:  # Blocked for >100ms
            print(f"Blocking operation detected in {name}! Blocked for {current - last_yield:.2f}s")
        last_yield = current
        yielded = True

    async def monitor():
        nonlocal yielded
        while True:
            if yielded:
                yielded = False
                await asyncio.sleep(0.05)  # Check every 50ms
                check_yield()

    try:
        monitoring_task = asyncio.create_task(monitor())
        yield
    finally:
        if monitoring_task:
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
        check_yield()
