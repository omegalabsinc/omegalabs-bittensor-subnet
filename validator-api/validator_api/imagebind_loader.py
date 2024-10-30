from typing import Optional
from fastapi import HTTPException
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from omega.imagebind_wrapper import ImageBind


class ImageBindLoader:
    def __init__(self):
        self._imagebind: Optional[ImageBind] = None
        self._loading_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._thread_pool = ThreadPoolExecutor(max_workers=1)

    async def get_imagebind(self) -> ImageBind:
        """
        Asynchronously get or initialize ImageBind instance.
        Handles concurrent requests efficiently.
        """
        if self._imagebind is not None:
            return self._imagebind

        if self._loading_task is None:
            self._loading_task = asyncio.create_task(self._load_imagebind_wrapper())

        raise HTTPException(
            status_code=503,
            detail="ImageBind loading has started. Please try again later."
        )

    def _load_imagebind_blocking(self) -> ImageBind:
        """Blocking method to load ImageBind in a separate thread."""
        return ImageBind(v2=True)

    async def _load_imagebind_wrapper(self) -> None:
        """Wrapper to run the blocking load in a thread pool."""
        try:
            # Run the blocking operation in a thread pool
            loop = asyncio.get_running_loop()
            self._imagebind = await loop.run_in_executor(
                self._thread_pool,
                self._load_imagebind_blocking
            )
        finally:
            self._loading_task = None
