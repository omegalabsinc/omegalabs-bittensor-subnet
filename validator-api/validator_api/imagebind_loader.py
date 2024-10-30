from typing import Optional
from fastapi import HTTPException
import asyncio
from omega.imagebind_wrapper import ImageBind


class ImageBindLoader:
    def __init__(self):
        self._imagebind: Optional[ImageBind] = None
        self._loading_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def get_imagebind(self) -> ImageBind:
        """
        Asynchronously get or initialize ImageBind instance.
        Handles concurrent requests efficiently.
        """
        if self._imagebind is not None:
            return self._imagebind
            
        async with self._lock:
            # Double-check pattern
            if self._imagebind is not None:
                return self._imagebind
                
            if self._loading_task is not None:
                # If already loading, wait for it to complete
                try:
                    await self._loading_task
                except Exception as e:
                    self._loading_task = None
                    raise HTTPException(
                        status_code=503, 
                        detail=f"Failed to load ImageBind: {str(e)}"
                    )
                return self._imagebind
                
            # Start loading
            self._loading_task = asyncio.create_task(self._load_imagebind())
            try:
                await self._loading_task
                return self._imagebind
            except Exception as e:
                self._loading_task = None
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to load ImageBind: {str(e)}"
                )

    async def _load_imagebind(self) -> None:
        """Internal method to load ImageBind."""
        print("Loading ImageBind")
        try:
            self._imagebind = ImageBind(v2=True)
        finally:
            self._loading_task = None
