# services/resource_manager.py
import asyncio
import gc
from typing import Callable, Dict, Awaitable, Optional

from src.config.logging_config import get_logger


class _ResourceManager:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._callbacks: Dict[str, Callable[[], Optional[Awaitable[None]]]] = {}
        self.logger = get_logger(__name__)
        self._current_owner: Optional[str] = None

    def register(self, role: str, cleanup_callback: Callable[[], Optional[Awaitable[None]]]):
        self.logger.info(f"Registering cleanup callback for role: '{role}'")
        self._callbacks[role] = cleanup_callback

    async def claim(self, role: str):
        # If the calling role already owns the resource, do nothing.
        if self._current_owner == role:
            return

        self.logger.info(f"'{role}' is claiming resource lock...")
        async with self._lock:
            self.logger.info(f"'{role}' acquired resource lock. Previous owner: '{self._current_owner}'.")
            
            # Clean up all other registered services
            for other_role, cb in list(self._callbacks.items()):
                if other_role == role:
                    continue
                
                self.logger.info(f"Requesting cleanup for '{other_role}'...")
                try:
                    result = cb()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    self.logger.warning(f"Cleanup for '{other_role}' failed: {e}", exc_info=True)

            # A more aggressive system-level garbage collection
            self.logger.info("Performing aggressive garbage collection to free system RAM.")
            gc.collect()
            
            self._current_owner = role
            self.logger.info(f"Cleanup complete. '{role}' is now the exclusive owner.")


resource_manager = _ResourceManager()



