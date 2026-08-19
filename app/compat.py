import asyncio
import sys


def configure_windows_event_loop() -> None:
    """psycopg async cannot run on Windows ProactorEventLoop."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
