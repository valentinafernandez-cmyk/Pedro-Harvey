import asyncio
import os
import signal
import logging
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from cli import main_async_flow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CryptoScheduler")

load_dotenv()
TOKENS = ["bitcoin", "ethereum", "cardano"]

# Global event to signal shutdown
shutdown_event = asyncio.Event()


async def run_daily_crypto_fetch():
    API_KEY = os.getenv("COINGECKO_API_KEY")
    if not API_KEY:
        logger.critical("COINGECKO_API_KEY environment variable is not set!")
        return

    today_str = date.today().strftime("%Y-%m-%d")
    data_dir = Path("./data")

    logger.info(f"--- Starting daily task for {today_str} ---")

    tasks = [
        main_async_flow(
            coin_id=token,
            start_date=today_str,
            end_date=None,
            max_workers=1,
            api_key=API_KEY,
            db_flag=True,
            output_dir=data_dir,
            missing_only=False,
        )
        for token in TOKENS
    ]

    await asyncio.gather(*tasks)

    logger.info("--- Daily task finished successfully ---")


def handle_shutdown():
    logger.info("Shutdown signal received. Triggering graceful stop...")
    shutdown_event.set()


async def main():
    # Register OS signals for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_shutdown)
        except NotImplementedError:
            pass

    scheduler = AsyncIOScheduler()
    # Run cron at 3AM argentina
    scheduler.add_job(run_daily_crypto_fetch, "cron", hour=6, minute=0)

    scheduler.start()
    logger.info("Scheduler daemon successfully started. Waiting for tasks...")

    await shutdown_event.wait()

    logger.info("Shutting down APScheduler execution pools...")
    scheduler.shutdown()
    logger.info("Application cleanly stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Application terminated abruptly via KeyboardInterrupt.")
