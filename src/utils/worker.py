import os
import sys
from rq import Worker
from redis import Redis
from dotenv import load_dotenv


load_dotenv()

redis_host = os.getenv("REDIS_HOST", "redis")
redis_port = os.getenv("REDIS_PORT", "6379")
redis_password = os.getenv("REDIS_PASSWORD", "redis")
redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/0"


redis_conn = Redis.from_url(redis_url)
queue_name = "training-queue"


if __name__ == "__main__":
    worker = Worker([queue_name], connection=redis_conn)
    worker.work()
