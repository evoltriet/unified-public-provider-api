"""Gunicorn configuration for the unified CMS provider API."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gunicorn configuration
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:5000")
workers = int(os.getenv("GUNICORN_WORKERS", "2"))
worker_class = "sync"
worker_connections = 100
timeout = 300  # Increased: 5 minutes for large searches
keepalive = 5
max_requests = 100  # Reload workers more frequently to manage memory
max_requests_jitter = 10

# Logging
accesslog = "-"  # stdout
errorlog = "-"  # stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'


# Worker initialization
def post_worker_init(worker):
    """Load providers, clinics, and systems in each worker process."""
    from src.api import initialize_data

    loaded = initialize_data()
    print(f"[Worker {worker.pid}] CMS provider data loaded={loaded}")


def worker_int(worker):
    """Handle worker shutdown gracefully."""
    print(f"[Worker {worker.pid}] Shutting down")
