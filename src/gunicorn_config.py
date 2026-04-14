"""
Optimized gunicorn configuration for NPI Registry API

Key optimizations:
- Longer timeout for large dataset searches
- Reduced workers to manage memory
- Better error handling
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gunicorn configuration
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:5000")
workers = int(
    os.getenv("GUNICORN_WORKERS", "2")
)  # Reduced: each worker loads 8.5M rows
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
    """
    Called just after a worker has initialized.
    This ensures NPI data is loaded in each worker process.
    """
    # Import here to avoid circular imports
    from src.api import load_npi_data

    # Load data in this worker process
    load_npi_data()
    print(f"[Worker {worker.pid}] NPI data loaded")


def worker_int(worker):
    """Handle worker shutdown gracefully"""
    print(f"[Worker {worker.pid}] Shutting down")
