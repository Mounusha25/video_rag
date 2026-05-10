import os
import sys

# Add the project root to sys.path so top-level modules (db, time_stamp_grouping) are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backend.api import app
