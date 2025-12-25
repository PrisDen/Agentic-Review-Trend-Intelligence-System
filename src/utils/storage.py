"""
Storage utility.

File I/O helpers for daily reviews, daily counts, and other data persistence.
"""

import json
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Manages file I/O for all data persistence except the registry.
    
    Handles:
    - Raw reviews (data/raw/YYYY-MM-DD.json)
    - Daily counts (data/daily_counts/YYYY-MM-DD.json)
    """
    
    def __init__(self, data_root: str):
        """
        Initialize storage manager.
        
        Args:
            data_root: Root data directory (e.g., /path/to/data)
        """
        self.data_root = data_root
        self.raw_dir = os.path.join(data_root, "raw")
        self.daily_counts_dir = os.path.join(data_root, "daily_counts")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.daily_counts_dir, exist_ok=True)
        
        logger.info(f"Initialized StorageManager with data_root={data_root}")
    
    def save_raw_reviews(self, reviews: List[Dict], date: str) -> None:
        """
        Save raw reviews for a specific date.
        
        Args:
            reviews: List of review dicts
            date: Date in YYYY-MM-DD format
        """
        filepath = os.path.join(self.raw_dir, f"{date}.json")
        
        try:
            with open(filepath, 'w') as f:
                json.dump(reviews, f, indent=2)
            logger.info(f"Saved {len(reviews)} raw reviews to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save raw reviews for {date}: {e}")
            raise
    
    def load_raw_reviews(self, date: str) -> Optional[List[Dict]]:
        """
        Load raw reviews for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format
        
        Returns:
            List of review dicts, or None if file doesn't exist
        """
        filepath = os.path.join(self.raw_dir, f"{date}.json")
        
        if not os.path.exists(filepath):
            logger.warning(f"No raw reviews found for {date}")
            return None
        
        try:
            with open(filepath, 'r') as f:
                reviews = json.load(f)
            logger.debug(f"Loaded {len(reviews)} raw reviews from {filepath}")
            return reviews
        except Exception as e:
            logger.error(f"Failed to load raw reviews for {date}: {e}")
            return None
    
    def save_daily_count(self, count_data: Dict, date: str) -> None:
        """
        Save daily topic counts.
        
        Args:
            count_data: Daily count data dict (see aggregation.py for schema)
            date: Date in YYYY-MM-DD format
        """
        filepath = os.path.join(self.daily_counts_dir, f"{date}.json")
        
        try:
            with open(filepath, 'w') as f:
                json.dump(count_data, f, indent=2)
            logger.info(f"Saved daily counts to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save daily counts for {date}: {e}")
            raise
    
    def load_daily_count(self, date: str) -> Optional[Dict]:
        """
        Load daily topic counts for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format
        
        Returns:
            Daily count data dict, or None if file doesn't exist
        """
        filepath = os.path.join(self.daily_counts_dir, f"{date}.json")
        
        if not os.path.exists(filepath):
            logger.debug(f"No daily counts found for {date}")
            return None
        
        try:
            with open(filepath, 'r') as f:
                count_data = json.load(f)
            return count_data
        except Exception as e:
            logger.error(f"Failed to load daily counts for {date}: {e}")
            return None
    
    def get_all_daily_count_dates(self) -> List[str]:
        """
        Get all dates that have daily count files.
        
        Returns:
            Sorted list of dates in YYYY-MM-DD format
        """
        dates = []
        for filename in os.listdir(self.daily_counts_dir):
            if filename.endswith('.json') and filename != '.gitkeep':
                date = filename.replace('.json', '')
                dates.append(date)
        
        return sorted(dates)


# Design Rationale and Trade-offs:
#
# 1. Why separate StorageManager instead of inline file I/O?
#    - Single responsibility: All file I/O in one place
#    - Easy to test (mock the storage manager)
#    - Easy to swap storage backend (e.g., S3, database) in V2
#    - Trade-off: Extra abstraction layer, but worth it for testability
#
# 2. Why create directories in __init__ instead of on first write?
#    - Fail fast if permissions are wrong
#    - Avoids scattered os.makedirs() calls throughout codebase
#    - Trade-off: Creates empty dirs even if never used, but negligible
#
# 3. Why return None instead of raising on missing files?
#    - Missing data is expected (e.g., day not processed yet)
#    - Caller can decide whether to treat as error or use default
#    - Aligns with graceful degradation principle
#    - Trade-off: Caller must check for None, but more flexible
#
# 4. Why not implement caching?
#    - Daily counts are loaded once during trend aggregation
#    - Caching adds complexity without benefit for sequential access
#    - Trade-off: Re-reads from disk if called multiple times, acceptable
