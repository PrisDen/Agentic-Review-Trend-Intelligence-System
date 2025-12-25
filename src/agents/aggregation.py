"""
Daily Topic Counter and Trend Aggregator.

Counts topic occurrences per day and aggregates 30-day trends.
"""

import logging
from typing import List, Dict
from collections import Counter
from datetime import datetime, timedelta
import pandas as pd
import os

from src.registry.topic_registry import TopicRegistry
from src.utils.storage import StorageManager

logger = logging.getLogger(__name__)


class DailyTopicCounter:
    """
    Counts canonical topic occurrences for a single day.
    """
    
    def count(
        self,
        topic_ids: List[str],
        date: str,
        total_reviews: int,
        total_statements: int,
        metadata: Dict = None
    ) -> Dict:
        """
        Count topic occurrences and create daily count data.
        
        Args:
            topic_ids: List of topic IDs (from consolidation)
            date: Date in YYYY-MM-DD format
            total_reviews: Number of reviews processed
            total_statements: Number of statements extracted
            metadata: Optional processing metadata
        
        Returns:
            Daily count data dict
        """
        # Count topic occurrences
        topic_counts = Counter(topic_ids)
        
        # Build daily count data
        daily_count_data = {
            "date": date,
            "total_reviews_processed": total_reviews,
            "total_statements_extracted": total_statements,
            "topic_counts": dict(topic_counts),  # Convert Counter to dict
            "metadata": metadata or {}
        }
        
        logger.info(
            f"Counted {len(topic_counts)} unique topics for {date} "
            f"(total mentions: {len(topic_ids)})"
        )
        
        return daily_count_data


class TrendAggregator:
    """
    Aggregates daily topic counts into 30-day trend table.
    """
    
    def __init__(self, storage: StorageManager, registry: TopicRegistry):
        """
        Initialize trend aggregator.
        
        Args:
            storage: Storage manager for loading daily counts
            registry: Topic registry for label lookups
        """
        self.storage = storage
        self.registry = registry
    
    def generate_trend_table(
        self,
        target_date: str,
        window_days: int = 30,
        output_dir: str = "output"
    ) -> str:
        """
        Generate 30-day trend table for target date.
        
        Args:
            target_date: End date in YYYY-MM-DD format
            window_days: Number of days to include (default: 30)
            output_dir: Directory to save CSV output
        
        Returns:
            Path to generated CSV file
        """
        logger.info(f"Generating {window_days}-day trend table for {target_date}")
        
        # Calculate date range (T-window_days to T)
        end_date = datetime.strptime(target_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=window_days - 1)
        
        # Generate list of dates in range
        date_range = []
        current = start_date
        while current <= end_date:
            date_range.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        logger.info(f"Date range: {date_range[0]} to {date_range[-1]} ({len(date_range)} days)")
        
        # Load all daily counts
        daily_counts = []
        missing_dates = []
        
        for day in date_range:
            count_data = self.storage.load_daily_count(day)
            if count_data:
                daily_counts.append(count_data)
            else:
                # Missing day - create empty count
                logger.warning(f"No data for {day}, using zeros")
                missing_dates.append(day)
                daily_counts.append(self._create_empty_count(day))
        
        # Collect all topic IDs that appear in any of the days
        all_topic_ids = set()
        for day_data in daily_counts:
            all_topic_ids.update(day_data['topic_counts'].keys())
        
        logger.info(f"Found {len(all_topic_ids)} unique topics across all days")
        
        # Build trend matrix
        rows = []
        for topic_id in all_topic_ids:
            topic = self.registry.get_topic(topic_id)
            if not topic:
                logger.warning(f"Topic {topic_id} not found in registry, skipping")
                continue
            
            row = {
                'Topic': topic.canonical_label,
                'Type': topic.type
            }
            
            # Add daily counts
            for day_data in daily_counts:
                date_str = day_data['date']
                count = day_data['topic_counts'].get(topic_id, 0)
                row[date_str] = count
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        if df.empty:
            logger.warning("No topics found, creating empty trend table")
            df = pd.DataFrame(columns=['Topic', 'Type'] + date_range)
        else:
            # Calculate total mentions across all days
            date_columns = [col for col in df.columns if col not in ['Topic', 'Type']]
            df['Total'] = df[date_columns].sum(axis=1)
            
            # Sort by total mentions (descending)
            df = df.sort_values('Total', ascending=False)
        
        # Save to CSV
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"trend_{target_date}.csv")
        df.to_csv(output_path, index=False)
        
        logger.info(
            f"Trend table saved to {output_path} "
            f"({len(df)} topics, {len(date_range)} days, "
            f"{len(missing_dates)} missing days)"
        )
        
        # Save metadata
        metadata_path = os.path.join(output_dir, f"trend_{target_date}_metadata.json")
        metadata = {
            "target_date": target_date,
            "window_days": window_days,
            "date_range": {
                "start": date_range[0],
                "end": date_range[-1]
            },
            "missing_dates": missing_dates,
            "coverage": f"{len(date_range) - len(missing_dates)}/{len(date_range)} days ({100 * (len(date_range) - len(missing_dates)) / len(date_range):.1f}%)",
            "total_topics": len(df),
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
        
        return output_path
    
    def _create_empty_count(self, date: str) -> Dict:
        """Create empty daily count for missing date."""
        return {
            "date": date,
            "total_reviews_processed": 0,
            "total_statements_extracted": 0,
            "topic_counts": {},
            "metadata": {"note": "Missing data - no reviews processed"}
        }


# Design Rationale and Trade-offs:
#
# 1. Why separate DailyTopicCounter and TrendAggregator classes?
#    - DailyCounter: Single-day operation (used during daily processing)
#    - TrendAggregator: Multi-day operation (used at end of pipeline)
#    - Different responsibilities, different usage patterns
#    - Trade-off: Two classes instead of one, but clearer separation
#
# 2. Why use Counter for topic counting?
#    - Built-in Python, optimized for frequency counting
#    - Clean, readable code (Counter(list) is self-explanatory)
#    - Trade-off: Could use plain dict, but Counter is more idiomatic
#
# 3. Why pandas DataFrame for trend table?
#    - CSV export is trivial (df.to_csv)
#    - Easy sorting, filtering, statistical operations
#    - Standard tool for tabular data in Python
#    - Trade-off: Adds pandas dependency, but widely used and reliable
#
# 4. Why fill missing days with zeros instead of nulls?
#    - CSV simplicity (no special null handling)
#    - Trend visualization tools expect numeric values
#    - 0 == "no complaints observed" (acceptable interpretation)
#    - Trade-off: Can't distinguish "no data" from "no complaints", mitigated by metadata
#
# 5. Why save metadata alongside CSV?
#    - Transparency: User sees coverage %
#    - Debugging: Can identify which days are missing
#    - Reproducibility: Know exactly what went into the trend table
#    - Trade-off: Extra file to manage, but valuable for trust
#
# 6. Why sort by Total (descending)?
#    - PMs care most about highest-volume topics
#    - Trend table is more actionable when sorted by impact
#    - Trade-off: Could sort by date or alphabetically, but Total is most useful
