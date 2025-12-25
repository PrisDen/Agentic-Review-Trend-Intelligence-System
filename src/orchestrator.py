"""
Pipeline Orchestrator.

Coordinates sequential execution of all agents across daily batches.
"""

import logging
from typing import List
from datetime import datetime, timedelta

from src.agents.ingestion import IngestionAgent
from src.agents.normalization import SemanticNormalizationAgent
from src.agents.candidate_generation import TopicCandidateGenerator
from src.agents.consolidation import TopicConsolidationAgent
from src.agents.aggregation import DailyTopicCounter, TrendAggregator
from src.registry.topic_registry import TopicRegistry
from src.utils.embeddings import EmbeddingGenerator
from src.utils.storage import StorageManager
from src.models.review import Review
import config.settings as settings

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates the daily batch processing pipeline.
    
    Coordinates:
    1. Ingestion → 2. Normalization → 3. Candidate Generation
    → 4. Consolidation → 5. Daily Counting → 6. Registry Save
    
    After all days: Trend Aggregation
    """
    
    def __init__(self, api_key: str, data_root: str, registry_path: str):
        """
        Initialize pipeline orchestrator.
        
        Args:
            api_key: Google API key for LLMs and embeddings
            data_root: Root directory for data storage
            registry_path: Path to topic registry JSON
        """
        self.api_key = api_key
        self.data_root = data_root
        self.registry_path = registry_path
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        
        self.storage = StorageManager(data_root)
        self.registry = TopicRegistry(registry_path)
        
        self.ingestion_agent = IngestionAgent(
            use_mock_data=settings.USE_MOCK_DATA
        )
        
        self.normalization_agent = SemanticNormalizationAgent(
            api_key=api_key,
            model_name=settings.NORMALIZATION_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            min_confidence=settings.NORMALIZATION_MIN_CONFIDENCE,
            max_retries=settings.NORMALIZATION_MAX_RETRIES
        )
        
        self.candidate_generator = TopicCandidateGenerator(
            api_key=api_key,
            model_name=settings.CANDIDATE_GENERATION_MODEL,
            temperature=settings.LLM_TEMPERATURE
        )
        
        self.embedding_generator = EmbeddingGenerator(
            api_key=api_key,
            model_name=settings.EMBEDDING_MODEL,
            embedding_dimensions=settings.EMBEDDING_DIMENSIONS
        )
        
        self.consolidation_agent = TopicConsolidationAgent(
            api_key=api_key,
            embedding_generator=self.embedding_generator,
            model_name=settings.CONSOLIDATION_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            hard_merge_threshold=settings.HARD_MERGE_THRESHOLD,
            low_threshold=settings.LOW_THRESHOLD,
            top_k=settings.TOP_K_SIMILAR_TOPICS
        )
        
        self.daily_counter = DailyTopicCounter()
        
        self.trend_aggregator = TrendAggregator(
            storage=self.storage,
            registry=self.registry
        )
        
        logger.info("Pipeline initialized successfully")
    
    def run(
        self,
        app_package: str,
        target_date: str,
        start_date: str = None,
        window_days: int = 30
    ) -> str:
        """
        Run complete pipeline from start_date to target_date.
        
        Args:
            app_package: Google Play package name
            target_date: End date (YYYY-MM-DD)
            start_date: Start date (YYYY-MM-DD), defaults to target_date - window_days
            window_days: Number of days to process
        
        Returns:
            Path to generated trend table CSV
        """
        # Calculate date range
        end_date = datetime.strptime(target_date, "%Y-%m-%d")
        
        if start_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start = end_date - timedelta(days=window_days - 1)
        
        date_range = self._generate_date_range(start, end_date)
        
        logger.info(
            f"Starting pipeline for {app_package}: "
            f"{len(date_range)} days from {date_range[0]} to {date_range[-1]}"
        )
        
        # Process each day sequentially
        for day in date_range:
            try:
                self._process_single_day(app_package, day)
            except Exception as e:
                logger.error(f"Failed to process {day}: {e}")
                if settings.CONTINUE_ON_DAY_FAILURE:
                    logger.warning(f"Continuing to next day (graceful degradation)")
                    continue
                else:
                    raise
        
        # Generate trend table
        logger.info(f"Generating trend table for {target_date}")
        output_path = self.trend_aggregator.generate_trend_table(
            target_date=target_date,
            window_days=window_days,
            output_dir=str(settings.OUTPUT_ROOT)
        )
        
        logger.info(f"Pipeline complete! Trend table: {output_path}")
        return output_path
    
    def _process_single_day(self, app_package: str, date: str) -> None:
        """
        Process all reviews for a single day.
        
        Implements the sequential agent flow:
        Ingestion → Normalization → Candidate Gen → Consolidation → Counting
        """
        logger.info(f"Processing day: {date}")
        start_time = datetime.now()
        
        # STAGE 1: Ingestion
        raw_reviews = self.ingestion_agent.fetch_reviews(
            app_package=app_package,
            date=date,
            limit=settings.MOCK_REVIEWS_PER_DAY
        )
        
        if not raw_reviews:
            logger.warning(f"No reviews found for {date}")
            self._create_empty_daily_count(date)
            return
        
        # Save raw reviews
        raw_review_dicts = [
            {
                "review_id": r.review_id,
                "text": r.text,
                "rating": r.rating,
                "date": r.date,
                "author": r.author
            }
            for r in raw_reviews
        ]
        self.storage.save_raw_reviews(raw_review_dicts, date)
        logger.info(f"Ingested {len(raw_reviews)} reviews")
        
        # STAGE 2: Normalization
        all_statements = []
        for review in raw_reviews:
            statements = self.normalization_agent.normalize(review)
            all_statements.extend(statements)
        
        if not all_statements:
            logger.warning(f"No statements extracted for {date}")
            self._create_empty_daily_count(date)
            return
        
        logger.info(f"Normalized {len(all_statements)} statements")
        
        # STAGE 3: Candidate Generation
        candidates = []
        for statement in all_statements:
            candidate = self.candidate_generator.generate(statement)
            if candidate:
                candidates.append(candidate)
        
        if not candidates:
            logger.warning(f"No candidates generated for {date}")
            self._create_empty_daily_count(date)
            return
        
        logger.info(f"Generated {len(candidates)} topic candidates")
        
        # STAGE 4: Topic Consolidation
        topic_ids = []
        for candidate in candidates:
            topic_id = self.consolidation_agent.consolidate(
                candidate=candidate,
                registry=self.registry,
                current_date=date
            )
            topic_ids.append(topic_id)
        
        unique_topics = len(set(topic_ids))
        logger.info(f"Consolidated to {unique_topics} unique topics")
        
        # STAGE 5: Daily Counting
        processing_time = (datetime.now() - start_time).total_seconds()
        
        daily_count_data = self.daily_counter.count(
            topic_ids=topic_ids,
            date=date,
            total_reviews=len(raw_reviews),
            total_statements=len(all_statements),
            metadata={
                "processing_time_seconds": processing_time,
                "new_topics_created": self._count_new_topics_for_date(date),
                "app_package": app_package
            }
        )
        
        self.storage.save_daily_count(daily_count_data, date)
        
        # STAGE 6: Registry Persistence
        try:
            self.registry.save()
            logger.info(f"Day {date} complete: {len(raw_reviews)} reviews → {unique_topics} topics")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to save registry for {date}: {e}")
            if settings.CRASH_ON_REGISTRY_ERROR:
                raise
    
    def _create_empty_daily_count(self, date: str) -> None:
        """Create empty daily count for days with no data."""
        empty_count = {
            "date": date,
            "total_reviews_processed": 0,
            "total_statements_extracted": 0,
            "topic_counts": {},
            "metadata": {"note": "No actionable reviews"}
        }
        self.storage.save_daily_count(empty_count, date)
    
    def _generate_date_range(
        self,
        start: datetime,
        end: datetime
    ) -> List[str]:
        """Generate list of dates between start and end (inclusive)."""
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return dates
    
    def _count_new_topics_for_date(self, date: str) -> int:
        """Count topics created on this date."""
        count = 0
        for topic in self.registry.topics.values():
            if topic.created_on == date:
                count += 1
        return count


# Design Rationale and Trade-offs:
#
# 1. Why initialize all agents in __init__ instead of per-day?
#    - Agents are stateless (except registry)
#    - Initialization overhead avoided (API clients reused)
#    - Simpler code (no repeated initialization)
#    - Trade-off: Higher memory usage, but negligible for our agents
#
# 2. Why process days sequentially instead of parallel?
#    - Registry is stateful (topics evolve over time)
#    - Parallel processing would require complex locking
#    - Sequential matches pulsegen.md spec
#    - Trade-off: Slower (2 hours for 30 days), but correct
#
# 3. Why continue_on_day_failure instead of crash?
#    - Partial data better than no data (97% vs 0%)
#    - Aligns with trend continuity principle
#    - PMs can still see trends with gaps
#    - Trade-off: Silent data loss, mitigated by logging
#
# 4. Why save registry after EVERY day instead of at end?
#    - Crash safety: Don't lose 30 days of processing
#    - Incremental progress (can resume from last good day)
#    - Trade-off: More I/O, but registry is small (~1MB)
#
# 5. Why not implement checkpointing within a day?
#    - Single day processes fast (<5 minutes for 1000 reviews)
#    - Restart overhead is acceptable
#    - Simpler implementation (no state management)
#    - Trade-off: Wasted work on crash, but minimal
