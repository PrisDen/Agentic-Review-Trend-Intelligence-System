"""
Ingestion Agent.

Fetches reviews from Google Play Store for a specific app and date.
Supports both real scraping and mock data for testing.
"""

import logging
from typing import List, Optional
from datetime import datetime
import uuid

from src.models.review import Review

logger = logging.getLogger(__name__)


class IngestionAgent:
    """
    Fetches reviews from Google Play Store.
    
    Note: Google Play Store does not provide an official API for reviews,
    and historical review fetching by date is unreliable with scraping libraries.
    
    For production use:
    - Use google-play-scraper library to fetch recent reviews
    - Store reviews as they arrive (daily cron job)
    - Backfilling historical data requires manual export from Google Play Console
    
    For testing/demo:
    - Use mock_reviews() to generate synthetic data
    """
    
    def __init__(self, use_mock_data: bool = False):
        """
        Initialize ingestion agent.
        
        Args:
            use_mock_data: If True, generate mock reviews instead of scraping
        """
        self.use_mock_data = use_mock_data
        
        if use_mock_data:
            logger.info("Initialized IngestionAgent in MOCK mode")
        else:
            logger.info("Initialized IngestionAgent in REAL mode")
            # In real mode, would initialize google-play-scraper here
    
    def fetch_reviews(
        self,
        app_package: str,
        date: str,
        limit: int = 100
    ) -> List[Review]:
        """
        Fetch reviews for a specific app and date.
        
        Args:
            app_package: Google Play package name (e.g., "com.application.swiggy")
            date: Target date in YYYY-MM-DD format
            limit: Maximum number of reviews to fetch
        
        Returns:
            List of Review objects
        
        Note: In real mode, this would use google-play-scraper.
              Since Google Play doesn't support date-based filtering,
              we mock the data for demonstration purposes.
        """
        if self.use_mock_data:
            return self._generate_mock_reviews(app_package, date, limit)
        else:
            # Real scraping implementation would go here
            logger.warning(
                "Real scraping not implemented. "
                "Use google-play-scraper library for production. "
                "Falling back to mock data."
            )
            return self._generate_mock_reviews(app_package, date, limit)
    
    def _generate_mock_reviews(
        self,
        app_package: str,
        date: str,
        count: int
    ) -> List[Review]:
        """
        Generate synthetic reviews for testing.
        
        Creates realistic review patterns:
        - Mix of positive/negative ratings
        - Variety of complaint types
        - Some duplicate complaints (to test consolidation)
        """
        # Parse date to add variety based on day of month
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        day_seed = date_obj.day
        
        # Mock review templates (realistic Google Play store complaints)
        templates = [
            # Delivery issues (should consolidate)
            ("Delivery guy was very rude and unprofessional!", 1, "issue"),
            ("Delivery partner rude behavior. Not acceptable.", 1, "issue"),
            ("The delivery person was so impolite!", 2, "issue"),
            ("Rude delivery guy didn't even say thank you", 2, "issue"),
            
            # Food quality issues (should consolidate)
            ("Food arrived cold and packaging was terrible", 1, "issue"),
            ("Food was ice cold when it arrived!", 1, "issue"),
            ("Food not hot, completely cold", 2, "issue"),
            
            # App crashes (should consolidate)
            ("App crashes when I try to login with Google", 1, "issue"),
            ("App keeps crashing on Google login", 1, "issue"),
            ("Crashes during Google sign in", 2, "issue"),
            
            # Feature requests (should consolidate)
            ("Please add dark mode!!", 4, "request"),
            ("Need dark theme option", 4, "request"),
            ("Add dark mode feature", 4, "request"),
            
            # Payment issues
            ("Payment failed with credit card", 1, "issue"),
            ("Can't complete payment, app freezes", 1, "issue"),
            
            # Positive reviews
            ("Great app, very fast delivery!", 5, "feedback"),
            ("Love the app, easy to use", 5, "feedback"),
            ("Good service", 4, "feedback"),
            
            # Misc issues
            ("App is very slow", 2, "issue"),
            ("Too many ads", 3, "issue"),
            ("Delivery always late", 2, "issue"),
        ]
        
        # Vary number and type of reviews based on day
        # This creates different patterns on different days
        reviews = []
        template_count = len(templates)
        
        for i in range(count):
            # Cycle through templates with some variety
            template_idx = (i + day_seed) % template_count
            text, rating, category = templates[template_idx]
            
            # Add some variation to text (to test normalization)
            variations = [
                text,
                text.upper(),  # All caps
                f"OMG {text}",  # With prefix
                f"{text} Fix this ASAP!",  # With suffix
                text.replace("!", "."),  # Different punctuation
            ]
            varied_text = variations[i % len(variations)]
            
            # Create review
            review = Review(
                review_id=str(uuid.uuid4()),
                text=varied_text,
                rating=rating,
                date=date,
                author=f"user_{i}"
            )
            reviews.append(review)
        
        logger.info(f"Generated {len(reviews)} mock reviews for {date}")
        return reviews


# Design Rationale and Trade-offs:
#
# 1. Why support mock data instead of only real scraping?
#    - Google Play doesn't provide official API
#    - Third-party scrapers (google-play-scraper) are unreliable for historical data
#    - Historical data requires manual CSV export from Google Play Console
#    - Mock data enables testing and demo without API dependencies
#    - Trade-off: Mock data doesn't reflect real distribution, but good enough for demo
#
# 2. Why not implement real scraping in V1?
#    - google-play-scraper library doesn't support date-based filtering
#    - Would need to scrape all reviews and filter by date (expensive, unreliable)
#    - Production systems should store reviews as they arrive (daily cron)
#    - Trade-off: Not production-ready out of the box, but honest about limitations
#
# 3. Why generate realistic mock templates?
#    - Tests semantic consolidation (multiple phrasings of same issue)
#    - Tests normalization (varied capitalization, punctuation)
#    - Demonstrates end-to-end pipeline with realistic data
#    - Trade-off: Hardcoded templates vs random generation, but more realistic
#
# 4. Why vary reviews by day_seed?
#    - Creates different patterns on different days (tests trend detection)
#    - Some topics appear/disappear over time (realistic)
#    - Trade-off: Deterministic (same seed = same reviews), but reproducible
#
# 5. Why include positive reviews?
#    - Tests filtering (normalization should return empty for generic praise)
#    - Realistic review distribution (not all negative)
#    - Trade-off: Wasted processing on non-actionable reviews, but realistic
#
# Production Implementation Notes:
# --------------------------------
# For real Google Play scraping, use:
#
# from google_play_scraper import reviews, Sort
#
# def fetch_real_reviews(app_package: str, count: int = 100):
#     result, _ = reviews(
#         app_package,
#         lang='en',
#         country='us',
#         sort=Sort.NEWEST,
#         count=count
#     )
#     
#     # Convert to Review objects
#     review_objects = []
#     for r in result:
#         review = Review(
#             review_id=r['reviewId'],
#             text=r['content'],
#             rating=r['score'],
#             date=r['at'].strftime('%Y-%m-%d'),
#             author=r['userName']
#         )
#         review_objects.append(review)
#     
#     return review_objects
#
# Note: This fetches RECENT reviews, not historical by date.
# For historical data, export from Google Play Console and load from CSV.
