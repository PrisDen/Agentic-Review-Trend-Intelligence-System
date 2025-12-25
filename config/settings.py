"""
Configuration settings for PulseGen.

Centralized configuration for all agents and pipeline parameters.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "output"

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# LLM Models
NORMALIZATION_MODEL = "gemini-1.5-flash"
CANDIDATE_GENERATION_MODEL = "gemini-1.5-flash"
CONSOLIDATION_MODEL = "gemini-1.5-flash"

# Embedding Configuration
EMBEDDING_MODEL = "models/text-embedding-004"
EMBEDDING_DIMENSIONS = 768

# Temperature settings (0.0 for deterministic)
LLM_TEMPERATURE = 0.0

# Normalization Agent
NORMALIZATION_MIN_CONFIDENCE = "medium"  # "high", "medium", or "low"
NORMALIZATION_MAX_RETRIES = 3
NORMALIZATION_TIMEOUT_SECONDS = 30

# Consolidation Agent Thresholds
HARD_MERGE_THRESHOLD = 0.85  # Auto-merge above this similarity
LOW_THRESHOLD = 0.70  # Auto-create below this similarity
TOP_K_SIMILAR_TOPICS = 5  # Number of similar topics for LLM adjudication
CONSOLIDATION_MAX_RETRIES = 3

# Pipeline Configuration
DEFAULT_START_DATE = "2024-06-01"
DEFAULT_WINDOW_DAYS = 30
CONTINUE_ON_DAY_FAILURE = True  # Graceful degradation
CRASH_ON_REGISTRY_ERROR = True  # Data integrity critical

# Ingestion
USE_MOCK_DATA = True  # Set to False for real Google Play scraping
MOCK_REVIEWS_PER_DAY = 50  # Number of mock reviews to generate

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# Design Rationale and Trade-offs:
#
# 1. Why environment variable for API key?
#    - Security: Don't hardcode secrets in code
#    - Flexibility: Different keys for dev/prod
#    - Standard practice for API credentials
#    - Trade-off: Requires setting env var, but safer
#
# 2. Why centralized config instead of argparse everywhere?
#    - Single source of truth for all parameters
#    - Easy to tune without editing multiple files
#    - Can load from config file in V2
#    - Trade-off: Less granular control, but simpler
#
# 3. Why Path objects instead of strings?
#    - Type safety (Path operations are type-checked)
#    - OS-agnostic path operations
#    - Trade-off: Need to convert to str sometimes, but safer
#
# 4. Why hardcode thresholds instead of auto-tuning?
#    - V1 simplicity (no ML required)
#    - Empirically validated values (0.70, 0.85)
#    - Can be tuned per domain manually
#    - Trade-off: Not adaptive, but predictable
