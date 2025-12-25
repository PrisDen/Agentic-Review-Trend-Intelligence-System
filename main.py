"""
PulseGen - Review Trend Analysis Agent

CLI entry point for running the agentic pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.orchestrator import PipelineOrchestrator
import config.settings as settings


def setup_logging(log_level: str = "INFO"):
    """Configure logging for the entire application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=settings.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pulsegen.log")
        ]
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PulseGen - Agentic AI Review Trend Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 30 days of reviews for Swiggy app
  python main.py --app com.application.swiggy --target-date 2024-07-01

  # Process custom date range
  python main.py --app com.application.swiggy \\
                 --start-date 2024-06-01 \\
                 --target-date 2024-06-30

  # Use different window size
  python main.py --app com.application.swiggy \\
                 --target-date 2024-07-01 \\
                 --window-days 60

Note: Set GOOGLE_API_KEY environment variable before running.
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--app",
        required=True,
        help="Google Play package name (e.g., com.application.swiggy)"
    )
    
    parser.add_argument(
        "--target-date",
        required=True,
        help="Target date for trend analysis (YYYY-MM-DD)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--start-date",
        help="Start date (YYYY-MM-DD). Defaults to target-date minus window-days"
    )
    
    parser.add_argument(
        "--window-days",
        type=int,
        default=settings.DEFAULT_WINDOW_DAYS,
        help=f"Number of days to analyze (default: {settings.DEFAULT_WINDOW_DAYS})"
    )
    
    parser.add_argument(
        "--data-root",
        default=str(settings.DATA_ROOT),
        help=f"Data directory (default: {settings.DATA_ROOT})"
    )
    
    parser.add_argument(
        "--registry-path",
        default=str(settings.DATA_ROOT / "topic_registry.json"),
        help="Path to topic registry JSON"
    )
    
    parser.add_argument(
        "--log-level",
        default=settings.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=f"Logging level (default: {settings.LOG_LEVEL})"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate API key
    if not settings.GOOGLE_API_KEY:
        logger.error(
            "GOOGLE_API_KEY environment variable not set. "
            "Please set it before running PulseGen."
        )
        sys.exit(1)
    
    # Print banner
    print("=" * 60)
    print("PulseGen - Agentic AI Review Trend Analysis")
    print("=" * 60)
    print(f"App: {args.app}")
    print(f"Target Date: {args.target_date}")
    print(f"Window: {args.window_days} days")
    if args.start_date:
        print(f"Start Date: {args.start_date}")
    print(f"Mock Data: {settings.USE_MOCK_DATA}")
    print("=" * 60)
    print()
    
    try:
        # Initialize orchestrator
        logger.info("Initializing PulseGen pipeline...")
        orchestrator = PipelineOrchestrator(
            api_key=settings.GOOGLE_API_KEY,
            data_root=args.data_root,
            registry_path=args.registry_path
        )
        
        # Run pipeline
        output_path = orchestrator.run(
            app_package=args.app,
            target_date=args.target_date,
            start_date=args.start_date,
            window_days=args.window_days
        )
        
        # Success
        print()
        print("=" * 60)
        print("✅ Pipeline completed successfully!")
        print("=" * 60)
        print(f"Trend table: {output_path}")
        print(f"Metadata: {output_path.replace('.csv', '_metadata.json')}")
        print(f"Topic registry: {args.registry_path}")
        print("=" * 60)
        
        logger.info("PulseGen completed successfully")
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        print("\n⚠️  Pipeline interrupted")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n❌ Pipeline failed: {e}")
        print("Check pulsegen.log for details")
        sys.exit(1)


if __name__ == "__main__":
    main()


# Design Rationale and Trade-offs:
#
# 1. Why argparse instead of click or typer?
#    - Standard library (no extra dependencies)
#    - Sufficient for our needs (simple CLI)
#    - Well-documented and widely understood
#    - Trade-off: Less fancy than click, but simpler
#
# 2. Why validate API key at startup instead of lazy?
#    - Fail fast (don't process 10 days then crash)
#    - Clear error message upfront
#    - Better UX (user knows immediately)
#    - Trade-off: Can't run without API key even for --help
#
# 3. Why log to both stdout and file?
#    - stdout: Real-time monitoring during run
#    - file: Debugging after completion
#    - Trade-off: Double I/O, but logs are small
#
# 4. Why print banner instead of just logging?
#    - User-friendly progress indication
#    - Highlights key parameters before starting
#    - Makes CLI feel polished
#    - Trade-off: Slightly verbose, but helpful
#
# 5. Why exit codes (0 for success, 1 for failure)?
#    - Shell scripting integration
#    - CI/CD pipeline compatibility
#    - Standard UNIX convention
#    - Trade-off: None, just good practice
