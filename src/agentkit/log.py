import logging

from .getenv import get_env

logger = logging.getLogger("agentkit")
logger.addHandler(logging.NullHandler())  # silent unless the app opts in


def configure_logging(level: str | None = None) -> None:
    """Call once in your app. Level falls back to LOG_LEVEL env, then INFO."""
    level = (level or get_env("LOG_LEVEL", "INFO")).upper()  # type: ignore[union-attr]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    logger.setLevel(level)
