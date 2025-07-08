import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Python path:")
for path in sys.path:
    logger.info(path)

try:
    from scripts.multifactor import AdvancedMultiFactorPMM
    logger.info("Successfully imported AdvancedMultiFactorPMM")
except Exception as e:
    logger.error(f"Failed to import AdvancedMultiFactorPMM: {str(e)}")

try:
    from hummingbot.connector.connector_base import ConnectorBase
    logger.info("Successfully imported ConnectorBase")
except Exception as e:
    logger.error(f"Failed to import ConnectorBase: {str(e)}") 