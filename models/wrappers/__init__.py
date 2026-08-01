import logging
from config.settings import settings

logger = logging.getLogger(__name__)

logger.info(f"🔥 目前實例化使用的 learning_rate: {settings.default_params}")