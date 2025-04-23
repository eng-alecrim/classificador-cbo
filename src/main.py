import os

from common.logging import configure_logging
from dotenv import find_dotenv, load_dotenv
from loguru import logger

load_dotenv(find_dotenv())
project_name = os.getenv("PROJECT_NAME", "classificador-cbo")

configure_logging(project_name=project_name, log_to_file=True)


def main() -> None:
    logger.debug("Hello from classificador-cbo!")
    return None


if __name__ == "__main__":
    main()
