import json
import logging

from main import aggregate_data
from shared.aws import extract_index


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event, context):
    logger.info("Aggregation request received", extra={"event": event})
    index = extract_index(event)
    if not index:
        logger.warning("Aggregation request missing index")
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "index is required"}),
        }

    logger.info("Starting aggregation", extra={"index": index})
    aggregate_data(index)
    logger.info("Aggregation completed", extra={"index": index})

    return {
        "statusCode": 200,
        "body": json.dumps({"status": "ok", "index": index}),
    }
