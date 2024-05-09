import gzip
import json
import logging
from typing import Any, Dict, Iterator

import fsspec

logger = logging.getLogger(__name__)


def read_json_records(file_pattern: str) -> Iterator[Dict[str, Any]]:
    """Read json records from file_pattern."""
    logger.info("Reading json records from %s", file_pattern)
    count = 0
    for of in fsspec.open_files(file_pattern):
        with of as f:
            if of.path.endswith(".gz"):
                f = gzip.GzipFile(fileobj=f)
            for line in f:
                yield json.loads(line.strip())
                count += 1
        of.close()
    logger.info("Read %d records", count)
