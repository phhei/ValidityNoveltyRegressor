from pandas import DataFrame
from loguru import logger


def truncate_df(df: DataFrame, max_number: int = -1) -> DataFrame:
    if max_number >= 1:
        logger.debug("max-number is set to {}", max_number)
        if max_number >= len(df):
            logger.warning("You want more samples ({}) than you have ({}) -- skip max_number-param",
                           max_number, len(df))
        else:
            r_df = df[:max_number]
            logger.debug("Truncated to {} samples", len(r_df))
    else:
        logger.trace("No truncation of samples -- skip negative max_number-param")

    return r_df