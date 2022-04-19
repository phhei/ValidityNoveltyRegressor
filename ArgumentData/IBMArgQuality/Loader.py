from typing import Literal, Optional
from loguru import logger
import pandas

from transformers import PreTrainedTokenizer
from ArgumentData.GeneralDataset import ValidityNoveltyDataset
from ArgumentData.Utils import truncate_df

main_path = "ArgumentData/IBMArgQuality/original/arg_quality_rank_30k.csv"
extension_path = "ArgumentData/IBMArgQuality/extension/arguments.tsv"


def load_dataset(tokenizer: PreTrainedTokenizer, max_length_sample: Optional[int] = None,
                 max_number: int = -1, mace_ibm_threshold: Optional[float] = None,
                 include_topic: bool = False,
                 continuous_val_nov: bool = True, continuous_sample_weight: bool = False) -> ValidityNoveltyDataset:
    main_df = pandas.read_csv(main_path)
    logger.info("Read {} arguments from the main file \"{}\"", len(main_df), main_path)

    if max_number < 0:
        extension_df = pandas.read_csv(extension_path, sep="\t", index_col="Argument ID")
        extension_df_len = len(extension_df)
        extension_df = extension_df[extension_df.Part != "usa"]
        logger.info("Loaded {} usable extensions from \"{}\" ({} before filtering)",
                    len(extension_df), extension_path, extension_df_len)
    else:
        logger.warning("You limit your sample amount to {} - hence, we don't load an extension here ;)", max_number)
        extension_df = None

    main_df = truncate_df(main_df, max_number=max_number)

    #TODO
