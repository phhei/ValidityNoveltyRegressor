from typing import Literal, Optional
from loguru import logger
import pandas

from transformers import PreTrainedTokenizer
from ArgumentData.GeneralDataset import ValidityNoveltyDataset
from ArgumentData.Utils import truncate_df

main_path = "ArgumentData/IBMArgQuality/original/arg_quality_rank_30k.csv"
extension_path = "ArgumentData/IBMArgQuality/extension/arguments.tsv"


def load_dataset(tokenizer: PreTrainedTokenizer, split: Literal["all", "train", "dev", "test"] = "all",
                 max_length_sample: Optional[int] = None,
                 max_number: int = -1, quality_threshold: Optional[float] = None,
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

    if split != "all":
        logger.trace("You want to filter a certain split: {}", split)
        main_df = main_df[main_df.set == split]
        extension_df = None if extension_df is None else extension_df[extension_df.Usage == split]
        logger.info("Your selected split includes {} samples (main) and {} samples (extension)",
                    len(main_df), "n/a" if extension_df is None else len(extension_df))

    if quality_threshold is not None:
        logger.trace("You want to filter the samples (quality >= {})", quality_threshold)
        if quality_threshold <= 0 or quality_threshold > 1:
            logger.warning("Your quality threshold of {} makes no sense since it's in [{},{}]-area",
                           quality_threshold, round(main_df["WA"].min(), 2), round(main_df["WA"].max(), 2))
        else:
            main_df = main_df[main_df.WA >= quality_threshold]
            logger.info("The quality check reduces further the sample amount to {} samples", len(main_df))

    main_df = truncate_df(main_df, max_number=max_number)

    samples = []

    for sid, row in main_df.iterrows():
        logger.trace("Processing sample {}", sid)

        samples.append(ValidityNoveltyDataset.Sample(
            premise=row["argument"],
            conclusion=row["topic"],
            validity=max(0, min(1, .5+(row["stance_WA"]*row["stance_WA_conf"]*(.5*row["WA"]/2))*2))
            if continuous_val_nov else max(0, row["stance_WA"]),
            novelty=None,
            weight=.5 + row["WA"]/2 + row["stance_WA"]/3 if continuous_sample_weight else 1 + row["stance_WA"]/3
        ))

        logger.trace("Produced a new main sample: {}", samples[-1])

    logger.info("Having {} samples now", len(samples))

    for sid, row in ([] if extension_df is None else extension_df.iterrows()):
        logger.trace("Processing sample {} (extension)", sid)

        samples.append(ValidityNoveltyDataset.Sample(
            premise=row["Premise"],
            conclusion=row["Conclusion"],
            validity=int(row["Stance"] == "in favor of"),
            novelty=None,
            weight=.4 + int(row["Part"] != "china") if continuous_sample_weight else .5
        ))

        logger.trace("Produced a new extension sample: {}", samples[-1])

    logger.info("Having {} samples now", len(samples))

    data = ValidityNoveltyDataset(
        samples=samples,
        tokenizer=tokenizer,
        max_length=max_length_sample or 96,
        name="IBM-Arg-Quality"
    )

    logger.success("Successfully created the dataset {}, but {} samples are redundant -> removed",
                   data, data.deduplicate(original_data=True, extracted_data=True))

    return data
