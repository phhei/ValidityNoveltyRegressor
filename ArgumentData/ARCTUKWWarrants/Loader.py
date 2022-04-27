from typing import Optional

import pandas
from loguru import logger
from transformers import PreTrainedTokenizer

from ArgumentData.GeneralDataset import ValidityNoveltyDataset
from ArgumentData.Utils import truncate_df

path = "ArgumentData/ARCTUKWWarrants/final_data_with_hr.csv"


def load_dataset(tokenizer: PreTrainedTokenizer, max_length_sample: Optional[int] = None,
                 max_number: int = -1, mace_ibm_threshold: Optional[float] = None,
                 include_topic: bool = False,
                 continuous_val_nov: bool = True, continuous_sample_weight: bool = False) -> ValidityNoveltyDataset:
    df = pandas.read_csv(path)
    logger.info("Load {} lines from \"{}\"", len(df), path)

    if mace_ibm_threshold is not None:
        logger.info("You want to filter the data, excluding all samples with a mace_ibm_score lower than {}. "
                    "With this, you ensure high-quality-samples", mace_ibm_threshold)
        logger.trace("MACE [...] as a scoring function for the quality of an argument based on crowd annotations. "
                     "MACE is an unsupervised item-response generative model which predicts the probability for each "
                     "label given the annotations. MACE also estimates a reliability score for each annotator which it "
                     "then uses to weigh this annotator’s judgments. "
                     "We use the probability MACE outputs for the positive label as the MACE-P scoring function.")

        if mace_ibm_threshold <= 0 or mace_ibm_threshold > 1:
            logger.warning("Your mace-threshold is {}. However, the mace is a probability score between 0 and 1, "
                           "hence, your threshold makes no sense!", mace_ibm_threshold)
        else:
            df = df[df.mace_ibm >= mace_ibm_threshold]
            logger.success("Filtered low-qualitative successfully, {} remain", len(df))

    df = truncate_df(df, max_number=max_number)

    samples = []

    for sid, row in df.iterrows():
        logger.trace("Let's generate samples for the {}. line", sid)
        try:
            samples.append(ValidityNoveltyDataset.Sample(
                premise="{}{}".format("{}: ".format(row["topic"]) if include_topic else "", row["premise"]),
                conclusion=row["claim"],
                validity=.75 + .25*row["mace_ibm"] if continuous_val_nov else 1,
                novelty=0 if (row["claim"] in row["premise"]) else (.8+.1*row["mace_ibm"] if continuous_val_nov else 1),
                weight=row["mace_ibm"] if continuous_sample_weight else 1
            ))

            samples.append(ValidityNoveltyDataset.Sample(
                premise="{}{} {} {} {}".format(
                    "{}: ".format(row["topic"]) if include_topic else "", row["premise"],
                    row["claim_keyword"], row["hidden_reasoning_keyword"], row["premise_keyword"]
                ),
                conclusion=row["claim"],
                validity=.75 + .25 * row["mace_ibm"] if continuous_val_nov else 1,
                novelty=0 if (row["claim"] in row["premise"]) else (.33 if continuous_val_nov else 0),
                weight=.75*row["mace_ibm"] if continuous_sample_weight else .5
            ))
        except TypeError:
            logger.opt(exception=True).warning("Corrupted CSV: {}->{}", sid, row)

        logger.debug("Added two samples: \"{}\" and \"{}\"", *samples[-2:])

    logger.debug("Successfully crawled through {} data-lines, succeeding with {} samples", len(df), len(samples))

    data = ValidityNoveltyDataset(
        samples=samples,
        tokenizer=tokenizer,
        max_length=max_length_sample or 100,
        name="ARCT-More"
    )

    logger.success("Successfully created the dataset {}, but {} samples are redundant -> removed",
                   data, data.deduplicate(original_data=True, extracted_data=True))

    return data
