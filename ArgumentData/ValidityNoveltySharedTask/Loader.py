import pandas

from typing import Literal, Optional, List
from loguru import logger

from transformers import PreTrainedTokenizer
from ArgumentData.GeneralDataset import ValidityNoveltyDataset
from ArgumentData.Sample import Sample
from ArgumentData.Utils import truncate_dataset

base_path_absolute_ratings = "ArgumentData/ValidityNoveltySharedTask/TaskA_"
base_path_relative_ratings = "ArgumentData/ValidityNoveltySharedTask/TaskB_"


def load_dataset(split: Literal["train", "dev", "test", "test_without_dev_topics"],
                 tokenizer: PreTrainedTokenizer,
                 max_length_sample: Optional[int] = None,
                 max_number: int = -1,
                 min_confidence: Literal["defeasible", "majority", "confident", "very confident"] = "majority",
                 include_topic: bool = True,
                 continuous_val_nov: bool = False, continuous_sample_weight: bool = False) -> ValidityNoveltyDataset:
    logger.trace("OK, let's load {}", split)

    df_absolute: pandas.DataFrame = pandas.read_csv(
        filepath_or_buffer="{}{}.csv".format(base_path_absolute_ratings,
                                             "test" if split == "test_without_dev_topics" else split),
        encoding="utf-8",
        encoding_errors="ignore"
    )

    df_relative: pandas.DataFrame = pandas.read_csv(
        filepath_or_buffer="{}{}.csv".format(base_path_relative_ratings,
                                             "test" if split == "test_without_dev_topics" else split),
        encoding="utf-8",
        encoding_errors="ignore"
    )

    logger.info("Load {} samples with {} samples for a more fine-grained validity/ novelty",
                len(df_absolute), len(df_relative))

    if min_confidence != "defeasible":
        accepted_confidence_types: List[str] = ["very confident"]
        if min_confidence == "confident" or min_confidence == "majority":
            accepted_confidence_types.append("confident")
        if min_confidence == "majority":
            accepted_confidence_types.append("majority")

        logger.trace("We have to filter our data since you want only samples which are: ",
                     " or ".join(accepted_confidence_types))
        df_absolute.query(
            expr="({})".format(
                ") and (".join(map(lambda aspect: " or ".join(map(lambda le: "`{}` == '{}'".format(aspect, le),
                                                                  accepted_confidence_types)),
                                   ["Validity-Confidence", "Novelty-Confidence"]))),
            inplace=True
        )
        logger.debug("Successfully ensures {}. {} samples left", accepted_confidence_types, len(df_absolute))

    if split == "test_without_dev_topics":
        logger.trace("You want to discard all samples topic-premise-shared with the dev-set...")
        df_absolute.query(expr="`Topic-in-dev-split` == 'no'", inplace=True)
        logger.info("{} samples left after removing the dev-overlap.", len(df_absolute))

    df_absolute = truncate_dataset(data=df_absolute, max_number=max_number)

    data: List[Sample] = []

    try:
        for row_id, row in df_absolute.iterrows():
            logger.trace("Process line {}", row_id)

            base_validity = (row["Validity"]+1)/2 if row["Validity"] != 0 else None
            base_novelty = (row["Novelty"]+1)/2

            ref_comparable_concl1: Optional[pandas.DataFrame] = None
            ref_comparable_concl2: Optional[pandas.DataFrame] = None
            if continuous_val_nov:
                prem = row["Premise"]
                concl = row["Conclusion"]
                ref_comparable_concl1: pandas.DataFrame = df_relative.query(
                    expr="`Premise` == @prem and `Conclusion 1` == @concl",
                    inplace=False
                )
                ref_comparable_concl2: pandas.DataFrame = df_relative.query(
                    expr="`Premise` == @prem and `Conclusion 2` == @concl",
                    inplace=False
                )
                logger.debug("Found {} and {} comparable samples to this sample",
                             len(ref_comparable_concl1), len(ref_comparable_concl2))

            data.append(Sample(
                premise="{}{}".format("{}: ".format(row["topic"]) if include_topic else "", row["Premise"]),
                conclusion=row["Conclusion"],
                validity=base_validity if base_novelty is None or not continuous_val_nov else
                (base_validity+1/3)/(5/3) +
                ((ref_comparable_concl1["Votes_Concl1IsMoreValid"].sum() +
                  ref_comparable_concl2["Votes_Concl2IsMoreValid"].sum()) /
                 (5*3*(len(ref_comparable_concl1)+len(ref_comparable_concl2)))) -
                ((ref_comparable_concl1["Votes_Concl2IsMoreValid"].sum() +
                  ref_comparable_concl2["Votes_Concl1IsMoreValid"].sum()) /
                 (5*3*(len(ref_comparable_concl1)+len(ref_comparable_concl2)))),
                novelty=base_novelty if not continuous_val_nov else
                (base_novelty+1/3)/(5/3) +
                ((ref_comparable_concl1["Votes_Concl1IsMoreNovel"].sum() +
                  ref_comparable_concl2["Votes_Concl2IsMoreNovel"].sum()) /
                 (5*3*(len(ref_comparable_concl1)+len(ref_comparable_concl2)))) -
                ((ref_comparable_concl1["Votes_Concl2IsMoreNovel"].sum() +
                  ref_comparable_concl2["Votes_Concl1IsMoreNovel"].sum()) /
                 (5*3*(len(ref_comparable_concl1)+len(ref_comparable_concl2)))),
                weight=(1 +
                        int(row["Validity-Confidence"] == "majority")*.25 +
                        int(row["Validity-Confidence"] == "confident") +
                        int(row["Validity-Confidence"] == "very confident")*2 +
                        int(row["Novelty-Confidence"] == "majority")*.25 +
                        int(row["Novelty-Confidence"] == "confident") +
                        int(row["Novelty-Confidence"] == "very confident")*2) if continuous_sample_weight else 3,
                source="ValNovOwnDataV2[Premise-->Conclusion]"
            ))
            logger.debug("Successfully added a original sample: {}", data[-1])
    except KeyError:
        logger.opt(exception=True).error("Your database \"{}...\" is corrupted! Skip remaining lines!",
                                         base_path_absolute_ratings)

    dataset = ValidityNoveltyDataset(
        samples=data,
        tokenizer=tokenizer,
        max_length=132 + 8 * int(include_topic) if max_length_sample is None else max_length_sample,
        name="annotated_{}".format(split)
    )

    logger.success("Successfully retrieved {} samples: {}", len(dataset), dataset.get_sample_class_distribution())

    return dataset
