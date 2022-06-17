import pathlib
from typing import Optional

import pandas
from loguru import logger
from nltk import sent_tokenize
from transformers import PreTrainedTokenizer

from ArgumentData.GeneralDataset import ValidityNoveltyDataset
from ArgumentData.Sample import Sample
from ArgumentData.Utils import truncate_dataset

annotation_file = "ArgumentData/StudentEssays/sufficient/annotations.csv"
main_file = "ArgumentData/StudentEssays/sufficient/all-samples.tsv"
essays_path = "ArgumentData/StudentEssays/essays"


def load_dataset(tokenizer: PreTrainedTokenizer, max_length_sample: Optional[int] = None,
                 max_number: int = -1, exclude_samples_without_detail_annotation_info: bool = True,
                 continuous_val_nov: bool = False, continuous_sample_weight: bool = False) -> ValidityNoveltyDataset:
    main_df = pandas.read_csv(main_file, sep="\t", encoding_errors="ignore", index_col=["ESSAY", "ARGUMENT"])
    main_df.fillna(value="sufficient", inplace=True)
    details = pandas.read_csv(annotation_file, sep=";", quoting=1, index_col=["ESSAY", "ARGUMENT"])
    logger.trace("For the file \"{}\": Sufficiency means in this case a sufficiency flaw, ergo insufficient",
                 annotation_file)
    details.fillna(value=1, inplace=True)
    details.replace(to_replace="Sufficiency", value=0, inplace=True)

    logger.info("Loaded {} samples with {} of them with detailed information", len(main_df), len(details))

    main_df = main_df.join(other=details, how="left", rsuffix="_details")
    logger.debug("Joined together: {}", main_df)

    essays_df = {
        int(file.stem[-3:]): pandas.read_csv(
            str(file), sep="\t", header=None,  names=["ID", "AU-start-end", "text", "overflow"]
        ).query(expr="ID in ({})".format(", ".join(map(lambda i: "\"T{}\"".format(i), range(30)))), inplace=False)
        for file in pathlib.Path(essays_path).glob(pattern="*.ann")
    }

    logger.debug("Loaded {} essays", len(essays_df))
    logger.trace(" :: ".join(map(lambda k: str(k), essays_df.keys())))

    if exclude_samples_without_detail_annotation_info:
        main_df = main_df[main_df.notna().all(axis="columns")]
        logger.info("Excluded samples without any details od annotation, having {} left", len(main_df))

    main_df = truncate_dataset(main_df, max_number=max_number)

    samples = []

    for sid, row in main_df.iterrows():
        # noinspection PyUnresolvedReferences
        essay_number = sid[0]

        logger.trace("Processing sample \"{}\" -- essay number {}", sid, essay_number)

        essay_df: pandas.DataFrame = essays_df[essay_number]

        try:
            sentences = [
                (
                    es["AU-start-end"].iloc[0].split(" ")[0]
                    if len(es := essay_df.drop(index=[_sid for _sid, _row in essay_df.iterrows()
                                                      if _row["text"].lower() not in s.lower()],
                                               inplace=False)) >= 1 else None,
                    s
                ) for s in sent_tokenize(row["TEXT"] if pandas.isna(row["TEXT_details"]) else row["TEXT_details"])
            ]
            logger.trace(sentences)
            sentences_none = [s for t, s in sentences if t is None]

            if len(sentences_none) >= 1:
                logger.warning("Following sentences in sample \"{}\" couldn't be related to the "
                               "related argument unit type: {}",
                               sid, " // ".join(sentences_none))

            premise = " ".join([s for t, s in sentences if t == "Premise"])
            claim = " ".join([s for t, s in sentences if t == "Claim"])
            major_claim = " ".join([s for t, s in sentences if t == "MajorClaim"])

            logger.debug("Identified following parts for sample {}: \"{}\"->\"{}\"->\"{}\"",
                         sid, premise, claim, major_claim)

            if sum((int(len(premise) >= 1), int(len(claim) >= 1), int(len(major_claim) >= 1))) >= 2:
                samples.append(Sample(
                    premise=(premise + " " + claim) if len(major_claim) >= 1 else premise,
                    conclusion=major_claim if len(major_claim) >= 1 else claim,
                    validity=int(row["ANNOTATION"] == "sufficient") if not continuous_val_nov or pandas.isna(row["A1"])
                    else ((row["A1"]+row["A2"]+row["A3"])/3),
                    novelty=None,
                    weight=.5+(int(pandas.notna(row["A1"]))+int(pandas.notna(row["A2"]))+int(pandas.notna(row["A3"])))/3
                    if continuous_sample_weight else .75,
                    source="StudentEssays[]"
                ))
                logger.trace("Produced a new sample: {}", samples[-1])
            else:
                logger.warning("Can't create a sample for \"{}\": not enough retrieved parts: {}/{}/{}",
                               sid,
                               "premise-{}".format(len(premise)) if len(premise) >= 1 else "premise missing",
                               "conclusion-{}".format(len(claim)) if len(claim) >= 1 else "conclusion missing",
                               "major-conclusion-{}".format(len(major_claim)) if len(major_claim) >= 1
                               else "major-conclusion missing")
        except KeyError:
            logger.opt(exception=True).critical("Corrupted input file: {}", main_file)
        except LookupError:
            logger.opt(exception=True).error("Please resolve this NLTK-dependency!")

    data = ValidityNoveltyDataset(
        samples=samples,
        tokenizer=tokenizer,
        max_length=max_length_sample or 129,
        name="Student-essays{}{}{}".format(
            "_" if continuous_val_nov or continuous_sample_weight else "",
            "C" if continuous_val_nov else "", "CW" if continuous_sample_weight else ""
        )
    )

    logger.success("Successfully created the dataset with {} out of {} samples", len(data), len(main_df))

    return data
