from typing import Literal, Optional
from loguru import logger
import pandas

from transformers import PreTrainedTokenizer
from ArgumentData.GeneralDataset import ValidityNoveltyDataset
from ArgumentData.Utils import truncate_df

train_path = "ArgumentData/ARCT/train-full.tsv"
train_path_adversarial = "ArgumentData/ARCT/adversarial_dataset/train-adv-negated.tsv"
dev_path = "ArgumentData/ARCT/dev-full.tsv"
dev_path_adversarial = "ArgumentData/ARCT/adversarial_dataset/dev-adv-negated.tsv"
test_path = None
test_path_adversarial = "ArgumentData/ARCT/adversarial_dataset/test-adv-negated.tsv"

claim_map = "ArgumentData/ARCT/adversarial_dataset/claim_map.csv"


def load_dataset(split: Literal["train", "dev", "test"], tokenizer: PreTrainedTokenizer,
                 include_adversarial_data: bool = True,
                 max_length_sample: Optional[int] = None, max_number: int = -1,
                 include_topic: bool = False, include_debate_info: bool = False,
                 continuous_val_nov: bool = True, continuous_sample_weight: bool = False) -> ValidityNoveltyDataset:
    if split == "train":
        selected_data_path = train_path
        selected_data_path_adversarial = train_path_adversarial
    elif split == "dev":
        selected_data_path = dev_path
        selected_data_path_adversarial = dev_path_adversarial
    else:
        selected_data_path = test_path
        selected_data_path_adversarial = test_path_adversarial

    if selected_data_path is not None:
        df = pandas.read_csv(selected_data_path, index_col="#id", sep="\t")
        logger.success("Successfully loaded {} samples from \"{}\"", len(df), selected_data_path)
    else:
        logger.warning("Selected split: {} -- we don't know a dataset for this split!", split)
        df = pandas.DataFrame(
            data=[("dummy-sample", "A results in B.", "A does not result in B.", 0, "We assume A.", "B occurs.",
                   "Will B occur?", "This is just a dummy example")],
            columns=["#id", "warrant0", "warrant1", "correctLabelW0orW1", "reason", "claim",
                     "debateTitle", "debateInfo"]
        ).set_index("#id", inplace=False)

    if include_adversarial_data:
        logger.debug("You want to include the adversarial data (inverted claims), too! OK, let's do this.")
        if selected_data_path_adversarial is not None:
            try:
                old_len_df = len(df)
                df = pandas.concat(
                    objs=(
                        df,
                        pandas.read_csv(selected_data_path_adversarial, index_col="#id", sep="\t").
                        query(expr="adversarial==True", inplace=False).drop(columns=["adversarial"], inplace=False)
                    ),
                    axis="index",
                    ignore_index=False,
                    verify_integrity=True,
                    copy=False
                )
                logger.success("Successfully appended {} adversarial samples!", len(df)-old_len_df)
            except ValueError:
                logger.opt(exception=True).error("Something doesn't fit here - \"{}\" seems to be corrupted",
                                                 selected_data_path_adversarial)
        else:
            logger.warning("You want to add adversarial examples -- but there aren't some!")
    else:
        logger.debug("Adding adversarial samples is not desired...")

    claim_map_df = pandas.read_csv(claim_map, index_col="original")
    logger.trace("Loaded {} negated claims", claim_map_df)

    data_df = truncate_df(df, max_number)

    samples = []

    try:
        for sid, row_data in data_df.iterrows():
            logger.debug("Process samples {}", sid)
            weight_ref = .9 if str(sid).endswith("adversarial") and continuous_sample_weight else 1

            logger.trace("Let's add a standard sample (valid and novel)")
            samples.append(
                ValidityNoveltyDataset.Sample(
                    premise="{}{}{} {}".format(
                        "{}: ".format(row_data["debateTitle"]) if include_topic else "",
                        "({}) ".format(row_data["debateInfo"]) if include_debate_info else "",
                        row_data["reason"],
                        row_data["warrant1"] if bool(row_data["correctLabelW0orW1"]) else row_data["warrant0"]),
                    conclusion=row_data["claim"],
                    validity=1,
                    novelty=(.5 if include_topic else .85) if continuous_val_nov else 1,
                    weight=weight_ref
                )
            )
            logger.trace("Added a ARCT-sample: {}", samples[-1])

            logger.trace("Let's mess up: destroy the validity with the false warrant!")
            samples.append(
                ValidityNoveltyDataset.Sample(
                    premise="{}{}{} {}".format(
                        "{}: ".format(row_data["debateTitle"]) if include_topic else "",
                        "({}) ".format(row_data["debateInfo"]) if include_debate_info else "",
                        row_data["reason"],
                        row_data["warrant0"] if bool(row_data["correctLabelW0orW1"]) else row_data["warrant1"]),
                    conclusion=row_data["claim"],
                    validity=.05 if continuous_val_nov else 0,
                    novelty=(.5 if include_topic else .85) if continuous_val_nov else 1,
                    weight=weight_ref
                )
            )
            logger.trace("Added a false-warrant-ARCT-sample: {}", samples[-1])

            logger.trace("No let's destroy the novelty, too!")
            try:
                samples.append(
                    ValidityNoveltyDataset.Sample(
                        premise="{}{}{} {}".format(
                            "{}: ".format(row_data["debateTitle"]) if include_topic else "",
                            "({}) ".format(row_data["debateInfo"]) if include_debate_info else "",
                            row_data["reason"],
                            row_data["warrant0"]),
                        conclusion=row_data["warrant1"],
                        validity=0,
                        novelty=.05 if continuous_val_nov else 0,
                        weight=weight_ref*(.8 if continuous_sample_weight else 1)
                    )
                )
                samples.append(
                    ValidityNoveltyDataset.Sample(
                        premise="{}{}{} {}".format(
                            "{}: ".format(row_data["debateTitle"]) if include_topic else "",
                            "({}) ".format(row_data["debateInfo"]) if include_debate_info else "",
                            row_data["reason"],
                            row_data["warrant1"]),
                        conclusion=row_data["warrant0"],
                        validity=0,
                        novelty=.05 if continuous_val_nov else 0,
                        weight=weight_ref*(.8 if continuous_sample_weight else 1)
                    )
                )
                samples.append(
                    ValidityNoveltyDataset.Sample(
                        premise="{}{}{} {}".format(
                            "{}: ".format(row_data["debateTitle"]) if include_topic else "",
                            "({}) ".format(row_data["debateInfo"]) if include_debate_info else "",
                            row_data["reason"],
                            row_data["claim"]),
                        conclusion=claim_map_df.loc[row_data["claim"]]["negated"],
                        validity=0,
                        novelty=.05 if continuous_val_nov else 0,
                        weight=weight_ref
                    )
                )
            except KeyError:
                logger.opt(exception=False).info("Unfortunately, we don't know the inverted conclusion for "
                                                 "the sample {}", sid)
    except KeyError:
        logger.opt(exception=True).error("Your database \"{}\" is corrupted! Skip remaining lines!",
                                         selected_data_path)

    data = ValidityNoveltyDataset(
        samples=samples,
        tokenizer=tokenizer,
        max_length=100 + 8 * int(include_topic) + 16 * int(include_debate_info)
        if max_length_sample is None else max_length_sample,
        name="ARCT{}_{}".format("+adversarial" if include_adversarial_data else "", split)
    )

    logger.success("Successfully created the dataset: {}", data)

    return data
