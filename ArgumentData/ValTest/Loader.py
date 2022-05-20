from typing import Literal, Optional
from loguru import logger
import pandas

from transformers import PreTrainedTokenizer
from ArgumentData.GeneralDataset import ValidityNoveltyDataset
from ArgumentData.Utils import truncate_df

dev_path = "ArgumentData/ValTest/dev_better_balanced.csv"
test_path = "ArgumentData/ValTest/test_better_balanced-with-dev-topics.csv"
data_path = "ArgumentData/ValTest/validity_novelty_corpus.csv"
comparative_path = "ArgumentData/ValTest/validity_novelty_corpus-comparative.csv"


def load_dataset(split: Literal["dev", "test"], tokenizer: PreTrainedTokenizer, max_length_sample: Optional[int] = None,
                 max_number: int = -1, deduplicate: bool = True, include_topic: bool = True,
                 continuous_val_nov: bool = True, continuous_sample_weight: bool = False) -> ValidityNoveltyDataset:
    df = pandas.read_csv(data_path, sep="|", index_col="argument_ID", quotechar=None, quoting=3)
    df_comparative = pandas.read_csv(comparative_path,
                                     sep="|", index_col="argument_sample_ID", quotechar=None, quoting=3)
    data_index = pandas.read_csv(dev_path if split == "dev" else test_path, index_col="index")
    data_df = df.join(other=data_index, how="inner").join(other=df_comparative, how="left", rsuffix="_comp")

    logger.info("Loaded {} samples from \"{}\"", len(data_df), data_path)

    data_df = truncate_df(data_df, max_number)

    samples = []

    try:
        for sid, row_data in data_df.iterrows():
            logger.debug("Process samples {}", sid)
            num_voters = sum(
                [row_data["number_pos_validity_votes"],
                 row_data["number_neutral_validity_votes"],
                 row_data["number_neg_validity_votes"]]
            )
            num_comparable_voters = sum(
                [row_data["votes_underperforming_in_validity"]
                 if pandas.notna(row_data["votes_underperforming_in_validity"]) else 0,
                 row_data["votes_equal_in_validity"]
                 if pandas.notna(row_data["votes_equal_in_validity"]) else 0,
                 row_data["votes_outperforming_in_validity"]
                 if pandas.notna(row_data["votes_outperforming_in_validity"]) else 0])
            logger.trace("num_voters: {}/ num_comparable_voters: {}", num_voters, num_comparable_voters)
            if continuous_val_nov:
                if num_comparable_voters > 0:
                    validity = \
                        (.9*row_data["number_pos_validity_votes"]/num_voters +
                         .5*row_data["number_neutral_validity_votes"]/num_voters) * \
                        (1+(1/9)*row_data["votes_outperforming_in_validity"]/num_comparable_voters -
                         (1/9)*row_data["votes_underperforming_in_validity"]/num_comparable_voters)
                    novelty = \
                        (.9*row_data["number_pos_novelty_votes"]/num_voters +
                         .5*row_data["number_neutral_novelty_votes"]/num_voters) * \
                        (1+(1/9)*row_data["votes_outperforming_in_novelty"]/num_comparable_voters -
                         (1/9)*row_data["votes_underperforming_in_novelty"]/num_comparable_voters)
                else:
                    validity = \
                        (row_data["number_pos_validity_votes"]/num_voters +
                         .5*row_data["number_neutral_validity_votes"]/num_voters)
                    novelty = \
                        (row_data["number_pos_validity_votes"]/num_voters +
                         .5*row_data["number_neutral_validity_votes"]/num_voters)
            else:
                validity = \
                    min(1, max(0, row_data["number_pos_validity_votes"] - row_data["number_neg_validity_votes"] + 0.5))
                novelty = \
                    min(1, max(0, row_data["number_pos_novelty_votes"] - row_data["number_neg_novelty_votes"] + 0.5))
            if continuous_sample_weight:
                if num_comparable_voters > 0:
                    weight = \
                        5 - ((row_data["number_neutral_validity_votes"] + row_data["number_neutral_novelty_votes"])
                             * (3. / (2 * num_voters)))
                else:
                    weight = \
                        3 - ((row_data["number_neutral_validity_votes"] + row_data["number_neutral_novelty_votes"])
                             * (2. / (2 * num_voters)))
            else:
                weight = 3
            samples.append(ValidityNoveltyDataset.Sample(
                premise="{}: {}".format(row_data["topic"], row_data["premise"])
                if include_topic else row_data["premise"],
                conclusion=row_data["conclusion_text"],
                validity=validity,
                novelty=novelty,
                weight=weight,
                source="ValNovOwnData[Premise-->Conclusion]"
            ))
            logger.debug("Added sample: {}", samples[-1])
            if continuous_val_nov:
                logger.debug("... with {}% validity and {}% novelty", round(100*validity), round(100*novelty))
    except KeyError:
        logger.opt(exception=True).error("Your database \"{}\" is corrupted! Skip remaining lines!",
                                         dev_path if split == "dev" else test_path)

    data = ValidityNoveltyDataset(
        samples=samples,
        tokenizer=tokenizer,
        max_length=132+8*int(include_topic) if max_length_sample is None else max_length_sample,
        name="annotated_{}".format(split)
    )

    if deduplicate:
        data_len = len(data)
        data.deduplicate(original_data=True, extracted_data=True)
        logger.info("Deduplicate... before deduplication, we had {} samples, now {} samples (-{}%)",
                    data_len, len(data), round(100*len(data)/data_len))

    logger.success("Successfully created the dataset: {}", data)

    return data
