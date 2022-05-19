from argparse import ArgumentParser
from typing import Tuple, List

from HGTrainer import RobertaForValNovRegression, ValNovOutput
from pathlib import Path
from loguru import logger

import transformers
import pandas

if __name__ == "__main__":
    arg_parser = ArgumentParser(
        description="This python script will help you to load a (with Main.py-fine-tuned) model to make "
                    "Validity-Novelty-Predictions on a given CSV",
        add_help=True,
        allow_abbrev=False,
        exit_on_error=True
    )

    arg_parser.add_argument("model", type=Path, help="The path (directory) with the saved transformer-model")
    arg_parser.add_argument("data", type=Path, help="THE CSV which should be classified."
                                                    "MUST CONTAIN \"Premise\" and \"Conclusion\"-column")
    arg_parser.add_argument("length", type=int, help="The maximum token length of an sample input",
                            nargs="?", default=100)
    arg_parser.add_argument("--batch_size", default=32, type=int, required=False,
                            help="How many samples should be processed at once")

    parsed_args = arg_parser.parse_args()

    model: Path = parsed_args.model
    data: Path = parsed_args.data

    logger.info("Start classifying the file \"{}\"", data.name)

    df = pandas.read_csv(str(data.absolute()), encoding="utf-8", encoding_errors="replace")
    logger.debug("Read {} lines", len(df))

    hg_model: RobertaForValNovRegression = RobertaForValNovRegression.from_pretrained(str(model.absolute()))
    hg_model.loss = "ignore"
    logger.info("Loaded the learning model: {}", hg_model.name_or_path)
    hg_tokenizer: transformers.RobertaTokenizer = \
        transformers.AutoTokenizer.from_pretrained("{}-base".format(hg_model.config.model_type))

    validity = []
    novelty = []
    batch_part = ([], [])

    def process(batch: Tuple[List[str], List[str]]) -> None:
        logger.trace("Received following samples: {}",
                     "|".join(map(lambda i: "{}-->{}".format(batch[0][i], batch[1][i]), range(len(batch[0])))))

        # noinspection PyCallingNonCallable
        output: ValNovOutput = hg_model(**hg_tokenizer(text=batch[0],
                                                       text_pair=batch[1],
                                                       max_length=parsed_args.length,
                                                       padding=True,
                                                       truncation=True,
                                                       is_split_into_words=False,
                                                       return_tensors="pt"))

        out_validity = output.validity.tolist()
        if not isinstance(out_validity, List):
            out_validity = [out_validity]

        out_novelty = output.novelty.tolist()
        if not isinstance(out_novelty, List):
            out_novelty = [out_novelty]

        logger.info("Processed successfully {} samples (mean-val: {}/mean-nov: {}//{})",
                    len(batch[0]),
                    round(sum(out_validity)/len(out_validity), 2), round(sum(out_novelty)/len(out_novelty), 2),
                    output.logits.shape, sum)
        validity.extend(out_validity)
        novelty.extend(out_novelty)

    for row_id, row in df.iterrows():
        logger.trace("Process line {}", row_id)

        batch_part[0].append(row["Premise"] if row.notna()["Premise"] else "failure")
        batch_part[1].append(row["Conclusion"] if row.notna()["Conclusion"] else "failure")
        if len(batch_part[0]) < parsed_args.batch_size:
            logger.trace("Just add it to the batch")
            continue

        logger.debug("OK, batch is full, let's predict!")

        process(batch_part)
        batch_part[0].clear()
        batch_part[1].clear()

        logger.trace("Reset the lists...")

    if len(batch_part[0]) >= 1:
        logger.warning("There are still {} samples left (not full batch) - process it...", len(batch_part[0]))
        process(batch_part)

    out_path = data.parent.joinpath("{}_{}.csv".format(data.stem, model.stem))
    df["predicted validity"] = validity
    df["predicted novelty"] = novelty
    df.to_csv(path_or_buf=str(out_path.absolute()), encoding="utf-8", index=False)
    logger.success("Successfully processed {} samples, wrote to: {}", len(df), out_path)
