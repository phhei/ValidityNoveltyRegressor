import pathlib
from argparse import ArgumentParser

import torch
import torch.cuda
import numpy

from typing import List, Tuple, Dict

from transformers import BatchEncoding
from ArgumentData.GeneralDataset import ValidityNoveltyDataset
from HGTrainer import _val_nov_metric, ValNovOutput, RobertaForValNovRegression
from loguru import logger

if __name__ == "__main__":
    arg_parser = ArgumentParser(
        description="This python script will find the best dev-fit-model among several repetitions of the same config."
                    "Is only compatible with >=V0.4.",
        add_help=True,
        allow_abbrev=False,
        exit_on_error=True
    )

    arg_parser.add_argument("root", type=pathlib.Path, help="The root path (directory) containing various runs")
    arg_parser.add_argument("--batch_size", default=32, type=int, required=False,
                            help="How many samples should be processed at once")
    arg_parser.add_argument("--key", action="store", required=False, default="f1_macro", type=str,
                            help="Metric jey which should be considered. Greater should be better.")
    parsed_args = arg_parser.parse_args()

    root_path: pathlib.Path = parsed_args.root
    batch_size = parsed_args.batch_size
    metric_key: str = parsed_args.key

    if not root_path.exists():
        logger.error("\"{}\" doesn't exist", root_path.absolute())

    logger.add(sink=root_path.joinpath("best_models.txt"),
               level="SUCCESS",
               colorize=False,
               catch=True,
               encoding="utf-8",
               mode="a",
               filter="__main__")

    loss = []
    logger.success("Let's go - start crawling \"{}\"", root_path)

    for dataset_config_path in root_path.iterdir():
        logger.info("Let's explore: {}", dataset_config_path.name)
        if dataset_config_path.is_file():
            logger.warning("Unexpected file in this base: \"{}\"", dataset_config_path.absolute())
            continue
        for sample_size_path in dataset_config_path.iterdir():
            logger.info("Let's explore: {}", sample_size_path.name)
            if dataset_config_path.is_file():
                logger.warning("Unexpected file in this sample size base: \"{}\"", sample_size_path.absolute())
                continue

            try_folders: List[pathlib.Path] = [d for d in sample_size_path.iterdir() if d.is_dir()]
            if len(try_folders) == 0:
                logger.warning("[} is empty!", sample_size_path.absolute())
                continue

            try_folders.sort(key=lambda d: int(d.name[d.name.index("-")+1:]), reverse=True)
            logger.debug("Found following tries: {}", ", ".join(map(lambda d: d.name, try_folders)))
            logger.info("Consider: {}", try_folders[0])

            validation_performances: List[Tuple[pathlib.Path, Dict[str, float], Dict[str, float]]] = []

            for repetition_folder in try_folders[0].iterdir():
                if repetition_folder.is_dir() and repetition_folder.name.startswith("repetition-"):
                    datasets = ValidityNoveltyDataset.load(path=repetition_folder)
                    logger.debug("Loads following datasets: {}({}% synthetic)/{}/{}",
                                 datasets["train"],
                                 round(100*sum([int("#" in s.source) for s in datasets["train"].samples_extraction]) /
                                       len(datasets["train"].samples_extraction)),
                                 datasets["train"],
                                 datasets["test"])

                    transformer = RobertaForValNovRegression.from_pretrained(str(repetition_folder.absolute()))
                    if torch.cuda.is_available():
                        transformer.to("cuda")
                    logger.trace("Loaded: {}", transformer)

                    validity_predictions: Dict = dict()
                    novelty_predictions: Dict = dict()
                    should_validity: Dict = dict()
                    should_novelty: Dict = dict()

                    for split in ["dev", "test"]:
                        logger.debug("Analyse {}", split)
                        for i in range(0, len(datasets[split]), batch_size):
                            logger.trace("Batch {}/{}", i, len(datasets[split]))
                            should_validity[split] = should_validity.get(split, list())
                            should_novelty[split] = should_novelty.get(split, list())

                            input_ids = []
                            attention_mask = []
                            for ii in range(i, i+batch_size):
                                if ii >= len(datasets[split]):
                                    break
                                batch: BatchEncoding = datasets[split][ii]
                                should_validity[split] += [batch.pop("validity").item()]
                                should_novelty[split] += [batch.pop("novelty").item()]
                                input_ids.append(batch.pop("input_ids"))
                                attention_mask.append(batch.pop("attention_mask"))

                            output: ValNovOutput = transformer(
                                input_ids=torch.stack(input_ids, dim=0).to(
                                    "cuda" if torch.cuda.is_available() else "cpu"),
                                attention_mask=torch.stack(attention_mask, dim=0).to(
                                    "cuda" if torch.cuda.is_available() else "cpu")
                            )
                            logger.trace("Computed {} samples", batch_size)
                            validity_predictions[split] = validity_predictions.get(split, list())
                            validity_predictions[split] += (y if isinstance(y := output.validity.tolist(), List)
                                                            else [y])
                            novelty_predictions[split] = novelty_predictions.get(split, list())
                            novelty_predictions[split] += (y if isinstance(y := output.novelty.tolist(), List) else [y])
                            logger.trace("Completed batch")
                        logger.trace("Completed {}", split)

                    validation_performances.append((
                        repetition_folder,
                        _val_nov_metric(is_validity=numpy.fromiter(validity_predictions["dev"], float),
                                        should_validity=numpy.fromiter(should_validity["dev"], float),
                                        is_novelty=numpy.fromiter(novelty_predictions["dev"], float),
                                        should_novelty=numpy.fromiter(should_novelty["dev"], float)),
                        _val_nov_metric(is_validity=numpy.fromiter(validity_predictions["test"], float),
                                        should_validity=numpy.fromiter(should_validity["test"], float),
                                        is_novelty=numpy.fromiter(novelty_predictions["test"], float),
                                        should_novelty=numpy.fromiter(should_novelty["test"], float))
                    ))
                    logger.debug("Completed folder \"{}\" with DEV[{}], TEST[{}]", *validation_performances[-1])
                else:
                    logger.debug("Found a {} which isn't processable: {}",
                                 "file" if repetition_folder.is_file() else "dir",
                                 repetition_folder.name)
            logger.debug("Went through all {} repetitions. Analyse it now...", len(validation_performances))

            dev_sort = validation_performances.copy()
            dev_sort.sort(key=lambda t: t[1][metric_key], reverse=True)
            test_sort = validation_performances.copy()
            test_sort.sort(key=lambda t: t[2][metric_key], reverse=True)

            if dev_sort[0][0] == test_sort[0][0]:
                logger.success("{}->{}: Found the best test fit! Its {}-score is: {}",
                               dataset_config_path.name, sample_size_path.name,
                               metric_key, round(dev_sort[0][2][metric_key], 3))
                loss.append(0)
            else:
                logger.warning("{}->{}: Didn't found the best test fit. Selected {}-model reach {} on test split, "
                               "while test-scores are from {}-{} (in {})",
                               dataset_config_path.name, sample_size_path.name,
                               dev_sort[0][0].name,
                               round(dev_sort[0][2][metric_key], 4),
                               round(test_sort[-1][2][metric_key], 3), round(test_sort[0][2][metric_key], 4),
                               test_sort[0][0].name)
                loss.append(test_sort[0][2][metric_key]-dev_sort[0][2][metric_key])

    logger.success("Processed all models in {}. {}% we failed to find the best test fit, "
                   "having a drawback from {}-{}-{}",
                   root_path.name, round(100*sum([int(lo > 0) for lo in loss])/len(loss)),
                   round(min([lo for lo in loss if lo > 0]), 4),
                   round(sum([lo for lo in loss if lo > 0])/sum([int(lo > 0) for lo in loss]), 4),
                   round(max([lo for lo in loss if lo > 0]), 4))
