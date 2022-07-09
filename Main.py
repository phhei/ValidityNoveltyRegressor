import argparse
import datetime
import numbers
import pprint
import shutil
import sys
from pathlib import Path
from pprint import pformat
from shutil import move
from typing import Dict, List, Union

import numpy
import transformers
from loguru import logger

from ArgumentData import Utils
from ArgumentData.ARCT.Loader import load_dataset as load_arct
from ArgumentData.ARCTUKWWarrants.Loader import load_dataset as load_arct_ukw
from ArgumentData.ExplaGraphs.Loader import load_dataset as load_explagraphs
from ArgumentData.GeneralDataset import ValidityNoveltyDataset
from ArgumentData.IBMArgQuality.Loader import load_dataset as load_ibm
from ArgumentData.StudentEssays.Loader import load_dataset as load_essays
from ArgumentData.ValidityNoveltySharedTask.Loader import load_dataset as load_annotation
from HGTrainer import ValNovTrainer, RobertaForValNovRegression, val_nov_metric

VERSION: str = "V0.4.2"

if __name__ == "__main__":
    argv = argparse.ArgumentParser(
        prog="ValidityNoveltyRegressor",
        description="You want to know whether your conclusion candidate given a premise is valid and/or novel? "
                    "Use this tool! (NLP->Argument Mining)",
        add_help=True,
        allow_abbrev=True,
        exit_on_error=True
    )
    argv.add_argument("-t", "--transformer", action="store", default="roberta-base", type=str, required=False,
                      help="The used transformer model (should be a RoBERTa one) from https://huggingface.co/")
    argv.add_argument("-bs", "--batch_size", action="store", default=1, type=int, required=False,
                      help="The batch size - please be aware of the trade-off: high batch sizes are fast calculated, "
                           "but require lots of memory!")
    argv.add_argument("-lr", "--learning_rate", action="store", default=1e-5, type=float, required=False,
                      help="The learning rate for training")
    argv.add_argument("-v", "--verbose", action="store_true", default=False, required=False,
                      help="Verbose mode - more logging output from hugging-face")
    argv.add_argument("--save_memory_training", action="store_true", default=False, required=False,
                      help="gradient_checkpointing")
    argv.add_argument("--use_preloaded_dataset", action="store", nargs="+", type=Path, default=[],
                      required=False,
                      help="If you want to use a preloaded dataset, give here the root path. "
                           "You can have also multiple runs with several datasets (list).")
    argv.add_argument("--use_ExplaGraphs", action="store", nargs="?", default="n/a", type=str, required=False,
                      help="using the ExplaGraphs for training. You can provide additional arguments for this dataset "
                           "by replacing all the whitespaces with '#', starting with '#'")
    argv.add_argument("--use_ARCT", action="store", nargs="?", default="n/a", type=str, required=False,
                      help="using the the ARCT-dataset for training. You can provide additional arguments "
                           "for this dataset by replacing all the whitespaces with '#', starting with '#'")
    argv.add_argument("--use_ARCTUKW", action="store", nargs="?", default="n/a", type=str, required=False,
                      help="using the the UKW-Warrants-dataset for training. You can provide additional arguments "
                           "for this dataset by replacing all the whitespaces with '#', starting with '#'")
    argv.add_argument("--use_IBM", action="store", nargs="?", default="n/a", type=str, required=False,
                      help="using the the IBM-Argument-Quality-dataset for training. "
                           "You can provide additional arguments for this dataset by replacing all the whitespaces "
                           "with '#', starting with '#'")
    argv.add_argument("--use_essays", action="store", nargs="?", default="n/a", type=str, required=False,
                      help="using the the Student-Essays-dataset for training. "
                           "You can provide additional arguments for this dataset by replacing all the whitespaces "
                           "with '#', starting with '#'")
    argv.add_argument("--use_annotation_train", action="store", nargs="?", default="n/a", type=str, required=False,
                      help="using the validation und test set. You can provide additional arguments for this dataset "
                           "by replacing all the whitespaces with '#', starting with '#'")
    argv.add_argument("--use_annotation_ValTest", action="store", nargs="?", default="n/a", type=str, required=False,
                      help="using the validation und test set. You can provide additional arguments for this dataset "
                           "by replacing all the whitespaces with '#', starting with '#'")
    argv.add_argument("--generate_more_training_samples", action="store_true", default=False, required=False,
                      help="Forces the training set to automatically generate (more) samples")
    argv.add_argument("--random_truncation", action="store_true", default=False, required=False,
                      help="Shuffles the data before truncating it if an restricted number to use from a "
                           "chosen source is given.")
    argv.add_argument("--intelligent_truncation", action="store_true", default=False, required=False,
                      help="Do a representative clustering + weighting before truncating data if a restricted "
                           "number to use from a chosen source is given - hence, cherry-picks the samples")
    argv.add_argument("--ignore_weights", action="store", nargs="?", type=float, default=False, const=1, required=False,
                      help="If you set this option, you disable the different sample weights, "
                           "hence each sample contributes with the same (potential) strengths to the training weights."
                           "You can further define the value of this fixed weight (optional).")
    argv.add_argument("--equalize_source_distribution", action="append", nargs="?", type=str, default=[],
                      const="train", required=False,
                      help="Forces the train-split (or other splits if defined) to have an equalized sample "
                           "source distribution. This is done BEFORE sampling")
    argv.add_argument("-s", "--sample", action="store", nargs="?", default="n/a", type=str, required=False,
                      help="Rescale the training data - You can provide additional arguments for this dataset "
                           "by replacing all the whitespaces with '#', starting with '#'")
    argv.add_argument("--cold_start", action="store_true", default=False, required=False,
                      help="Shuffles the training data before processing it "
                           "(with the https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/"
                           "optimizer_schedules#transformers.get_cosine_schedule_with_warmup). "
                           "This is done AFTER sampling -- not implemented until yet.")
    argv.add_argument("--warm_start", action="store_true", default=False, required=False,
                      help="Do a representative clustering + weighting to have an informed training start."
                           "This done AFTER sampling -- not implemented until yet.")
    argv.add_argument("-skip", "--skip_training", action="store_true", default=False,  required=False,
                      help="If you want to skip the training (directly to testing), take this argument!")
    argv.add_argument("--save", action="store", nargs="?", type=str, default="n/a", required=False,
                      help="Saves the used/ produced data. You can exclude the neural model by specifying this "
                           "argument by \"no-model\" or the avoiding the dataset by \"no-dataset\"")
    argv.add_argument("--analyse", action="store", nargs="*", type=str, default=["test"], required=False,
                      help="You want to have a depth analysis of your data before training/ testing on it? "
                           "Please define the split(s) here.")
    argv.add_argument("--clean", action="store_true", default=False, required=False,
                      help="Removes all checkpoints (not the best model) to free disk space")
    argv.add_argument("--repetitions", action="store", type=int, default=1,
                      help="How many times should the experiment be repeated? Recommend for statistical clarification, "
                           "especially in combination with sampling")

    logger.info("Welcome to the ValidityNoveltyRegressor -- you made {} specific choices", len(sys.argv)-1)

    args = argv.parse_args()

    train: Union[ValidityNoveltyDataset, List[ValidityNoveltyDataset]]
    dev: Union[ValidityNoveltyDataset, List[ValidityNoveltyDataset]]
    test: Union[ValidityNoveltyDataset, List[ValidityNoveltyDataset]]

    if len(args.use_preloaded_dataset) >= 1:
        logger.warning("OK, let's load the {} datasets from {} (all --use-parameters are ignored)",
                       len(args.use_preloaded_dataset), args.use_preloaded_dataset)
        train = []
        dev = []
        test = []
        for pld in args.use_preloaded_dataset:
            datasets = ValidityNoveltyDataset.load(path=pld)
            train.append(datasets["train"])
            dev.append(datasets["dev"])
            test.append(datasets["test"])

        if len(args.use_preloaded_dataset) != args.repetitions:
            logger.warning("You want to have {0} repetitions, but give us {1} datasets - "
                           "so you will have {1} repetitions",
                           args.repetitions, len(args.use_preloaded_dataset))
    else:
        tokenizer = transformers.RobertaTokenizer.from_pretrained(args.transformer)

        train = ValidityNoveltyDataset(samples=[], tokenizer=tokenizer, max_length=156, name="Train")
        dev = ValidityNoveltyDataset(samples=[], tokenizer=tokenizer, max_length=156, name="Eval")
        test = ValidityNoveltyDataset(samples=[], tokenizer=tokenizer, max_length=156, name="Test")

        if args.random_truncation or args.intelligent_truncation:
            Utils.sampling_technique = "mixed" if args.random_truncation and args.intelligent_truncation else \
                ("most informative" if args.intelligent_truncation else "random")
            logger.info("You changed the truncation technique (if you need less samples from a dataset) to: {}",
                        Utils.sampling_technique)

        if args.use_ExplaGraphs != "n/a":
            if args.use_ExplaGraphs is None:
                logger.debug("You want to use the entire ExplaGraphs as a part of the training set - fine")
                for split in ["train", "dev"]:
                    train += load_explagraphs(split=split, tokenizer=tokenizer)
            else:
                logger.debug("You want to use the ExplaGraphs as a part of the training set with "
                             "following specifications: {}", args.use_ExplaGraphs)
                arg_explagraphs = argparse.ArgumentParser(add_help=False, allow_abbrev=True, exit_on_error=False)
                arg_explagraphs.add_argument("-s", "--split", action="store", default="train", type=str,
                                             choices=["train", "dev", "all"], required=False)
                arg_explagraphs.add_argument("-l", "--max_length_sample", action="store", default=96, type=int,
                                             required=False)
                arg_explagraphs.add_argument("-n", "--max_number", action="store", default=-1, type=int,
                                             required=False)
                arg_explagraphs.add_argument("--generate_non_novel_non_valid_samples_by_random", action="store_true",
                                             default=False, required=False)
                arg_explagraphs.add_argument("--continuous_val_nov", action="store", default=-1, type=float,
                                             required=False)
                arg_explagraphs.add_argument("--continuous_sample_weight", action="store_true", default=False,
                                             required=False)
                parsed_args_explagraphs = arg_explagraphs.parse_args(
                    args.use_ExplaGraphs[1:].split("#") if args.use_ExplaGraphs.startswith("#")
                    else args.use_ExplaGraphs.split("#")
                )
                for split in (
                        ["train", "dev"] if parsed_args_explagraphs.split == "all" else [parsed_args_explagraphs.split]
                ):
                    train += load_explagraphs(
                        split=split, tokenizer=tokenizer,
                        max_length_sample=parsed_args_explagraphs.max_length_sample,
                        max_number=parsed_args_explagraphs.max_number,
                        generate_non_novel_non_valid_samples_by_random=
                        parsed_args_explagraphs.generate_non_novel_non_valid_samples_by_random,
                        continuous_val_nov=False if parsed_args_explagraphs.continuous_val_nov < 0 else
                        parsed_args_explagraphs.continuous_val_nov,
                        continuous_sample_weight=parsed_args_explagraphs.continuous_sample_weight
                    )

        if args.use_ARCT != "n/a":
            if args.use_ARCT is None:
                logger.debug("You want to use the entire ARCT-dataset as a part of the training set - fine")
                train += load_arct(split="train", tokenizer=tokenizer)
                train += load_arct(split="dev", tokenizer=tokenizer)
                train += load_arct(split="test", tokenizer=tokenizer)
            else:
                logger.debug("You want to use the ARCT-dataset as a part of the training set with "
                             "following specifications: {}", args.use_ARCT)
                arg_ARCT = argparse.ArgumentParser(add_help=False, allow_abbrev=True, exit_on_error=False)
                arg_ARCT.add_argument("-s", "--split", action="store", default="train", type=str,
                                      choices=["all", "train", "dev", "test"], required=False)
                arg_ARCT.add_argument("-l", "--max_length_sample", action="store", default=108, type=int,
                                      required=False)
                arg_ARCT.add_argument("-n", "--max_number", action="store", default=-1, type=int, required=False)
                arg_ARCT.add_argument("--exclude_adversarial_data", action="store_true", default=False, required=False)
                arg_ARCT.add_argument("--include_topic", action="store_true", default=False, required=False)
                arg_ARCT.add_argument("--include_debate_info", action="store_true", default=False, required=False)
                arg_ARCT.add_argument("--continuous_val_nov", action="store_true", default=False, required=False)
                arg_ARCT.add_argument("--continuous_sample_weight", action="store_true", default=False, required=False)
                parsed_args_arct = arg_ARCT.parse_args(
                    args.use_ARCT[1:].split("#") if args.use_ARCT.startswith("#") else args.use_ARCT.split("#")
                )
                for split in \
                        (["train", "dev", "test"] if parsed_args_arct.split == "all" else [parsed_args_arct.split]):
                    train += load_arct(
                        split=parsed_args_arct.split, tokenizer=tokenizer,
                        max_length_sample=parsed_args_arct.max_length_sample,
                        max_number=parsed_args_arct.max_number,
                        continuous_val_nov=parsed_args_arct.continuous_val_nov,
                        continuous_sample_weight=parsed_args_arct.continuous_sample_weight,
                        include_adversarial_data=not parsed_args_arct.exclude_adversarial_data,
                        include_topic=parsed_args_arct.include_topic,
                        include_debate_info=parsed_args_arct.include_debate_info
                    )

        if args.use_ARCTUKW != "n/a":
            if args.use_ARCTUKW is None:
                logger.debug("You want to use the entire ARCT-UKW-dataset as a part of the training set - fine")
                train += load_arct_ukw(tokenizer=tokenizer)
            else:
                logger.debug("You want to use the ARCT-dataset as a part of the training set with "
                             "following specifications: {}", args.use_ARCT)
                arg_ARCTUKW = argparse.ArgumentParser(add_help=False, allow_abbrev=True, exit_on_error=False)
                arg_ARCTUKW.add_argument("-l", "--max_length_sample", action="store", default=108, type=int,
                                         required=False)
                arg_ARCTUKW.add_argument("-n", "--max_number", action="store", default=-1, type=int, required=False)
                arg_ARCTUKW.add_argument("-th", "--quality_threshold", action="store", default=None, type=float,
                                         required=False)
                arg_ARCTUKW.add_argument("--include_topic", action="store_true", default=False, required=False)
                arg_ARCTUKW.add_argument("--continuous_val_nov", action="store_true", default=False, required=False)
                arg_ARCTUKW.add_argument("--continuous_sample_weight", action="store_true", default=False,
                                         required=False)
                parsed_args_arct_ukw = arg_ARCTUKW.parse_args(
                    args.use_ARCTUKW[1:].split("#") if args.use_ARCTUKW.startswith("#") else args.use_ARCTUKW.split("#")
                )
                train += load_arct_ukw(
                    tokenizer=tokenizer,
                    max_length_sample=parsed_args_arct_ukw.max_length_sample,
                    max_number=parsed_args_arct_ukw.max_number,
                    continuous_val_nov=parsed_args_arct_ukw.continuous_val_nov,
                    continuous_sample_weight=parsed_args_arct_ukw.continuous_sample_weight,
                    include_topic=parsed_args_arct_ukw.include_topic,
                    mace_ibm_threshold=parsed_args_arct_ukw.quality_threshold
                )

        if args.use_IBM != "n/a":
            if args.use_IBM is None:
                logger.debug("You want to use the entire IBM-Arg-dataset as a part of the training set - fine")
                train += load_ibm(tokenizer=tokenizer)
            else:
                logger.debug("You want to use the ARCT-dataset as a part of the training set with "
                             "following specifications: {}", args.use_ARCT)
                arg_IBM = argparse.ArgumentParser(add_help=False, allow_abbrev=True, exit_on_error=False)
                arg_IBM.add_argument("-l", "--max_length_sample", action="store", default=108, type=int, required=False)
                arg_IBM.add_argument("-n", "--max_number", action="store", default=-1, type=int, required=False)
                arg_IBM.add_argument("-th", "--quality_threshold", action="store", default=None, type=float,
                                     required=False)
                arg_IBM.add_argument("-s", "--split", action="store", default="all", type=str,
                                     choices=["all", "train", "dev", "test"], required=False)
                arg_IBM.add_argument("--continuous_val_nov", action="store_true", default=False, required=False)
                arg_IBM.add_argument("--continuous_sample_weight", action="store_true", default=False, required=False)
                parsed_args_ibm = arg_IBM.parse_args(
                    args.use_IBM[1:].split("#") if args.use_IBM.startswith("#") else args.use_IBM.split("#")
                )
                train += load_ibm(
                    tokenizer=tokenizer,
                    split=parsed_args_ibm.split,
                    max_length_sample=parsed_args_ibm.max_length_sample,
                    max_number=parsed_args_ibm.max_number,
                    continuous_val_nov=parsed_args_ibm.continuous_val_nov,
                    continuous_sample_weight=parsed_args_ibm.continuous_sample_weight,
                    quality_threshold=parsed_args_ibm.quality_threshold
                )

        if args.use_essays != "n/a":
            if args.use_essays is None:
                logger.debug("You want to use the entire IBM-Arg-dataset as a part of the training set - fine")
                train += load_essays(tokenizer=tokenizer)
            else:
                logger.debug("You want to use the Student-Essays-dataset as a part of the training set with "
                             "following specifications: {}", args.use_essays)
                arg_essays = argparse.ArgumentParser(add_help=False, allow_abbrev=True, exit_on_error=False)
                arg_essays.add_argument("-l", "--max_length_sample", action="store", default=108, type=int,
                                        required=False)
                arg_essays.add_argument("-n", "--max_number", action="store", default=-1, type=int, required=False)
                arg_essays.add_argument("--include_samples_without_detail_annotation_info", action="store_false",
                                        default=True, required=False)
                arg_essays.add_argument("--continuous_val_nov", action="store_true", default=False, required=False)
                arg_essays.add_argument("--continuous_sample_weight", action="store_true", default=False,
                                        required=False)
                parsed_args_essays = arg_essays.parse_args(
                    args.use_essays[1:].split("#") if args.use_essays.startswith("#") else args.use_essays.split("#")
                )
                train += load_essays(
                    tokenizer=tokenizer,
                    max_length_sample=parsed_args_essays.max_length_sample,
                    max_number=parsed_args_essays.max_number,
                    exclude_samples_without_detail_annotation_info=
                    parsed_args_essays.include_samples_without_detail_annotation_info,
                    continuous_val_nov=parsed_args_essays.continuous_val_nov,
                    continuous_sample_weight=parsed_args_essays.continuous_sample_weight,
                )

        ###############################################################################################################

        arg_annotation = argparse.ArgumentParser(add_help=False, allow_abbrev=True, exit_on_error=False)
        arg_annotation.add_argument("-l", "--max_length_sample", action="store", default=132, type=int,
                                    required=False)
        arg_annotation.add_argument("-n", "--max_number", action="store", default=-1, type=int, required=False)
        arg_annotation.add_argument("-t", "--include_topic", action="store_true", default=False,
                                    required=False)
        arg_annotation.add_argument("--min_confidence", action="store", default="defeasible", required=False)
        arg_annotation.add_argument("--continuous_val_nov", action="store_true", default=False,
                                    required=False)
        arg_annotation.add_argument("--continuous_sample_weight", action="store_true", default=False,
                                    required=False)
        if len(sys.argv) <= 1 or args.use_annotation_train != "n/a":
            if len(sys.argv) <= 1 or args.use_annotation_train is None:
                logger.info("OK, let's use the best training data as it is")
                train += load_annotation(split="train", tokenizer=tokenizer)
            else:
                parsed_args_annotation = arg_annotation.parse_args(
                    args.use_annotation_train[1:].split("#") if args.use_annotation_train.startswith("#")
                    else args.use_annotation_train.split("#")
                )

                train += load_annotation(
                    split="train",
                    tokenizer=tokenizer,
                    max_length_sample=parsed_args_annotation.max_length_sample,
                    max_number=parsed_args_annotation.max_number,
                    include_topic=parsed_args_annotation.include_topic,
                    continuous_val_nov=parsed_args_annotation.continuous_val_nov,
                    continuous_sample_weight=parsed_args_annotation.continuous_sample_weight,
                    min_confidence=parsed_args_annotation.min_confidence
                )

        if len(sys.argv) <= 1 or args.use_annotation_ValTest != "n/a":
            if args.use_annotation_ValTest != "n/a":
                logger.debug("You want to use the ValTest as a part of the validation/ test set")
            else:
                logger.info("Default included dataset")

            if len(sys.argv) <= 1 or args.use_annotation_ValTest is None:
                dev += load_annotation(split="dev", tokenizer=tokenizer)
                test += load_annotation(split="test", tokenizer=tokenizer)
            else:
                arg_annotation.add_argument("--no_dev_overlap", action="store_true", default=False, required=False)
                parsed_args_annotation = arg_annotation.parse_args(
                    args.use_annotation_ValTest[1:].split("#") if args.use_annotation_ValTest.startswith("#")
                    else args.use_annotation_ValTest.split("#")
                )

                dev += load_annotation(
                    split="dev",
                    tokenizer=tokenizer,
                    max_length_sample=parsed_args_annotation.max_length_sample,
                    max_number=parsed_args_annotation.max_number,
                    include_topic=parsed_args_annotation.include_topic,
                    continuous_val_nov=parsed_args_annotation.continuous_val_nov,
                    continuous_sample_weight=parsed_args_annotation.continuous_sample_weight,
                    min_confidence=parsed_args_annotation.min_confidence
                )

                test += load_annotation(
                    split="test_without_dev_topics" if parsed_args_annotation.no_dev_overlap else "test",
                    tokenizer=tokenizer,
                    max_length_sample=parsed_args_annotation.max_length_sample,
                    max_number=parsed_args_annotation.max_number,
                    include_topic=parsed_args_annotation.include_topic,
                    continuous_val_nov=parsed_args_annotation.continuous_val_nov,
                    continuous_sample_weight=parsed_args_annotation.continuous_sample_weight,
                    min_confidence=parsed_args_annotation.min_confidence
                )

    if args.generate_more_training_samples:
        logger.info("You want to automatically generate more training samples - ok, let's do it!")
        samples_generated = train.generate_more_samples()
        logger.info("Training set has the size of {} (+{}) now ({})", len(train), samples_generated,
                    train.get_sample_class_distribution(for_original_data=True))
        if args.repetitions >= 2:
            train.samples_original = train.samples_extraction.copy()

    # #################################################################################################################

    if args.sample != "n/a":
        logger.info("You want to sample your training data ({})", len(train))
        sampling = True

        if args.sample is None:
            parsed_args_sample = None
        else:
            arg_sample = argparse.ArgumentParser(add_help=False, allow_abbrev=True, exit_on_error=False)
            arg_sample.add_argument("-n", "--number", action="store", type=int, required=False)
            arg_sample.add_argument("-f", "--fraction", action="store", type=float, required=False)
            arg_sample.add_argument("--classes", action="store", nargs="*", default="n/a", required=False)
            arg_sample.add_argument("--not_forced_balanced_dataset", action="store_false", default=True, required=False)
            arg_sample.add_argument("--automatic_samples", action="store_true",
                                    default=args.generate_more_training_samples, required=False)

            parsed_args_sample = arg_sample.parse_args(
                args.sample[1:].split("#") if args.sample.startswith("#") else args.sample.split("#")
            )
    else:
        sampling = False
        parsed_args_sample = None

    output_dir: Union[Path, List[Path]]

    if len(args.use_preloaded_dataset) == 0:
        output_dir: Path = train.dataset_path(
            base_path=Path(".out", VERSION, args.transformer),
            num_samples=None if args.sample == "n/a" else (len(train.samples_original)/2
                                                           if parsed_args_sample is None else parsed_args_sample.number)
        )

        if output_dir.exists():
            logger.warning("The dictionary \"{}\" exists already - [re]move it!", output_dir.absolute())
            target = Path(
                ".out", VERSION, "_old", args.transformer,
                "{} {}".format(
                    datetime.datetime.now().isoformat(sep="_", timespec="minutes").replace(":", "-"),
                    output_dir.name
                )
            )
            logger.info("Moved to: {}", move(str(output_dir.absolute()), str(target.absolute())))
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = []
        for pld in args.use_preloaded_dataset:
            output_dir_temp: Path = Path(pld)

            if args.save != "n/a" and args.save != "no-dataset":
                logger.debug("Don't save/ analyse loaded datasets again!")

            output_dir_temp = output_dir_temp.joinpath("_reload")
            output_dir_temp = output_dir_temp.joinpath(
                "{}-{}".format(args.transformer,
                               len(list(output_dir_temp.glob(pattern="{}-*".format(args.transformer)))))
            )
            output_dir.append(output_dir_temp)
            logger.info("Final output path added: {}", output_dir_temp)

    # #################################################################################################################

    def run(repetition_index: int = -1) -> Dict:
        logger.info("Let's start repetition {}", repetition_index)

        _train = train[repetition_index] if isinstance(train, List) else train
        _dev = dev[repetition_index] if isinstance(dev, List) else dev
        _test = test[repetition_index] if isinstance(test, List) else test
        _output_dir = output_dir[repetition_index] if isinstance(output_dir, List) else output_dir

        for equiv_split in args.equalize_source_distribution:
            logger.info("Ok, you want to equalize the sample-source-distribution of the \"{}\"-split", equiv_split)
            result = None
            if "train" in equiv_split:
                result = _train.equalize_source_distribution()
            elif "dev" in equiv_split:
                result = _dev.equalize_source_distribution()
            elif "test" in equiv_split:
                result = _test.equalize_source_distribution()

            if result is None:
                logger.warning("Your defined split \"{}\" was not found -- "
                               "please chose among \"train\", \"dev\" and \"test\"", equiv_split)
            elif result:
                logger.success("Successfully equalize the source distribution of the \"{}\"-split", equiv_split)
            else:
                logger.warning("Split \"{}\" is not source-equalized!", equiv_split)

        if sampling:
            if parsed_args_sample is None:
                final_train_samples = _train.sample(seed=42+repetition_index)
            else:
                final_train_samples = _train.sample(
                    number_or_fraction=parsed_args_sample.number if parsed_args_sample.number is not None else
                    (parsed_args_sample.fraction if parsed_args_sample.fraction is not None else 1.),
                    forced_balanced_class_distribution=parsed_args_sample.not_forced_balanced_dataset,
                    force_classes=parsed_args_sample.not_forced_balanced_dataset
                    if parsed_args_sample.classes == "n/a" else
                    (True if len(parsed_args_sample.classes) == 0 else
                     [(int(parsed_args_sample.classes[j]) if parsed_args_sample.classes[j].lstrip(" -+").isdigit()
                       else parsed_args_sample.classes[j],
                       int(parsed_args_sample.classes[j + 1]) if parsed_args_sample.classes[j + 1].lstrip(" -+").isdigit()
                       else parsed_args_sample.classes[j + 1])
                      for j in range(0, len(parsed_args_sample.classes) - 1, 2)]),
                    allow_automatically_created_samples=parsed_args_sample.automatic_samples,
                    seed=42+repetition_index
                )

            logger.trace("Final training-samples: {}", " +++ ".join(map(lambda t: str(t), final_train_samples)))

        if args.ignore_weights:
            _train.fix_sample_weights(fixed_value=args.ignore_weights)

        logger.info("To summarize, you have {} training data, {} validation data and {} test data", len(_train),
                    len(_dev), len(_test))
        if not args.skip_training and len(_train) == 0:
            logger.warning("You want to train without training data! Please explicit define some with your parameters!")

        if len(_test) == 0:
            logger.warning("No test data given...")

        if len(args.use_preloaded_dataset) == 0:
            if repetition_index >= 0:
                _output_dir = _output_dir.joinpath("repetition-{}".format(repetition_index))
            else:
                _output_dir = _output_dir

            if args.save != "n/a" and args.save != "no-dataset":
                logger.debug("You want to save the dataset")
                _train.save(path=_output_dir.joinpath("_train"))
                _dev.save(path=_output_dir.joinpath("_dev"))
                _test.save(path=_output_dir.joinpath("_test"))

            if (isinstance(args.analyse, str) and args.analyse == "train") or "train" in args.analyse:
                _train.depth_analysis_data(show_heatmaps=False, handling_not_known_data=.5,
                                           save_heatmaps=str(_output_dir.joinpath("train_analyse.png").absolute()))
            if (isinstance(args.analyse, str) and (args.analyse.startswith("dev") or args.analyse.startswith("val"))) \
                    or "dev" in args.analyse or "development" in args.analyse \
                    or "val" in args.analyse or "validation" in args.analyse:
                _dev.depth_analysis_data(show_heatmaps=False,
                                         save_heatmaps=str(_output_dir.joinpath("dev_analyse.png").absolute()))
            if (isinstance(args.analyse, str) and args.analyse == "test") or "test" in args.analyse:
                _test.depth_analysis_data(show_heatmaps=False,
                                          save_heatmaps=str(_output_dir.joinpath("test_analyse.png").absolute()))
        else:
            logger.debug("Let's continue with \"{}\"", _output_dir)

        trainer = ValNovTrainer(
            model=RobertaForValNovRegression.from_pretrained(pretrained_model_name_or_path=args.transformer),
            args=transformers.TrainingArguments(
                output_dir=str(_output_dir),
                do_train=True,
                do_eval=True,
                do_predict=True,
                evaluation_strategy="steps",
                eval_steps=int((len(_train)/args.batch_size)/4),
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.learning_rate/100,
                num_train_epochs=5,
                lr_scheduler_type="cosine",
                warmup_steps=min(100, len(_train)/2),
                log_level="debug" if args.verbose else "warning",
                log_level_replica="info" if args.verbose else "warning",
                logging_strategy="steps",
                logging_steps=5 if args.verbose else 25,
                logging_first_step=True,
                logging_nan_inf_filter=True,
                save_strategy="steps",
                save_steps=int((len(_train)/args.batch_size)/4),
                save_total_limit=6+int(args.verbose),
                label_names=["validity", "novelty"],
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1_macro",
                greater_is_better=True,
                label_smoothing_factor=0.0,
                push_to_hub=False,
                gradient_checkpointing=args.save_memory_training
            ),
            train_dataset=_train,
            eval_dataset=None if len(_dev) == 0 else _dev,
            compute_metrics=val_nov_metric,
            callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=4, early_stopping_threshold=.001)]
        )

        logger.success("Successfully initialized the trainer: {}", trainer)

        if not args.skip_training:
            logger.info("Let's start the training ({} samples)", len(_train))

            train_out = trainer.train(ignore_keys_for_eval=["logits", "loss", "hidden_states", "attentions"])

            logger.success("Finished the training -- ending with loss {}", round(train_out.training_loss, 4))
            trainer.save_metrics(split="train", metrics=trainer.metrics_format(metrics=train_out.metrics))

            for early_stopping_callback \
                    in [cb for cb in trainer.callback_handler.callbacks
                        if isinstance(cb, transformers.EarlyStoppingCallback)]:
                removed_cb = trainer.pop_callback(early_stopping_callback)
                logger.debug("Removed the following callback handler: {}. It's an early stopping training callback and "
                             "since we're leaving the training procedure, we don't need them anymore.", removed_cb)

        if len(_test) >= 10:
            logger.info("OK, let's test the best model to test on: {}/ {}", _dev, _test)
            try:
                logger.debug("First, calculate the dev-performance ({})", _dev)
                test_metrics: Dict = trainer.metrics_format(
                    metrics=trainer.evaluate(eval_dataset=_dev,
                                             ignore_keys=["logits", "loss", "hidden_states", "attentions"],
                                             metric_key_prefix="dev")
                )
                test_metrics["data_train"] = str(_train)
                test_metrics["data_dev"] = str(_dev)
                test_metrics["data_test"] = str(_test)
                logger.debug("Now the standard test ({})", _test)
                test_metrics.update(trainer.metrics_format(
                    metrics=trainer.evaluate(eval_dataset=_test,
                                             ignore_keys=["logits", "loss", "hidden_states", "attentions"],
                                             metric_key_prefix="test")
                ))
                logger.debug("Standard-test DONE -- so we have a clever hans now? Let's continue with two fool-tests: "
                             "{}",
                             _test)
                for fool in [("test_wo_premise", (False, True)), ("test_wo_conclusion", (True, False))]:
                    _test.include_premise = fool[1][0]
                    _test.include_conclusion = fool[1][1]
                    logger.debug("Clever-hans-check: {}", fool[0])
                    fool_test_results: Dict = trainer.metrics_format(
                        metrics=trainer.evaluate(eval_dataset=_test,
                                                 ignore_keys=["logits", "loss", "hidden_states", "attentions"],
                                                 metric_key_prefix=fool[0])
                    )
                    try:
                        logger.debug("Don't need following stats: {}/{}/{}/{}/{}/{}/{}",
                                     fool_test_results.pop("{}_size".format(fool[0])),
                                     fool_test_results.pop("{}_mse_validity".format(fool[0])),
                                     fool_test_results.pop("{}_mse_novelty".format(fool[0])),
                                     fool_test_results.pop("{}_approximately_hits_validity".format(fool[0])),
                                     fool_test_results.pop("{}_approximately_hits_novelty".format(fool[0])),
                                     fool_test_results.pop("{}_exact_hits_validity".format(fool[0])),
                                     fool_test_results.pop("{}_exact_hits_novelty".format(fool[0])))
                    except KeyError:
                        logger.opt(exception=True).warning("Please update this result-dict-interface!")
                    test_metrics.update(fool_test_results)
                _test.include_premise = True
                _test.include_conclusion = True

                try:
                    logger.success("Test 3 times on {} test samples: {}",
                                   len(_test),
                                   ", ".join(map(lambda mv: "{}: {}".format(
                                       mv[0], round(mv[1], 3)), test_metrics.items())))
                except TypeError:
                    logger.opt(exception=False).warning("Strange test-metrics-outputs-dict. Should be str->float, "
                                                        "but we have following: {}",
                                                        pformat(test_metrics, indent=2, compact=False))
                trainer.save_metrics(split="test", metrics=test_metrics)
            except RuntimeError:
                logger.opt(exception=True).error("Something went wrong with the model - "
                                                 "please try the model stand-alone (in {})",
                                                 _output_dir.absolute())
                test_metrics = dict()
            except IndexError:
                logger.opt(exception=True).error("Corrupted test data? {}", _test)
                test_metrics = dict()
        else:
            test_metrics = dict()

        if args.clean or (args.save != "n/a" and args.save != "no-model"):
            logger.debug("You want to save the model")
            # noinspection PyBroadException
            try:
                trainer.save_model()
                logger.success("Successfully stored the model in: {}", _output_dir)
            except Exception:
                logger.opt(exception=True).error("Can't save to {}", _output_dir.absolute())

        try:
            info_code = _output_dir.joinpath("sys-args.txt").write_text(
                data="Run from {} with following inputs args:\n{}\n\n Train-data:{}/Dev-data:{}/Test-data:{}".format(
                    datetime.datetime.now().isoformat(),
                    "\n".join(sys.argv[1:]) if len(sys.argv) >= 2 else "no input args",
                    _train, _dev, _test
                ),
                encoding="utf-8",
                errors="ignore"
            )
            logger.info("OK, we're at the end - let's close the process by writing an Info-file at: {}",
                        _output_dir.joinpath("sys-args.txt").absolute())
            logger.trace("Write-code: {}", info_code)
        except IOError:
            logger.opt(exception=False).info("No-info-file provided")

        if args.clean:
            logger.debug("You want to clean \"{}\"", _output_dir.absolute())
            for checkpoint in _output_dir.iterdir():
                if "checkpoint" in checkpoint.name:
                    if checkpoint.is_dir():
                        logger.trace("Found a checkpoint-dir: {}", checkpoint.name)
                        try:
                            shutil.rmtree(path=str(checkpoint.absolute()), ignore_errors=False)
                        except OSError:
                            logger.warning("Failed to clean \"{}\"", checkpoint.absolute())
                        except Exception:
                            logger.opt(exception=True).error("Critical cleaning")
                    else:
                        logger.info("A checkpoint-like link is not a directory, but a file: {}", checkpoint.name)
            logger.debug("Cleaning done...")

        if repetition_index >= 0:
            logger.success("Repetition {} done", repetition_index)
            _train.reset_to_original_data()
            _dev.reset_to_original_data()
            _test.reset_to_original_data()
            logger.trace("Reset data: {}/{}/{}", _train, _dev, _test)
        return test_metrics

    # #################################################################################################################

    if args.repetitions >= 2 or len(args.use_preloaded_dataset) >= 2:
        logger.info("You want to repeat your experiments multiple times ({}). OK...",
                    args.repetitions if len(args.use_preloaded_dataset) == 0 else len(args.use_preloaded_dataset))
        results = dict()
        fail_trains = 0

        for i in range(args.repetitions if len(args.use_preloaded_dataset) == 0 else len(args.use_preloaded_dataset)):
            res = run(i)
            if "never_predicted_classes" in res and res["never_predicted_classes"] >= 3:
                logger.warning("Repetition {} results in a classifier with always a static prediction - "
                               "throw results away!", i)
                fail_trains += 1
                for k in res.keys():
                    if k in results:
                        results[k].append(-1)
                    else:
                        logger.debug("Now result-key: \"{}\"", k)
                        results[k] = [-1]
            else:
                for k, v in res.items():
                    logger.trace("Found following result-key: {}", k)
                    if k in results:
                        results[k].append(v)
                    else:
                        logger.debug("Now result-key: \"{}\"", k)
                        results[k] = [v]

        logger.success("Run successfully the experiment {} times, single results in following path: {}",
                       args.repetitions if len(args.use_preloaded_dataset) == 0 else len(args.use_preloaded_dataset),
                       output_dir.name if isinstance(output_dir, Path) else output_dir[0].name)

        final_results = {
            "fail_trains": fail_trains,
            "repetitions": args.repetitions if len(args.use_preloaded_dataset) == 0 else len(args.use_preloaded_dataset)
        }

        if len(args.use_preloaded_dataset) >= 1:
            final_results["preloaded_datasets"] = args.use_preloaded_dataset

        for k, v_list in results.items():
            if all(map(lambda val: isinstance(val, numbers.Number), v_list)):
                n = numpy.fromiter(v_list, dtype=int if all(map(lambda _v: isinstance(_v, int), v_list)) else float)
                masked_n = numpy.ma.masked_array(n, mask=[i == -1 for i in n])
                final_results[k] = {
                    "min": n.min().round(4),
                    "repetition_index_min": n.argmin(),
                    "mean": masked_n.mean().round(5),
                    "max": n.max().round(4),
                    "repetition_index_max": n.argmax(),
                    # "derivation": masked_n.std().round(3),
                    "values": n.round(3)
                }
            else:
                final_results[k] = v_list

        final_results_str = pprint.pformat(
            final_results,
            indent=2,
            width=120,
            depth=3,
            sort_dicts=True
        )

        logger.success("Final results: {}", final_results)
        for out_dir in (output_dir if isinstance(output_dir, List) else [output_dir]):
            out_dir.joinpath("aggregated_stats.txt").write_text(data=final_results_str, encoding="utf-8",
                                                                errors="ignore")
    else:
        run()
