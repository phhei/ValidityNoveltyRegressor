import argparse
import datetime
import numbers
import pprint
import shutil
import sys
from pathlib import Path
from pprint import pformat
from shutil import move
from typing import Dict

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

VERSION: str = "V0.4.0"

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
    argv.add_argument("--use_preloaded_dataset", action="store", type=Path, default=None, required=False,
                      help="If you want to use a preloaded dataset, give here the root path")
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

    if args.use_preloaded_dataset is None:
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
                train += load_explagraphs(split="train", tokenizer=tokenizer)
                train += load_explagraphs(split="dev", tokenizer=tokenizer)
            else:
                logger.debug("You want to use the ExplaGraphs as a part of the training set with "
                             "following specifications: {}", args.use_ExplaGraphs)
                arg_explagraphs = argparse.ArgumentParser(add_help=False, allow_abbrev=True, exit_on_error=False)
                arg_explagraphs.add_argument("-s", "--split", action="store", default="train", type=str,
                                             choices=["train", "dev"], required=False)
                arg_explagraphs.add_argument("-l", "--max_length_sample", action="store", default=96, type=int,
                                             required=False)
                arg_explagraphs.add_argument("-n", "--max_number", action="store", default=-1, type=int,
                                             required=False)
                arg_explagraphs.add_argument("--generate_non_novel_non_valid_samples_by_random", action="store_true",
                                             default=False, required=False)
                arg_explagraphs.add_argument("--continuous_val_nov", action="store", default=-1, type=float, required=False)
                arg_explagraphs.add_argument("--continuous_sample_weight", action="store_true", default=False,
                                             required=False)
                parsed_args_explagraphs = arg_explagraphs.parse_args(
                    args.use_ExplaGraphs[1:].split("#") if args.use_ExplaGraphs.startswith("#")
                    else args.use_ExplaGraphs.split("#")
                )
                train += load_explagraphs(
                    split=parsed_args_explagraphs.split, tokenizer=tokenizer,
                    max_length_sample=parsed_args_explagraphs.max_length_sample,
                    max_number=parsed_args_explagraphs.max_number,
                    generate_non_novel_non_valid_samples_by_random=parsed_args_explagraphs.generate_non_novel_non_valid_samples_by_random,
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
                arg_ARCT.add_argument("-l", "--max_length_sample", action="store", default=108, type=int, required=False)
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
                arg_ARCTUKW.add_argument("-l", "--max_length_sample", action="store", default=108, type=int, required=False)
                arg_ARCTUKW.add_argument("-n", "--max_number", action="store", default=-1, type=int, required=False)
                arg_ARCTUKW.add_argument("-th", "--quality_threshold", action="store", default=None, type=float,
                                         required=False)
                arg_ARCTUKW.add_argument("--include_topic", action="store_true", default=False, required=False)
                arg_ARCTUKW.add_argument("--continuous_val_nov", action="store_true", default=False, required=False)
                arg_ARCTUKW.add_argument("--continuous_sample_weight", action="store_true", default=False, required=False)
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
                arg_IBM.add_argument("-th", "--quality_threshold", action="store", default=None, type=float, required=False)
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
                arg_essays.add_argument("-l", "--max_length_sample", action="store", default=108, type=int, required=False)
                arg_essays.add_argument("-n", "--max_number", action="store", default=-1, type=int, required=False)
                arg_essays.add_argument("--include_samples_without_detail_annotation_info", action="store_false",
                                        default=True, required=False)
                arg_essays.add_argument("--continuous_val_nov", action="store_true", default=False, required=False)
                arg_essays.add_argument("--continuous_sample_weight", action="store_true", default=False, required=False)
                parsed_args_essays = arg_essays.parse_args(
                    args.use_essays[1:].split("#") if args.use_essays.startswith("#") else args.use_essays.split("#")
                )
                train += load_essays(
                    tokenizer=tokenizer,
                    max_length_sample=parsed_args_essays.max_length_sample,
                    max_number=parsed_args_essays.max_number,
                    exclude_samples_without_detail_annotation_info=parsed_args_essays.include_samples_without_detail_annotation_info,
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
    else:
        logger.warning("OK, let's load the dataset from \"{}\" (all --use-parameters are ignored)",
                       args.use_preloaded_dataset.name)
        datasets = ValidityNoveltyDataset.load(path=args.use_preloaded_dataset)
        train: ValidityNoveltyDataset = datasets["train"]
        dev: ValidityNoveltyDataset = datasets["dev"]
        test: ValidityNoveltyDataset = datasets["test"]

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

    output_dir: Path = Path()

    if args.use_preloaded_dataset is None:
        output_dir: Path = train.dataset_path(base_path=Path(".out", VERSION, args.transformer),
                                              num_samples=None if args.sample == "n/a" else (len(train.samples_original)/2 if parsed_args_sample is None else parsed_args_sample.number))

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
        output_dir: Path = args.use_preloaded_dataset

        if args.save != "n/a" and args.save != "no-dataset":
            logger.debug("Don't save/ analyse loaded datasets again!")

        output_dir = output_dir.joinpath("_reload")
        output_dir = output_dir.joinpath(
            "{}-{}".format(args.transformer, len(list(output_dir.glob(pattern="{}-*".format(args.transformer)))))
        )
        logger.info("Final output path: {}", output_dir)

    # #################################################################################################################

    def run(repetition_index: int = -1) -> Dict:
        logger.info("Let's start repetition {}", repetition_index)

        for equiv_split in args.equalize_source_distribution:
            logger.info("Ok, you want to equalize the sample-source-distribution of the \"{}\"-split", equiv_split)
            result = None
            if "train" in equiv_split:
                result = train.equalize_source_distribution()
            elif "dev" in equiv_split:
                result = dev.equalize_source_distribution()
            elif "test" in equiv_split:
                result = test.equalize_source_distribution()

            if result is None:
                logger.warning("Your defined split \"{}\" was not found -- "
                               "please chose among \"train\", \"dev\" and \"test\"", equiv_split)
            elif result:
                logger.success("Successfully equalize the source distribution of the \"{}\"-split", equiv_split)
            else:
                logger.warning("Split \"{}\" is not source-equalized!", equiv_split)

        if sampling:
            if parsed_args_sample is None:
                final_train_samples = train.sample(seed=42+repetition_index)
            else:
                final_train_samples = train.sample(
                    number_or_fraction=parsed_args_sample.number if parsed_args_sample.number is not None else
                    (parsed_args_sample.fraction if parsed_args_sample.fraction is not None else 1.),
                    forced_balanced_class_distribution=parsed_args_sample.not_forced_balanced_dataset,
                    force_classes=parsed_args_sample.not_forced_balanced_dataset
                    if parsed_args_sample.classes == "n/a" else
                    (True if len(parsed_args_sample.classes) == 0 else
                     [(int(parsed_args_sample.classes[i]) if parsed_args_sample.classes[i].lstrip(" -+").isdigit()
                       else parsed_args_sample.classes[i],
                       int(parsed_args_sample.classes[i+1]) if parsed_args_sample.classes[i+1].lstrip(" -+").isdigit()
                       else parsed_args_sample.classes[i+1])
                      for i in range(0, len(parsed_args_sample.classes)-1, 2)]),
                    allow_automatically_created_samples=parsed_args_sample.automatic_samples,
                    seed=42+repetition_index
                )

            logger.trace("Final training-samples: {}", " +++ ".join(map(lambda t: str(t), final_train_samples)))

        logger.info("To summarize, you have {} training data, {} validation data and {} test data", len(train),
                    len(dev), len(test))
        if not args.skip_training and len(train) == 0:
            logger.warning("You want to train without training data! Please explicit define some with your parameters!")

        if len(test) == 0:
            logger.warning("No test data given...")

        if args.use_preloaded_dataset is None:
            if repetition_index >= 0:
                _output_dir = output_dir.joinpath("repetition-{}".format(repetition_index))
            else:
                _output_dir = output_dir

            if args.save != "n/a" and args.save != "no-dataset":
                logger.debug("You want to save the dataset")
                train.save(path=_output_dir.joinpath("_train"))
                dev.save(path=_output_dir.joinpath("_dev"))
                test.save(path=_output_dir.joinpath("_test"))

            if (isinstance(args.analyse, str) and args.analyse == "train") or "train" in args.analyse:
                train.depth_analysis_data(show_heatmaps=False, handling_not_known_data=.5,
                                          save_heatmaps=str(_output_dir.joinpath("train_analyse.png").absolute()))
            if (isinstance(args.analyse, str) and (args.analyse.startswith("dev") or args.analyse.startswith("val"))) \
                    or "dev" in args.analyse or "development" in args.analyse \
                    or "val" in args.analyse or "validation" in args.analyse:
                dev.depth_analysis_data(show_heatmaps=False,
                                        save_heatmaps=str(_output_dir.joinpath("dev_analyse.png").absolute()))
            if (isinstance(args.analyse, str) and args.analyse == "test") or "test" in args.analyse:
                test.depth_analysis_data(show_heatmaps=False,
                                         save_heatmaps=str(_output_dir.joinpath("test_analyse.png").absolute()))
        else:
            logger.debug("Let's continue with \"{}\"", args.use_preloaded_dataset)

        trainer = ValNovTrainer(
            model=RobertaForValNovRegression.from_pretrained(pretrained_model_name_or_path=args.transformer),
            args=transformers.TrainingArguments(
                output_dir=str(_output_dir),
                do_train=True,
                do_eval=True,
                do_predict=True,
                evaluation_strategy="steps",
                eval_steps=int((len(train)/args.batch_size)/4),
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.learning_rate/100,
                num_train_epochs=5,
                lr_scheduler_type="cosine",
                warmup_steps=min(100, len(train)/2),
                log_level="debug" if args.verbose else "warning",
                log_level_replica="info" if args.verbose else "warning",
                logging_strategy="steps",
                logging_steps=5 if args.verbose else 25,
                logging_first_step=True,
                logging_nan_inf_filter=True,
                save_strategy="steps",
                save_steps=int((len(train)/args.batch_size)/4),
                save_total_limit=6+int(args.verbose),
                label_names=["validity", "novelty"],
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1_macro",
                greater_is_better=True,
                label_smoothing_factor=0.0,
                push_to_hub=False,
                gradient_checkpointing=args.save_memory_training
            ),
            train_dataset=train,
            eval_dataset=None if len(dev) == 0 else dev,
            compute_metrics=val_nov_metric,
            callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=4, early_stopping_threshold=.001)]
        )

        logger.success("Successfully initialized the trainer: {}", trainer)

        if not args.skip_training:
            logger.info("Let's start the training ({} samples)", len(train))

            train_out = trainer.train(ignore_keys_for_eval=["logits", "loss", "hidden_states", "attentions"])

            logger.success("Finished the training -- ending with loss {}", round(train_out.training_loss, 4))
            trainer.save_metrics(split="train", metrics=trainer.metrics_format(metrics=train_out.metrics))

            for early_stopping_callback \
                    in [cb for cb in trainer.callback_handler.callbacks
                        if isinstance(cb, transformers.EarlyStoppingCallback)]:
                removed_cb = trainer.pop_callback(early_stopping_callback)
                logger.debug("Removed the following callback handler: {}. It's an early stopping training callback and "
                             "since we're leaving the training procedure, we don't need them anymore.", removed_cb)

        if len(test) >= 10:
            logger.info("OK, let's test the best model with: {}", test)
            try:
                test_metrics: Dict = trainer.metrics_format(
                    metrics=trainer.evaluate(eval_dataset=test,
                                             ignore_keys=["logits", "loss", "hidden_states", "attentions"],
                                             metric_key_prefix="test")
                )
                test.include_premise = False
                logger.debug("Standard-test DONE -- so we have a clever hans now? Let's continue with the fool-test: {}",
                             test)
                test_metrics.update(trainer.metrics_format(
                    metrics=trainer.evaluate(eval_dataset=test,
                                             ignore_keys=["logits", "loss", "hidden_states", "attentions"],
                                             metric_key_prefix="test_wo_premise")
                ))
                test.include_premise = True
                test.include_conclusion = False
                logger.debug("Another clever-hans-check. Let's continue with the fool-test: {}", test)
                test_metrics.update(trainer.metrics_format(
                    metrics=trainer.evaluate(eval_dataset=test,
                                             ignore_keys=["logits", "loss", "hidden_states", "attentions"],
                                             metric_key_prefix="test_wo_conclusion")
                ))
                test.include_premise = True
                test.include_conclusion = True

                try:
                    logger.success("Test 3 times on {} test samples: {}", len(test),
                                   ", ".join(map(lambda mv: "{}: {}".format(mv[0], round(mv[1], 3)), test_metrics.items())))
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
                logger.opt(exception=True).error("Corrupted test data? {}", test)
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
                    train, dev, test
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
            train.reset_to_original_data()
            dev.reset_to_original_data()
            test.reset_to_original_data()
            logger.trace("Resetted data: {}/{}/{}", train, dev, test)
        return test_metrics

    # #################################################################################################################

    if args.repetitions >= 2:
        logger.info("You want to repeat your experiments multiple times ({}). OK...", args.repetitions)
        results = dict()
        fail_trains = 0

        for i in range(args.repetitions):
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
                       args.repetitions, output_dir.name)

        parent_output_path = output_dir

        final_results = {"fail_trains": fail_trains, "repetitions": args.repetitions}

        for k, v_list in results.items():
            if all(map(lambda v: isinstance(v, numbers.Number), v_list)):
                n = numpy.fromiter(v_list, dtype=int if all(map(lambda _v: isinstance(_v, int), v_list)) else float)
                masked_n = numpy.ma.masked_array(n, mask=[i == -1 for i in n])
                final_results[k] = {
                    "min": n.min().round(3),
                    "repetition_index_min": n.argmin(),
                    "mean": masked_n.mean().round(4),
                    "max": n.max().round(3),
                    "repetition_index_max": n.argmax(),
                    "derivation": masked_n.std().round(3)
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
        parent_output_path.joinpath("aggregated_stats.txt").write_text(data=final_results_str, encoding="utf-8",
                                                                       errors="ignore")
    else:
        run()