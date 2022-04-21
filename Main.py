import datetime
import sys
from pathlib import Path
from shutil import move
from pprint import pformat

from HGTrainer import ValNovTrainer, RobertaForValNovRegression, val_nov_metric
from ArgumentData.GeneralDataset import ValidityNoveltyDataset
from ArgumentData.ExplaGraphs.Loader import load_dataset as load_explagraphs
from ArgumentData.ValTest.Loader import load_dataset as load_valtest
from ArgumentData.ARCT.Loader import load_dataset as load_arct
from ArgumentData.ARCTUKWWarrants.Loader import load_dataset as load_arct_ukw
from ArgumentData.IBMArgQuality.Loader import load_dataset as load_ibm
from ArgumentData.StudentEssays.Loader import load_dataset as load_essays
from loguru import logger

import transformers
import argparse

VERSION: str = "V0.1"

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
    argv.add_argument("--use_ValTest", action="store", nargs="?", default="n/a", type=str, required=False,
                      help="using the validation und test set. You can provide additional arguments for this dataset "
                           "by replacing all the whitespaces with '#', starting with '#'")
    argv.add_argument("--use_ValForTrain", action="store_true", default=False, required=False,
                      help="The ground truth data (dataset annotated with validity and novelty) isn't for training "
                           "(bias) unless you want to handle the validation data as training data as well")
    argv.add_argument("--generate_more_training_samples", action="store_true", default=False, required=False,
                      help="Forces the training set to automatically generate (more) samples")
    argv.add_argument("--cold_start", action="store_true", default=False,  required=False,
                      help="Shuffles the training data before processing it "
                           "(with the https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/"
                           "optimizer_schedules#transformers.get_cosine_schedule_with_warmup)")
    argv.add_argument("--warm_start", action="store_true", default=False,  required=False,
                      help="Do a representative clustering + weighting to have an informed training start")
    argv.add_argument("-s", "--sample", action="store", nargs="?", default="n/a", type=str, required=False,
                      help="Rescale the training data - You can provide additional arguments for this dataset "
                           "by replacing all the whitespaces with '#', starting with '#'")
    argv.add_argument("-skip", "--skip_training", action="store_true", default=False,  required=False)
    argv.add_argument("--save", action="store", nargs="?", type=str, default="n/a", required=False)
    argv.add_argument("--analyse", action="store", nargs="*", type=str, default=["test"], required=False,
                      help="You want to have a depth analysis of your data before training/ testing on it? "
                           "Please define the split(s) here.")

    logger.info("Welcome to the ValidityNoveltyRegressor -- you made {} specific choices", len(sys.argv)-1)

    args = argv.parse_args()
    tokenizer = transformers.RobertaTokenizer.from_pretrained(args.transformer)

    train = ValidityNoveltyDataset(samples=[], tokenizer=tokenizer, max_length=156, name="Train")
    evaluation = ValidityNoveltyDataset(samples=[], tokenizer=tokenizer, max_length=156, name="Eval")
    test = ValidityNoveltyDataset(samples=[], tokenizer=tokenizer, max_length=156, name="Test")

    if args.use_ExplaGraphs != "n/a":
        if args.use_ExplaGraphs is None:
            logger.debug("You want to use the entire ExplaGraphs as a part of the training set - fine")
            train += load_arct(split="train", tokenizer=tokenizer)
            train += load_arct(split="dev", tokenizer=tokenizer)
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
                continuous_val_nov=False if parsed_args_explagraphs.continuous_val_nov < 0 else parsed_args_explagraphs.continuous_val_nov,
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
            for split in (["train", "dev", "test"] if parsed_args_arct.split == "all" else [parsed_args_arct.split]):
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
                exclude_samples_without_detail_annotation_info=
                parsed_args_essays.include_samples_without_detail_annotation_info,
                continuous_val_nov=parsed_args_essays.continuous_val_nov,
                continuous_sample_weight=parsed_args_essays.continuous_sample_weight,
            )

    if len(sys.argv) <= 1 or args.use_ValTest != "n/a":
        if args.use_ValTest != "n/a":
            logger.debug("You want to use the ValTest as a part of the validation/ test set")
        else:
            logger.info("Default included dataset")

        if len(sys.argv) <= 1 or args.use_ValTest is None:
            ds = load_valtest(split="dev", tokenizer=tokenizer)
            if args.use_ValForTrain:
                train += ds
            evaluation += ds
            test += load_valtest(split="test", tokenizer=tokenizer)
        else:
            arg_ValTest = argparse.ArgumentParser(add_help=False, allow_abbrev=True, exit_on_error=False)
            arg_ValTest.add_argument("-l", "--max_length_sample", action="store", default=132, type=int,
                                     required=False)
            arg_ValTest.add_argument("-n", "--max_number", action="store", default=-1, type=int, required=False)
            arg_ValTest.add_argument("-t", "--include_topic", action="store_true",  default=False,
                                     required=False)
            arg_ValTest.add_argument("--continuous_val_nov", action="store_true", default=False,
                                     required=False)
            arg_ValTest.add_argument("--continuous_sample_weight", action="store_true", default=False,
                                     required=False)
            parsed_args_ValTest = arg_ValTest.parse_args(
                args.use_ValTest[1:].split("#") if args.use_ValTest.startswith("#") else args.use_ValTest.split("#")
            )
            ds = load_valtest(
                split="dev", tokenizer=tokenizer,
                max_length_sample=parsed_args_ValTest.max_length_sample,
                max_number=parsed_args_ValTest.max_number,
                include_topic=parsed_args_ValTest.include_topic,
                continuous_val_nov=parsed_args_ValTest.continuous_val_nov,
                continuous_sample_weight=parsed_args_ValTest.continuous_sample_weight
            )
            if args.use_ValForTrain:
                train += ds
            evaluation += ds
            test += load_valtest(
                split="test", tokenizer=tokenizer,
                max_length_sample=parsed_args_ValTest.max_length_sample,
                max_number=parsed_args_ValTest.max_number,
                include_topic=parsed_args_ValTest.include_topic,
                continuous_val_nov=parsed_args_ValTest.continuous_val_nov,
                continuous_sample_weight=parsed_args_ValTest.continuous_sample_weight
            )

    if args.generate_more_training_samples:
        logger.info("You want to automatically generate more training samples - ok, let's do it!")
        samples_generated = train.generate_more_samples()
        logger.info("Training set has the size of {} (+{}) now ({})", len(train), samples_generated,
                    train.get_sample_class_distribution(for_original_data=True))
        train.reset_to_original_data()

    if args.cold_start:
        raise NotImplementedError("Inspired by active learning -- shuffle...")

    if args.warm_start:
        raise NotImplementedError("Inspired by active learning -- representative sampling")

    if args.sample != "n/a":
        logger.info("You want to sample your training data ({})", len(train))

        if args.sample is None:
            final_train_samples = train.sample()
        else:
            arg_sample = argparse.ArgumentParser(add_help=False, allow_abbrev=True, exit_on_error=False)
            arg_sample.add_argument("-n", "--number", action="store", type=int, required=False)
            arg_sample.add_argument("-f", "--fraction", action="store", type=float, required=False)
            arg_sample.add_argument("--not_forced_balanced_dataset", action="store_false", default=True,
                                    required=False)

            parsed_args_sample = arg_sample.parse_args(
                args.sample[1:].split("#") if args.sample.startswith("#") else args.sample.split("#")
            )
            final_train_samples = train.sample(
                number_or_fraction=parsed_args_sample.number if parsed_args_sample.number is not None else
                (parsed_args_sample.fraction if parsed_args_sample.fraction is not None else 1.),
                forced_balanced_dataset=parsed_args_sample.not_forced_balanced_dataset,
                allow_automatically_created_samples=args.generate_more_training_samples or
                                                    (parsed_args_sample.fraction is not None and parsed_args_sample.fraction >= 1)
            )

        logger.trace("Final training-samples: {}", " +++ ".join(map(lambda t: str(t), final_train_samples)))

    logger.info("To summarize, you have {} training data, {} validation data and {} test data", len(train),
                len(evaluation), len(test))
    if not args.skip_training and len(train) == 0:
        logger.warning("You want to train without training data! Please explicit define some with your parameters!")

    if len(test) == 0:
        logger.warning("No test data given...")

    output_dir: Path = Path(".out",
                            VERSION,
                            args.transformer,
                            str(train).replace("(", "_").replace(")", "_").replace("*", ""))
    if output_dir.exists():
        logger.warning("The dictionary \"{}\" exists already - [re]move it!", output_dir.absolute())
        target = Path(".out", VERSION, "_old", args.transformer,
                      "{} {}".format(datetime.datetime.now().isoformat(sep="_", timespec="minutes").replace(":", "-"),
                                     output_dir.name))
        logger.info("Moved to: {}", move(str(output_dir.absolute()), str(target.absolute())))
    output_dir.mkdir(parents=True, exist_ok=True)

    if (isinstance(args.analyse, str) and args.analyse == "train") or "train" in args.analyse:
        train.depth_analysis_data(show_heatmaps=False, handling_not_known_data=.5,
                                  save_heatmaps=str(output_dir.joinpath("train_analyse.png").absolute()))
    if (isinstance(args.analyse, str) and (args.analyse.startswith("dev") or args.analyse.startswith("val"))) \
            or "dev" in args.analyse or "development" in args.analyse \
            or "val" in args.analyse or "validation" in args.analyse:
        evaluation.depth_analysis_data(show_heatmaps=False,
                                       save_heatmaps=str(output_dir.joinpath("dev_analyse.png").absolute()))
    if (isinstance(args.analyse, str) and args.analyse == "test") or "test" in args.analyse:
        test.depth_analysis_data(show_heatmaps=False,
                                 save_heatmaps=str(output_dir.joinpath("test_analyse.png").absolute()))

    trainer = ValNovTrainer(
        model=RobertaForValNovRegression.from_pretrained(pretrained_model_name_or_path=args.transformer),
        args=transformers.TrainingArguments(
            output_dir=str(output_dir),
            do_train=True,
            do_eval=True,
            do_predict=True,
            evaluation_strategy="epoch",
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
            logging_nan_inf_filter=True,
            save_strategy="epoch",
            save_total_limit=1+int(args.verbose),
            label_names=["validity", "novelty"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_approximately_hits",
            greater_is_better=True,
            label_smoothing_factor=0.0,
            push_to_hub=False,
            gradient_checkpointing=args.save_memory_training
        ),
        train_dataset=train,
        eval_dataset=None if len(evaluation) == 0 else evaluation,
        compute_metrics=val_nov_metric,
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=.01)]
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
        test_metrics = trainer.metrics_format(
            metrics=trainer.evaluate(eval_dataset=test,
                                     ignore_keys=["logits", "loss", "hidden_states", "attentions"],
                                     metric_key_prefix="test")
        )

        try:
            logger.success("Test on {} test samples: {}", len(test),
                           ", ".join(map(lambda mv: "{}: {}".format(mv[0], round(mv[1], 3)), test_metrics.items())))
        except TypeError:
            logger.opt(exception=False).warning("Strange test-metrics-outputs-dict. Should be str->float, "
                                                "but we have following: {}",
                                                pformat(test_metrics, indent=2, compact=False))
        trainer.save_metrics(split="test", metrics=test_metrics)

    if args.save != "n/a":
        logger.debug("You want to save the model to {}", args.save)
        try:
            trainer.save_model(output_dir=args.save if args.save is not None else None)
            logger.success("Successfully stored the model in: {}",
                           Path(args.save).absolute() if args.save is not None else output_dir)
        except Exception:
            logger.opt(exception=True).error("Can't save to {}", args.save)

    try:
        info_code = output_dir.joinpath("sys-args.txt").write_text(
            data="Run from {} with following inputs args:\n{}\n\n Train-data:{}/Dev-data:{}/Test-data:{}".format(
                datetime.datetime.now().isoformat(),
                "\n".join(sys.argv[1:]) if len(sys.argv) >= 2 else "no input args",
                train, evaluation, test
            ),
            encoding="utf-8",
            errors="ignore"
        )
        logger.info("OK, we're at the end - let's close the process by writing an Info-file at: {}",
                    output_dir.joinpath("sys-args.txt").absolute())
        logger.trace("Write-code: {}", info_code)
    except IOError:
        logger.opt(exception=False).info("No-info-file provided")
