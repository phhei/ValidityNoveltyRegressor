import sys
from pathlib import Path

from HGTrainer import ValNovTrainer, RobertaForValNovRegression, val_nov_metric
from ArgumentData.GeneralDataset import ValidityNoveltyDataset
from ArgumentData.ExplaGraphs.Loader import load_dataset as load_explagraphs
from ArgumentData.ValTest.Loader import load_dataset as load_valtest
from loguru import logger

import transformers
import argparse

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
    argv.add_argument("--use_ValTest", action="store", nargs="?", default="n/a", type=str, required=False,
                      help="using the validation und test set. You can provide additional arguments for this dataset "
                           "by replacing all the whitespaces with '#', starting with '#'")
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

    logger.info("Welcome to the ValidityNoveltyRegressor -- you made {} specific choices", len(sys.argv)-1)

    args = argv.parse_args()
    tokenizer = transformers.RobertaTokenizer.from_pretrained(args.transformer)

    train = ValidityNoveltyDataset(samples=[], tokenizer=tokenizer, max_length=156, name="Train")
    evaluation = ValidityNoveltyDataset(samples=[], tokenizer=tokenizer, max_length=156, name="Eval")
    test = ValidityNoveltyDataset(samples=[], tokenizer=tokenizer, max_length=156, name="Test")

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
            arg_explagraphs.add_argument("--continuous_val_nov", action="store", default=-1, type=float,
                                         required=False)
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

    if len(sys.argv) <= 1 or args.use_ValTest != "n/a":
        if args.use_ValTest != "n/a":
            logger.debug("You want to use the ValTest as a part of the validation/ test set")
        else:
            logger.info("Default included dataset")

        if len(sys.argv) <= 1 or args.use_ValTest is None:
            evaluation += load_valtest(split="dev", tokenizer=tokenizer)
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
            evaluation += load_valtest(
                split="dev", tokenizer=tokenizer,
                max_length_sample=parsed_args_ValTest.max_length_sample,
                max_number=parsed_args_ValTest.max_number,
                include_topic=parsed_args_ValTest.include_topic,
                continuous_val_nov=parsed_args_ValTest.continuous_val_nov,
                continuous_sample_weight=parsed_args_ValTest.continuous_sample_weight
            )
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
                number_or_fraction=parsed_args_sample.number if "number" in parsed_args_sample else
                (parsed_args_sample.fraction if "fraction" in parsed_args_sample else 1.),
                forced_balanced_dataset=parsed_args_sample.not_forced_balanced_dataset,
                allow_automatically_created_samples=args.generate_more_training_samples or
                                                    ("fraction" in parsed_args_sample and parsed_args_sample.fraction >= 1)
            )

        logger.trace("Final training-samples: {}", " +++ ".join(map(lambda t: str(t), final_train_samples)))

    logger.info("To summarize, you have {} training data, {} validation data and {} test data", len(train),
                len(evaluation), len(test))
    if not args.skip_training and len(train) == 0:
        logger.warning("You want to train without training data! Please explicit define some with your parameters!")

    if len(test) == 0:
        logger.warning("No test data given...")

    trainer = ValNovTrainer(
        model=RobertaForValNovRegression.from_pretrained(pretrained_model_name_or_path=args.transformer),
        args=transformers.TrainingArguments(
            output_dir=".out/{}/".format(args.transformer),
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
            logging_steps=10 if args.verbose else 50,
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

    if len(test) >= 10:
        test_metrics = trainer.metrics_format(
            metrics=trainer.evaluate(eval_dataset=test,
                                     ignore_keys=["logits", "loss", "hidden_states", "attentions"],
                                     metric_key_prefix="test_")
        )

        logger.success("Test on {} test samples: {}", len(test),
                       ", ".join(map(lambda mv: "{}: {}".format(mv[0], round(mv[1], 3)), test_metrics.items())))
        trainer.save_metrics(split="test", metrics=test_metrics)

    if args.save != "n/a":
        logger.debug("You want to save the model to {}", args.save)
        try:
            trainer.save_model(output_dir=args.save if args.save is not None else None)
            logger.success("Successfully stored the model in: {}", Path(args.save).absolute())
        except Exception:
            logger.opt(exception=True).error("Can't save to {}", args.save)
