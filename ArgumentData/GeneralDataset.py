import math
import random
from collections import Counter
from functools import reduce
from pathlib import Path
from typing import Optional, Any, Iterable, List, Union, Dict, Tuple

import matplotlib.pylab as plt
import matplotlib.pyplot
import numpy
import pandas
import seaborn
import torch
from loguru import logger
from pandas import DataFrame
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import PreTrainedTokenizer, AutoTokenizer

from ArgumentData.Sample import Sample
from ArgumentData.Utils import truncate_dataset
from HGTrainer import _val_nov_metric


class ValidityNoveltyDataset(Dataset):
    def __init__(self, samples: Iterable[Sample], tokenizer: PreTrainedTokenizer, max_length: int = 512,
                 name: str = "no name available") -> None:
        super().__init__()

        logger.debug("Init a new dataset instance with {} samples", len(samples))
        logger.debug("Tokenizer: {} ({})", tokenizer.name_or_path, max_length)

        self.name = name
        self.sample_weight: Optional[float] = None

        self.samples_original: List[Sample] = \
            samples if isinstance(samples, List) else list(samples)
        self.samples_extraction: List[Sample] = \
            samples.copy() if isinstance(samples, List) else list(samples)

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.include_premise: bool = True
        self.include_conclusion: bool = True

    def __getitem__(self, index: Any) -> T_co:
        # x
        try:
            sample = self.samples_extraction[index]
        except IndexError:
            logger.opt(exception=False).error("Invalid index {}: {}", index, self.name)
            return self.__getitem__(index=max(0, index-1))

        try:
            # x
            ret = self.tokenizer(
                text=sample.premise if self.include_premise else "",
                text_pair=sample.conclusion if self.include_conclusion else "",
                padding="max_length", truncation="longest_first", max_length=self.max_length,
                is_split_into_words=False, return_tensors="pt",
                return_overflowing_tokens=False, return_special_tokens_mask=False,
                return_offsets_mapping=False, return_length=False, verbose=True
            )
            # y
            ret.update(
                {
                    "validity": torch.FloatTensor([torch.nan if sample.validity is None else sample.validity]),
                    "novelty": torch.FloatTensor([torch.nan if sample.novelty is None else sample.novelty]),
                    "weight": torch.FloatTensor([sample.weight if self.sample_weight is None else self.sample_weight])
                }
            )
        except ValueError:
            logger.opt(exception=False).error("Corrupted sample at position {}: {} - "
                                              "please remove it next time!", index, sample)
            ret = self.tokenizer(
                text="Bugs are not nice.", text_pair="We should avoid errors in coding.",
                padding="max_length", truncation="longest_first", max_length=self.max_length,
                is_split_into_words=False, return_tensors="pt",
                return_overflowing_tokens=False, return_special_tokens_mask=False,
                return_offsets_mapping=False, return_length=False, verbose=True
            )
            ret.update(
                {
                    "validity": torch.FloatTensor([1.]),
                    "novelty": torch.FloatTensor([1.]),
                    "weight": torch.FloatTensor([1e-4])
                }
            )

        ret.data = {k: torch.squeeze(v) for k, v in ret.items()}

        return ret

    def __add__(self, other: Dataset[T_co]) -> Dataset:
        if not isinstance(other, ValidityNoveltyDataset):
            return super().__add__(other)

        logger.info("You want to extend the dataset \"{}\" with \"{}\" ({}+{} = {} instances)",
                    self.name, other.name, len(self), len(other), len(self) + len(other))

        self.name += " + {}".format(other.name.strip())
        if self.tokenizer != other.tokenizer:
            logger.warning("The tokenizer of the added dataset is different -- ignore \"{}\"!", other.tokenizer)
        if self.max_length != other.max_length:
            logger.warning("The max-length of tokenizer of the added dataset is different -- take the max of {} -- {}",
                           self.max_length, other.max_length)
            self.max_length = max(self.max_length, other.max_length)

        self.samples_original.extend(other.samples_original)
        self.samples_extraction.extend(other.samples_extraction)

        return self

    def __len__(self) -> int:
        return len(self.samples_extraction)

    def __str__(self) -> str:
        return "{} ({} out of {} selected){}{}".format(
            self.name,
            len(self.samples_extraction), len(self.samples_original),
            "" if self.include_premise else " -- without premises!",
            "" if self.include_conclusion else " -- without conclusions!"
        )

    def fix_sample_weights(self, fixed_value: float) -> None:
        """
        Fixes the sample weights for each sample to a given valuen in the dataset (regardless what the actual
        sample weight is)

        :param fixed_value: the fixed values
        :return: nothing
        """
        if self.sample_weight is None:
            logger.warning("You change this dataset from flexible (dataset/ instance-specific) sample weight to a"
                           "fixed sample weight of {}", fixed_value)
        if fixed_value < 0:
            logger.warning("You set your sample weight to a negative value! This will result in avoiding the "
                           "ground truth-models!")

        self.name += " (w{})".format(fixed_value)
        self.sample_weight = fixed_value

        logger.debug("Successfully updated \"{}\"", self)

    def dataset_path(self, base_path: Optional[Path], num_samples: Optional[int] = None) -> Path:
        """
        Generates a proper dataset path (for storing stats, fine-tuned models ect.)

        :param base_path: a base path
        :param num_samples: how many samples are you expect to hvae in the dataset? If no number si given, the number
        is the actual number of sample (in the extracted split)
        :return: a good path
        """
        if base_path is None:
            base_path = Path()
            logger.info("Set the ground path to \"{}\"", base_path.absolute())

        base_path = base_path.joinpath(self.name.replace("(", "_").replace(")", "_").replace("*", ""),
                                       "{} samples".format(len(self.samples_extraction)
                                                           if num_samples is None else num_samples))
        if not self.include_premise:
            base_path =  base_path.joinpath("wo premise")
        if not self.include_conclusion:
            base_path = base_path.joinpath("wo conclusion")
        logger.debug("Home directory of the dataset: \"{}\"", base_path)
        if not base_path.exists():
            logger.debug("This directory doesn't exist - create")
            base_path.mkdir(parents=True, exist_ok=False)
            sub_counter = 0
        else:
            sub_counter = len(list(base_path.glob(pattern="try*")))
            logger.trace("Found {} tries", sub_counter)

        return base_path.joinpath("try-{}".format(sub_counter))

    def reset_to_original_data(self):
        """
        A dataset contains a set of original data (unaffected by sampling, the pure data at initialization time) and an
        extraction, for example forced by sampling. This method resets the extraction to the original data

        :return: nothing -> see self.samples_extraction
        """
        self.sample(number_or_fraction=float(1), forced_balanced_class_distribution=False, force_classes=False,
                    allow_automatically_created_samples=False)
        logger.success("Reset \"{}\" to its original data successfully", self.name)

    # noinspection PyBroadException
    def generate_more_samples(self) -> int:
        """
        Generates for each sample in the original data 4 new/ similar ones (or at least tries it)
        -- valid and novel
        -- valid and not novel
        -- not valid and novel
        -- not valid and not novel

        This affects only the sample_extraction, not the sample_original!

        :return: The number of samples added to the sample_extraction
        """
        new_generated = 0

        for i, sample in enumerate(self.samples_original):
            try:
                self.samples_extraction.append(sample.automatically_create_valid_novel_sample())
                logger.trace("Created a new valid and novel sample: \"{}\"", self.samples_extraction[-1])
                new_generated += 1
            except Exception:
                logger.opt(exception=False).info("No new valid and novel sample for \"{}\"", sample)
            try:
                self.samples_extraction.append(sample.automatically_create_valid_non_novel_sample(
                    other_random_sample=random.choice(self.samples_original)
                    if len(self.samples_original) >= 10 else None
                ))
                logger.trace("Created a new valid and non-novel sample: \"{}\"", self.samples_extraction[-1])
                new_generated += 1
            except Exception:
                logger.opt(exception=False).info("No new valid and non-novel sample for \"{}\"", sample)
            try:
                self.samples_extraction.append(sample.automatically_create_non_valid_novel_sample())
                logger.trace("Created a new non-valid and novel sample: \"{}\"", self.samples_extraction[-1])
                new_generated += 1
            except Exception:
                logger.opt(exception=False).info("No new non-valid and novel sample for \"{}\"", sample)
            try:
                self.samples_extraction.append(sample.automatically_create_non_valid_non_novel_sample(
                    other_random_sample=random.choice(self.samples_original)
                    if len(self.samples_original) >= 10 else None
                ))
                logger.trace("Created a new non-valid and non-novel sample: \"{}\"", self.samples_extraction[-1])
                new_generated += 1
            except Exception:
                logger.opt(exception=False).info("No new non-valid and novel sample for \"{}\"", sample)

            logger.debug("Processed / regenerated sample {} ({}%)", sample, round(100*i/len(self.samples_original), 1))

        logger.success("Successfully generated {} new sample. Be aware! We added the new samples only to the original "
                       "data, not to the extracted data. If you want to load them, execute self.reset_to_original_data",
                       new_generated)

        return new_generated

    def deduplicate(self, original_data: bool = True, extracted_data: bool = True) -> int:
        """
        Deduplicate the samples in the dataset.

        :param original_data: deduplication in the original data
        :param extracted_data: deduplication in th extracted data
        :return: the number of removed samples in total
        """
        logger.trace("Let's deduplicate {}/ {} samples...",
                     len(self.samples_original) if original_data else 0,
                     len(self.samples_extraction) if extracted_data else 0)

        removed_samples = 0
        len_self_samples_original = len(self.samples_original)
        len_self_samples_extraction = len(self.samples_extraction)

        if original_data:
            self.samples_original = list(set(self.samples_original))
            self.name += " (deduplicated)"
            removed_samples += len_self_samples_original - len(self.samples_original)
        if extracted_data:
            self.samples_extraction = list(set(self.samples_extraction))
            removed_samples += len_self_samples_extraction - len(self.samples_original)

        try:
            logger.success("Successfully duplicated dataset and removed {} samples ({}%)",
                           removed_samples,
                           round(100*removed_samples /
                                 sum([len_self_samples_original if original_data else 0,
                                      len_self_samples_extraction if extracted_data else 0]), 1))
        except ArithmeticError:
            logger.opt(exception=True).warning("I guess both original_data and extracted_data were False?")

        return removed_samples

    def get_sample_class_distribution(self, for_original_data: bool = False) -> Counter:
        """
        Calculates the class distribution (validity, novelty) of the samples

        :param for_original_data: if True, counts in the original data, else in the extracted data
        :return: a dictionary with the counts the class occurrences in the samples
        """
        data = self.samples_original if for_original_data else self.samples_extraction

        logger.trace("Looking at {} samples", len(data))

        ret = Counter(
            [("valid" if c_s.is_valid(none_is_not=True) else ("?" if c_s.validity is None else "not valid"),
              ("novel" if c_s.is_novel(none_is_not=True) else ("?" if c_s.novelty is None else "not novel")))
             for c_s in data]
        )

        logger.debug("Calculated the class distribution for the {} data, most common class is: {}",
                     "original" if for_original_data else "extracted", ret.most_common(1)[0][0])

        return ret

    def depth_analysis_data(self, for_original_data: bool = False, steps: int = 5,
                            show_heatmaps: bool = True, handling_not_known_data: Optional[float] = None,
                            save_heatmaps: Optional[str] = None) -> None:
        """
        Having a depth analyses of the dataset.

        :param for_original_data: if True, counts in the original data, else in the extracted data
        :param steps: validity and novelty are continuous in the range of 0 to 1.
        How many steps/ buckets should we use to make this discrete for the analyse?
        :param show_heatmaps: Creates and shows a heatmap-file, if True
        :param handling_not_known_data: Samples can have a unknown validity/ novelty. If you set a number here,
        those unknown values will be treated as the given number, else such samples will be ignored in the process
        :param save_heatmaps: Saves the heatmaps (please give a target directory here)
        :return: nothing
        """
        data = self.samples_original if for_original_data else self.samples_extraction

        if len(data) == 0:
            logger.warning("Not applicable in an empty dataset!")
        else:
            logger.trace("Test {} instances", len(data))

        if handling_not_known_data is None:
            logger.trace("You wan to ignore the data with unknown validity/ novelty - ok!")
            data = [d for d in data if d.validity is not None and d.novelty is not None]
            logger.info("Discarded {} instances, {} remain",
                        (len(self.samples_original) if for_original_data else len(self.samples_extraction))-len(data),
                        len(data))
            if len(data) == 0:
                logger.warning("You discarded all your data by ignoring unknown data-fields. "
                               "Please try to set the handling_not_known_data to a float "
                               "(treat unknown fields as this number for testing -- does not change the data itself)")
                return
        else:
            logger.trace("You want to treat unknown data fields as {}", handling_not_known_data)
            data = [Sample(
                premise=d.premise,
                conclusion=d.conclusion,
                validity=handling_not_known_data if d.validity is None else d.validity,
                novelty=handling_not_known_data if d.novelty is None else d.novelty
            ) for d in data]
            logger.debug("Converted {} samples", len(data))

        heatmap = numpy.zeros((steps, steps), dtype=float)
        for i, val in enumerate(numpy.arange(0, 1, 1 / steps)):
            include_max_border_validity = (i >= steps - 1)
            for j, nov in enumerate(numpy.arange(0, 1, 1 / steps)):
                include_max_border_novelty = (j >= steps - 1)
                heatmap[i][j] = \
                        100*len([d for d in data
                                 if val <= d.validity < val + (1 + int(include_max_border_validity)) / steps and
                                 nov <= d.novelty < nov + (1 + int(include_max_border_novelty)) / steps])/len(data)

        with numpy.printoptions(precision=1, threshold=20, edgeitems=5, sign="-"):
            logger.success("Calculated the percentage-heatmap (row-number: validity, col-number: novelty): {}", heatmap)

        logger.trace("OK, now, let's test some voters...")

        heatmap_prediction = numpy.zeros((steps + 1, steps + 1), dtype=float)
        should_val_vector = numpy.array([d.validity for d in data], dtype=float, copy=False)
        should_nov_vector = numpy.array([d.novelty for d in data], dtype=float, copy=False)
        for i, val in enumerate(numpy.arange(0, 1 + 1 / steps, 1 / steps)):
            for j, nov in enumerate(numpy.arange(0, 1 + 1 / steps, 1 / steps)):
                results = _val_nov_metric(is_validity=numpy.ones((len(data),), dtype=float) * val,
                                          is_novelty=numpy.ones((len(data),), dtype=float) * nov,
                                          should_validity=should_val_vector,
                                          should_novelty=should_nov_vector)
                logger.info("OK, a predictor predicting always validity = {} and novelty = {} would have: {}",
                            round(val, 2), round(nov, 2), results)
                try:
                    heatmap_prediction[i][j] = 100 * results["approximately_hits"]
                    logger.debug("Number of approximately_hits: {}%", round(100*results["approximately_hits"]))
                except KeyError:
                    logger.opt(exception=True).warning("Return (dict) of function \"_val_nov_metric\" changed - "
                                                       "please update the code!")

        if save_heatmaps is not None or show_heatmaps:
            logger.trace("You want to see the heatmap - ok, lets load the matplotlib")
            fig, (ax1, ax2) = matplotlib.pyplot.subplots(1, 2)
            seaborn.heatmap(heatmap, vmin=0, vmax=min(100, numpy.max(heatmap)*2),
                            annot=True, fmt=".0f", linewidths=.25, cbar=False,
                            xticklabels=["{}>={}".format("nov" if i == 0 else "", round(i, 2))
                                         for i in numpy.arange(0, 1, 1 / steps)],
                            yticklabels=["{}>={}".format("val" if i == 0 else "", round(i, 2))
                                         for i in numpy.arange(0, 1, 1 / steps)],
                            ax=ax1)
            ax1.set_title("Sample distribution (%)")
            logger.trace("Heatmap was produced on axis {}...", ax1)

            logger.trace("You want to see the heatmap of prediction-approximately_hits")
            seaborn.heatmap(heatmap_prediction, vmin=0, vmax=100, annot=True, fmt=".0f", linewidths=.25, cbar=False,
                            xticklabels=["{}={}".format("nov" if i == 0 else "", round(i, 2))
                                         for i in numpy.arange(0, 1 + 1 / steps, 1 / steps)],
                            yticklabels=["{}={}".format("val" if i == 0 else "", round(i, 2))
                                         for i in numpy.arange(0, 1 + 1 / steps, 1 / steps)],
                            ax=ax2)
            ax2.set_title("Approximately hits of a static predictor (%)")
            fig.set_size_inches(15, 7.5)

            if show_heatmaps:
                plt.show()
                logger.info("Heatmap was shown...")
            if save_heatmaps is not None:
                Path(save_heatmaps).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_heatmaps, dpi=200)
                logger.info("Heatmaps saved at \"{}\"", save_heatmaps)

        logger.debug("Close the statistics-process...")

    def equalize_source_distribution(self, minimum_number: Optional[int] = None,
                                     maximum_number: Optional[int] = None, allow_duplicates: bool = True) -> bool:
        """
        A dataset can consist various data sample sources. With this method, you can equalize the sample source
        distribution in the extraction (not in the original data)

        :param minimum_number: minimum accepted number for a source.
        If not given, the minimum number is the base source with the fewest samples
        :param maximum_number: maximum accepted number for a source.
        If not given, the maximum number is the base source with the fewest samples
        :param allow_duplicates: in case you have too few samples from a specific source,
        allow to copy samples to get more
        :return: True, if the method was successfully (all sources are in the given range). If False, this failed
        """
        if len(self.samples_extraction):
            logger.info("Your dataset is empty (at least the current extracted samples)!")
            return False

        ret = True

        base_sources = Counter(
            [s_e.source[:(s_e.source.index("[") if "[" in s_e.source else len(s_e.source))]
             for s_e in self.samples_extraction]
        )

        logger.info("Found {} base sources from {}-{} samples, e.g.: {}",
                    len(base_sources),
                    min(base_sources.values()),
                    max(base_sources.values()),
                    " and ".join(map(lambda source: "{} ({}x)".format(*source),
                                     base_sources.most_common(min(len(base_sources), 3)))))

        minimum_number = minimum_number or min(base_sources.values())
        maximum_number = maximum_number or min(base_sources.values())

        if minimum_number > maximum_number:
            raise AttributeError("The maximum number ({}) should be higher or equal than the minimum number ({})",
                                 maximum_number, minimum_number)

        for base_source, count in base_sources.items():
            if minimum_number <= count <= maximum_number:
                logger.debug("Base source \"{}\" is correctly represented ({} <= {} <= {})",
                             base_source, minimum_number, count, maximum_number)
            elif minimum_number > count:
                if allow_duplicates:
                    samples_needed = minimum_number-count
                    logger.warning("We have to duplicate {} times because \"{}\" is underrepresented",
                                   samples_needed, base_source)

                    source_samples = \
                        [d_s for d_s in self.samples_extraction
                         if d_s.source[:d_s.source.index("[") if "[" in d_s.source else len(d_s.source)] == base_source]

                    while samples_needed >= count:
                        logger.trace("Add all available samples from \"{}\"", base_source)
                        self.samples_extraction.extend(source_samples)
                        samples_needed -= len(source_samples)

                    if samples_needed >= 1:
                        self.samples_extraction.extend(source_samples[:samples_needed])
                else:
                    logger.error("Base source \"{}\" is underrepresented ({} < {})",
                                 base_source, count, minimum_number)
                    ret = False
            elif maximum_number < count:
                logger.warning("Base source \"{}\" is overrepresented - let's change this! (delete {} samples)",
                               base_source, count - maximum_number)
                source_samples = \
                    [d_s for d_s in self.samples_extraction
                     if d_s.source[:d_s.source.index("[") if "[" in d_s.source else len(d_s.source)] == base_source]

                for s in source_samples:
                    logger.trace("First, remove {}", s)
                    self.samples_extraction.remove(s)

                self.samples_extraction.extend(truncate_dataset(data=source_samples, max_number=maximum_number))
                logger.debug("Finally removed {} {}-source-samples (out of {})",
                             count - maximum_number, base_source, count)

        logger.debug("OK, all samples in extract ({}) are better source-balanced now", len(self.samples_extraction))

        return ret

    def sample(self, number_or_fraction: Union[int, float] = .5,
               allow_automatically_created_samples: bool = False,
               forced_balanced_class_distribution: bool = True,
               force_classes: Union[bool, List[Tuple[Union[int, str], Union[int, str]]]] = True,
               seed: int = 42) -> List[Sample]:
        """
        Samples from the extracted data (only affects the extracted data)

        :param number_or_fraction:
        :param allow_automatically_created_samples: if necessary, fills underrepresented classes with automatic
        created samples
        :param forced_balanced_class_distribution: if True, the class distribution must be balanced
        :param force_classes: ... between the given classes
        :param seed: the sampled data is randomly sampled from the extracted data beforehand -
        here we can define the random seed
        :return: The list of sampled samples
        """
        def equal_class_distribution_dataset(minimum_number: int, maximum_number: Optional[int] = None) -> None:
            logger.debug("You want to have {} sample(s) of following combinations: {}",
                         "{} - {}".format(minimum_number, maximum_number)
                         if maximum_number is not None else ">= {}".format(minimum_number),
                         "all" if isinstance(force_classes, bool) else
                         " + ".join(map(
                             lambda f: "{}/{}".format(
                                 f[0] if isinstance(f[0], str) else
                                 ("valid" if f[0] == 1 else ("not valid" if f[0] == -1 else "unknown validity")),
                                 f[1] if isinstance(f[1], str) else
                                 ("novel" if f[1] == 1 else ("not novel" if f[1] == -1 else "unknown novelty"))
                             ),
                             force_classes
                         )))
            _count_extract: Counter = self.get_sample_class_distribution(for_original_data=False)
            force_classes_set = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)] \
                if isinstance(force_classes, bool) else force_classes
            maximum_number = maximum_number or len(self.samples_extraction)
            for validity, novelty in force_classes_set:
                validity = validity if isinstance(validity, int) \
                    else (-1 if "not" in validity else (0 if "unknown" in validity else 1))
                novelty = novelty if isinstance(novelty, int) \
                    else (-1 if "not" in novelty else (0 if "unknown" in novelty else 1))
                logger.trace("Are there {} sample(s) having {} validity and {} novelty?",
                             "{} - {}".format(minimum_number, maximum_number),
                             "" if validity == 1 else ("no" if validity == -1 else "unknown"),
                             "" if novelty == 1 else ("no" if novelty == -1 else "unknown"))
                _count = _count_extract[
                    (
                        "valid" if validity == 1 else ("not valid" if validity == -1 else "?"),
                        "novel" if novelty == 1 else ("not novel" if novelty == -1 else "?")
                    )
                ]
                if minimum_number <= _count <= maximum_number:
                    logger.trace("Perfect, nothing to do: {}<={}<={}", minimum_number, _count, maximum_number)
                elif _count < minimum_number:
                    logger.debug("We have too few {}-{}-samples: we need {} but have only {}",
                                 "valid" if validity == 1 else ("non-valid" if validity == -1 else ""),
                                 "novel" if novelty == 1 else ("non-novel" if novelty == -1 else ""),
                                 minimum_number,
                                 _count)
                    still_needed = minimum_number - _count

                    samples_extraction_without_unknown = \
                        self.samples_extraction if len(
                            filtered := [sample for sample in self.samples_extraction
                                         if sample.validity is not None and sample.novelty is not None]
                        ) < 3 else filtered
                    samples_extraction_validity_unknown = \
                        self.samples_extraction if len(
                            filtered := [sample for sample in self.samples_extraction
                                         if sample.validity is None and sample.novelty is not None]
                        ) < 3 else filtered
                    samples_extraction_novelty_unknown = \
                        self.samples_extraction if len(
                            filtered := [sample for sample in self.samples_extraction
                                         if sample.validity is not None and sample.novelty is None]
                        ) < 3 else filtered

                    if allow_automatically_created_samples:
                        while still_needed > 0:
                            try:
                                stem = random.choice(samples_extraction_validity_unknown
                                                     if validity == 0 else
                                                     (samples_extraction_novelty_unknown
                                                      if novelty == 0 else samples_extraction_without_unknown))
                                logger.trace("OK, trying to make something out of: {}", stem)
                                if validity == 1 and novelty == 1:
                                    self.samples_extraction.append(stem.automatically_create_valid_novel_sample())
                                elif validity == 1 and novelty == 0:
                                    _stem = stem if stem.is_valid(none_is_not=True) and stem.novelty is not None else \
                                        stem.automatically_create_valid_non_novel_sample(
                                            other_random_sample=random.choice(self.samples_original)
                                        )
                                    self.samples_extraction.append(Sample(
                                        premise=_stem.premise,
                                        conclusion=_stem.conclusion,
                                        validity=_stem.validity,
                                        novelty=None,
                                        weight=_stem.weight,
                                        source=_stem.source
                                    ))
                                elif validity == 1 and novelty == -1:
                                    self.samples_extraction.append(stem.automatically_create_valid_non_novel_sample(
                                        other_random_sample=random.choice(self.samples_original)
                                    ))
                                elif validity == 0 and novelty == 1:
                                    _stem = stem if stem.is_novel(none_is_not=True) and stem.validity is not None else \
                                        stem.automatically_create_non_valid_novel_sample()
                                    self.samples_extraction.append(Sample(
                                        premise=_stem.premise,
                                        conclusion=_stem.conclusion,
                                        validity=None,
                                        novelty=_stem.novelty,
                                        weight=_stem.weight,
                                        source=_stem.source
                                    ))
                                elif validity == 0 and novelty == 0:
                                    logger.warning("Generating a sample without known validity and novelty "
                                                   "doesn't make sense for training, anyway, we mask {}", stem)
                                    self.samples_extraction.append(Sample(
                                        premise=stem.premise,
                                        conclusion=stem.conclusion,
                                        validity=None,
                                        novelty=None,
                                        weight=1e-4,
                                        source=stem.source
                                    ))
                                elif validity == 0 and novelty == -1:
                                    _stem = stem if not stem.is_novel(none_is_not=True) \
                                                    and stem.validity is not None else \
                                        stem.automatically_create_non_valid_non_novel_sample(
                                            other_random_sample=random.choice(self.samples_original)
                                        )
                                    self.samples_extraction.append(Sample(
                                        premise=_stem.premise,
                                        conclusion=_stem.conclusion,
                                        validity=None,
                                        novelty=_stem.novelty,
                                        weight=_stem.weight,
                                        source=_stem.source
                                    ))
                                elif validity == -1 and novelty == 1:
                                    self.samples_extraction.append(stem.automatically_create_non_valid_novel_sample())
                                elif validity == -1 and novelty == 0:
                                    _stem = stem if not stem.is_valid(none_is_not=True) \
                                                    and stem.novelty is not None else \
                                        stem.automatically_create_non_valid_non_novel_sample(
                                            other_random_sample=random.choice(self.samples_original)
                                        )
                                    self.samples_extraction.append(Sample(
                                        premise=_stem.premise,
                                        conclusion=_stem.conclusion,
                                        validity=_stem.validity,
                                        novelty=None,
                                        weight=_stem.weight,
                                        source=_stem.source
                                    ))
                                elif validity == -1 and novelty == -1:
                                    self.samples_extraction.append(stem.automatically_create_non_valid_non_novel_sample(
                                        other_random_sample=random.choice(self.samples_original)
                                    ))
                                else:
                                    logger.error("Unexpected combination: {}/{}", validity, novelty)
                                    break
                                still_needed -= 1
                            except Exception:
                                logger.opt(exception=True).warning("Failed to automatically create such a sample, "
                                                                   "next try please!")
                        logger.success("Successfully created {} new samples which are {} and {}",
                                       minimum_number - _count,
                                       "valid" if validity == 1 else ("not valid" if validity == -1 else "?valid?"),
                                       "novel" if novelty == 1 else ("not novel" if novelty == -1 else "?novel?"))
                    else:
                        useful_samples = [sample for sample in self.samples_original
                                          if sample not in self.samples_extraction and
                                          ((sample.is_valid(none_is_not=True) and validity == 1) or
                                           (sample.validity is None and validity == 0) or
                                           (not sample.is_valid(none_is_not=True) and sample.validity is not None and
                                            validity == -1)) and
                                          ((sample.is_novel(none_is_not=True) and novelty == 1) or
                                           (sample.novelty is None and novelty == 0) or
                                           (not sample.is_novel(none_is_not=True) and sample.novelty is not None and
                                            novelty == -1))]

                        if len(useful_samples) < still_needed:
                            logger.warning("You want to add at least {} samples which are {} and {} but you only have "
                                           "{} left! Consider can only those...",
                                           still_needed,
                                           "valid" if validity == 1 else ("not valid" if validity == -1 else "?valid?"),
                                           "novel" if novelty == 1 else ("not novel" if novelty == -1 else "?novel?"),
                                           len(useful_samples))
                            self.samples_extraction.extend(useful_samples)
                        elif len(useful_samples) == still_needed:
                            logger.info("Perfect - you need {} further samples for {}/{}, "
                                        "the exact number that we found!",
                                        still_needed,
                                        "validity" if validity == 1 else ("non-validity" if validity == -1 else "-"),
                                        "novelty" if novelty == 1 else ("non-novelty" if novelty == -1 else "-"))
                        else:
                            logger.debug("More the enough: chose {} out of {}", still_needed, len(useful_samples))
                            self.samples_extraction.extend(random.sample(useful_samples, k=still_needed))
                else:
                    logger.info("We have too much {}-{}-samples: {} > {}",
                                "valid" if validity == 1 else ("non-valid" if validity == -1 else ""),
                                "novel" if novelty == 1 else ("non-novel" if novelty == -1 else ""),
                                _count, maximum_number)
                    used_samples = [sample for sample in self.samples_extraction if
                                    ((sample.is_valid(none_is_not=True) and validity == 1) or
                                     (sample.validity is None and validity == 0) or
                                     (not sample.is_valid(none_is_not=True) and sample.validity is not None and
                                      validity == -1)) and
                                    ((sample.is_novel(none_is_not=True) and novelty == 1) or
                                     (sample.novelty is None and novelty == 0) or
                                     (not sample.is_novel(none_is_not=True) and sample.novelty is not None
                                      and novelty == -1))]
                    remove_samples = random.sample(used_samples, k=_count-maximum_number)
                    logger.trace("Remove following samples: {}", remove_samples)
                    for rem_sample in remove_samples:
                        try:
                            self.samples_extraction.remove(rem_sample)
                        except ValueError:
                            logger.opt(exception=True).warning("We cannot remove {}", rem_sample)
            logger.success("Closed all {} cases of forcing sample balance ({}-{}). "
                           "Be beware: we only manipulate the extraction-part of the dataset, "
                           "not the original data ({})",
                           len(force_classes_set), minimum_number, maximum_number,
                           len(self.samples_extraction), len(self.samples_original))

        number = number_or_fraction if isinstance(number_or_fraction, int) else \
            round(number_or_fraction*len(self.samples_original))
        if number > len(self.samples_original):
            if allow_automatically_created_samples:
                logger.debug("You want to have more samples than you have in your database: {} vs. {}. "
                             "This will force automatically created samples!",
                             number, len(self.samples_original))
            else:
                logger.warning("You want to sample more samples ({}) than you have ({}) - this is not possible "
                               "without additional sample creations which you've deactivated!",
                               number, len(self.samples_original))
                number = len(self.samples_original)

        if number == len(self.samples_original) and not forced_balanced_class_distribution:
            logger.warning("You want to reset your dataset to the original data ({}->{} samples)",
                           len(self.samples_extraction), len(self.samples_original))
            self.samples_extraction = list(self.samples_original)
            return self.samples_extraction

        random.seed(seed)
        logger.debug("OK, you will select {} (out of {})", number, len(self.samples_original))
        logger.debug("Random state: {}", random.getstate())

        if not forced_balanced_class_distribution:
            self.samples_extraction = random.sample(self.samples_original, k=number)
            if isinstance(force_classes, List) or force_classes:
                equal_class_distribution_dataset(minimum_number=1)
                return self.samples_extraction
            else:
                count_extract = self.get_sample_class_distribution(for_original_data=False)
                count_original = self.get_sample_class_distribution(for_original_data=True)
                logger.warning("You don't care about the novel/ valid-distribution. "
                               "So, we have: not-valid-not-novel: {}%->{}%, not-valid-novel: {}%->{}%,"
                               "valid-not-novel: {}%->{}%, valid-novel: {}%->{}%",
                               round(100 * count_original[("not valid", "not novel")] / len(self.samples_original)),
                               round(100 * count_extract[("not valid", "not novel")] / len(self.samples_extraction)),
                               round(100 * count_original[("not valid", "novel")] / len(self.samples_original)),
                               round(100 * count_extract[("not valid", "novel")] / len(self.samples_extraction)),
                               round(100 * count_original[("valid", "not novel")] / len(self.samples_original)),
                               round(100 * count_extract[("valid", "not novel")] / len(self.samples_extraction)),
                               round(100 * count_original[("valid", "novel")] / len(self.samples_original)),
                               round(100 * count_extract[("valid", "novel")] / len(self.samples_extraction)))
                return self.samples_extraction

        logger.trace("You want force a balanced sample by having {}",
                     self.get_sample_class_distribution(for_original_data=False))

        all_classes_set = {(1, 1), (1, 0), (1, -1), (0, 1), (0, 0), (0, -1), (-1, 1), (-1, 0), (-1, -1)}
        chosen_class_set = {(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)}\
            if isinstance(force_classes, bool) and force_classes else \
            ({} if isinstance(force_classes, bool) and not force_classes else
             {((-1 if "not" in val else (0 if "unknown" in val else 1)) if isinstance(val, str) else val,
               (-1 if "not" in nov else (0 if "unknown" in nov else 1)) if isinstance(nov, str) else nov)
              for val, nov in force_classes})
        not_chosen_class_set = all_classes_set.difference(chosen_class_set)

        logger.debug("forcing equal distributions only in {} class-cases (not in {})", len(chosen_class_set),
                     not_chosen_class_set)

        samples_in_not_chosen_set = reduce(lambda l1, l2: l1+l2, map(
            lambda cc: [sample for sample in self.samples_extraction if
                        ((sample.is_valid(none_is_not=True) and cc[0] == 1) or
                         (sample.validity is None and cc[0] == 0) or
                         (not sample.is_valid(none_is_not=True) and sample.validity is not None and cc[0] == -1)) and
                        ((sample.is_novel(none_is_not=True) and cc[1] == 1) or
                         (sample.novelty is None and cc[1] == 0) or
                         (not sample.is_novel(none_is_not=True) and sample.novelty is not None and cc[1] == -1))],
            not_chosen_class_set
        ))
        logger.debug("Having {} samples matching non of the classes which should be equally distributed",
                     len(samples_in_not_chosen_set))

        if number / 5 < len(samples_in_not_chosen_set):
            logger.info("There are too many samples of \"don't-care\"-classes ({} >> {}/5) - we have to cut here!",
                        len(samples_in_not_chosen_set), number)
            samples_to_delete = random.sample(samples_in_not_chosen_set,
                                              k=len(samples_in_not_chosen_set)-int(number / 5))
            for s in samples_to_delete:
                logger.trace("Removing unnecessary sample: {}", s)
                self.samples_extraction.remove(s)
            samples_in_not_chosen_set_in_extraction = len(samples_in_not_chosen_set)-len(samples_to_delete)
        else:
            samples_in_not_chosen_set_in_extraction = len(samples_in_not_chosen_set)

        number_samples_class = (number - samples_in_not_chosen_set_in_extraction)/len(chosen_class_set)

        equal_class_distribution_dataset(minimum_number=math.floor(number_samples_class),
                                         maximum_number=math.ceil(number_samples_class))

        return self.samples_extraction

    def save(self, path: Optional[Union[Path, str]] = None) -> Path:
        """
        Saves the dataset with all its components, including the original and extracted data in a CSV

        :param path: the target path (directory)
        :return: the path which the dataset was saved
        """
        logger.info("Prepare \"{}\" for saving", self)

        if path is None:
            logger.trace("No path given...")
            path: Path = Path("saved_datasets",
                              self.name.replace("<", "l").replace(">", "g").replace(":", "_").replace("\"", "qq").
                              replace("/", "_").replace("\\", "_").replace("|", "-").replace("?", " ").
                              replace("*", " "))
            logger.debug("Selected path: {}", path)
        elif isinstance(path, str):
            logger.trace("We have to convert the path-string to a path-object first!")
            path = Path(path)

        logger.debug("Let's create the root \"{}\"", path.name)

        path.mkdir(parents=True, exist_ok=True)

        logger.info("Write name \"{}\": {}", self.name,
                    path.joinpath("name.prop").write_text(data=self.name, encoding="utf8", errors="ignore"))
        logger.info("Write max_length \"{}\": {}", self.max_length,
                    path.joinpath("max_length.prop").write_text(data=str(self.max_length),
                                                                encoding="utf8", errors="ignore"))
        logger.info("Write tokenizer \"{}\": {}", self.tokenizer.name_or_path,
                    path.joinpath("tokenizer.prop").write_text(data=self.tokenizer.name_or_path,
                                                               encoding="utf8", errors="ignore"))

        DataFrame.from_records(
            data=[(s.premise, s.conclusion, s.validity, s.novelty, s.weight, s.source) for s in self.samples_original],
            columns=["Premise", "Conclusion", "Validity", "novelty", "Weight", "Source"]
        ).to_csv(
            path_or_buf=str(path.joinpath("samples_original.csv").absolute()),
            index=False,
            encoding="utf8",
            errors="ignore"
        )

        DataFrame.from_records(
            data=[(s.premise, s.conclusion, s.validity, s.novelty, s.weight, s.source)
                  for s in self.samples_extraction],
            columns=["Premise", "Conclusion", "Validity", "Novelty", "Weight", "Source"]
        ).to_csv(
            path_or_buf=str(path.joinpath("samples_extraction.csv").absolute()),
            index=False,
            encoding="utf8",
            errors="ignore"
        )

        logger.success("Wrote [{}] files in \"{}\"", ", ".join(map(lambda f: f.name, path.iterdir())), path.name)

        return path

    @staticmethod
    def load(path: Union[Path, str]) -> Dict[str, Any]:
        """
        Load datasets which were previously saved by the save-method

        :param path: the directory to load from
        :return: a dictionary with 3 datasets: - train - dev - test
        """
        if isinstance(path, str):
            logger.trace("Have to convert to a PATH-object first")
            path = Path(path)

        logger.debug("Let's load from \"{}\"", path.name)

        if not path.exists():
            raise FileNotFoundError("Please chose a already existing directory ({})".format(path.absolute()))
        if not path.is_dir():
            raise AttributeError(
                "{} has to be a directory, containing a \"_dev\", \"_test\" and \"_train\"-folder".format(
                    path.absolute()
                )
            )

        ret_datasets = dict()

        for split in ["train", "dev", "test"]:
            logger.debug("Let's load the {}-split", split)
            try:
                sub_path = path.joinpath("_{}".format(split))
                name: str = sub_path.joinpath("name.prop").read_text(encoding="utf-8", errors="ignore") \
                    if sub_path.joinpath("name.prop").exists() else "name not found"
                max_length: int = int(sub_path.joinpath("max_length.prop").read_text(encoding="utf-8", errors="ignore"))
                tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                    sub_path.joinpath("tokenizer.prop").read_text(encoding="utf-8", errors="strict")
                )
                logger.info("Loaded the basic properties of \"{}\". Now, the data is left", name)
                sample_data = dict()
                for part in ["samples_extraction.csv", "samples_original.csv"]:
                    data_sub_path = sub_path.joinpath(part)
                    logger.trace("Let's load {}", data_sub_path)
                    samples = []
                    for sid, row in pandas.read_csv(str(data_sub_path.absolute()), encoding="utf-8").iterrows():
                        logger.trace("Read line \"{}\"", sid)
                        samples.append(Sample(
                            premise=row["Premise"],
                            conclusion=row["Conclusion"],
                            validity=row["Validity"],
                            novelty=row["Novelty"] if "Novelty" in row else row["novelty"],
                            weight=row["Weight"],
                            source=row["Source"] if "Source" in row else "n/a"
                        ))
                        logger.trace("Appended: {}", samples[-1])
                    logger.info("Collected {} samples for \"{}\" --> {}", len(samples), name, part)
                    sample_data[part] = samples

                ret = ValidityNoveltyDataset(samples=sample_data["samples_original.csv"],
                                             tokenizer=tokenizer, max_length=max_length, name=name)
                ret.samples_extraction = sample_data["samples_extraction.csv"]

                logger.success("For the split \"{}\": {}", split, ret)

                ret_datasets[split] = ret
            except Exception:
                logger.opt(exception=True).critical("Split \"{}\" was not loadable!", split)
                ret_datasets[split] = ValidityNoveltyDataset(samples=[],
                                                             tokenizer=AutoTokenizer.from_pretrained("roberta-base"),
                                                             name="ERROR - original data was not loadable")

        logger.success("Loaded all {} datasets: {}", len(ret_datasets),
                       " *** ".join(map(lambda v: "\"{}\"".format(v), ret_datasets.values())))

        return ret_datasets
