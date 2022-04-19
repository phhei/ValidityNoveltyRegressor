import random
from functools import reduce
from pathlib import Path
from typing import Optional, Any, Iterable, List, Union, Dict

import matplotlib.pylab as plt
import matplotlib.pyplot
import numpy
import seaborn
import torch
from loguru import logger
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import PreTrainedTokenizer

from HGTrainer import _val_nov_metric


class ValidityNoveltyDataset(Dataset):
    class Sample:
        def __init__(self, premise: str, conclusion: str, validity: Optional[float] = None,
                     novelty: Optional[float] = None, weight: float = 1.):
            self.premise: str = premise
            self.conclusion = conclusion
            self.validity: Optional[float] = min(1, max(0, validity)) if validity is not None else validity
            self.novelty: Optional[float] = min(1, max(0, novelty)) if novelty is not None else novelty
            self.weight: float = weight

        def __str__(self) -> str:
            return "{}-->{} ({}/{})".format(
                self.premise,
                self.conclusion,
                "Val: {}".format(round(self.validity, 3)) if self.validity is not None else "-",
                "Nov: {}".format(round(self.novelty, 3)) if self.novelty is not None else "-"
            )

        def __eq__(self, o: object) -> bool:
            return isinstance(o, ValidityNoveltyDataset.Sample) \
                   and o.premise == self.premise and o.conclusion == self.conclusion

        def __hash__(self) -> int:
            return hash(self.premise) + hash(self.conclusion)

        def is_valid(self, none_is_not: bool = False) -> Optional[bool]:
            return (False if none_is_not else None) if self.validity is None else self.validity >= .5

        def is_novel(self, none_is_not: bool = False) -> Optional[bool]:
            return (False if none_is_not else None) if self.novelty is None else self.novelty >= .5

        def automatically_create_non_valid_non_novel_sample(self):
            raise NotImplementedError()

        def automatically_create_non_valid_novel_sample(self):
            raise NotImplementedError()

        def automatically_create_valid_non_novel_sample(self):
            raise NotImplementedError()

        def automatically_create_valid_novel_sample(self):
            raise NotImplementedError()

    def __init__(self, samples: Iterable[Sample], tokenizer: PreTrainedTokenizer, max_length: int = 512,
                 name: str = "no name available") -> None:
        super().__init__()

        logger.debug("Init a new dataset instance with {} samples", len(samples))
        logger.debug("Tokenizer: {} ({})", tokenizer.name_or_path, max_length)

        self.name = name

        self.samples_original: List[ValidityNoveltyDataset.Sample] = \
            samples if isinstance(samples, List) else list(samples)
        self.samples_extraction: List[ValidityNoveltyDataset.Sample] = \
            samples.copy() if isinstance(samples, List) else list(samples)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index: Any) -> T_co:
        # x
        sample = self.samples_extraction[index]
        ret = self.tokenizer(
            text=sample.premise, text_pair=sample.conclusion,
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
                "weight": torch.FloatTensor([sample.weight])
            }
        )

        ret.data = {k: torch.squeeze(v) for k, v in ret.items()}

        return ret

    def __add__(self, other: Dataset[T_co]) -> Dataset:
        if not isinstance(other, ValidityNoveltyDataset):
            return super().__add__(other)

        logger.info("You want to extend the dataset \"{}\" with \"{}\" ({}+{} = {} instances)",
                    self.name, other.name, len(self), len(other), len(self) + len(other))

        self.name += " + {}".format(other.name)
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
        return "{} ({} out of {} selected)".format(self.name, len(self.samples_extraction), len(self.samples_original))

    def reset_to_original_data(self):
        self.sample(number_or_fraction=float(1), forced_balanced_dataset=False, allow_empty_combinations=True,
                    allow_automatically_created_samples=False)
        logger.success("Reset \"{}\" to its original data successfully", self.name)

    def generate_more_samples(self) -> int:
        new_generated = 0

        for i, sample in enumerate(self.samples_original):
            try:
                self.samples_original.append(sample.automatically_create_valid_novel_sample())
                logger.trace("Created a new valid and novel sample: \"{}\"", self.samples_original[-1])
                new_generated += 1
            except NotImplementedError:
                logger.opt(exception=False).info("No new valid and novel sample for \"{}\"", sample)
            try:
                self.samples_original.append(sample.automatically_create_valid_non_novel_sample())
                logger.trace("Created a new valid and non-novel sample: \"{}\"", self.samples_original[-1])
                new_generated += 1
            except NotImplementedError:
                logger.opt(exception=False).info("No new valid and non-novel sample for \"{}\"", sample)
            try:
                self.samples_original.append(sample.automatically_create_non_valid_novel_sample())
                logger.trace("Created a new non-valid and novel sample: \"{}\"", self.samples_original[-1])
                new_generated += 1
            except NotImplementedError:
                logger.opt(exception=False).info("No new non-valid and novel sample for \"{}\"", sample)
            try:
                self.samples_original.append(sample.automatically_create_non_valid_non_novel_sample())
                logger.trace("Created a new non-valid and non-novel sample: \"{}\"", self.samples_original[-1])
                new_generated += 1
            except NotImplementedError:
                logger.opt(exception=False).info("No new non-valid and novel sample for \"{}\"", sample)

            logger.debug("Processed / regenerated sample {} ({}%)", sample, round(100*i/len(self.samples_original), 1))

        logger.success("Successfully generated {} new sample. Be aware! We added the new samples only to the original "
                       "data, not to the extracted data. If you want to load them, execute self.reset_to_original_data",
                       new_generated)

        if new_generated >= 1:
            self.name += " + gen. samples"

        return new_generated

    def deduplicate(self, original_data: bool = True, extracted_data: bool = True) -> int:
        logger.trace("Let's deduplicate {}/ {} samples...",
                     len(self.samples_original) if original_data else 0,
                     len(self.samples_extraction) if extracted_data else 0)

        removed_samples = 0
        len_self_samples_original = len(self.samples_original)
        len_self_samples_extraction = len(self.samples_extraction)

        if original_data:
            self.samples_original = list(set(self.samples_original))
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

    def get_sample_class_distribution(self, for_original_data: bool = False) -> Dict[str, Dict[str, int]]:
        data = self.samples_original if for_original_data else self.samples_extraction

        logger.trace("Looking at {} samples", len(data))

        return {
            "valid": {
                "novel": len(list(filter(
                    lambda sample: sample.is_valid(none_is_not=True) and sample.is_novel(none_is_not=True),
                    data
                ))),
                "not novel": len(list(filter(
                    lambda sample: sample.is_valid(none_is_not=True) and
                                   (False if sample.is_novel(none_is_not=False) is None
                                    else not sample.is_novel(none_is_not=True)),
                    data
                ))),
                "n/a": len(list(filter(
                    lambda sample: sample.is_valid(none_is_not=True) and sample.is_novel(none_is_not=False) is None,
                    data
                )))
            },
            "not valid": {
                "novel": len(list(filter(
                    lambda sample: (False if sample.is_valid(none_is_not=False) is None
                                    else not sample.is_valid(none_is_not=True)) and sample.is_novel(none_is_not=True),
                    data
                ))),
                "not novel": len(list(filter(
                    lambda sample: (False if sample.is_valid(none_is_not=False) is None
                                    else not sample.is_valid(none_is_not=True)) and
                                   (False if sample.is_novel(none_is_not=False) is None
                                    else not sample.is_novel(none_is_not=True)),
                    data
                ))),
                "n/a": len(list(filter(
                    lambda sample: (False if sample.is_valid(none_is_not=False) is None
                                    else not sample.is_valid(none_is_not=True)) and
                                   sample.is_novel(none_is_not=False) is None,
                    data
                )))
            },
            "n/a": {
                "novel": len(list(filter(
                    lambda sample: sample.is_valid(none_is_not=False) is None and sample.is_novel(none_is_not=True),
                    data
                ))),
                "not novel": len(list(filter(
                    lambda sample: sample.is_valid(none_is_not=False) is None and
                                   (False if sample.is_novel(none_is_not=False) is None
                                    else not sample.is_novel(none_is_not=True)),
                    data
                ))),
                "n/a": len(list(filter(
                    lambda sample: sample.is_valid(none_is_not=False) is None
                                   and sample.is_novel(none_is_not=False) is None,
                    data
                )))
            }
        }

    def depth_analysis_data(self, for_original_data: bool = False, steps: int = 5,
                            show_heatmaps: bool = True, handling_not_known_data: Optional[float] = None,
                            save_heatmaps: Optional[str] = None) -> None:
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
            data = [ValidityNoveltyDataset.Sample(
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

    def sample(self, number_or_fraction: Union[int, float] = .5, forced_balanced_dataset: bool = True,
               allow_automatically_created_samples: bool = False,
               allow_empty_combinations: bool = True, seed: int = 42) -> List[Sample]:
        number = min(number_or_fraction, len(self.samples_original)) \
            if isinstance(number_or_fraction, int) else min(len(self.samples_original),
                                                            round(number_or_fraction*len(self.samples_original)))

        if number == len(self.samples_original) and not forced_balanced_dataset:
            logger.warning("You want to reset your dataset to the original data ({}->{} samples)",
                           len(self.samples_extraction), len(self.samples_original))
            self.samples_extraction = list(self.samples_original)
            return self.samples_extraction

        random.seed(seed)
        logger.debug("OK, you will select {} (out of {})", number, len(self.samples_original))
        logger.debug("Random stare: {}", random.getstate())

        if not forced_balanced_dataset:
            self.samples_extraction = random.sample(self.samples_original, k=number)
            count_extract = self.get_sample_class_distribution(for_original_data=False)
            count_original = self.get_sample_class_distribution(for_original_data=True)
            logger.warning("You don't care about the novel/ valid-distribution. "
                           "So, we have: not-valid-not-novel: {}%->{}%, not-valid-novel: {}%->{}%,"
                           "valid-not-novel: {}%->{}%, valid-novel: {}%->{}%",
                           round(100*count_original["not valid"]["not novel"]/len(self.samples_original)),
                           round(100*count_extract["not valid"]["not novel"]/len(self.samples_extraction)),
                           round(100 * count_original["not valid"]["novel"] / len(self.samples_original)),
                           round(100 * count_extract["not valid"]["novel"] / len(self.samples_extraction)),
                           round(100 * count_original["valid"]["not novel"] / len(self.samples_original)),
                           round(100 * count_extract["valid"]["not novel"] / len(self.samples_extraction)),
                           round(100 * count_original["valid"]["novel"] / len(self.samples_original)),
                           round(100 * count_extract["valid"]["novel"] / len(self.samples_extraction)))
            return self.samples_extraction

        count_original = self.get_sample_class_distribution(for_original_data=True)
        logger.trace("You want force a balanced sample by having {}", count_original)

        logger.trace("Reset the extracted samples collection (delete {} samples)", len(self.samples_extraction))
        self.samples_extraction = []

        number_samples_class = \
            int(
                number/(
                    sum(map(lambda v: sum(map(lambda n: n >= 1, v.values())), count_original.values()))
                    if allow_empty_combinations else 3*3
                )
            )

        if not allow_automatically_created_samples and \
                any(map(lambda v: any(map(lambda n: n < number_samples_class, v.values())), count_original.values())):
            try:
                number_samples_class = max(
                    reduce(lambda v1, v2:
                           ([n for n in v1.values() if not allow_empty_combinations or n >= 1]
                            if isinstance(v1, Dict) else v1) +
                           ([n for n in v2.values() if not allow_empty_combinations or n >= 1]
                            if isinstance(v2, Dict) else v2),
                           count_original.values())
                )
                logger.warning("Your dataset hasn't enough samples to give balanced {} samples. "
                               "We can only offer {} samples", number, 9*number_samples_class)
            except ValueError:
                logger.opt(exception=True).error("Seems to be that you would like to sample from an empty dataset! "
                                                 "Makes no sense!")
                return self.samples_extraction

        if number_samples_class == 0:
            logger.warning("With this parameter combination "
                           "(forced_balanced_dataset: {} + allow_empty_combinations: {}) "
                           "we cannot satisfy your sample-query. Please consider another parameter setting!",
                           forced_balanced_dataset, allow_empty_combinations)
            return self.samples_extraction

        for valid_cls, novel_cls in reduce(
                lambda k1, k2: ([(k1, kk1) for kk1 in count_original[k1].keys()] if isinstance(k1, str) else k1) +
                               ([(k2, kk2) for kk2 in count_original[k2].keys()] if isinstance(k2, str) else k2),
                count_original.keys()
        ):
            cls_subset = self.samples_original
            if valid_cls == "valid":
                cls_subset = [s for s in cls_subset if s.is_valid(none_is_not=True)]
            elif valid_cls == "not valid":
                cls_subset = [s for s in cls_subset if s.is_valid(none_is_not=False) is False]
            elif valid_cls == "n/a":
                cls_subset = [s for s in cls_subset if s.is_valid(none_is_not=False) is None]
            else:
                logger.error("Unknown configuration flag for validity: {}", valid_cls)
            if novel_cls == "novel":
                cls_subset = [s for s in cls_subset if s.is_valid(none_is_not=True)]
            elif novel_cls == "not novel":
                cls_subset = [s for s in cls_subset if s.is_valid(none_is_not=False) is False]
            elif novel_cls == "n/a":
                cls_subset = [s for s in cls_subset if s.is_valid(none_is_not=False) is None]
            else:
                logger.error("Unknown configuration flag for novelty: {}", novel_cls)

            logger.debug("{} samples left for {}->{}, sample {}", len(cls_subset), valid_cls, novel_cls,
                         number_samples_class)

            if len(cls_subset) == 0 and allow_empty_combinations:
                logger.info("{}->{} is an empty combination. Skip...", valid_cls, novel_cls)
                continue
            elif len(cls_subset) == 0:
                logger.error("Having an empty combination ({}->{}) but no samples although they are required!")
                return self.samples_extraction

            if len(cls_subset) < number_samples_class:
                logger.info("{}->{} has to few samples, we have to generate {} more!",
                            valid_cls, novel_cls, number_samples_class-len(cls_subset))
                if valid_cls == "n/a" or novel_cls == "n/a":
                    logger.trace("But not for n/a")
                    self.samples_extraction.extend(cls_subset)
                    continue
                while len(cls_subset) < number_samples_class:
                    try:
                        if valid_cls == "valid" and novel_cls == "novel":
                            cls_subset.append(
                                random.choice(self.samples_original).automatically_create_valid_novel_sample()
                            )
                        elif valid_cls == "valid" and novel_cls == "not novel":
                            cls_subset.append(
                                random.choice(self.samples_original).automatically_create_valid_non_novel_sample()
                            )
                        elif valid_cls == "not valid" and novel_cls == "novel":
                            cls_subset.append(
                                random.choice(self.samples_original).automatically_create_non_valid_novel_sample()
                            )
                        elif valid_cls == "not valid" and novel_cls == "not novel":
                            cls_subset.append(
                                random.choice(self.samples_original).automatically_create_non_valid_non_novel_sample()
                            )
                        else:
                            raise AttributeError("unknown/ undefined cls: {}-> {}".format(valid_cls, novel_cls))
                    except Exception:
                        logger.opt(exception=True).warning("Bad choice -- try another... ({} left)",
                                                           number_samples_class-len(cls_subset))

                logger.success("Successfully created the missing samples, e.g.: {}", cls_subset[-1])
                self.samples_extraction.extend(cls_subset)
            else:
                logger.debug("You have enough samples of this class {}->{} ({}, {} needed)",
                             valid_cls, novel_cls, len(cls_subset), number_samples_class)
                self.samples_extraction.extend(random.sample(cls_subset, k=number_samples_class))

            logger.success("Collected successfully {} samples for {} -> {}",
                           min(len(cls_subset), number_samples_class), valid_cls, novel_cls)

        logger.success("Finished the sampling - ending up with {} samples ({} requested)",
                       len(self.samples_extraction), number)

        return self.samples_extraction
