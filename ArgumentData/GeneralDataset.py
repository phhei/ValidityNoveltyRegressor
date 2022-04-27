import math
import random
from functools import reduce
from pathlib import Path
from typing import Optional, Any, Iterable, List, Union, Dict, Tuple

import matplotlib.pylab as plt
import matplotlib.pyplot
import numpy
import seaborn
import torch
from loguru import logger
from nltk import sent_tokenize
from pandas import DataFrame
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import PreTrainedTokenizer

from ArgumentData.StringUtils import paraphrase, summarize, add_prefix, \
    remove_non_content_words_manipulate_punctuation, wordnet_changes_text
from HGTrainer import _val_nov_metric


class ValidityNoveltyDataset(Dataset):
    class Sample:
        def __init__(self, premise: str, conclusion: str, validity: Optional[float] = None,
                     novelty: Optional[float] = None, weight: float = 1.):
            #TODO add source (where a sample comes from...)
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

        def automatically_create_non_valid_non_novel_sample(self, other_random_sample: Optional[Any] = None):
            if self.is_valid(none_is_not=True) and self.is_novel(none_is_not=True):
                raise NotImplementedError("Negation script is not implemented until yet")
            elif self.is_valid(none_is_not=True) and not self.is_novel(none_is_not=True):
                raise NotImplementedError("Negation script is not implemented until yet")
            elif not self.is_valid(none_is_not=True) and self.is_novel(none_is_not=True):
                raise NotImplementedError("Negation script is not implemented until yet")
            elif not self.is_valid(none_is_not=True) and not self.is_novel(none_is_not=True):
                selection = random.randint(1, 3+int(other_random_sample is not None))
                if selection == 1:
                    threshold = random.randint(2, 7)/10
                    max_synsets = random.randint(4, 10)
                    changed_parts = random.randint(1, 3)
                    return ValidityNoveltyDataset.Sample(
                        premise=wordnet_changes_text(text=self.premise,
                                                     direction="similar",
                                                     change_threshold=threshold,
                                                     maximum_synsets_to_fix=max_synsets)
                        if changed_parts in [2, 3] else self.premise,
                        conclusion=wordnet_changes_text(text=self.conclusion,
                                                        direction="similar",
                                                        change_threshold=threshold,
                                                        maximum_synsets_to_fix=max_synsets)
                        if changed_parts in [1, 3] else self.conclusion,
                        validity=self.validity,
                        novelty=self.novelty,
                        weight=(1-((changed_parts/10)*max_synsets/10*(1-threshold)))*self.weight
                    )
                elif selection == 2:
                    changed_parts = random.randint(1, 3)
                    return ValidityNoveltyDataset.Sample(
                        premise=add_prefix(text=self.premise, part="premise")
                        if changed_parts in [2, 3] else self.premise,
                        conclusion=add_prefix(text=self.conclusion, part="conclusion")
                        if changed_parts in [1, 3] else self.conclusion,
                        validity=self.validity,
                        novelty=self.novelty,
                        weight=self.weight
                    )
                elif selection == 3:
                    changed_parts = random.randint(1, 3) if len(self.premise) <= 50 else 2
                    return ValidityNoveltyDataset.Sample(
                        premise=paraphrase(text=self.premise, avoid_equal_return=False,
                                           maximize_dissimilarity=False, fast=True)
                        if changed_parts in [1, 3] else self.premise,
                        conclusion=paraphrase(text=self.conclusion, avoid_equal_return=True,
                                              maximize_dissimilarity=False)
                        if changed_parts in [2, 3] else self.conclusion,
                        validity=self.validity,
                        novelty=self.novelty,
                        weight=(.9-((changed_parts-1)/4))*self.weight
                    )
                elif selection == 4:
                    return ValidityNoveltyDataset.Sample(
                        premise="{} {}".format(self.premise,
                                               other_random_sample.conclusion
                                               if isinstance(other_random_sample, ValidityNoveltyDataset.Sample)
                                               else other_random_sample),
                        conclusion=self.conclusion,
                        validity=self.validity,
                        novelty=self.novelty,
                        weight=self.weight
                    )
                else:
                    raise ValueError("Unexpected mode {}".format(selection))
            else:
                raise AttributeError("Unexpected configuration: {}".format(self))

        def automatically_create_non_valid_novel_sample(self):
            if self.is_valid(none_is_not=True) and self.is_novel(none_is_not=True):
                raise NotImplementedError("Negation (premise or conclusion) script is not implemented until yet")
            elif self.is_valid(none_is_not=True) and not self.is_novel(none_is_not=True):
                raise NotImplementedError("???")
            elif not self.is_valid(none_is_not=True) and self.is_novel(none_is_not=True):
                if self.novelty >= .8:
                    threshold = random.randint(2, 7) / 10
                    max_synsets = random.randint(2, 6)
                    changed_parts = random.randint(1, 3)
                    return ValidityNoveltyDataset.Sample(
                        premise=wordnet_changes_text(text=self.premise,
                                                     direction="similar",
                                                     change_threshold=threshold,
                                                     maximum_synsets_to_fix=max_synsets)
                        if changed_parts in [2, 3] else self.premise,
                        conclusion=wordnet_changes_text(text=self.conclusion,
                                                        direction="similar",
                                                        change_threshold=threshold,
                                                        maximum_synsets_to_fix=max_synsets)
                        if changed_parts in [1, 3] else self.conclusion,
                        validity=self.validity,
                        novelty=self.novelty,
                        weight=(1 - ((changed_parts / 10) * max_synsets / 6 * (1 - threshold))) * self.weight
                    )
                else:
                    changed_parts = random.randint(1, 3)
                    return ValidityNoveltyDataset.Sample(
                        premise=add_prefix(text=self.premise, part="premise")
                        if changed_parts in [2, 3] else self.premise,
                        conclusion=add_prefix(text=self.conclusion, part="conclusion")
                        if changed_parts in [1, 3] else self.conclusion,
                        validity=self.validity,
                        novelty=self.novelty,
                        weight=self.weight
                    )
            elif not self.is_valid(none_is_not=True) and not self.is_novel(none_is_not=True):
                premise_sents = sent_tokenize(text=self.premise, language="english")
                if len(premise_sents) == 1:
                    raise AttributeError("\"{}\" contains too few premises", self.premise)

                return ValidityNoveltyDataset.Sample(
                    premise=" ".join(premise_sents[:-1]),
                    conclusion="{} Additionally, {}{}".format(premise_sents[-1], self.conclusion[0].lower(),
                                                              self.conclusion[1:]),
                    validity=self.validity or (self.validity*.5),
                    novelty=1 if self.novelty is None else .6+(self.novelty*.4),
                    weight=self.weight * .5
                )
            else:
                raise AttributeError("Unexpected configuration: {}".format(self))

        def automatically_create_valid_non_novel_sample(self, other_random_sample: Optional[Any] = None):
            if self.is_valid(none_is_not=True) and self.is_novel(none_is_not=True):
                premise = [
                    self.premise,
                    paraphrase(text=self.conclusion, avoid_equal_return=False, maximize_dissimilarity=False, fast=True)
                ]
                premise[-1] += "." if premise[-1][-1] not in [".", "!", "?"] else ""
                random.shuffle(premise)
                return ValidityNoveltyDataset.Sample(
                    premise="{} {}".format(*premise),
                    conclusion=self.conclusion,
                    validity=1,
                    novelty=0 if any(map(lambda p: p in self.conclusion, premise)) else .1,
                    weight=self.weight
                )
            elif self.is_valid(none_is_not=True) and not self.is_novel(none_is_not=True):
                selection = random.randint(1, 3 + int(other_random_sample is not None))
                if selection == 1:
                    threshold = random.randint(4, 8) / 10
                    max_synsets = random.randint(2, 6)
                    changed_parts = random.randint(1, 3)
                    return ValidityNoveltyDataset.Sample(
                        premise=wordnet_changes_text(text=self.premise,
                                                     direction="similar",
                                                     change_threshold=threshold,
                                                     maximum_synsets_to_fix=max_synsets)
                        if changed_parts in [2, 3] else self.premise,
                        conclusion=wordnet_changes_text(text=self.conclusion,
                                                        direction="similar",
                                                        change_threshold=threshold,
                                                        maximum_synsets_to_fix=max_synsets)
                        if changed_parts in [1, 3] else self.conclusion,
                        validity=self.validity,
                        novelty=self.novelty,
                        weight=(1 - ((changed_parts / 6) * max_synsets / 6 * (1 - threshold))) * self.weight
                    )
                elif selection == 2:
                    changed_parts = random.randint(1, 3)
                    return ValidityNoveltyDataset.Sample(
                        premise=add_prefix(text=self.premise, part="premise")
                        if changed_parts in [2, 3] else self.premise,
                        conclusion=add_prefix(text=self.conclusion, part="conclusion")
                        if changed_parts in [1, 3] else self.conclusion,
                        validity=self.validity,
                        novelty=self.novelty,
                        weight=self.weight
                    )
                elif selection == 3:
                    changed_parts = random.randint(1, 3) if len(self.premise) <= 50 else 2
                    return ValidityNoveltyDataset.Sample(
                        premise=paraphrase(text=self.premise, avoid_equal_return=False,
                                           maximize_dissimilarity=False, fast=True)
                        if changed_parts in [1, 3] else self.premise,
                        conclusion=paraphrase(text=self.conclusion, avoid_equal_return=False,
                                              maximize_dissimilarity=False, fast=True)
                        if changed_parts in [2, 3] else self.conclusion,
                        validity=self.validity,
                        novelty=self.novelty,
                        weight=(.9 - ((changed_parts - 1) / 4)) * self.weight
                    )
                elif selection == 4:
                    return ValidityNoveltyDataset.Sample(
                        premise="{} Furthermore, {}".format(
                            self.premise,
                            other_random_sample.premise
                            if isinstance(other_random_sample, ValidityNoveltyDataset.Sample)
                            else other_random_sample),
                        conclusion=self.conclusion,
                        validity=self.validity,
                        novelty=self.novelty,
                        weight=.9 * self.weight
                    )
                else:
                    raise ValueError("Unexpected mode {}".format(selection))
            elif not self.is_valid(none_is_not=True):
                selection = random.randint(1, 4)
                if selection <= 3:
                    premise = [
                        self.premise,
                        paraphrase(text=self.conclusion, avoid_equal_return=False,
                                   maximize_dissimilarity=False, fast=True)
                    ]
                    premise[-1] += "." if premise[-1][-1] not in [".", "!", "?"] else ""
                    random.shuffle(premise)
                    return ValidityNoveltyDataset.Sample(
                        premise="{} {}".format(*premise),
                        conclusion=self.conclusion,
                        validity=1,
                        novelty=0 if any(map(lambda p: p in self.conclusion, premise)) else .1,
                        weight=.9 * self.weight
                    )
                else:
                    return ValidityNoveltyDataset.Sample(
                        premise=self.premise,
                        conclusion=summarize(text=self.premise),
                        validity=1,
                        novelty=.05,
                        weight=.1+.05*self.weight
                    )
            else:
                raise AttributeError("Unexpected configuration: {}".format(self))

        def automatically_create_valid_novel_sample(self):
            if self.is_valid(none_is_not=True) and self.is_novel(none_is_not=True):
                selection = random.randint(1, 3)
                if selection == 1:
                    threshold = random.randint(1, 5)/10
                    max_synsets = random.randint(3, 8)
                    conclusion = wordnet_changes_text(text=self.conclusion,
                                                      direction="more_general",
                                                      change_threshold=threshold,
                                                      maximum_synsets_to_fix=max_synsets)
                    return ValidityNoveltyDataset.Sample(
                        premise=self.premise,
                        conclusion=conclusion if conclusion != self.conclusion
                        else remove_non_content_words_manipulate_punctuation(self.conclusion),
                        validity=max(.5, self.validity*0.95) if conclusion != self.conclusion else self.validity,
                        novelty=min(1, self.novelty*1.05) if conclusion != self.conclusion else self.novelty,
                        weight=(1-.5*(max_synsets/8*(1-threshold)))*self.weight
                    )
                elif selection == 2:
                    changed_parts = random.randint(1, 3)
                    return ValidityNoveltyDataset.Sample(
                        premise=add_prefix(text=self.premise, part="premise")
                        if changed_parts in [2, 3] else self.premise,
                        conclusion=add_prefix(text=self.conclusion, part="conclusion")
                        if changed_parts in [1, 3] else self.conclusion,
                        validity=self.validity,
                        novelty=self.novelty,
                        weight=self.weight
                    )
                elif selection == 3:
                    changed_parts = random.randint(1, 3)
                    return ValidityNoveltyDataset.Sample(
                        premise=remove_non_content_words_manipulate_punctuation(self.premise)
                        if changed_parts in [2, 3] else self.premise,
                        conclusion=remove_non_content_words_manipulate_punctuation(self.conclusion)
                        if changed_parts in [1, 3] else self.conclusion,
                        validity=max(.5, self.validity*.99),
                        novelty=self.novelty,
                        weight=(1-changed_parts/9)*self.weight
                    )
                else:
                    raise ValueError("Unexpected mode {}".format(selection))
            elif self.is_valid(none_is_not=True) and not self.is_novel(none_is_not=True):
                raise NotImplementedError("???")
            elif not self.is_valid(none_is_not=True) and self.is_novel(none_is_not=True):
                raise NotImplementedError("???")
            elif not self.is_valid(none_is_not=True) and not self.is_novel(none_is_not=True):
                raise NotImplementedError("???")
            else:
                raise AttributeError("Unexpected configuration: {}".format(self))

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
        self.sample(number_or_fraction=float(1), forced_balanced_dataset=False, force_classes=False,
                    allow_automatically_created_samples=False)
        logger.success("Reset \"{}\" to its original data successfully", self.name)

    def generate_more_samples(self) -> int:
        new_generated = 0

        for i, sample in enumerate(self.samples_original):
            try:
                self.samples_original.append(sample.automatically_create_valid_novel_sample())
                logger.trace("Created a new valid and novel sample: \"{}\"", self.samples_original[-1])
                new_generated += 1
            except Exception:
                logger.opt(exception=False).info("No new valid and novel sample for \"{}\"", sample)
            try:
                self.samples_original.append(sample.automatically_create_valid_non_novel_sample(
                    other_random_sample=random.choice(self.samples_original)
                    if len(self.samples_original) >= 10 else None
                ))
                logger.trace("Created a new valid and non-novel sample: \"{}\"", self.samples_original[-1])
                new_generated += 1
            except Exception:
                logger.opt(exception=False).info("No new valid and non-novel sample for \"{}\"", sample)
            try:
                self.samples_original.append(sample.automatically_create_non_valid_novel_sample())
                logger.trace("Created a new non-valid and novel sample: \"{}\"", self.samples_original[-1])
                new_generated += 1
            except Exception:
                logger.opt(exception=False).info("No new non-valid and novel sample for \"{}\"", sample)
            try:
                self.samples_original.append(sample.automatically_create_non_valid_non_novel_sample(
                    other_random_sample=random.choice(self.samples_original)
                    if len(self.samples_original) >= 10 else None
                ))
                logger.trace("Created a new non-valid and non-novel sample: \"{}\"", self.samples_original[-1])
                new_generated += 1
            except Exception:
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

    def sample(self, number_or_fraction: Union[int, float] = .5,
               allow_automatically_created_samples: bool = False,
               forced_balanced_dataset: bool = True,
               force_classes: Union[bool, List[Tuple[Union[int, str], Union[int, str]]]] = True,
               seed: int = 42) -> List[Sample]:
        def equal_dataset(minimum_number: int, maximum_number: Optional[int] = None) -> None:
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
            _count_extract = self.get_sample_class_distribution(for_original_data=False)
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
                    "valid" if validity == 1 else ("not valid" if validity == -1 else "n/a")
                ][
                    "novel" if novelty == 1 else ("not novel" if novelty == -1 else "n/a")
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
                    if allow_automatically_created_samples:
                        while still_needed > 0:
                            try:
                                stem = random.choice(self.samples_extraction)
                                #TODO avoid unknown stems...
                                logger.trace("OK, trying to make something out of: {}", stem)
                                if validity == 1 and novelty == 1:
                                    self.samples_extraction.append(stem.automatically_create_valid_novel_sample())
                                elif validity == 1 and novelty == 0:
                                    _stem = stem if stem.is_valid(none_is_not=True) and stem.novelty is not None else \
                                        stem.automatically_create_valid_non_novel_sample(
                                            other_random_sample=random.choice(self.samples_original)
                                        )
                                    self.samples_extraction.append(ValidityNoveltyDataset.Sample(
                                        premise=_stem.premise,
                                        conclusion=_stem.conclusion,
                                        validity=_stem.validity,
                                        novelty=None,
                                        weight=_stem.weight
                                    ))
                                elif validity == 1 and novelty == -1:
                                    self.samples_extraction.append(stem.automatically_create_valid_non_novel_sample(
                                        other_random_sample=random.choice(self.samples_original)
                                    ))
                                elif validity == 0 and novelty == 1:
                                    _stem = stem if stem.is_novel(none_is_not=True) and stem.validity is not None else \
                                        stem.automatically_create_non_valid_novel_sample()
                                    self.samples_extraction.append(ValidityNoveltyDataset.Sample(
                                        premise=_stem.premise,
                                        conclusion=_stem.conclusion,
                                        validity=None,
                                        novelty=_stem.novelty,
                                        weight=_stem.weight
                                    ))
                                elif validity == 0 and novelty == 0:
                                    logger.warning("Generating a sample without known validity and novelty "
                                                   "doesn't make sense for training, anyway, we mask {}", stem)
                                    self.samples_extraction.append(ValidityNoveltyDataset.Sample(
                                        premise=stem.premise,
                                        conclusion=stem.conclusion,
                                        validity=None,
                                        novelty=None,
                                        weight=1e-4
                                    ))
                                elif validity == 0 and novelty == -1:
                                    _stem = stem if not stem.is_novel(none_is_not=True) and stem.validity is not None else \
                                        stem.automatically_create_non_valid_non_novel_sample(
                                            other_random_sample=random.choice(self.samples_original)
                                        )
                                    self.samples_extraction.append(ValidityNoveltyDataset.Sample(
                                        premise=_stem.premise,
                                        conclusion=_stem.conclusion,
                                        validity=None,
                                        novelty=_stem.novelty,
                                        weight=_stem.weight
                                    ))
                                elif validity == -1 and novelty == 1:
                                    self.samples_extraction.append(stem.automatically_create_non_valid_novel_sample())
                                elif validity == -1 and novelty == 0:
                                    _stem = stem if not stem.is_valid(none_is_not=True) and stem.novelty is not None else \
                                        stem.automatically_create_non_valid_non_novel_sample(
                                            other_random_sample=random.choice(self.samples_original)
                                        )
                                    self.samples_extraction.append(ValidityNoveltyDataset.Sample(
                                        premise=_stem.premise,
                                        conclusion=_stem.conclusion,
                                        validity=_stem.validity,
                                        novelty=None,
                                        weight=_stem.weight
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
                        useful_samples = [s for s in self.samples_original
                                          if s not in self.samples_extraction and
                                          ((s.is_valid(none_is_not=True) and validity == 1) or
                                           (s.validity is None and validity == 0) or
                                           (not s.is_valid(none_is_not=True) and s.validity is not None and
                                            validity == -1)) and
                                          ((s.is_novel(none_is_not=True) and novelty == 1) or
                                           (s.novelty is None and novelty == 0) or
                                           (not s.is_novel(none_is_not=True) and s.novelty is not None and
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
                    used_samples = [s for s in self.samples_extraction if
                                    ((s.is_valid(none_is_not=True) and validity == 1) or
                                     (s.validity is None and validity == 0) or
                                     (not s.is_valid(none_is_not=True) and s.validity is not None and
                                      validity == -1)) and
                                    ((s.is_novel(none_is_not=True) and novelty == 1) or
                                     (s.novelty is None and novelty == 0) or
                                     (not s.is_novel(none_is_not=True) and s.novelty is not None and novelty == -1))]
                    remove_samples = random.sample(used_samples, k=_count-maximum_number)
                    logger.trace("Remove following samples: {}", remove_samples)
                    for s in remove_samples:
                        try:
                            self.samples_extraction.remove(s)
                        except ValueError:
                            logger.opt(exception=True).warning("We cannot remove {}", s)
            logger.success("Closed all {} cases of forcing sample balance ({}-{}). "
                           "Be beware: we only manipulate the extraction-part of the dataset, "
                           "not the original data ({})",
                           len(force_classes_set), minimum_number, maximum_number,
                           len(self.samples_extraction), len(self.samples_original))

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
        logger.debug("Random state: {}", random.getstate())

        if not forced_balanced_dataset:
            self.samples_extraction = random.sample(self.samples_original, k=number)
            if isinstance(force_classes, List) or force_classes:
                equal_dataset(minimum_number=1)
                return self.samples_extraction
            else:
                count_extract = self.get_sample_class_distribution(for_original_data=False)
                count_original = self.get_sample_class_distribution(for_original_data=True)
                logger.warning("You don't care about the novel/ valid-distribution. "
                               "So, we have: not-valid-not-novel: {}%->{}%, not-valid-novel: {}%->{}%,"
                               "valid-not-novel: {}%->{}%, valid-novel: {}%->{}%",
                               round(100 * count_original["not valid"]["not novel"] / len(self.samples_original)),
                               round(100 * count_extract["not valid"]["not novel"] / len(self.samples_extraction)),
                               round(100 * count_original["not valid"]["novel"] / len(self.samples_original)),
                               round(100 * count_extract["not valid"]["novel"] / len(self.samples_extraction)),
                               round(100 * count_original["valid"]["not novel"] / len(self.samples_original)),
                               round(100 * count_extract["valid"]["not novel"] / len(self.samples_extraction)),
                               round(100 * count_original["valid"]["novel"] / len(self.samples_original)),
                               round(100 * count_extract["valid"]["novel"] / len(self.samples_extraction)))
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
            lambda cc: [s for s in self.samples_extraction if
                        ((s.is_valid(none_is_not=True) and cc[0] == 1) or
                         (s.validity is None and cc[0] == 0) or
                         (not s.is_valid(none_is_not=True) and s.validity is not None and cc[0] == -1)) and
                        ((s.is_novel(none_is_not=True) and cc[1] == 1) or
                         (s.novelty is None and cc[1] == 0) or
                         (not s.is_novel(none_is_not=True) and s.novelty is not None and cc[1] == -1))],
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
        else:
            samples_to_delete = []

        number_samples_class = (number - (len(samples_in_not_chosen_set)-len(samples_to_delete)))/len(chosen_class_set)

        equal_dataset(minimum_number=math.floor(number_samples_class), maximum_number=math.ceil(number_samples_class))

        return self.samples_extraction

    def save(self, path: Optional[Union[Path, str]] = None) -> Path:
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
            data=[(s.premise, s.conclusion, s.validity, s.novelty, s.weight) for s in self.samples_original],
            columns=["Premise", "Conclusion", "Validity", "novelty", "Weight"]
        ).to_csv(
            path_or_buf=str(path.joinpath("samples_original.csv").absolute()),
            index=False,
            encoding="utf8",
            errors="ignore"
        )

        DataFrame.from_records(
            data=[(s.premise, s.conclusion, s.validity, s.novelty, s.weight) for s in self.samples_extraction],
            columns=["Premise", "Conclusion", "Validity", "novelty", "Weight"]
        ).to_csv(
            path_or_buf=str(path.joinpath("samples_extraction.csv").absolute()),
            index=False,
            encoding="utf8",
            errors="ignore"
        )

        logger.success("Wrote [{}] files in \"{}\"", ", ".join(map(lambda f: f.name, path.iterdir())), path.name)

        return path
