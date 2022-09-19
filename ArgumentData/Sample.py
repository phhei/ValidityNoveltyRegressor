import random
from typing import Optional, Any

import nltk
from loguru import logger
from nltk import sent_tokenize

from ArgumentData.StringUtils import paraphrase, negate, wordnet_changes_text, add_prefix, summarize, \
    remove_non_content_words_manipulate_punctuation


class Sample:
    """
    A single sample
    (instance consisting premise, conclusion, an (optional) validity value, an (optional) novelty value
    and a sample weight (training impact))
    """
    def __init__(self, premise: str, conclusion: str, validity: Optional[float] = None,
                 novelty: Optional[float] = None, weight: float = 1., source: str = "unknown"):
        self.premise: str = premise
        self.conclusion = conclusion
        self.validity: Optional[float] = min(1, max(0, validity)) if validity is not None else validity
        self.novelty: Optional[float] = min(1, max(0, novelty)) if novelty is not None else novelty
        self.weight: float = weight
        self.source = source

    def __str__(self) -> str:
        return "{}-->{} ({}/{})".format(
            self.premise,
            self.conclusion,
            "Val: {}".format(round(self.validity, 3)) if self.validity is not None else "-",
            "Nov: {}".format(round(self.novelty, 3)) if self.novelty is not None else "-"
        )

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Sample) \
               and o.premise == self.premise and o.conclusion == self.conclusion

    def __hash__(self) -> int:
        return hash(self.premise) + hash(self.conclusion)

    def is_valid(self, none_is_not: bool = False) -> Optional[bool]:
        """
        Returns whether the sample is valid or not

        :param none_is_not: if the validity is unknown, you can treat this as "not valid"
        :return: the validity
        """
        return (False if none_is_not else None) if self.validity is None else self.validity >= .5

    def is_novel(self, none_is_not: bool = False) -> Optional[bool]:
        """
        Returns whether the sample is novel or not

        :param none_is_not: if the novelty is unknown, you can treat this as "not novel"
        :return: the novelty
        """
        return (False if none_is_not else None) if self.novelty is None else self.novelty >= .5

    def automatically_create_non_valid_non_novel_sample(self, other_random_sample: Optional[Any] = None):
        """
        See the clone& mutate procedure (synthetic data augmentation)

        :param other_random_sample: in some cases we just append a random conclusion from another random sample -
        you can define one here
        :return: a new non-valid, non-novel sample
        """
        if self.is_valid(none_is_not=True) and self.is_novel(none_is_not=True):
            include_paraphrase = random.choice([True, False])
            premises = [self.premise,
                        paraphrase(negate(self.conclusion)) if include_paraphrase else negate(self.conclusion)]
            random.shuffle(premises)
            return Sample(
                premise=" ".join(premises),
                conclusion=self.conclusion,
                validity=.1 * self.validity * int(include_paraphrase),
                novelty=0.05 * int(include_paraphrase),
                weight=self.weight * (.9 - .15 * int(include_paraphrase)),
                source="{}#{}".format(self.source, "negated conclusion-> premise")
            )
        elif self.is_valid(none_is_not=True) and not self.is_novel(none_is_not=True):
            return Sample(
                premise=self.premise,
                conclusion=negate(self.conclusion),
                validity=(1 - self.validity) ** 2,
                novelty=self.novelty,
                weight=.9 * self.weight,
                source="{}#{}".format(self.source, "negated conclusion")
            )
        elif not self.is_valid(none_is_not=True) and self.is_novel(none_is_not=True):
            include_paraphrase = random.choice([True, False])
            premises = [self.premise,
                        paraphrase(negate(self.conclusion)) if include_paraphrase else negate(self.conclusion)]
            random.shuffle(premises)
            return Sample(
                premise=" ".join(premises),
                conclusion=self.conclusion,
                validity=0,
                novelty=0.05 * int(include_paraphrase),
                weight=self.weight * (.9 - .15 * int(include_paraphrase)),
                source="{}#{}".format(self.source, "negated conclusion-> premise")
            )
        elif not self.is_valid(none_is_not=True) and not self.is_novel(none_is_not=True):
            selection = random.randint(1, 3 + int(other_random_sample is not None))
            if selection == 1:
                threshold = random.randint(2, 7) / 10
                max_synsets = random.randint(4, 10)
                changed_parts = random.randint(1, 3)
                return Sample(
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
                    weight=(1 - ((changed_parts / 10) * max_synsets / 10 * (1 - threshold))) * self.weight,
                    source="{}#{}".format(self.source, "WordNet-Similar>{}".format(threshold))
                )
            elif selection == 2:
                changed_parts = random.randint(1, 3)
                return Sample(
                    premise=add_prefix(text=self.premise, part="premise")
                    if changed_parts in [2, 3] else self.premise,
                    conclusion=add_prefix(text=self.conclusion, part="conclusion")
                    if changed_parts in [1, 3] else self.conclusion,
                    validity=self.validity,
                    novelty=self.novelty,
                    weight=self.weight,
                    source="{}#{}".format(self.source, "prefix added")
                )
            elif selection == 3:
                changed_parts = random.randint(1, 3) if len(self.premise) <= 50 else 2
                return Sample(
                    premise=paraphrase(text=self.premise, avoid_equal_return=False,
                                       maximize_dissimilarity=False, fast=True)
                    if changed_parts in [1, 3] else self.premise,
                    conclusion=paraphrase(text=self.conclusion, avoid_equal_return=True,
                                          maximize_dissimilarity=False)
                    if changed_parts in [2, 3] else self.conclusion,
                    validity=self.validity,
                    novelty=self.novelty,
                    weight=(.9 - ((changed_parts - 1) / 4)) * self.weight,
                    source="{}#{}".format(self.source, "paraphrased")
                )
            elif selection == 4:
                return Sample(
                    premise="{} {}".format(self.premise,
                                           other_random_sample.conclusion
                                           if isinstance(other_random_sample, Sample)
                                           else other_random_sample),
                    conclusion=self.conclusion,
                    validity=self.validity,
                    novelty=self.novelty,
                    weight=self.weight,
                    source="{}#{}".format(self.source, "random conclusion to premise added")
                )
            else:
                raise ValueError("Unexpected mode {}".format(selection))
        else:
            raise AttributeError("Unexpected configuration: {}".format(self))

    def automatically_create_non_valid_novel_sample(self):
        """
        See the clone& mutate procedure (synthetic data augmentation)

        :return: a new non valid but novel sample
        """
        if self.is_valid(none_is_not=True) and self.is_novel(none_is_not=True):
            return Sample(
                premise=self.premise,
                conclusion=negate(self.conclusion),
                validity=(1 - self.validity) ** 2,
                novelty=max(.5, .9 * self.novelty),
                weight=.25 * self.weight,
                source="{}#{}".format(self.source, "negated conclusion")
            )
        elif self.is_valid(none_is_not=True) and not self.is_novel(none_is_not=True):
            raise NotImplementedError("???")
        elif not self.is_valid(none_is_not=True) and self.is_novel(none_is_not=True):
            if self.novelty >= .8:
                threshold = random.randint(2, 7) / 10
                max_synsets = random.randint(2, 6)
                changed_parts = random.randint(1, 3)
                return Sample(
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
                    weight=(1 - ((changed_parts / 10) * max_synsets / 6 * (1 - threshold))) * self.weight,
                    source="{}#{}".format(self.source, "WordNet-Similar>{}".format(threshold))
                )
            else:
                changed_parts = random.randint(1, 3)
                return Sample(
                    premise=add_prefix(text=self.premise, part="premise")
                    if changed_parts in [2, 3] else self.premise,
                    conclusion=add_prefix(text=self.conclusion, part="conclusion")
                    if changed_parts in [1, 3] else self.conclusion,
                    validity=self.validity,
                    novelty=self.novelty,
                    weight=self.weight,
                    source="{}#{}".format(self.source, "prefix added")
                )
        elif not self.is_valid(none_is_not=True) and not self.is_novel(none_is_not=True):
            try:
                premise_sents = sent_tokenize(text=self.premise, language="english")
            except LookupError:
                logger.opt(exception=True).debug("Need to download the nltk-stuff first!")
                nltk.download("punkt")
                premise_sents = sent_tokenize(text=self.premise, language="english")

            if len(premise_sents) == 1:
                raise AttributeError("\"{}\" contains too few premises", self.premise)

            return Sample(
                premise=" ".join(premise_sents[:-1]),
                conclusion="{} Additionally, {}{}".format(premise_sents[-1], self.conclusion[0].lower(),
                                                          self.conclusion[1:]),
                validity=self.validity or (self.validity * .5),
                novelty=1 if self.novelty is None else .6 + (self.novelty * .4),
                weight=self.weight * .5,
                source="{}#{}".format(self.source, "Premise-Sentence-Shift")
            )
        else:
            raise AttributeError("Unexpected configuration: {}".format(self))

    def automatically_create_valid_non_novel_sample(self, other_random_sample: Optional[Any] = None):
        """
        See the clone& mutate procedure (synthetic data augmentation)

        :param other_random_sample: in some cases we just append/ substitute a random conclusion from another random
        sample - you can define one here
        :return: a new valid but non-novel sample
        """
        if self.is_valid(none_is_not=True) and self.is_novel(none_is_not=True):
            premise = [
                self.premise,
                paraphrase(text=self.conclusion, avoid_equal_return=False, maximize_dissimilarity=False, fast=True)
            ]
            premise[-1] += "." if premise[-1][-1] not in [".", "!", "?"] else ""
            random.shuffle(premise)
            return Sample(
                premise="{} {}".format(*premise),
                conclusion=self.conclusion,
                validity=1,
                novelty=0 if any(map(lambda p: p in self.conclusion, premise)) else .1,
                weight=self.weight,
                source="{}#{}".format(self.source, "paraphrased conclusion to premise added")
            )
        elif self.is_valid(none_is_not=True) and not self.is_novel(none_is_not=True):
            selection = random.randint(1, 3 + int(other_random_sample is not None))
            if selection == 1:
                threshold = random.randint(4, 8) / 10
                max_synsets = random.randint(2, 6)
                changed_parts = random.randint(1, 3)
                return Sample(
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
                    weight=(1 - ((changed_parts / 6) * max_synsets / 6 * (1 - threshold))) * self.weight,
                    source="{}#{}".format(self.source, "WordNet-Similar>{}".format(threshold))
                )
            elif selection == 2:
                changed_parts = random.randint(1, 3)
                return Sample(
                    premise=add_prefix(text=self.premise, part="premise")
                    if changed_parts in [2, 3] else self.premise,
                    conclusion=add_prefix(text=self.conclusion, part="conclusion")
                    if changed_parts in [1, 3] else self.conclusion,
                    validity=self.validity,
                    novelty=self.novelty,
                    weight=self.weight,
                    source="{}#{}".format(self.source, "prefix added")
                )
            elif selection == 3:
                changed_parts = random.randint(1, 3) if len(self.premise) <= 50 else 2
                return Sample(
                    premise=paraphrase(text=self.premise, avoid_equal_return=False,
                                       maximize_dissimilarity=False, fast=True)
                    if changed_parts in [1, 3] else self.premise,
                    conclusion=paraphrase(text=self.conclusion, avoid_equal_return=False,
                                          maximize_dissimilarity=False, fast=True)
                    if changed_parts in [2, 3] else self.conclusion,
                    validity=self.validity,
                    novelty=self.novelty,
                    weight=(.9 - ((changed_parts - 1) / 4)) * self.weight,
                    source="{}#{}".format(self.source, "paraphrased")
                )
            elif selection == 4:
                return Sample(
                    premise="{} Furthermore, {}".format(
                        self.premise,
                        other_random_sample.premise
                        if isinstance(other_random_sample, Sample)
                        else other_random_sample),
                    conclusion=self.conclusion,
                    validity=self.validity,
                    novelty=self.novelty,
                    weight=.9 * self.weight,
                    source="{}#{}".format(self.source, "random premise to premise added")
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
                return Sample(
                    premise="{} {}".format(*premise),
                    conclusion=self.conclusion,
                    validity=1,
                    novelty=0 if any(map(lambda p: p in self.conclusion, premise)) else .1,
                    weight=.9 * self.weight,
                    source="{}#{}".format(self.source, "paraphrased conclusion to premise added")
                )
            else:
                return Sample(
                    premise=self.premise,
                    conclusion=summarize(text=self.premise),
                    validity=1,
                    novelty=.05,
                    weight=.1 + .05 * self.weight,
                    source="{}#{}".format(self.source, "conclusion := summarized premise")
                )
        else:
            raise AttributeError("Unexpected configuration: {}".format(self))

    def automatically_create_valid_novel_sample(self):
        """
        See the clone& mutate procedure (synthetic data augmentation)

        :return: a new valid and novel sample
        """
        if self.is_valid(none_is_not=True) and self.is_novel(none_is_not=True):
            selection = random.randint(1, 3)
            if selection == 1:
                threshold = random.randint(1, 5) / 10
                max_synsets = random.randint(3, 8)
                conclusion = wordnet_changes_text(text=self.conclusion,
                                                  direction="more_general",
                                                  change_threshold=threshold,
                                                  maximum_synsets_to_fix=max_synsets)
                return Sample(
                    premise=self.premise,
                    conclusion=conclusion if conclusion != self.conclusion
                    else remove_non_content_words_manipulate_punctuation(self.conclusion),
                    validity=max(.5, self.validity * 0.95) if conclusion != self.conclusion else self.validity,
                    novelty=min(1, self.novelty * 1.05) if conclusion != self.conclusion else self.novelty,
                    weight=(1 - .5 * (max_synsets / 8 * (1 - threshold))) * self.weight,
                    source="{}#{}".format(self.source, "WordNet-Hypernym>{}".format(threshold))
                )
            elif selection == 2:
                changed_parts = random.randint(1, 3)
                return Sample(
                    premise=add_prefix(text=self.premise, part="premise")
                    if changed_parts in [2, 3] else self.premise,
                    conclusion=add_prefix(text=self.conclusion, part="conclusion")
                    if changed_parts in [1, 3] else self.conclusion,
                    validity=self.validity,
                    novelty=self.novelty,
                    weight=self.weight,
                    source="{}#{}".format(self.source, "prefix added")
                )
            elif selection == 3:
                changed_parts = random.randint(1, 3)
                return Sample(
                    premise=remove_non_content_words_manipulate_punctuation(self.premise)
                    if changed_parts in [2, 3] else self.premise,
                    conclusion=remove_non_content_words_manipulate_punctuation(self.conclusion)
                    if changed_parts in [1, 3] else self.conclusion,
                    validity=max(.5, self.validity * .99),
                    novelty=self.novelty,
                    weight=(1 - changed_parts / 9) * self.weight,
                    source="{}#{}".format(self.source, "fill-words removed")
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
