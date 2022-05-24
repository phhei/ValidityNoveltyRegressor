import random
from typing import Literal, Optional, List, Dict, Any, Union

import numpy
from loguru import logger
from pandas import DataFrame
from sentence_transformers import SentenceTransformer

from ArgumentData.Sample import Sample

sampling_technique: Literal["first", "random", "less informative",  "most informative", "mixed"] = "first"
sentence_transformer: Optional[SentenceTransformer] = None


def truncate_dataset(data: Union[DataFrame, List[Sample]], max_number: int = -1) -> Union[DataFrame, List[Sample]]:
    if max_number >= 1:
        logger.debug("max-number is set to {}", max_number)
        if max_number >= len(data):
            logger.warning("You want more samples ({}) than you have ({}) -- skip max_number-param",
                           max_number, len(data))
            return data
        else:
            if sampling_technique == "first":
                r_df = data[:max_number]
                logger.debug("Truncated to {} samples", len(r_df))
            elif sampling_technique == "random" or (sampling_technique == "mixed" and
                                                    (max_number / len(data) > 1 / 5 or max_number * len(data) >= 10 ** 6)):
                logger.trace("Ok, you want to have a cold start (term from active learning): "
                             "randomly sample {} out of {} samples", max_number, len(data))
                if isinstance(data, DataFrame):
                    r_df = data.sample(n=max_number, replace=False, random_state=max_number, ignore_index=False)
                else:
                    r_df = data.copy()
                    random.seed(len(max_number))
                    random.shuffle(r_df)
                    r_df = r_df[max_number]
            elif "informative" in sampling_technique or sampling_technique == "mixed":
                logger.trace("You want to have a warm start (term from active learning) - choose {}", max_number)
                if isinstance(data, DataFrame):
                    s_df = data.select_dtypes(include="object", exclude=["datetime", "timedelta"])
                    s_df["all_text"] = s_df.apply(
                        func=lambda x: " # ".join(x.dropna()),
                        axis="columns"
                    )
                    logger.trace("Reduced the text data to: {}", s_df["all_text"])
                else:
                    s_df = dict()
                    s_df["all_text"] = ["{} # {}".format(s.premise, s.conclusion) for s in data]

                global sentence_transformer

                if sentence_transformer is None:
                    logger.warning("We have to load the sentence transformer first!")
                    sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
                    logger.info("Successfully loaded {}", sentence_transformer)

                s_df["text_embedding"] = sentence_transformer.encode(
                    sentences=[text for text in s_df["all_text"]],
                    batch_size=64,
                    show_progress_bar=len(data) > 128,
                    convert_to_numpy=True,
                    convert_to_tensor=False,
                    normalize_embeddings=True
                ).tolist()
                logger.debug("Successfully generated {} sentence embeddings", len(data))

                chosen_indexes: List = \
                    [sid for sid, _ in data.sample(n=1, random_state=len(data), ignore_index=False).iterrows()] \
                        if isinstance(data, DataFrame) else [random.randint(0, len(data)-1)]
                cos_dist_dict: Dict[Any: Dict[Any: float]] = dict()

                def set_and_return(f_id: List) -> float:
                    f_id.sort()
                    if f_id[0] in cos_dist_dict.values():
                        if f_id[1] in cos_dist_dict[f_id[0]].values():
                            return cos_dist_dict[f_id[0]][f_id[1]]
                        else:
                            if isinstance(s_df, DataFrame):
                                v = numpy.dot(s_df.loc[f_id[0]]["text_embedding"], s_df.loc[f_id[1]]["text_embedding"])
                            else:
                                v = numpy.dot(s_df["text_embedding"][f_id[0]], s_df["text_embedding"][f_id[1]])
                            cos_dist_dict[f_id[0]][f_id[1]] = v
                            return v
                    else:
                        if isinstance(s_df, DataFrame):
                            v = numpy.dot(s_df.loc[f_id[0]]["text_embedding"], s_df.loc[f_id[1]]["text_embedding"])
                        else:
                            v = numpy.dot(s_df["text_embedding"][f_id[0]], s_df["text_embedding"][f_id[1]])
                        cos_dist_dict[f_id[0]] = {f_id[1]: v}
                        return v

                while len(chosen_indexes) < max_number:
                    logger.trace("There are still {} samples left to select", max_number-len(chosen_indexes))
                    if isinstance(s_df, DataFrame):
                        min_distance = [(sid, max([set_and_return([sid, i]) for i in chosen_indexes]))
                                        for sid, _ in s_df.iterrows() if sid not in chosen_indexes]
                    else:
                        min_distance = [(sid, max([set_and_return([sid, i]) for i in chosen_indexes]))
                                        for sid in range(len(data)) if sid not in chosen_indexes]
                    min_distance.sort(key=lambda t: t[1], reverse=sampling_technique.startswith("less"))
                    number = min(max_number-len(chosen_indexes), max(4, int((max_number-len(chosen_indexes))/8)))
                    chosen_indexes.extend([s for s, d in min_distance[:number]])
                    logger.debug("Takes {} samples (e.g. \"{}\") with the max-cosine similarities between {} and {}",
                                 number, min_distance[0][0], round(min_distance[0][1], 3),
                                 round(min_distance[number-1][1], 3))

                logger.debug("Successfully choose {} indexes: {}",
                             len(chosen_indexes),
                             ", ".join(map(lambda c: str(c), chosen_indexes)) if len(chosen_indexes) < 25 else
                             "{}...".format(", ".join(map(lambda c: str(c), chosen_indexes[:15]))))
                if isinstance(data, DataFrame):
                    r_df = data.loc[chosen_indexes]
                else:
                    r_df = [s for i, s in enumerate(data) if i in chosen_indexes]
            else:
                logger.warning("The sampling method \"{}\" is not available, "
                               "only \"first\", \"random\", \"less informative\",  \"most informative\" and \"mixed\""
                               "(combination of \"random\" and \"most informative\")",
                               sampling_technique)
                r_df = data
    else:
        logger.trace("No truncation of samples -- skip negative max_number-param")
        return data

    return r_df
