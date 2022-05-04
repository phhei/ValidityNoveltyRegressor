import numpy
from typing import Literal, Optional, List, Dict, Any

from pandas import DataFrame
from loguru import logger
from sentence_transformers import SentenceTransformer

sampling_technique: Literal["first", "random", "less informative",  "most informative", "mixed"] = "first"
sentence_transformer: Optional[SentenceTransformer] = None


def truncate_df(df: DataFrame, max_number: int = -1) -> DataFrame:
    if max_number >= 1:
        logger.debug("max-number is set to {}", max_number)
        if max_number >= len(df):
            logger.warning("You want more samples ({}) than you have ({}) -- skip max_number-param",
                           max_number, len(df))
            return df
        else:
            if sampling_technique == "first":
                r_df = df[:max_number]
                logger.debug("Truncated to {} samples", len(r_df))
            elif sampling_technique == "random" or (sampling_technique == "mixed" and
                                                    (max_number/len(df) > 1/5 or max_number*len(df) >= 10**6)):
                logger.trace("Ok, you want to have a cold start (term from active learning): "
                             "randomly sample {} out of {} samples", max_number, len(df))
                r_df = df.sample(n=max_number, replace=False, random_state=max_number, ignore_index=False)
            elif "informative" in sampling_technique or sampling_technique == "mixed":
                logger.trace("You want to have a warm start (term from active learning) - choose {}", max_number)
                s_df = df.select_dtypes(include="object", exclude=["datetime", "timedelta"])
                s_df["all_text"] = s_df.apply(
                    func=lambda x: " # ".join(x.dropna()),
                    axis="columns"
                )
                logger.trace("Reduced the text data to: {}", s_df["all_text"])

                global sentence_transformer

                if sentence_transformer is None:
                    logger.warning("We have to load the sentence transformer first!")
                    sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
                    logger.info("Successfully loaded {}", sentence_transformer)

                s_df["text_embedding"] = sentence_transformer.encode(
                    sentences=[text for text in s_df["all_text"]],
                    batch_size=64,
                    show_progress_bar=len(s_df) > 128,
                    convert_to_numpy=True,
                    convert_to_tensor=False,
                    normalize_embeddings=True
                ).tolist()
                logger.debug("Successfully generated {} sentence embeddings", len(s_df))

                chosen_indexes: List = \
                    [sid for sid, _ in df.sample(n=1, random_state=len(df), ignore_index=False).iterrows()]
                cos_dist_dict: Dict[Any: Dict[Any: float]] = dict()

                def set_and_return(f_id: List) -> float:
                    f_id.sort()
                    if f_id[0] in cos_dist_dict.values():
                        if f_id[1] in cos_dist_dict[f_id[0]].values():
                            return cos_dist_dict[f_id[0]][f_id[1]]
                        else:
                            v = numpy.dot(s_df.loc[f_id[0]]["text_embedding"], s_df.loc[f_id[1]]["text_embedding"])
                            cos_dist_dict[f_id[0]][f_id[1]] = v
                            return v
                    else:
                        v = numpy.dot(s_df.loc[f_id[0]]["text_embedding"], s_df.loc[f_id[1]]["text_embedding"])
                        cos_dist_dict[f_id[0]] = {f_id[1]: v}
                        return v

                while len(chosen_indexes) < max_number:
                    logger.trace("There are still {} samples left to select", max_number-len(chosen_indexes))
                    min_distance = [(sid, max([set_and_return([sid, i]) for i in chosen_indexes]))
                                    for sid, _ in s_df.iterrows() if sid not in chosen_indexes]
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
                r_df = df.loc[chosen_indexes]
            else:
                logger.warning("The sampling method \"{}\" is not available, "
                               "only \"first\", \"random\", \"less informative\",  \"most informative\" and \"mixed\""
                               "(combination of \"random\" and \"most informative\")",
                               sampling_technique)
                r_df = df
    else:
        logger.trace("No truncation of samples -- skip negative max_number-param")
        return df

    return r_df
