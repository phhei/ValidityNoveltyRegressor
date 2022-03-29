from typing import Literal, Optional, Union
from loguru import logger
import pandas
import itertools

from transformers import PreTrainedTokenizer
from functools import reduce
from ArgumentData.GeneralDataset import ValidityNoveltyDataset

dev_path = "ArgumentData/ExplaGraphs/dev.tsv"
train_path = "ArgumentData/ExplaGraphs/train.tsv"


def load_dataset(split: Literal["train", "dev"], tokenizer: PreTrainedTokenizer,
                 max_length_sample: Optional[int] = None, max_number: int = -1,
                 generate_non_novel_non_valid_samples_by_random: bool = False,
                 continuous_val_nov: Union[bool, float] = True,
                 continuous_sample_weight: bool = False) -> ValidityNoveltyDataset:
    data = pandas.read_csv(filepath_or_buffer=train_path if split == "train" else dev_path,
                           sep="\t", quotechar=None, quoting=3, header=None,
                           names=["Conclusion", "Premise", "Stance", "ExplaGraph"])

    logger.info("Loaded {} samples", len(data))

    general_val_nov_adjustment_multiplicative = \
        (.2 if continuous_val_nov else 0) \
            if isinstance(continuous_val_nov, bool) else min(1, max(0, continuous_val_nov))
    samples = []

    for sid, row_data in data.iterrows():
        logger.debug("Process sample {} now", sid)

        graph = [tuple([node.strip("() ") for node in edge.split(";")])
                 for edge in row_data["ExplaGraph"].str.split(")(")]
        logger.trace("{} -{}-> {}{}",
                     row_data["Premise"],
                     row_data["ExplaGraph"],
                     "NOT: " if row_data["Stance"] == "counter" else "",
                     row_data["Conclusion"])

        edges_connecting_premise_conclusion = \
            [edge for edge in graph if edge[0] in row_data["Premise"].str and edge[2] in row_data["Conclusion"]]
        common_sense_nodes = {source for source, _, _ in graph
                              if source not in row_data["Premise"].str and source not in row_data["Conclusion"].str}
        common_sense_nodes.update(
            {sink for _, _, sink in graph
             if sink not in row_data["Premise"].str and sink not in row_data["Conclusion"].str}
        )
        edges_containing_common_sense = [edge for edge in graph
                                         if edge[0] not in row_data["Premise"].str or
                                         edge[2] not in row_data["Conclusion"].str]
        edges_only_common_sense = [edge for edge in graph
                                   if edge[0] not in row_data["Premise"].str and edge[2] not in row_data["Conclusion"]]
        logger.trace("Graph containing {} edges ({} common-sense nodes: {})",
                     len(graph), len(common_sense_nodes), common_sense_nodes)

        traversals = [combination for combination in
                      reduce(lambda l1, l2: l1+l2,
                             [list(itertools.combinations(graph, i)) for i in range(2, len(graph)+1)])
                      if all([combination[j-1][2] == combination[j][0] for j in range(1, len(combination))])]
        logger.trace("Found following (sub)paths (1 or more hops) in the graph: {}",
                     "::".join(map(lambda path_chain: "|".join(map(lambda edge: "{}-{}->{}".format(*edge),
                                                                   path_chain)),
                                   traversals)))

        graph_is_linear = \
            len(graph) == 1 if len(traversals) == 0 else max(map(lambda l: len(l), traversals)) == len(graph)
        if graph_is_linear:
            logger.debug("Sample {} has a linear explanation graph", sid)

        graph_is_easy_to_undermine = len(graph) >= 4 and graph_is_linear and len(edges_containing_common_sense) >= 1
        if graph_is_easy_to_undermine:
            logger.warning("Sample {} is easy to undermine: {}-{}->{}",
                           sid,
                           row_data["Premise"],
                           row_data["ExplaGraph"],
                           "NOT: " if row_data["Stance"] == "counter" else "",
                           row_data["Conclusion"])

        premise = row_data["Premise"]
        conclusion = row_data["Conclusion"]

        if row_data["Stance"] == "support":
            validity = 1
            validity_adjustment_multiplicative = -general_val_nov_adjustment_multiplicative
        elif row_data["Stance"] == "counter":
            validity = 0
            validity_adjustment_multiplicative = general_val_nov_adjustment_multiplicative
        else:
            validity = .5
            validity_adjustment_multiplicative = 0
            logger.warning("Unexpected stance-label \"{}\" at sample {}", row_data["Stance"], sid)

        if len(common_sense_nodes) == 0 and len(traversals) == 0:
            novelty = 0
            novelty_adjustment_multiplicative = general_val_nov_adjustment_multiplicative

            premise += " {} {} {}".format(*graph[0])
        else:
            novelty = 1
            novelty_adjustment_multiplicative = -general_val_nov_adjustment_multiplicative

        validity_corrective = 0 if len(edges_connecting_premise_conclusion) >= 1 else \
            (.5*len(common_sense_nodes)/len(graph) +
             .5*int(graph_is_easy_to_undermine))
        novelty_corrective = max(0,
                                 .33*(len(graph)-len(common_sense_nodes))/len(graph) +
                                 .33*(len(graph)-len(edges_only_common_sense)-2) +
                                 .33*(len(graph)-max([0] + list(map(lambda l: len(l), traversals)))) -
                                 .1*len(graph)/7)

        weight = 2-.8*int(graph_is_easy_to_undermine)-.2*int(graph_is_linear) if continuous_sample_weight else 2

        logger.trace("Having all components together, creating the sample now!")

        samples.append(ValidityNoveltyDataset.Sample(
            premise=premise,
            conclusion=conclusion,
            validity=validity+validity_adjustment_multiplicative*validity_corrective,
            novelty=novelty+novelty_adjustment_multiplicative*novelty_corrective,
            weight=weight
        ))

        logger.debug("Successfully added a sample for row {}: {}", sid, samples[-1])
        logger.debug("Validity: {} / Novelty: {}", samples[-1].validity, samples[-1].novelty)

        if generate_non_novel_non_valid_samples_by_random:
            logger.trace("We have to create a nonsense-sample for row {}, too", sid)
            samples.append(ValidityNoveltyDataset.Sample(
                premise=premise,
                conclusion=data.sample(n=1)["conclusion"],
                validity=0,
                novelty=0,
                weight=.5 if continuous_sample_weight else 2
            ))
            logger.debug("Nonsense-sample generated: {}", samples[-1])

    logger.info("Generated {} samples out of {} dataset lines", len(samples), len(data))

    r_data = ValidityNoveltyDataset(
        samples=samples,
        tokenizer=tokenizer,
        max_length=96 if max_length_sample is None else max_length_sample,
        name="ExplaGraphs_{}".format(split)
    )

    logger.success("Successfully created the dataset: {}", r_data)

    return r_data
