"""
This script is for running a demo. A  running demo is located at
https://huggingface.co/spaces/pheinisch/ConclusionValidityNoveltyClassifier-Augmentation
"""

import gradio
import torch
import transformers

from HGTrainer import RobertaForValNovRegression, ValNovOutput


@torch.no_grad()
def classify(model, premise, conclusion):
    model_path = "pheinisch/"
    if model == "model trained only with task-internal data":
        model_path += "ConclusionValidityNoveltyClassifier-Augmentation-in_750"
    elif model == "model trained with 1k task-external+synthetic data":
        model_path += "ConclusionValidityNoveltyClassifier-Augmentation-ext_syn_1k"
    elif model == "model trained with 10k task-internal-external-synthetic data":
        model_path += "ConclusionValidityNoveltyClassifier-Augmentation-int_ext_syn_10k"
    elif model == "model trained with 750 task-internal-synthetic data":
        model_path += "ConclusionValidityNoveltyClassifier-Augmentation-in_syn_750"

    hg_model: RobertaForValNovRegression = RobertaForValNovRegression.from_pretrained(model_path)
    hg_model.loss = "ignore"
    hg_tokenizer: transformers.RobertaTokenizer = \
        transformers.AutoTokenizer.from_pretrained("{}-base".format(hg_model.config.model_type))

    # noinspection PyCallingNonCallable
    output: ValNovOutput = hg_model(**hg_tokenizer(text=[premise],
                                                   text_pair=[conclusion],
                                                   max_length=156,
                                                   padding=True,
                                                   truncation=True,
                                                   is_split_into_words=False,
                                                   return_tensors="pt"))

    validity = output.validity.cpu().item()
    novelty = output.novelty.cpu().item()

    return {
        "valid and novel": validity*novelty,
        "valid but not novel": validity*(1-novelty),
        "not valid but novel": (1-validity)*novelty,
        "rubbish": (1-validity)*(1-novelty)
    }


gradio.Interface(
    fn=classify,
    inputs=[
        gradio.Dropdown(choices=["model trained only with task-internal data",
                                 "model trained with 1k task-external+synthetic data",
                                 "model trained with 10k task-internal-external-synthetic data",
                                 "model trained with 750 task-internal-synthetic data"],
                        value="model trained with 750 task-internal-synthetic data",
                        label="Classifier"),
        gradio.Textbox(value="Whatever begins to exist has a cause of its existence. The universe began to exist.",
                       max_lines=5,
                       label="Premise"),
        gradio.Textbox(value="Therefore, the universe has a cause of its existence (God).",
                       max_lines=2,
                       label="Conclusion")
    ],
    outputs=gradio.Label(value="Try out your premise and conclusion ;)",
                         num_top_classes=2,
                         label="Validity-Novelty-Output"),
    examples=[
        ["model trained with 1k task-external+synthetic data",
         "Whatever begins to exist has a cause of its existence. The universe began to exist.",
         "Therefore, the universe has a cause of its existence (God)."],
        ["model trained with 750 task-internal-synthetic data",
         "Humans have the capability to think about more than food or mecanical stuff, they have the capability to pay "
         "attention to morality, having a conscience, ask for philosophical question, thinking about the sense of "
         "life, where I'm coming from, where I'll go...",
         "We should order a pizza"],
        ["model trained with 750 task-internal-synthetic data",
         "I plan to walk, but the weather forecast prognoses rain.",
         "I should take my umbrella"]
    ],
    title="Predicting validity and novelty",
    description="Demo for the paper: \"Data Augmentation for Improving the Prediction of Validity and Novelty of "
                "Argumentative Conclusions\". Consider the repo: https://github.com/phhei/ValidityNoveltyRegressor"
).launch()