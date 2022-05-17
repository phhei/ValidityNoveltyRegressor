from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

import torch
import numpy

from transformers import Trainer, PreTrainedModel, RobertaForSequenceClassification, BatchEncoding, RobertaConfig, \
    EvalPrediction
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput
from loguru import logger


def val_nov_loss(is_val: torch.Tensor, should_val: torch.Tensor, is_nov: torch.Tensor, should_nov: torch.Tensor,
                 weights: Optional[torch.Tensor] = None, reduce: bool = True) -> torch.Tensor:
    if weights is None:
        weights = torch.ones_like(should_val)
        logger.debug("No weights-vector - assume, all {} samples should count equally", weights.size())

    loss_validity = torch.pow(is_val - torch.where(torch.isnan(should_val), is_val, should_val), 2)
    loss_novelty = torch.pow(is_nov - torch.where(torch.isnan(should_nov), is_nov, should_nov), 2)

    logger.trace("loss_validity: {} / loss_novelty: {}", loss_validity, loss_novelty)

    loss = (.5 * (loss_validity * loss_novelty) + .5 * loss_validity + .5 * loss_novelty) * weights

    return torch.mean(loss) if reduce else loss


def val_nov_metric(eval_data: EvalPrediction) -> Dict[str, float]:
    if isinstance(eval_data.predictions, Tuple) and isinstance(eval_data.label_ids, Tuple) \
            or min(len(eval_data.predictions), len(eval_data.label_ids)) >= 2:
        logger.trace("Format is as processable ({}: {})", type(eval_data.predictions), len(eval_data.predictions))
        if len(eval_data.predictions) != 2:
            logger.debug("We expect 2 tuples, but get {}: {}", len(eval_data.predictions), eval_data.predictions)

        is_validity = eval_data.predictions[-2]
        should_validity = eval_data.label_ids[-2]
        is_novelty = eval_data.predictions[-1]
        should_novelty = eval_data.label_ids[-1]

        return _val_nov_metric(is_validity=is_validity, should_validity=should_validity,
                               is_novelty=is_novelty, should_novelty=should_novelty)
    else:
        logger.warning("This metric can't return all metrics properly, "
                       "because validity and novelty are not distinguishable")

        return {
            "size": numpy.size(eval_data.label_ids),
            "mse_validity": numpy.mean((eval_data.predictions-eval_data.label_ids) ** 2),
            "mse_novelty": numpy.mean((eval_data.predictions-eval_data.label_ids) ** 2),
            "error_validity": numpy.mean(numpy.abs(eval_data.predictions-eval_data.label_ids)),
            "error_novelty": numpy.mean(numpy.abs(eval_data.predictions-eval_data.label_ids)),
            "approximately_hits_validity": -1,
            "approximately_hits_novelty": -1,
            "exact_hits_validity": -1,
            "exact_hits_novelty": -1,
            "approximately_hits": numpy.count_nonzero(
                numpy.where(numpy.abs(eval_data.predictions-eval_data.label_ids) < .2, 1, 0)
            ) / numpy.size(eval_data.predictions),
            "exact_hits": numpy.count_nonzero(
                numpy.where(numpy.abs(eval_data.predictions-eval_data.label_ids) < .05, 1, 0)
            ) / numpy.size(eval_data.predictions),
            "accuracy_validity": -1,
            "accuracy_novelty": -1,
            "accuracy": -1
        }


def _val_nov_metric(is_validity: numpy.ndarray, should_validity: numpy.ndarray,
                    is_novelty: numpy.ndarray, should_novelty: numpy.ndarray) -> Dict[str, float]:
    return {
        "size": numpy.size(is_validity),
        "mse_validity": numpy.mean((is_validity - should_validity) ** 2),
        "mse_novelty": numpy.mean((is_novelty - should_novelty) ** 2),
        "error_validity": numpy.mean(numpy.abs(is_validity - should_validity)),
        "error_novelty": numpy.mean(numpy.abs(is_novelty - should_novelty)),
        "approximately_hits_validity": numpy.sum(
            numpy.where(numpy.abs(is_validity - should_validity) < .2, 1, 0)) / numpy.size(is_validity),
        "approximately_hits_novelty": numpy.sum(
            numpy.where(numpy.abs(is_novelty - should_novelty) < .2, 1, 0)) / numpy.size(is_novelty),
        "exact_hits_validity": numpy.sum(
            numpy.where(numpy.abs(is_validity - should_validity) < .05, 1, 0)) / numpy.size(is_validity),
        "exact_hits_novelty": numpy.sum(
            numpy.where(numpy.abs(is_novelty - should_novelty) < .05, 1, 0)) / numpy.size(is_novelty),
        "approximately_hits": numpy.sum(
            numpy.where(numpy.abs(is_validity - should_validity) + numpy.abs(is_novelty - should_novelty) < .25, 1, 0)
        ) / numpy.size(is_validity),
        "exact_hits": numpy.sum(
            numpy.where(numpy.abs(is_validity - should_validity) + numpy.abs(is_novelty - should_novelty) < .05, 1, 0)
        ) / numpy.size(is_validity),
        "accuracy_validity": numpy.sum(numpy.where(
            numpy.any(numpy.stack([
                numpy.all(numpy.stack([is_validity >= .5, should_validity >= .5]), axis=0),
                numpy.all(numpy.stack([is_validity < .5, should_validity < .5]), axis=0)
            ]), axis=0),
            1, 0
        )) / numpy.size(is_validity),
        "accuracy_novelty": numpy.sum(numpy.where(
            numpy.any(numpy.stack([
                numpy.all(numpy.stack([is_novelty >= .5, should_novelty >= .5]), axis=0),
                numpy.all(numpy.stack([is_novelty < .5, should_novelty < .5]), axis=0)
            ]), axis=0),
            1, 0
        )) / numpy.size(is_validity),
        "accuracy": numpy.sum(numpy.where(
            numpy.any(numpy.stack([
                numpy.all(numpy.stack([is_validity >= .5, should_validity >= .5, is_novelty >= .5, should_novelty >= .5]),
                          axis=0),
                numpy.all(numpy.stack([is_validity >= .5, should_validity >= .5, is_novelty < .5, should_novelty < .5]),
                          axis=0),
                numpy.all(numpy.stack([is_validity < .5, should_validity < .5, is_novelty >= .5, should_novelty >= .5]),
                          axis=0),
                numpy.all(numpy.stack([is_validity < .5, should_validity < .5, is_novelty < .5, should_novelty < .5]),
                          axis=0)
            ]), axis=0),
            1, 0
        )) / numpy.size(is_validity)
    }


# noinspection PyMethodMayBeStatic
class ValNovTrainer(Trainer):
    def compute_loss(self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor], return_outputs=False):
        try:
            validity = inputs.pop("validity")
            novelty = inputs.pop("novelty")
            weights = inputs.pop("weight")
            logger.trace("The batch contain following validity-scores ({}), novelty-scores ({}) and weights ({})",
                         validity, novelty, weights)

            outputs = model(**inputs)

            if isinstance(outputs, ValNovOutput) and outputs.loss is not None:
                logger.debug("The loss was already computed: {}", outputs.loss)
                return (outputs.loss, outputs) if return_outputs else outputs.loss

            if isinstance(outputs, ValNovOutput):
                is_val = outputs.validity
                is_nov = outputs.novelty
            else:
                logger.warning("The output of you model {} is a {}, bit should be a ValNovOutput",
                               model.name_or_path, type(outputs))
                is_val = outputs[0] if isinstance(outputs, Tuple) and len(outputs) >= 2 else outputs
                is_nov = outputs[1] if isinstance(outputs, Tuple) and len(outputs) >= 2 else outputs

            loss = val_nov_loss(is_val=is_val, is_nov=is_nov,
                                should_val=validity, should_nov=novelty,
                                weights=weights)

            return (loss, outputs) if return_outputs else loss
        except KeyError:
            logger.opt(exception=True).error("Something in your configuration / plugged model is false")

        return (torch.zeros((0,), dtype=torch.float), model(**inputs)) if return_outputs \
            else torch.zeros((0,), dtype=torch.float)


@dataclass
class ValNovOutput(SequenceClassifierOutput):
    validity: torch.FloatTensor = None
    novelty: torch.FloatTensor = None


class ValNovRegressor(torch.nn.Module):
    def __init__(self, transformer: PreTrainedModel,
                 loss: Literal["ignore", "compute", "compute and reduce"] = "ignore"):
        super(ValNovRegressor, self).__init__()

        self.transformer = transformer
        try:
            self.regression_layer_validity = torch.nn.Linear(in_features=transformer.config.hidden_size, out_features=1)
            self.regression_layer_novelty = torch.nn.Linear(in_features=transformer.config.hidden_size, out_features=1)
        except AttributeError:
            logger.opt(exception=True).warning("No hidden-size... please use a XXXForMaskedLM-Model!")
            self.regression_layer_validity = torch.nn.LazyLinear(out_features=1)
            self.regression_layer_novelty = torch.nn.LazyLinear(out_features=1)

        self.sigmoid = torch.nn.Sigmoid()
        if loss == "ignore":
            logger.info("torch-Module without an additional loss computation during the forward-pass - "
                        "has to be done explicitly in the training loop!")
        self.loss = loss

        logger.success("Successfully created {}", self)

    def forward(self, x: BatchEncoding) -> ValNovOutput:
        transformer_cls: BaseModelOutput = self.transformer(input_ids=x["input_ids"],
                                                            attention_mask=x["attention_mask"],
                                                            token_type_ids=x["token_type_ids"],
                                                            return_dict=True)

        cls_logits = transformer_cls.last_hidden_state[0]

        validity_logits = torch.squeeze(self.regression_layer_validity(cls_logits))
        novelty_logits = torch.squeeze(self.regression_layer_novelty(cls_logits))

        return ValNovOutput(
            logits=torch.stack([validity_logits, novelty_logits]),
            loss=val_nov_loss(is_val=self.sigmoid(validity_logits),
                              is_nov=self.sigmoid(novelty_logits),
                              should_val=x["validity"],
                              should_nov=x["novelty"],
                              weights=x.get("weight", None),
                              reduce=self.loss == "compute and reduce"
                              ) if self.loss != "ignore" and "validity" in x and "novelty" in x else None,
            hidden_states=transformer_cls.hidden_states,
            attentions=transformer_cls.attentions,
            validity=self.sigmoid(validity_logits),
            novelty=self.sigmoid(novelty_logits)
        )

    def __str__(self) -> str:
        return "() --> ({} --> validity/ {} --> novelty)".format(self.transformer.name_or_path,
                                                                 self.regression_layer_validity,
                                                                 self.regression_layer_novelty)


class RobertaForValNovRegression(RobertaForSequenceClassification):
    def __init__(self, *model_args, **model_kwargs):
        config = RobertaForValNovRegression.get_config()

        configs = [arg for arg in model_args if isinstance(arg, RobertaConfig)]
        if len(configs) >= 1:
            logger.warning("Found already {} config {}... extend it", len(configs), configs[0])
            model_args = [arg for arg in model_args if not isinstance(arg, RobertaConfig)]
            config = configs[0]
            config.num_labels = 2
            config.id2label = {
                0: "validity",
                1: "novelty"
            }
            config.return_dict = True

        super().__init__(config=config, *model_args, **model_kwargs)

        self.loss = "compute"
        self.sigmoid = torch.nn.Sigmoid()

    @classmethod
    def get_config(cls) -> RobertaConfig:
        config = RobertaConfig()
        config.finetuning_task = "Validity-Novelty-Prediction"
        config.num_labels = 2
        config.id2label = {
            0: "validity",
            1: "novelty"
        }
        config.return_dict = True

        return config

    def forward(self, **kwargs):
        logger.trace("Found {} forward-params", len(kwargs))
        if "labels" in kwargs:
            labels = kwargs.pop("labels")
            logger.warning("Found a disturbing param in forward-function: labels ({})", labels)
        if "return_dict" in kwargs:
            return_dict = kwargs.pop("return_dict")
            logger.warning("Found a disturbing param in forward-function: return_dict ({})", return_dict)

        should_validity = None
        if "validity" in kwargs:
            should_validity = kwargs.pop("validity")
            logger.trace("Found a target validity-vector: {}", should_validity)

        should_novelty = None
        if "novelty" in kwargs:
            should_novelty = kwargs.pop("novelty")
            logger.trace("Found a target novelty-vector: {}", should_novelty)

        weights = None
        if "weight" in kwargs:
            weights = kwargs.pop("weight")
            logger.trace("Found a sample-weights-vector: {}", weights)

        out: SequenceClassifierOutput = super().forward(**kwargs)
        is_validity = self.sigmoid(torch.squeeze(out.logits[:, 0]))
        is_novelty = self.sigmoid(torch.squeeze(out.logits[:, 1]))

        return ValNovOutput(
            attentions=out.attentions,
            hidden_states=out.hidden_states,
            logits=out.logits,
            loss=val_nov_loss(is_val=is_validity,
                              is_nov=is_novelty,
                              should_val=should_validity,
                              should_nov=should_novelty,
                              weights=weights,
                              reduce=self.loss == "compute and reduce"
                              ) if self.loss != "ignore" and should_validity is not None
                                   and should_novelty is not None else None,
            validity=is_validity,
            novelty=is_novelty
        )
