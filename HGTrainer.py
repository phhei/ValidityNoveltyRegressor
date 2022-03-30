from typing import Dict

import torch

from transformers import Trainer, PreTrainedModel
from loguru import logger


class ValNovTrainer(Trainer):
    def compute_loss(self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor], return_outputs=False):
        try:
            validity = inputs.pop("validity")
            novelty = inputs.pop("novelty")
            weights = inputs.pop("weight")
            logger.trace("The batch contain following validity-scores ({}), novelty-scores ({}) and weights ({})",
                         validity, novelty, weights)

            outputs = model(**inputs)

            loss_validity = torch.pow(outputs["validity"]-validity, 2)
            loss_novelty = torch.pow(outputs["novelty"]-novelty, 2)
            logger.trace("loss_validity: {} / loss_novelty: {}", loss_validity, loss_novelty)

            loss = (.5*(loss_validity*loss_novelty)+.5*loss_validity+.5*loss_novelty)*weights

            return (loss, outputs) if return_outputs else torch.nanmean(loss)
        except KeyError:
            logger.opt(exception=True).error("Something in your configuration / plugged model is false")

        return (torch.zeros((0,), dtype=torch.float), model(**inputs)) if return_outputs \
            else torch.zeros((0,), dtype=torch.float)