from pathlib import Path

import copy
import torch
from torch_optimizer import RAdam
from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange

from core.common.constants import *
from core.preprocess.dataset import CustomCollate, CustomDataset
from core.eval.evaluator import Evaluator
from core.preprocess.instances import batch_to_device


class Trainer(object):
    @staticmethod
    def save_model(config, model, epoch):
        model_filename = '{}-{}-h{}-seed{}-ep{}-{}.model'.format(
            config.model_suffix.lower(), config.data_suffix.lower(), config.num_state, config.seed, epoch, config.now)
        output_file = config.dir_output / Path(model_filename)
        torch.save(model.state_dict(), output_file)
        return model_filename

    @staticmethod
    def train(train_instances, dev_instances, model, config, logger):
        train_data = CustomDataset(data=train_instances)

        sampler = RandomSampler(train_data)

        batch_size = config.batch_size
        iterator = trange(config.num_epochs, desc='Epoch', disable=False)
        data_loader = DataLoader(dataset=train_data, sampler=sampler, batch_size=batch_size,
                                 collate_fn=CustomCollate.collate, pin_memory=True, num_workers=1)

        optimizer = RAdam(model.parameters(), lr=config.learning_rate)

        logger.info('***** Start Training *****')
        torch.autograd.set_detect_anomaly(True)
        model.train()
        losses = []
        best_eval_loss = 10000
        best_epoch = -1
        best_model = None
        for epoch in iterator:
            logger.info('***** Epoch: {} *****'.format(epoch))
            total_loss = 0.0
            total_items = 0
            for _, batch in enumerate(data_loader):
                batch = batch_to_device(batch, config.device)
                model.to(config.device)
                model.train()
                model.zero_grad()
                output = model(batch)
                logliks = output[LOG_LIKELIHOOD]
                loss = -logliks.sum() / output[BATCH_SIZE]
                loss.backward()
                optimizer.step()
                total_loss += -logliks.sum().item()
                total_items += output[BATCH_SIZE]
            total_loss /= total_items
            losses.append(total_loss)
            logger.info('Train-Loss:{}'.format(total_loss))

            # eval
            eval_result = Evaluator.evaluate(dev_instances, model, config, logger)
            eval_loss = eval_result[TOTAL_LOSS]
            if eval_loss < best_eval_loss:
                logger.info('Update model')
                best_eval_loss = eval_loss
                best_epoch = epoch
                best_model = copy.deepcopy(model)
            else:
                if config.patience < epoch - best_epoch:
                    logger.info('Early stopping, Best Epoch: {}'.format(best_epoch))
                    break
        logger.info('End Training, Best Epoch: {}'.format(best_epoch))
        model_filename = Trainer.save_model(config, best_model, best_epoch)
        return best_model, model_filename

