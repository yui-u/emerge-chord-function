import math

import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler

from core.common.constants import *
from core.preprocess.dataset import CustomCollate, CustomDataset
from core.preprocess.instances import batch_to_device


class Evaluator(object):
    @staticmethod
    def evaluate(instances, model, config, logger):
        eval_data = CustomDataset(data=instances)

        sampler = RandomSampler(eval_data)

        batch_size = config.batch_size
        data_loader = DataLoader(dataset=eval_data, sampler=sampler, batch_size=batch_size,
                                 collate_fn=CustomCollate.collate, pin_memory=True, num_workers=1)

        logger.info('***** Evaluating *****')
        model.zero_grad()
        losses = []
        total_loss = 0.0
        total_items = 0
        total_tokens = 0
        total_output = []
        for _, batch in enumerate(data_loader):
            batch = batch_to_device(batch, config.device)
            model.eval()
            with torch.no_grad():
                output = model(batch)
                viterbi_output = model.viterbi(batch)
                logliks = output[LOG_LIKELIHOOD]
                total_loss += -logliks.sum().item()
                total_items += output[BATCH_SIZE]
                sequence_length = batch['sequence_length'] - 1  # remove start symbol
                total_tokens += sequence_length.sum().item()
                total_output.append(viterbi_output)
        average_perplexity = math.exp(total_loss / total_tokens)
        total_loss /= total_items
        losses.append(total_loss)
        logger.info('Eval-Loss:{}, Perplexity: {}'.format(total_loss, average_perplexity))

        return {
            TOTAL_LOSS: total_loss,
            OUTPUT: total_output,
            PERPLEXITY: average_perplexity
        }

    @staticmethod
    def state_stats(config, vocab, eval_results):
        initial_state_count = [0 for _ in range(config.num_state)]
        state_emission_count = []
        state_transition_count = []
        state_residence_count = []
        for _ in range(config.num_state):
            state_emission_count.append([0] * (len(vocab.c2i) - 1))
            state_transition_count.append([0] * config.num_state)
            state_residence_count.append([0] * config.max_residential_time)

        eval_results = eval_results[OUTPUT]
        for e in eval_results:
            assert len(e[STATES]) == len(e[OBSERVATION])
            for bs, bo in zip(e[STATES], e[OBSERVATION]):
                assert len(bs) == len(bo)
                for s, o in zip(bs, bo):
                    if o != vocab.pad_index:  # PAD index is shared as the start symbol
                        state_emission_count[s][o] += 1

            for bs in e[STATES]:
                initial_state_count[bs[0]] += 1
                for t in range(1, len(bs)):
                    state_transition_count[bs[t - 1]][bs[t]] += 1

            if config.model_type == HSMM:
                assert len(e[STATES]) == len(e[RESIDENCES])
                for bs, br in zip(e[STATES], e[RESIDENCES]):
                    assert len(bs) == len(br)
                    for t in range(1, len(bs)):
                        if br[t - 1] == 0:
                            assert bs[t - 1] != bs[t]
                            state_residence_count[bs[t]][br[t]] += 1

        if config.model_type == HSMM:
            for s in range(config.num_state):
                state_transition_count[s][s] = 0   # Residential time HMM

        initial_index = ['state{}'.format(i) for i in range(config.num_state)]
        initial_columns = ['count']
        emission_index = ['s{}'.format(i) for i in range(config.num_state)]
        emission_columns = ['{}'.format(vocab.i2c[i]) for i in range(len(vocab.c2i) - 1)]
        transition_index = ['state{}'.format(i) for i in range(config.num_state)]
        transition_columns = ['s{}'.format(i) for i in range(config.num_state)]
        residence_index = ['state{}'.format(i) for i in range(config.num_state)]
        residence_columns = ['{}'.format((i + 1) * METRIC_BEAT_RATIO) for i in range(config.max_residential_time)]
        df_initial_state_count = pd.DataFrame(initial_state_count, index=initial_index, columns=initial_columns)
        df_state_emission_count = pd.DataFrame(state_emission_count, index=emission_index, columns=emission_columns)
        df_state_transition_count = pd.DataFrame(state_transition_count, index=transition_index, columns=transition_columns)
        if config.model_type == HSMM:
            df_state_residence_count = pd.DataFrame(state_residence_count, index=residence_index, columns=residence_columns)
        else:
            df_state_residence_count = None

        df_result = {
            "initial": df_initial_state_count,
            "transition": df_state_transition_count,
            "emission": df_state_emission_count,
            "residence": df_state_residence_count
        }
        return df_result



