import argparse
import pickle
import random
import math
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from core.common.constants import *
from core.model.nhmm import NeuralHMM, NeuralResidentialHMM
from core.postprocess.visualize_hmm import HmmVisualizer
from core.preprocess.reader import BachChoraleReader
from core.preprocess.vocab import VocabChord
from core.trainer.trainer import Trainer
from core.eval.evaluator import Evaluator
from util.config import ConfigNHMM
from util.logging import create_logger


def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.manual_seed_all(seed)


def add_arguments():
    parser = argparse.ArgumentParser(prog='run_nhmm')
    parser.add_argument('--device',
                        type=str,
                        metavar='DEVICE',
                        default='cpu',
                        help='`cuda:n` where `n` is an integer, or `cpu`')
    parser.add_argument('--dir_instance_output',
                        type=str,
                        default='out',
                        help='output directory path')
    parser.add_argument('--vocab_cover_ratio',
                        type=float,
                        default=0.95,
                        help='vocab accumulated coverage')
    parser.add_argument('--key_preprocessing',
                        type=str,
                        choices=[KEY_PREPROCESS_NORMALIZE],
                        default=KEY_PREPROCESS_NORMALIZE,
                        help='key pre-processing type')
    parser.add_argument('--scale',
                        type=str,
                        choices=[ANY],
                        default=ANY,
                        help='scale selection')
    parser.add_argument('--max_sequence_length',
                        type=int,
                        default=150,
                        help='maximum sequence length')
    parser.add_argument('--beat_type',
                        type=str,
                        choices=[BEAT_FOUR_FOUR],
                        default=BEAT_FOUR_FOUR,
                        help='allowed beat type')
    parser.add_argument('--model_type',
                        type=str,
                        choices=[HMM, HSMM],
                        default=HMM,
                        metavar='model type')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='seed')
    parser.add_argument('--base_model',
                        action='store_true',
                        help='base model')
    parser.add_argument('--use_lstm',
                        action='store_true',
                        help='use lstm')
    parser.add_argument('--use_histogram',
                        action='store_true',
                        help='use pitch-class histogram as a context for transition probability')
    parser.add_argument('--use_beat',
                        action='store_true',
                        help='use beat as a context for residence probability')
    parser.add_argument('--use_pitch',
                        action='store_true',
                        help='use pitch-class information for observation embeddings')
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help='batch size')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=500,
                        help='maximum epochs')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='learning rate')
    parser.add_argument('--num_state',
                        type=int,
                        default=4,
                        help='number of hidden states')
    parser.add_argument('--max_residential_time',
                        type=int,
                        default=16,
                        help='max residential time')
    parser.add_argument('--embedding_size',
                        type=int,
                        default=16,
                        help='state and observation embedding size')
    parser.add_argument('--transition_hidden_size',
                        type=int,
                        default=16,
                        help='transition network hidden size')
    parser.add_argument('--residence_hidden_size',
                        type=int,
                        default=16,
                        help='residence network hidden size')
    parser.add_argument('--lstm_hidden_size',
                        type=int,
                        default=16,
                        help='LSTM hidden size')
    parser.add_argument('--pitch_encoding_size',
                        type=int,
                        default=16,
                        help='pitch encoding size')
    parser.add_argument('--histogram_encoding_size',
                        type=int,
                        default=8,
                        help='histogram encoding size')
    parser.add_argument('--beat_encoding_size',
                        type=int,
                        default=8,
                        help='beat encoding size')
    parser.add_argument('--dropout_p',
                        type=float,
                        default=0.125,
                        help='dropout proportion')
    parser.add_argument('--patience',
                        type=int,
                        default=20,
                        help='early stop patience')
    parser.add_argument('--do_train', action='store_true',
                        help='run training')
    parser.add_argument('--do_eval', action='store_true',
                        help='run evaluation')
    parser.add_argument('--model_to_eval',
                        type=str,
                        default='',
                        help='model filename')
    parser.add_argument('--eval_scale',
                        type=str,
                        choices=[ANY, MAJOR, MINOR, DORIAN],
                        default=ANY,
                        help='scale for evaluation')
    return parser


def select_eval_scale(instances, eval_scale):
    selected_instances = []
    for instance in instances:
        scale = instance[META_DATA]['scale']
        modal = instance[META_DATA]['modal']
        allowed_scale = False
        if eval_scale == MAJOR:
            if scale == MAJOR and modal == NOMODAL:
                allowed_scale = True
        elif eval_scale == MINOR:
            if scale == MINOR and modal == NOMODAL:
                allowed_scale = True
        elif eval_scale == DORIAN:
            if modal == DORIAN:
                allowed_scale = True
        elif eval_scale == ANY:
            allowed_scale = True
        else:
            raise NotImplementedError

        if allowed_scale:
            selected_instances.append(instance)
    return selected_instances


def run_evaluate(
        config,
        vocab,
        logger,
        instances,
        instance_label,
        model,
        model_filename,
        save_entire_transition=False,
        save_entire_observation=False,
        save_top_observation=False,
        save_top_emission_probability=False,
        save_residence_probability_context_none=False,
        save_residence_probability_context_beat=False,
        top=3
):
    results = Evaluator.evaluate(instances, model, config, logger)
    logger.info('{}-{} loss by the best model: {}'.format(instance_label, config.eval_scale, results[TOTAL_LOSS]))
    logger.info('{}-{} perplexity by the best model: {}'.format(instance_label, config.eval_scale, results[PERPLEXITY]))
    eval_state_stats = Evaluator.state_stats(config, vocab, results)
    emission_state_stats = eval_state_stats['emission'].T
    transition_state_stats = eval_state_stats['transition'].T
    if save_entire_transition:
        # State transition counts
        HmmVisualizer.save_bar_plot(
            model_filename=model_filename,
            df_data=transition_state_stats,
            data_name='{}-{}'.format(instance_label, config.eval_scale),
            prob_name='transition',
            legend=False,
            figsize=(config.num_state * 2, config.num_state),
            layout=(int(math.ceil(config.num_state / 4.0)), 4),
            subplots=True
        )
    if save_entire_observation:
        # Emission counts
        HmmVisualizer.save_bar_plot(
            model_filename=model_filename,
            df_data=emission_state_stats,
            data_name='{}-{}'.format(instance_label, config.eval_scale),
            prob_name='emission',
            legend=False,
            figsize=(math.ceil(len(vocab.c2i) / 1.5), config.num_state * 5),
            layout=(int(math.ceil(config.num_state / 2.0)), 2),
            subplots=True
        )
    if save_top_observation:
        # Top K emission counts
        HmmVisualizer.save_bar_plot_top(
            model_filename=model_filename,
            df_data=emission_state_stats,
            instance_label=instance_label,
            eval_scale=config.eval_scale,
            figsize=(80, int(math.ceil(config.num_state * 1.8))),
            top=top
        )
    if save_top_emission_probability:
        # Top K emission probability
        HmmVisualizer.save_emission_prob_top(
            model=model,
            model_filename=model_filename,
            instance_label=instance_label,
            eval_scale=config.eval_scale,
            figsize=(80, int(math.ceil(config.num_state * 1.8))),
            top=top
        )
    if save_residence_probability_context_none:
        # Residence probability
        state_embeddings = model.state_embeddings
        model.residence_net.batch_size = 1
        index = ['state{}'.format(i) for i in range(model.num_state)]
        columns = [str((b + 1) * METRIC_BEAT_RATIO) for b in range(model.max_residential_time)]
        df_data_residence = pd.DataFrame(
            model.residence_net.residence_matrix(state_embeddings, context=None).squeeze().tolist(),
            index=index,
            columns=columns
        )
        HmmVisualizer.save_bar_plot(
            model_filename=model_filename,
            df_data=df_data_residence.T,
            data_name='{}-{}'.format(instance_label, config.eval_scale),
            prob_name='residence',
            legend=False,
            figsize=(config.max_residential_time * 1.5, int(math.ceil(config.num_state/0.8))),
            layout=(int(math.ceil(config.num_state / 4.0)), 4),
            subplots=True
        )
    if save_residence_probability_context_beat:
        # Residence probability (Hinton diagram)
        HmmVisualizer.save_residence_prob_context_beat_4_4(
            model=model,
            model_filename=model_filename,
            instance_label=instance_label,
            eval_scale=config.eval_scale,
            figsize=(20, int(math.ceil(config.num_state * 1.2))),
        )
    return eval_state_stats


def run_nhmm(args):
    set_seed(args.seed, args.device)

    if args.model_type == HSMM:
        args.metric = True
    else:
        args.metric = False

    config = ConfigNHMM(args)

    if args.do_eval:
        dest_log = Path(args.model_to_eval).parents[0] / Path('{}-{}-eval.log'.format(config.model_suffix, config.data_suffix))
    else:
        dest_log = config.dir_output / Path('{}-{}-h{}-seed{}-{}.log'.format(
            config.model_suffix, config.data_suffix, config.num_state, config.seed, config.now))

    logger = create_logger(dest_log=dest_log)
    config.write_log(logger)

    reader = BachChoraleReader(config)

    if config.scale != ANY:
        assert config.key_preprocessing == KEY_PREPROCESS_NORMALIZE

    if args.do_eval:
        data_dir = Path(args.model_to_eval).parents[1]
    else:
        data_dir = config.dir_instance_output

    train_instance_path = data_dir / Path('{}-{}.pkl'.format(TRAIN, config.data_suffix))
    dev_instance_path = data_dir / Path('{}-{}.pkl'.format(DEV, config.data_suffix))
    test_instance_path = data_dir / Path('{}-{}.pkl'.format(TEST, config.data_suffix))
    vocab_path = data_dir / Path('vocab-{}.pkl'.format(config.data_suffix))

    if train_instance_path.is_file():
        train_instances = pickle.load(train_instance_path.open('rb'))
        dev_instances = pickle.load(dev_instance_path.open('rb'))
        test_instances = pickle.load(test_instance_path.open('rb'))
        vocab = pickle.load(vocab_path.open('rb'))
        assert vocab.vocab_cover_ratio == config.vocab_cover_ratio
    else:
        vocab = VocabChord(config)
        if config.metric:
            train_instances, dev_instances, test_instances = \
                reader.create_hmm_instance_and_vocab_with_metric(logger, vocab=vocab)
        else:
            train_instances, dev_instances, test_instances = \
                reader.create_hmm_instance_and_vocab(logger, vocab=vocab)
        pickle.dump(train_instances, train_instance_path.open('wb'))
        pickle.dump(dev_instances, dev_instance_path.open('wb'))
        pickle.dump(test_instances, test_instance_path.open('wb'))
        pickle.dump(vocab, vocab_path.open('wb'))

    if args.do_train:
        assert config.scale == config.eval_scale
        if config.model_type == HSMM:
            model = NeuralResidentialHMM(config, vocab, args.device)
            if config.use_beat:
                save_residence_probability_context_beat = True
                save_residence_probability_context_none = False
            else:
                save_residence_probability_context_beat = False
                save_residence_probability_context_none = True
        else:
            model = NeuralHMM(config, vocab, args.device)
            save_residence_probability_context_beat = False
            save_residence_probability_context_none = False

        logger.info('Start training')
        model.to(args.device)
        best_model, model_filename = Trainer.train(train_instances, dev_instances, model, config, logger)
        # Eval by the best model
        run_evaluate(
            config=config,
            vocab=vocab,
            logger=logger,
            instances=dev_instances,
            instance_label=DEV,
            model=best_model,
            model_filename=config.dir_output / Path(model_filename).stem,
            save_entire_transition=True,
            save_entire_observation=True,
            save_top_observation=False,
            save_residence_probability_context_none=save_residence_probability_context_none,
            save_residence_probability_context_beat=save_residence_probability_context_beat
        )
        # Test by the best model
        run_evaluate(
            config=config,
            vocab=vocab,
            logger=logger,
            instances=test_instances,
            instance_label=TEST,
            model=best_model,
            model_filename=config.dir_output / Path(model_filename).stem,
            save_entire_transition=True,
            save_entire_observation=True,
            save_top_observation=False,
            save_residence_probability_context_none=save_residence_probability_context_none,
            save_residence_probability_context_beat=save_residence_probability_context_beat
        )

    if args.do_eval:
        dev_instances = select_eval_scale(dev_instances, config.eval_scale)
        test_instances = select_eval_scale(test_instances, config.eval_scale)
        if config.model_type == HSMM:
            model = NeuralResidentialHMM(config, vocab, args.device)
            if config.use_beat:
                save_residence_probability_context_beat = True
                save_residence_probability_context_none = False
            else:
                save_residence_probability_context_beat = False
                save_residence_probability_context_none = True
        else:
            model = NeuralHMM(config, vocab, args.device)
            save_residence_probability_context_beat = False
            save_residence_probability_context_none = False
        model.load_state_dict(torch.load(args.model_to_eval, map_location=torch.device(args.device)))
        model_filename = Path(args.model_to_eval)

        # Eval
        dev_dir = model_filename.parent / Path(DEV)
        if not dev_dir.is_dir():
            logger.info('create {}'.format(dev_dir.name))
            dev_dir.mkdir()
        eval_state_stats = run_evaluate(
            config=config,
            vocab=vocab,
            logger=logger,
            instances=dev_instances,
            instance_label=DEV,
            model=model,
            model_filename=dev_dir / model_filename.stem,
            save_entire_transition=True,
            save_entire_observation=True,
            save_top_observation=True,
            save_top_emission_probability=True,
            save_residence_probability_context_none=save_residence_probability_context_none,
            save_residence_probability_context_beat=save_residence_probability_context_beat
        )
        # Test
        test_dir = model_filename.parent / Path(TEST)
        if not test_dir.is_dir():
            logger.info('create {}'.format(test_dir.name))
            test_dir.mkdir()
        test_state_stats = run_evaluate(
            config=config,
            vocab=vocab,
            logger=logger,
            instances=test_instances,
            instance_label=TEST,
            model=model,
            model_filename=test_dir / model_filename.stem,
            save_entire_transition=True,
            save_entire_observation=True,
            save_top_observation=True,
            save_top_emission_probability=True,
            save_residence_probability_context_none=save_residence_probability_context_none,
            save_residence_probability_context_beat=save_residence_probability_context_beat
        )


if __name__ == '__main__':
    args = add_arguments().parse_args()
    run_nhmm(args)
