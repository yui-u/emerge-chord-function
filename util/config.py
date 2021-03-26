from datetime import datetime
from logging import Logger
from pathlib import Path

from core.common.constants import *


class ConfigNHMM:
    def __init__(self, args):
        self.device = args.device

        self.now = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

        # location
        self.dir_instance_output = args.dir_instance_output

        # dataset
        self.vocab_cover_ratio = args.vocab_cover_ratio
        self.key_preprocessing = args.key_preprocessing
        self.scale = args.scale
        self.eval_scale = args.eval_scale
        self.max_sequence_length = args.max_sequence_length
        self.metric = args.metric
        self.beat_type = args.beat_type

        # model
        self.model_type = args.model_type
        if self.model_type != HSMM:
            assert not args.metric
            assert not args.use_beat
        self.seed = args.seed
        self.base_model = args.base_model
        self.use_lstm = args.use_lstm
        self.use_histogram = args.use_histogram
        self.use_beat = args.use_beat
        self.use_pitch = args.use_pitch

        # train
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.num_state = args.num_state
        self.max_residential_time = args.max_residential_time
        self.embedding_size = args.embedding_size
        self.lstm_hidden_size = args.lstm_hidden_size
        self.transition_hidden_size = args.transition_hidden_size
        self.residence_hidden_size = args.residence_hidden_size
        self.histogram_encoding_size = args.histogram_encoding_size
        self.pitch_encoding_size = args.pitch_encoding_size
        self.beat_encoding_size = args.beat_encoding_size
        self.dropout_p = args.dropout_p
        self.patience = args.patience

        # output dir
        self.dir_output = self.dir_instance_output / Path('{}-{}'.format(self.model_suffix, self.data_suffix))

    @property
    def data_suffix(self):
        suf = self.scale

        if self.metric:
            suf = '{}-metric'.format(suf)

        return suf

    @property
    def model_suffix(self):
        if self.base_model:
            suf = 'baseline' + self.model_type
        else:
            suf = 'neural' + self.model_type

            if self.use_lstm:
                suf += '-lstm'

            if self.use_histogram:
                suf += '-histo'

            if self.use_pitch:
                suf += '-pitch'

            if self.use_beat:
                suf += '-beat'

        return suf

    def write_log(self, logger: Logger):
        s = '\n'.join(['vocab_cover_ratio: {}'.format(self.vocab_cover_ratio),
                       'key_preprocessing: {}'.format(self.key_preprocessing),
                       'scale: {}'.format(self.scale),
                       'eval_scale: {}'.format(self.eval_scale),
                       'max_sequence_length: {}'.format(self.max_sequence_length),
                       'beat_type: {}'.format(self.beat_type),
                       'model_type: {}'.format(self.model_type),
                       'seed: {}'.format(self.seed),
                       'base_model: {}'.format(self.base_model),
                       'use_lstm: {}'.format(self.use_lstm),
                       'use_histogram: {}'.format(self.use_histogram),
                       'use_beat: {}'.format(self.use_beat),
                       'use_pitch: {}'.format(self.use_pitch),
                       'batch_size: {}'.format(self.batch_size),
                       'num_epochs: {}'.format(self.num_epochs),
                       'learning_rate: {}'.format(self.learning_rate),
                       'num_state: {}'.format(self.num_state),
                       'max_residential_time: {}'.format(self.max_residential_time),
                       'embedding_size: {}'.format(self.embedding_size),
                       'histogram_encoding_size: {}'.format(self.histogram_encoding_size),
                       'pitch_encoding_size: {}'.format(self.pitch_encoding_size),
                       'beat_encoding_size: {}'.format(self.beat_encoding_size),
                       'lstm_hidden_size: {}'.format(self.lstm_hidden_size),
                       'transition_hidden_size: {}'.format(self.transition_hidden_size),
                       'residence_hidden_size: {}'.format(self.residence_hidden_size),
                       'dropout_p: {}'.format(self.dropout_p),
                       'patience: {}'.format(self.patience)])

        logger.info(s)
