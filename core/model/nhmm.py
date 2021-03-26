import torch
import torch.nn as nn
import torch.nn.functional as F

from core.common.constants import *


class TransMat(nn.Module):
    def __init__(self, config, device):
        super(TransMat, self).__init__()
        self._batch_size = None
        self.device = device
        self.num_state = config.num_state
        self.unnormalized_transition_matrix = torch.nn.Parameter(torch.randn(self.num_state, self.num_state))

    def forward(self, state_embeddings, alpha, context):
        # state_embeddings and context are not used
        return torch.einsum("bs,st->bt", alpha, self.transition_matrix(state_embeddings, context))

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, b):
        self._batch_size = b

    def transition_matrix(self, state_embeddings, context):
        # state_embeddings and context are not used
        return F.softmax(self.unnormalized_transition_matrix, dim=1)


class ResidentialTransMat(nn.Module):
    def __init__(self, config, device):
        super(ResidentialTransMat, self).__init__()
        self._batch_size = None
        self.device = device
        self.num_state = config.num_state
        self.unnormalized_transition_matrix = torch.nn.Parameter(torch.randn(self.num_state, self.num_state - 1))

    def forward(self, state_embeddings, alpha, context):
        # state_embeddings and context are not used
        return torch.einsum("bs,st->bt", alpha, self.transition_matrix(state_embeddings, context))

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, b):
        self._batch_size = b

    def transition_matrix(self, state_embeddings, context):
        # state_embeddings and context are not used
        mat_raw = F.softmax(self.unnormalized_transition_matrix, dim=1)
        mat_raw = torch.cat([torch.zeros(self.num_state, 1).to(self.device), mat_raw], dim=1)
        gather_index = torch.zeros(self.num_state, self.num_state)
        gather_index[0, :] = torch.arange(0, self.num_state)
        for s in range(1, self.num_state):
            gather_index[s, :s] = torch.arange(1, s + 1)
            gather_index[s, s + 1:] = torch.arange(s + 1, self.num_state)
        mat = torch.gather(mat_raw, dim=1, index=gather_index.long().to(self.device))
        return mat


class EmissionMat(nn.Module):
    def __init__(self, config, vocab, device):
        super(EmissionMat, self).__init__()
        self.device = device
        self.num_state = config.num_state
        self.num_out = len(vocab.c2i) - 1  # exclude PAD
        self.unnormalized_emission_matrix = torch.nn.Parameter(torch.randn(self.num_state, self.num_out))
        self._dummy_for_pad = torch.zeros((self.num_state, 1)).to(self.device)

    def forward(self, state_embeddings, observation_embeddings, x_t):
        # state embeddings and observation embeddings are not used
        mat = torch.cat([self.emission_matrix(state_embeddings, observation_embeddings), self._dummy_for_pad], dim=-1)
        return mat[:, x_t].transpose(0, 1)

    def emission_matrix(self, state_embeddings, observation_embeddings):
        # state embeddings and observation_embeddings are not used
        return F.softmax(self.unnormalized_emission_matrix, dim=1)


class ResidenceMat(nn.Module):
    def __init__(self, config, device):
        super(ResidenceMat, self).__init__()
        self.device = device
        self.num_state = config.num_state
        self.max_residential_time = config.max_residential_time
        self.unnormalized_residence_matrix = torch.nn.Parameter(torch.randn(self.num_state, self.max_residential_time))

    def forward(self, state_embeddings=None, context=None):
        return self.residence_matrix(state_embeddings, context)

    def residence_matrix(self, state_embeddings=None, context=None):
        return F.softmax(self.unnormalized_residence_matrix, dim=1)


class TransNet(nn.Module):
    def __init__(self, config, vocab, device):
        super(TransNet, self).__init__()
        self._batch_size = None
        self.device = device
        self.dropout_p = config.dropout_p
        self.vocab = vocab
        self.hidden_size = config.transition_hidden_size
        self.cat_hidden_size = config.embedding_size
        self.num_state = config.num_state
        self.use_lstm = config.use_lstm
        self.use_histogram = config.use_histogram

        if self.use_lstm:
            self.cat_hidden_size += config.lstm_hidden_size
        if self.use_histogram:
            self.cat_hidden_size += config.histogram_encoding_size

        if self.use_lstm:
            self._lstm_hidden = None
            self.lstm_hidden_size = config.lstm_hidden_size
            self.embedding_size = config.embedding_size
            self.lstm = nn.LSTMCell(
                input_size=self.embedding_size,
                hidden_size=self.lstm_hidden_size,
                bias=True
            )

        self.cat_layer = nn.Sequential(
            nn.Linear(self.cat_hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(p=self.dropout_p)
        )

        self.transition_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_size, self.num_state)
        )

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, b):
        self._batch_size = b

    def initialize_lstm_hidden(self, batch_size):
        h = torch.zeros((batch_size, self.lstm_hidden_size)).to(self.device)
        c = torch.zeros((batch_size, self.lstm_hidden_size)).to(self.device)
        self._lstm_hidden = (h, c)

    def update_lstm(self, xt_embedding, xt):
        xt = xt.unsqueeze(-1)
        (ho, co) = self._lstm_hidden
        (hn, cn) = self.lstm(xt_embedding, self._lstm_hidden)
        h = torch.where(xt == self.vocab.pad_index, ho, hn)
        c = torch.where(xt == self.vocab.pad_index, co, cn)
        self._lstm_hidden = (h, c)

    def forward(self, state_embeddings, alpha, context):
        return torch.einsum("bs,bst->bt", alpha, self.transition_matrix(state_embeddings, context))

    def transition_matrix(self, state_embeddings, context):
        e = state_embeddings.unsqueeze(0).repeat(self._batch_size, 1, 1)
        if self.use_lstm or self.use_histogram:
            e = [e]
            if self.use_histogram:
                context = context.unsqueeze(1).repeat(1, self.num_state, 1)
                e.append(context)
            if self.use_lstm:
                context_lstm = self._lstm_hidden[0].unsqueeze(1).repeat(1, self.num_state, 1)
                e.append(context_lstm)
            e = torch.cat(e, dim=-1)
        e = self.cat_layer(e)
        unnormalized_transition_matrix = self.transition_layer(e)
        return F.softmax(unnormalized_transition_matrix, dim=-1)


class ResidentialTransNet(nn.Module):
    def __init__(self, config, vocab, device):
        super(ResidentialTransNet, self).__init__()
        self._batch_size = None
        self.device = device
        self.dropout_p = config.dropout_p
        self.vocab = vocab
        self.hidden_size = config.transition_hidden_size
        self.cat_hidden_size = config.embedding_size
        self.lstm_hidden_size = config.lstm_hidden_size
        self.histogram_encoding_size = config.histogram_encoding_size
        self.num_state = config.num_state
        self.use_lstm = config.use_lstm
        self.use_histogram = config.use_histogram

        if self.use_lstm:
            self.cat_hidden_size += self.lstm_hidden_size
        if self.use_histogram:
            self.cat_hidden_size += self.histogram_encoding_size

        if self.use_lstm:
            self._lstm_hidden = None
            self.lstm_hidden_size = config.lstm_hidden_size
            self.embedding_size = config.embedding_size
            self.lstm = nn.LSTMCell(
                input_size=self.embedding_size,
                hidden_size=self.lstm_hidden_size,
                bias=True
            )

        self.cat_layer = nn.Sequential(
            nn.Linear(self.cat_hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(p=self.dropout_p)
        )

        self.transition_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_size, self.num_state - 1)
        )

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, b):
        self._batch_size = b

    def initialize_lstm_hidden(self, batch_size):
        h = torch.zeros((batch_size, self.lstm_hidden_size)).to(self.device)
        c = torch.zeros((batch_size, self.lstm_hidden_size)).to(self.device)
        self._lstm_hidden = (h, c)

    def update_lstm(self, xt_embedding, xt):
        xt = xt.unsqueeze(-1)
        (ho, co) = self._lstm_hidden
        (hn, cn) = self.lstm(xt_embedding, self._lstm_hidden)
        h = torch.where(xt == self.vocab.pad_index, ho, hn)
        c = torch.where(xt == self.vocab.pad_index, co, cn)
        self._lstm_hidden = (h, c)

    def forward(self, state_embeddings, alpha, context):
        return torch.einsum("bs,bst->bt", alpha, self.transition_matrix(state_embeddings, context))

    def transition_matrix(self, state_embeddings, context):
        e = state_embeddings.repeat(self._batch_size, 1, 1)
        if self.use_lstm or self.use_histogram:
            e = [e]
            if self.use_histogram:
                context = context.unsqueeze(1).repeat(1, self.num_state, 1)
                e.append(context)
            if self.use_lstm:
                context_lstm = self._lstm_hidden[0].unsqueeze(1).repeat(1, self.num_state, 1)
                e.append(context_lstm)
            e = torch.cat(e, dim=-1)
        e = self.cat_layer(e)
        mat_raw = F.softmax(self.transition_layer(e), dim=-1)
        mat_raw = torch.cat([torch.zeros((self._batch_size, self.num_state, 1)).to(self.device), mat_raw], dim=-1)
        gather_index = torch.zeros(self.num_state, self.num_state)
        gather_index[0, :] = torch.arange(0, self.num_state)
        for s in range(1, self.num_state):
            gather_index[s, :s] = torch.arange(1, s + 1)
            gather_index[s, s + 1:] = torch.arange(s + 1, self.num_state)
        gather_index = gather_index.unsqueeze(0).repeat(self._batch_size, 1, 1)
        mat = torch.gather(mat_raw, dim=-1, index=gather_index.long().to(self.device))
        return mat


class EmissionNet(nn.Module):
    def __init__(self, config, vocab, device):
        super(EmissionNet, self).__init__()
        self.device = device
        self.vocab = vocab
        self.num_state = config.num_state
        self.dropout_p = config.dropout_p
        self.bias = nn.Parameter(torch.randn(len(self.vocab.c2i) - 1))
        self._dummy_for_pad = torch.zeros((self.num_state, 1)).to(self.device)

    def forward(self, state_embeddings, observation_embeddings, x_t):
        mat = torch.cat([self.emission_matrix(state_embeddings, observation_embeddings), self._dummy_for_pad], dim=-1)
        return mat[:, x_t].transpose(0, 1)

    def emission_matrix(self, state_embeddings, observation_embeddings):
        score = torch.einsum("sd,vd->sv", state_embeddings, observation_embeddings) + self.bias.unsqueeze(0).repeat(self.num_state, 1)
        return F.softmax(score, dim=-1)


class ResidenceNet(nn.Module):
    def __init__(self, config, device):
        super(ResidenceNet, self).__init__()
        self._batch_size = None
        self.device = device
        self.dropout_p = config.dropout_p
        self.hidden_size = config.residence_hidden_size
        self.num_state = config.num_state
        self.max_residential_time = config.max_residential_time
        self.use_beat = config.use_beat

        if self.use_beat:
            self.cat_hidden_size = config.embedding_size + config.beat_encoding_size
        else:
            self.cat_hidden_size = config.embedding_size

        self.cat_layer = nn.Sequential(
            nn.Linear(self.cat_hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(p=self.dropout_p)
        )

        self.residence_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_size, self.max_residential_time)
        )

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, b):
        self._batch_size = b

    def forward(self, state_embeddings, context):
        return self.residence_matrix(state_embeddings, context)

    def residence_matrix(self, state_embeddings, context):
        e = state_embeddings.unsqueeze(0).repeat(self._batch_size, 1, 1)
        if context is not None:
            context = context.unsqueeze(1).repeat(1, self.num_state, 1)
            e = torch.cat([e, context], dim=-1)
        e = self.cat_layer(e)
        unnormalized_residence_matrix = self.residence_layer(e)
        return F.softmax(unnormalized_residence_matrix, dim=-1)


class ObservationEncoder(nn.Module):
    def __init__(self, config, vocab, device):
        super(ObservationEncoder, self).__init__()
        self.device = device
        self.vocab = vocab
        self.dropout_p = config.dropout_p
        self.pitch_encoding_size = config.pitch_encoding_size
        self.embedding_size = config.embedding_size
        self.use_pitch = config.use_pitch
        if self.use_pitch:
            vb = [v for k, v in self.vocab.i2b.items()]
            vb = vb[:-1]  # exclude PAD
            self.vocab_binary = torch.tensor(vb).to(self.device).float()
            self.observation_pitch_encoder = nn.Sequential(
                nn.Linear(12, self.pitch_encoding_size),
                nn.Tanh(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(self.pitch_encoding_size, self.pitch_encoding_size),
                nn.Tanh()
            )
        else:
            self.observation_symbol_embeddings = nn.Parameter(torch.randn(len(self.vocab.c2i) - 1, self.embedding_size))

    @property
    def value(self):
        if self.use_pitch:
            observation_encodings = self.observation_pitch_encoder(self.vocab_binary)
        else:
            observation_encodings = self.observation_symbol_embeddings
        return observation_encodings


class PitchClassHistogramEncoder(nn.Module):
    def __init__(self, config, device):
        super(PitchClassHistogramEncoder, self).__init__()
        self.device = device
        self.dropout_p = config.dropout_p
        self.encoding_size = config.histogram_encoding_size
        self.layer = nn.Sequential(
            nn.Linear(12, self.encoding_size),
            nn.Tanh(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.encoding_size, self.encoding_size),
        )

    def forward(self, pitch_histogram):
        return self.layer(pitch_histogram)


class BeatEncoder(nn.Module):
    def __init__(self, config, device):
        super(BeatEncoder, self).__init__()
        self.device = device
        self.dropout_p = config.dropout_p
        self.encoding_size = config.beat_encoding_size
        self.layer = nn.Sequential(
            nn.Linear(2, self.encoding_size),
            nn.Tanh(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.encoding_size, self.encoding_size),
        )

    def forward(self, timesig_numerator, beat):
        timesig_numerator = timesig_numerator.unsqueeze(-1)
        beat = beat.unsqueeze(-1)
        beat_context = torch.cat([timesig_numerator, beat], dim=-1)
        return self.layer(beat_context)


class NeuralHMM(nn.Module):
    def __init__(self, config, vocab, device):
        super(NeuralHMM, self).__init__()
        self.device = device
        self.dropout_p = config.dropout_p
        self.vocab = vocab
        self.num_state = config.num_state
        self.embedding_size = config.embedding_size
        self.use_lstm = config.use_lstm

        self.use_histogram = config.use_histogram
        if self.use_histogram:
            self.pitch_histo_encoder = PitchClassHistogramEncoder(config, device)

        self.base_model = config.base_model
        if self.base_model:
            assert not config.use_lstm
            assert not config.use_histogram

        self.unnormalized_initial_state_matrix = nn.Parameter(torch.randn(self.num_state))
        if self.base_model:
            self.transition_net = TransMat(config, device)
            self.emission_net = EmissionMat(config, vocab, device)
        else:
            self.transition_net = TransNet(config, vocab, device)
            self.emission_net = EmissionNet(config, vocab, device)

        self._state_embeddings = nn.Parameter(torch.randn(self.num_state, self.embedding_size))
        self._observation_encodings = ObservationEncoder(config, vocab, device)
        self._dummy_embedding = torch.zeros((1, self.embedding_size)).to(device)

        self.dropout = nn.Dropout(p=self.dropout_p)

    @property
    def state_embeddings(self):
        return self._state_embeddings

    @property
    def observation_encodings(self):
        return self._observation_encodings.value

    def forward(self, batch):
        lengths = batch["sequence_length"].long()
        batch_size = lengths.size(0)
        pitch_histogram = batch["pitch_histogram"]
        lengths = lengths - 1  # remove start pad symbol
        max_length = lengths.max().item()
        x = batch["observation_index"][:, 1:max_length + 1].long()  # remove start pad symbol
        self.transition_net.batch_size = batch_size

        state_embeddings = self.dropout(self.state_embeddings)
        observation_encodings = self.observation_encodings
        observation_encodings_with_pad = torch.cat([observation_encodings, self._dummy_embedding], dim=0)

        if self.use_lstm:
            xt_encoding = observation_encodings_with_pad[x][:, 0, :]
            xt = x[:, 0]
            self.transition_net.initialize_lstm_hidden(batch_size)
            self.transition_net.update_lstm(xt_encoding, xt)

        if self.use_histogram:
            pitch_histo_encoding = self.pitch_histo_encoder(pitch_histogram)
        else:
            pitch_histo_encoding = None

        initial_state = F.softmax(self.unnormalized_initial_state_matrix, dim=0)
        emission = self.emission_net(state_embeddings, observation_encodings, x[:, 0])
        alpha = initial_state * emission
        c = alpha.sum(dim=-1).unsqueeze(-1)
        alpha /= c
        accum_logc = torch.log(c.squeeze())
        mem_accum_logc = torch.zeros((max_length, batch_size)).to(self.device)
        mem_accum_logc[0] = accum_logc
        for t in range(1, max_length):
            # alpha
            alpha = self.transition_net(state_embeddings, alpha, context=pitch_histo_encoding)
            emission = self.emission_net(state_embeddings, observation_encodings, x[:, t])
            alpha = alpha * emission
            c = alpha.sum(dim=1).unsqueeze(-1)
            c = torch.where(0.0 < c, c, torch.ones_like(c))  # avoid zero division for PAD
            alpha = alpha / c
            accum_logc += torch.log(c.squeeze())
            mem_accum_logc[t] = accum_logc
            if self.use_lstm:
                xt_encoding = observation_encodings_with_pad[x][:, t, :]
                xt = x[:, t]
                self.transition_net.update_lstm(xt_encoding, xt)
        mem_accum_logc = mem_accum_logc.transpose(0, 1)
        gather_indices = (lengths - 1).unsqueeze(-1)
        logliks = mem_accum_logc.gather(1, gather_indices)
        perplexity = torch.exp(-logliks.squeeze() / lengths)
        return {
            LOG_LIKELIHOOD: logliks,
            PERPLEXITY: perplexity,
            BATCH_SIZE: batch_size
        }

    def viterbi_step(self, state_embeddings, omega, context):
        batch_size = omega.size(0)
        tmat = self.transition_net.transition_matrix(state_embeddings, context)
        if tmat.dim() < 3:
            tmat = tmat.unsqueeze(0).repeat(batch_size, 1, 1)
        omega = omega.unsqueeze(-1).repeat(1, 1, self.num_state)
        temp = torch.log(tmat) + omega
        omega_temp, omega_arg = temp.max(dim=1)
        return omega_temp, omega_arg, tmat

    def viterbi(self, batch):
        lengths = batch["sequence_length"].long()
        batch_size = lengths.size(0)
        pitch_histogram = batch["pitch_histogram"]
        lengths = lengths - 1  # remove start pad symbol
        max_length = lengths.max().item()
        x = batch["observation_index"][:, 1:max_length + 1].long()  # remove start pad symbol
        self.transition_net.batch_size = batch_size

        state_embeddings = self.state_embeddings
        observation_encodings = self.observation_encodings
        observation_encodings_with_pad = torch.cat([observation_encodings, self._dummy_embedding], dim=0)

        if self.use_lstm:
            xt_encoding = observation_encodings_with_pad[x][:, 0, :]
            xt = x[:, 0]
            self.transition_net.initialize_lstm_hidden(batch_size)
            self.transition_net.update_lstm(xt_encoding, xt)

        if self.use_histogram:
            pitch_histo_encoding = self.pitch_histo_encoder(pitch_histogram)
        else:
            pitch_histo_encoding = None

        initial_state = F.softmax(self.unnormalized_initial_state_matrix, dim=0)
        emission = self.emission_net(state_embeddings, observation_encodings, x[:, 0])
        omega = torch.log(initial_state) + torch.log(emission)
        omegas = [omega]
        omega_args = [torch.zeros((batch_size, self.num_state)).long().to(self.device)]
        tmats = [initial_state.unsqueeze(0).repeat(batch_size, 1).unsqueeze(1).repeat(1, self.num_state, 1).unsqueeze(0)]

        for t in range(1, max_length):
            # omega
            omega, omega_arg, tmat = self.viterbi_step(state_embeddings, omegas[-1], context=pitch_histo_encoding)
            emission = self.emission_net(state_embeddings, observation_encodings, x[:, t])
            omega += torch.log(emission)
            omegas.append(omega)
            omega_args.append(omega_arg)
            tmats.append(tmat.unsqueeze(0))
            if self.use_lstm:
                xt_encoding = observation_encodings_with_pad[x][:, t, :]
                xt = x[:, t]
                self.transition_net.update_lstm(xt_encoding, xt)

        # finish
        omega_args = (torch.stack(omega_args).transpose(0, 1)).transpose(1, 2)
        omegas = (torch.stack(omegas).transpose(0, 1)).transpose(1, 2)
        gather_indices = (lengths - 1).unsqueeze(-1).repeat(1, self.num_state).unsqueeze(-1)
        joint_probs = omegas.gather(2, gather_indices).squeeze()
        if joint_probs.dim() < 2:
            joint_probs = joint_probs.unsqueeze(0)
        best_probs, best_args = joint_probs.max(dim=-1)

        # reconstruction
        batch_states = [[best_arg] for best_arg in best_args.tolist()]
        for t in range(0, max_length - 1)[::-1]:
            for ib in range(batch_size):
                if t < (lengths[ib] - 1):
                    prev_state = batch_states[ib][-1]
                    state = omega_args[ib, prev_state, t + 1].item()
                    batch_states[ib].append(state)

        for ib in range(batch_size):
            batch_states[ib] = batch_states[ib][::-1]

        tmats = torch.cat(tmats, dim=0)
        observation_list = []
        transition_matrix_list = []
        for ib in range(batch_size):
            length = lengths[ib].item()
            observation_list.append(x[ib].tolist()[:length])
            transition_matrix_list.append(tmats[:length, ib, :, :].detach().cpu().numpy())

        return {
            OBSERVATION: observation_list,
            JOINT_PROB: best_probs.tolist(),
            STATES: batch_states,
            BATCH_SIZE: batch_size,
            TRANS_MAT: transition_matrix_list,
            META_DATA: batch[META_DATA]
        }


class NeuralResidentialHMM(nn.Module):
    def __init__(self, config, vocab, device):
        super(NeuralResidentialHMM, self).__init__()
        self.device = device
        self.dropout_p = config.dropout_p
        self.vocab = vocab
        self.num_state = config.num_state
        self.embedding_size = config.embedding_size
        self.max_residential_time = config.max_residential_time
        self.use_lstm = config.use_lstm

        self.use_histogram = config.use_histogram
        if self.use_histogram:
            self.pitch_histo_encoder = PitchClassHistogramEncoder(config, device)

        self.use_beat = config.use_beat
        if self.use_beat:
            self.beat_encoder = BeatEncoder(config, device)

        self.base_model = config.base_model
        if self.base_model:
            assert not config.use_lstm
            assert not config.use_histogram
            assert not config.use_beat

        self.unnormalized_initial_state_matrix = nn.Parameter(torch.randn(self.num_state))
        if self.base_model:
            self.transition_net = ResidentialTransMat(config, device)
            self.emission_net = EmissionMat(config, vocab, device)
            self.residence_net = ResidenceMat(config, device)
        else:
            self.transition_net = ResidentialTransNet(config, vocab, device)
            self.emission_net = EmissionNet(config, vocab, device)
            self.residence_net = ResidenceNet(config, device)

        self._state_embeddings = nn.Parameter(torch.randn(self.num_state, self.embedding_size))
        self._observation_encodings = ObservationEncoder(config, vocab, device)
        self._dummy_embedding = torch.zeros((1, self.embedding_size)).to(device)

        self.dropout = nn.Dropout(p=self.dropout_p)

    @property
    def state_embeddings(self):
        return self._state_embeddings

    @property
    def observation_encodings(self):
        return self._observation_encodings.value

    def forward(self, batch):
        lengths = batch["sequence_length"].long()
        batch_size = lengths.size(0)
        max_length = lengths.max().item()
        pitch_histogram = batch["pitch_histogram"]
        x = batch["observation_index"][:, :max_length].long()
        beat = batch["beat"][:, :max_length].float()
        timesig_numerator = batch["timesignature_numerator"].float()
        self.transition_net.batch_size = batch_size
        self.residence_net.batch_size = batch_size

        state_embeddings = self.dropout(self.state_embeddings)
        observation_encodings = self.observation_encodings
        observation_encodings_with_pad = torch.cat([observation_encodings, self._dummy_embedding], dim=0)

        if self.use_lstm:
            xt_encoding = observation_encodings_with_pad[x][:, 0, :]
            xt = x[:, 0]
            self.transition_net.initialize_lstm_hidden(batch_size)
            self.transition_net.update_lstm(xt_encoding, xt)

        if self.use_histogram:
            pitch_histo_encoding = self.pitch_histo_encoder(pitch_histogram)
        else:
            pitch_histo_encoding = None

        initial_state = F.softmax(self.unnormalized_initial_state_matrix, dim=0)
        initial_state = initial_state.unsqueeze(0).repeat(batch_size, 1)
        alpha_t = [initial_state]
        for r in range(1, self.max_residential_time):
            alpha_t.append(torch.zeros(batch_size, self.num_state).to(self.device))
        c_t = initial_state.sum(dim=1).unsqueeze(-1)
        alpha = [[a / c_t for a in alpha_t]]
        accum_logc = torch.log(c_t.squeeze())
        mem_accum_logc = torch.zeros((max_length, batch_size)).to(self.device)
        mem_accum_logc[0] = accum_logc

        # forward
        for t in range(1, max_length):
            if self.use_beat:
                beat_t = self.beat_encoder(timesig_numerator, beat[:, t])
            else:
                beat_t = None
            # alpha
            emission_probs = self.emission_net(state_embeddings, observation_encodings, x[:, t])
            residence_probs = self.residence_net(state_embeddings, context=beat_t)
            transition_forward = self.transition_net(state_embeddings, alpha[t - 1][0], context=pitch_histo_encoding)
            alpha_t = []
            c_t = torch.zeros(batch_size).to(self.device)
            for r in range(self.max_residential_time):
                if residence_probs.dim() < 3:
                    residence_pr = residence_probs[:, r].unsqueeze(0)
                else:
                    residence_pr = residence_probs[:, :, r]
                alpha_tr = transition_forward * residence_pr * emission_probs
                if r < (self.max_residential_time - 1):
                    alpha_tr = alpha_tr + alpha[t - 1][r + 1] * emission_probs
                alpha_t.append(alpha_tr)
                c_t = c_t + alpha_tr.sum(dim=1)
            c_t = torch.where(0.0 < c_t, c_t, torch.ones_like(c_t))  # avoid zero division for PAD
            c_t = c_t.unsqueeze(-1)
            alpha_t = [a / c_t for a in alpha_t]
            alpha.append(alpha_t)
            accum_logc += torch.log(c_t.squeeze())
            mem_accum_logc[t] = accum_logc
            if self.use_lstm:
                xt_encoding = observation_encodings_with_pad[x][:, t, :]
                xt = x[:, t]
                self.transition_net.update_lstm(xt_encoding, xt)

        mem_accum_logc = mem_accum_logc.transpose(0, 1)
        gather_indices = (lengths - 1).unsqueeze(-1)
        logliks = mem_accum_logc.gather(1, gather_indices)
        perplexity = torch.exp(-logliks.squeeze() / lengths)
        return {
            LOG_LIKELIHOOD: logliks,
            PERPLEXITY: perplexity,
            BATCH_SIZE: batch_size
        }

    def viterbi_step(self, state_embeddings, omega, context):
        batch_size = omega.size(0)
        tmat = self.transition_net.transition_matrix(state_embeddings, context)
        if tmat.dim() < 3:
            tmat = tmat.unsqueeze(0).repeat(batch_size, 1, 1)
        omega = omega.unsqueeze(-1).repeat(1, 1, self.num_state)
        temp = torch.log(tmat) + omega
        omega_temp, omega_arg = temp.max(dim=1)
        return omega_temp, omega_arg, tmat

    def viterbi(self, batch):
        lengths = batch["sequence_length"].long()
        batch_size = lengths.size(0)
        max_length = lengths.max().item()
        pitch_histogram = batch["pitch_histogram"]
        x = batch["observation_index"][:, :max_length].long()
        beat = batch["beat"][:, :max_length].float()
        timesig_numerator = batch["timesignature_numerator"].float()
        self.transition_net.batch_size = batch_size
        self.residence_net.batch_size = batch_size

        state_embeddings = self.state_embeddings
        observation_encodings = self.observation_encodings
        observation_encodings_with_pad = torch.cat([observation_encodings, self._dummy_embedding], dim=0)

        if self.use_lstm:
            xt_encoding = observation_encodings_with_pad[x][:, 0, :]
            xt = x[:, 0]
            self.transition_net.initialize_lstm_hidden(batch_size)
            self.transition_net.update_lstm(xt_encoding, xt)

        if self.use_histogram:
            pitch_histo_encoding = self.pitch_histo_encoder(pitch_histogram)
        else:
            pitch_histo_encoding = None

        initial_state = F.softmax(self.unnormalized_initial_state_matrix, dim=0)
        initial_state = initial_state.unsqueeze(0).repeat(batch_size, 1)
        omega_t = torch.zeros((batch_size, self.num_state, self.max_residential_time)).to(self.device)
        omega_t[:, :, 0] = initial_state
        omega_args_t = torch.zeros((batch_size, self.num_state, self.max_residential_time)).long().to(self.device)
        omega = [torch.log(omega_t)]
        omega_args = [omega_args_t]
        tmats = [initial_state.unsqueeze(1).repeat(1, self.num_state, 1).unsqueeze(0)]

        #  forward
        for t in range(1, max_length):
            if self.use_beat:
                beat_t = self.beat_encoder(timesig_numerator, beat[:, t])
            else:
                beat_t = None
            # omega
            emission_probs = self.emission_net(state_embeddings, observation_encodings, x[:, t])
            residence_probs = self.residence_net(state_embeddings, context=beat_t)
            transition_forward = self.viterbi_step(state_embeddings, omega[t - 1][:, :, 0], context=pitch_histo_encoding)
            omega_t = torch.zeros((batch_size, self.num_state, self.max_residential_time)).to(self.device)
            omega_args_t = torch.zeros((batch_size, self.num_state, self.max_residential_time)).long().to(self.device)
            for r in range(self.max_residential_time):
                if residence_probs.dim() < 3:
                    residence_pr = residence_probs[:, r].unsqueeze(0).repeat(batch_size, 1)
                else:
                    residence_pr = residence_probs[:, :, r]
                omega_tr, omega_args_tr, _ = transition_forward
                omega_tr += torch.log(residence_pr)
                omega_tr += torch.log(emission_probs)
                if r < (self.max_residential_time - 1):
                    omega_tr_res = omega[t - 1][:, :, r + 1] + torch.log(emission_probs)
                    omega_args_tr_res = torch.ones_like(omega_args_tr).long().to(self.device) * RESID_STATE
                    omega_tr_cat = torch.cat([omega_tr.unsqueeze(-1), omega_tr_res.unsqueeze(-1)], dim=-1)
                    omega_args_tr_cat = torch.cat([omega_args_tr.unsqueeze(-1), omega_args_tr_res.unsqueeze(-1)], dim=-1)
                    omega_tr, gather_index = omega_tr_cat.max(dim=-1)
                    omega_args_tr = omega_args_tr_cat.gather(-1, gather_index.unsqueeze(-1)).squeeze()
                omega_t[:, :, r] = omega_tr
                omega_args_t[:, :, r] = omega_args_tr

            omega.append(omega_t)
            omega_args.append(omega_args_t)
            _, _, tmat = transition_forward
            tmats.append(tmat.unsqueeze(0))

            if self.use_lstm:
                xt_encoding = observation_encodings_with_pad[x][:, t, :]
                xt = x[:, t]
                self.transition_net.update_lstm(xt_encoding, xt)

        # finish
        omega = torch.stack(omega, dim=-1)
        omega_args = torch.stack(omega_args, dim=-1)
        gather_indices = (lengths - 1).unsqueeze(-1).unsqueeze(-1).repeat(
            1, self.num_state, self.max_residential_time).unsqueeze(-1)
        joint_probs = omega.gather(-1, gather_indices).squeeze()
        if joint_probs.dim() < 3:
            joint_probs = joint_probs.unsqueeze(0)
        joint_probs = joint_probs[:, :, 0]
        best_probs, best_args = joint_probs.max(dim=1)

        # reconstruction
        batch_states = [[best_arg] for best_arg in best_args.tolist()]
        batch_residence = [[0] for _ in range(len(batch_states))]
        for t in range(0, max_length - 1)[::-1]:
            for ib in range(batch_size):
                if t < (lengths[ib] - 1):
                    state = batch_states[ib][-1]
                    res = batch_residence[ib][-1]
                    state_from = omega_args[ib, state, res, t + 1].item()
                    if state_from == RESID_STATE:
                        batch_states[ib].append(state)
                        batch_residence[ib].append(res + 1)
                    else:
                        batch_states[ib].append(state_from)
                        batch_residence[ib].append(0)

        for ib in range(batch_size):
            batch_states[ib] = batch_states[ib][::-1]
            batch_residence[ib] = batch_residence[ib][::-1]

        tmats = torch.cat(tmats, dim=0)
        observation_list = []
        transition_matrix_list = []
        for ib in range(batch_size):
            length = lengths[ib].item()
            observation_list.append(x[ib].tolist()[:length])
            transition_matrix_list.append(tmats[:length, ib, :, :].detach().cpu().numpy())

        return {
            OBSERVATION: observation_list,
            JOINT_PROB: best_probs.tolist(),
            STATES: batch_states,
            RESIDENCES: batch_residence,
            BATCH_SIZE: batch_size,
            TRANS_MAT: transition_matrix_list,
            META_DATA: batch[META_DATA]
        }
