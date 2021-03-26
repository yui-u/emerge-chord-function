from core.common.constants import *
import numpy as np


class VocabChord(object):
    def __init__(self, config):
        self.scale = config.scale
        self.key_preprocessing = config.key_preprocessing
        self.chord_counts = {}
        self.chord_cover_ratios = {}
        self.vocab_cover_ratio = config.vocab_cover_ratio
        self.c2i = {}
        self.i2c = {}
        self.i2b = {}
        self.unk_index = None
        self.bos_index = None
        self.pad_index = None
        self.fixed = False

    @property
    def unk_binary_pitch_class(self):
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    @property
    def pad_binary_pitch_class(self):
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def update_counts(self, chord, duration):
        assert not self.fixed
        chord = tuple(sorted(list(chord)))
        if chord in self.chord_counts:
            self.chord_counts[chord] += duration
        else:
            self.chord_counts[chord] = duration

    def fix_index(self):
        assert not self.fixed
        sum_durations = sum(list(self.chord_counts.values()))
        chord_counts = sorted(self.chord_counts.items(), key=lambda x: x[1], reverse=True)
        accum_durations = 0.0
        cut_index = -1
        for i, (k, v) in enumerate(chord_counts):
            accum_durations += v
            cover_ratio = accum_durations / sum_durations
            self.chord_cover_ratios[k] = cover_ratio
            if cut_index < 0 and self.vocab_cover_ratio <= cover_ratio:
                cut_index = i

        for i, (k, v) in enumerate(chord_counts):
            if i <= cut_index:
                assert k not in self.c2i
                self.c2i[k] = i
                self.i2c[i] = k
                self.i2b[i] = self.chord_to_pitchlist(k)
            else:
                break

        # UNK
        i = cut_index + 1
        self.c2i[UNK_STR] = i
        self.i2c[i] = UNK_STR
        self.i2b[i] = self.unk_binary_pitch_class
        self.unk_index = i

        # pad should take the last index
        i = i + 1
        self.c2i[PAD_STR] = i
        self.i2c[i] = PAD_STR
        self.i2b[i] = self.pad_binary_pitch_class
        self.pad_index = i

        self.fixed = True

    def pitchlist_to_index(self, pitch_classes):
        chord = np.where(pitch_classes.numpy() > 0)[0]
        chord = tuple(chord.tolist())
        if chord in self.c2i:
            return self.c2i[chord]
        else:
            return self.c2i[UNK_STR]

    def chord_to_pitchlist(self, chord):
        pitchlist = np.zeros(12, dtype=int)
        for p in list(chord):
            assert pitchlist[p] == 0
            pitchlist[p] = 1
        return pitchlist.tolist()
