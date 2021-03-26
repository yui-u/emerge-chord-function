META_DATA = 'metadata'
OUTPUT = 'output'
TOTAL_LOSS = 'total-loss'
OBSERVATION = 'observation'
JOINT_PROB = 'joint-probability'
PERPLEXITY = 'perplexity'
BATCH_SIZE = 'batch_size'

MAJOR = 'major'
MINOR = 'minor'
ANY = 'any'
NOMODAL = 'nomodal'
DORIAN = 'dorian'
LYDIAN = 'lydian'
MIXOLYDIAN = 'mixolydian'
PHRYGIAN = 'phrygian'
AMBIGUOUS = 'ambiguous'
SHARP_CIRCLE = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#']
FLAT_CIRCLE = ['C', 'F', 'B-', 'E-', 'A-', 'D-', 'G-', 'C-']

# Dataset
KEY_PREPROCESS_NORMALIZE = 'key-preprocess-normalized'

# NeuralHMM
HMM = 'hmm'
HSMM = 'hsmm'
PARAMS = 'params'
PRIOR = 'prior'
TRANSITION = 'transition'
EMISSION = 'emission'
LOG_LIKELIHOOD = 'loglik'
STATES = 'states'
RESIDENCES = 'residences'
PRIOR_MAT = 'prior-matrix'
TRANS_MAT = 'transition-matrix'
EMIT_MAT = 'emission-matrix'
RESID_MAT = 'residence-matrix'
BEAT_ALL = 'beat-all'
BEAT_FOUR_FOUR = 'beat-four-four'

MIDI_REST_INDEX = -1
MIDI_SPECIAL_INDEX = -2
MIDI_PAD_INDEX = -3
UNK_STR = '<unk>'
PAD_STR = '<pad>'
DURATIONS_SPECIAL_SYMBOL = 1.0  # dummy duration for special symbols
DURATIONS_PAD = 0.0
METRIC_BEAT_RATIO = 0.25
BEAT_SPECIAL_SYMBOL = 0.0
RESID_STATE = -1

TRAIN = "train"
DEV = "dev"
TEST = "test"
