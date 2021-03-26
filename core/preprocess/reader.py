import math

from logging import Logger
from typing import Dict

import numpy as np
from music21.corpus import parse
from music21.corpus.chorales import ChoraleListRKBWV
from music21.interval import Interval
from music21.pitch import Pitch
from music21.chord import Chord
from music21.note import Note, Rest
from music21.stream import Measure, Part

from core.common.constants import *
from core.preprocess.instances import ListInstance, ValueInstance

from core.preprocess.vocab import VocabChord


class BachChoraleReader(object):
    def __init__(self, config):
        self._config = config
        self._key_preprocessing = config.key_preprocessing

        self._major_minor_dict = {
            1: MAJOR,
            2: MAJOR,
            3: MINOR,
            4: MAJOR,
            5: MAJOR,
            6: MAJOR,
            7: MAJOR,
            8: MINOR,
            9: MAJOR,
            10: MINOR,
            11: MAJOR,
            12: MINOR,
            13: MINOR,
            14: MAJOR,
            15: MINOR,
            16: MINOR,
            17: MINOR,
            18: MAJOR,
            19: MINOR,
            20: MAJOR,
            21: MINOR,
            22: MAJOR,
            23: MINOR,
            24: MAJOR,
            25: MINOR,
            26: MAJOR,
            27: MAJOR,
            28: MINOR,
            29: MAJOR,
            30: MINOR,
            31: MINOR,
            32: MAJOR,
            33: MINOR,
            34: MINOR,
            35: MAJOR,
            36: MAJOR,
            37: MINOR,
            38: MAJOR,
            39: MINOR,
            40: MAJOR,
            41: MINOR,
            42: MAJOR,
            43: MAJOR,
            44: MAJOR,
            45: MINOR,
            46: MAJOR,
            47: MINOR,
            48: MINOR,
            49: MINOR,
            50: MAJOR,
            51: MAJOR,
            52: MAJOR,
            53: MINOR,
            54: MAJOR,
            55: MINOR,
            56: MINOR,
            57: MINOR,
            58: MAJOR,
            59: MINOR,
            60: MAJOR,
            61: MAJOR,
            62: MINOR,
            63: MAJOR,
            64: MAJOR,
            65: MAJOR,
            66: MINOR,
            67: MAJOR,
            68: MAJOR,
            69: MAJOR,
            70: MAJOR,
            71: MINOR,
            72: MINOR,
            73: MINOR,
            74: MAJOR,
            75: MINOR,
            76: MAJOR,
            77: MAJOR,
            78: MINOR,
            79: MINOR,
            80: MAJOR,
            81: MINOR,
            82: MINOR,
            83: MAJOR,
            84: MAJOR,
            85: MAJOR,
            86: MAJOR,
            87: MINOR,
            88: MINOR,
            89: MINOR,
            90: MAJOR,
            91: MINOR,
            92: MINOR,
            93: MAJOR,
            94: MINOR,
            95: MAJOR,
            96: MINOR,
            97: MAJOR,
            98: MAJOR,
            99: MINOR,
            100: MINOR,
            101: MAJOR,
            102: MAJOR,
            103: MAJOR,
            104: MINOR,
            105: MINOR,
            106: MAJOR,
            107: MAJOR,
            108: MAJOR,
            109: MINOR,
            110: MINOR,
            111: MINOR,
            112: MINOR,
            113: MINOR,
            114: MINOR,
            115: MINOR,
            116: MAJOR,
            117: MAJOR,
            118: MAJOR,
            119: MINOR,
            120: MINOR,
            121: MAJOR,
            122: MINOR,
            123: MINOR,
            124: MAJOR,
            125: MAJOR,
            126: MINOR,
            127: MAJOR,
            128: MAJOR,
            129: MINOR,
            130: MINOR,
            131: MAJOR,
            132: MINOR,
            133: MINOR,
            134: MINOR,
            135: MAJOR,
            136: MAJOR,
            137: MAJOR,
            138: MINOR,
            139: MAJOR,
            140: MAJOR,
            141: MAJOR,
            142: MINOR,
            143: MAJOR,
            144: MAJOR,
            145: MINOR,
            146: MINOR,
            147: MAJOR,
            148: MAJOR,
            149: MINOR,
            150: MAJOR,
            151: MAJOR,
            152: MAJOR,
            153: MAJOR,
            154: MINOR,
            155: MINOR,
            156: MAJOR,
            157: MAJOR,
            158: MAJOR,
            159: MAJOR,
            160: MAJOR,
            161: MINOR,
            162: MINOR,
            163: MINOR,
            164: MAJOR,
            165: MAJOR,
            166: MINOR,
            167: MINOR,
            168: MINOR,
            169: MAJOR,
            170: MINOR,
            171: MINOR,
            172: MINOR,
            173: MAJOR,
            174: MINOR,
            175: MAJOR,
            176: MAJOR,
            177: MAJOR,
            178: MINOR,
            179: MAJOR,
            180: MINOR,
            181: MINOR,
            182: MINOR,
            183: MAJOR,
            184: MINOR,
            185: MINOR,
            186: MINOR,
            187: MAJOR,
            188: MAJOR,
            189: MAJOR,
            190: MINOR,
            191: MINOR,
            192: MAJOR,
            193: MINOR,
            194: MINOR,
            195: MAJOR,
            196: MINOR,
            197: MINOR,
            198: MINOR,
            199: MINOR,
            200: MAJOR,
            201: MAJOR,
            202: MAJOR,
            203: MINOR,
            204: MINOR,
            205: MINOR,
            206: MINOR,
            207: MINOR,
            208: MINOR,
            209: MAJOR,
            210: MINOR,
            211: MAJOR,
            212: MAJOR,
            213: MINOR,
            214: MINOR,
            215: MINOR,
            216: MAJOR,
            217: MAJOR,
            218: MINOR,
            219: MINOR,
            220: MINOR,
            221: MINOR,
            222: MAJOR,
            223: MAJOR,
            224: MAJOR,
            225: MINOR,
            226: MINOR,
            227: MINOR,
            228: MINOR,
            229: MINOR,
            230: MINOR,
            231: MAJOR,
            232: MINOR,
            233: MAJOR,
            234: MAJOR,
            235: MAJOR,
            236: MINOR,
            237: MINOR,
            238: MINOR,
            239: MAJOR,
            240: MINOR,
            241: MINOR,
            242: MINOR,
            243: MINOR,
            244: MINOR,
            245: MINOR,
            246: MAJOR,
            247: MAJOR,
            248: MAJOR,
            249: MAJOR,
            250: MAJOR,
            251: MINOR,
            252: MAJOR,
            253: MINOR,
            254: MAJOR,
            255: MAJOR,
            256: MAJOR,
            257: MAJOR,
            258: MAJOR,
            259: MINOR,
            260: MAJOR,
            261: MINOR,
            262: MINOR,
            263: MINOR,
            264: MAJOR,
            265: MINOR,
            266: MINOR,
            267: MINOR,
            268: MAJOR,
            269: MINOR,
            270: MINOR,
            271: MINOR,
            272: MAJOR,
            273: MAJOR,
            274: MAJOR,
            275: MAJOR,
            276: MAJOR,
            277: MAJOR,
            278: MAJOR,
            279: MAJOR,
            280: MAJOR,
            281: MINOR,
            282: MAJOR,
            283: MINOR,
            284: MAJOR,
            285: MINOR,
            286: MINOR,
            287: MINOR,
            288: MAJOR,
            289: MAJOR,
            290: MAJOR,
            291: MAJOR,
            292: MINOR,
            293: MAJOR,
            294: MINOR,
            295: MINOR,
            296: MAJOR,
            297: MINOR,
            298: MAJOR,
            299: MAJOR,
            300: MINOR,
            301: MINOR,
            302: MINOR,
            303: MAJOR,
            304: MINOR,
            305: MAJOR,
            306: MAJOR,
            307: MINOR,
            308: MAJOR,
            309: MAJOR,
            310: MAJOR,
            311: MAJOR,
            312: MAJOR,
            313: MAJOR,
            314: MINOR,
            315: MAJOR,
            316: MAJOR,
            317: MAJOR,
            318: MAJOR,
            319: MAJOR,
            320: MINOR,
            321: MINOR,
            322: MAJOR,
            323: MAJOR,
            324: MINOR,
            325: MINOR,
            326: MAJOR,
            327: MAJOR,
            328: MAJOR,
            329: MAJOR,
            330: MAJOR,
            331: MINOR,
            332: MINOR,
            333: MAJOR,
            334: MAJOR,
            335: MAJOR,
            336: MINOR,
            337: MAJOR,
            338: MAJOR,
            339: MINOR,
            340: MINOR,
            341: MAJOR,
            342: MAJOR,
            343: MAJOR,
            344: MAJOR,
            345: MINOR,
            346: MINOR,
            347: MAJOR,
            348: MAJOR,
            349: MINOR,
            350: MAJOR,
            351: MAJOR,
            352: MINOR,
            353: MAJOR,
            354: MAJOR,
            355: MAJOR,
            356: MINOR,
            357: MAJOR,
            358: MINOR,
            359: MINOR,
            360: MINOR,
            361: MAJOR,
            362: MAJOR,
            363: MAJOR,
            364: MINOR,
            365: MAJOR,
            366: MAJOR,
            367: MINOR,
            368: MAJOR,
            369: MINOR,
            370: MINOR,
            371: MINOR,
        }

        self._modal_dict = {
            1: NOMODAL,
            2: NOMODAL,
            3: DORIAN,
            4: NOMODAL,
            5: NOMODAL,
            6: NOMODAL,
            7: NOMODAL,
            8: DORIAN,
            9: NOMODAL,
            10: PHRYGIAN,
            11: NOMODAL,
            12: NOMODAL,
            13: NOMODAL,
            14: NOMODAL,
            15: DORIAN,
            16: NOMODAL,
            17: PHRYGIAN,
            18: NOMODAL,
            19: DORIAN,
            20: NOMODAL,
            21: NOMODAL,
            22: NOMODAL,
            23: NOMODAL,
            24: NOMODAL,
            25: NOMODAL,
            26: NOMODAL,
            27: NOMODAL,
            28: NOMODAL,
            29: NOMODAL,
            30: NOMODAL,
            31: NOMODAL,
            32: NOMODAL,
            33: NOMODAL,
            34: PHRYGIAN,
            35: NOMODAL,
            36: NOMODAL,
            37: NOMODAL,
            38: NOMODAL,
            39: NOMODAL,
            40: NOMODAL,
            41: NOMODAL,
            42: NOMODAL,
            43: NOMODAL,
            44: NOMODAL,
            45: NOMODAL,
            46: NOMODAL,
            47: NOMODAL,
            48: NOMODAL,
            49: DORIAN,
            50: NOMODAL,
            51: NOMODAL,
            52: NOMODAL,
            53: NOMODAL,
            54: NOMODAL,
            55: NOMODAL,
            56: NOMODAL,
            57: NOMODAL,
            58: NOMODAL,
            59: NOMODAL,
            60: NOMODAL,
            61: NOMODAL,
            62: NOMODAL,
            63: NOMODAL,
            64: NOMODAL,
            65: NOMODAL,
            66: DORIAN,
            67: NOMODAL,
            68: NOMODAL,
            69: NOMODAL,
            70: NOMODAL,
            71: NOMODAL,
            72: NOMODAL,
            73: NOMODAL,
            74: NOMODAL,
            75: NOMODAL,
            76: NOMODAL,
            77: NOMODAL,
            78: NOMODAL,
            79: NOMODAL,
            80: NOMODAL,
            81: NOMODAL,
            82: NOMODAL,
            83: NOMODAL,
            84: NOMODAL,
            85: NOMODAL,
            86: NOMODAL,
            87: DORIAN,
            88: NOMODAL,
            89: NOMODAL,
            90: NOMODAL,
            91: NOMODAL,
            92: NOMODAL,
            93: NOMODAL,
            94: NOMODAL,
            95: NOMODAL,
            96: DORIAN,
            97: NOMODAL,
            98: NOMODAL,
            99: NOMODAL,
            100: NOMODAL,
            101: NOMODAL,
            102: NOMODAL,
            103: NOMODAL,
            104: NOMODAL,
            105: NOMODAL,
            106: NOMODAL,
            107: NOMODAL,
            108: NOMODAL,
            109: NOMODAL,
            110: DORIAN,
            111: NOMODAL,
            112: NOMODAL,
            113: AMBIGUOUS,
            114: NOMODAL,
            115: NOMODAL,
            116: NOMODAL,
            117: NOMODAL,
            118: NOMODAL,
            119: DORIAN,
            120: NOMODAL,
            121: NOMODAL,
            122: DORIAN,
            123: NOMODAL,
            124: NOMODAL,
            125: NOMODAL,
            126: NOMODAL,
            127: NOMODAL,
            128: NOMODAL,
            129: NOMODAL,
            130: NOMODAL,
            131: NOMODAL,
            132: NOMODAL,
            133: DORIAN,
            134: NOMODAL,
            135: NOMODAL,
            136: NOMODAL,
            137: NOMODAL,
            138: NOMODAL,
            139: NOMODAL,
            140: NOMODAL,
            141: NOMODAL,
            142: NOMODAL,
            143: NOMODAL,
            144: NOMODAL,
            145: NOMODAL,
            146: NOMODAL,
            147: NOMODAL,
            148: NOMODAL,
            149: NOMODAL,
            150: LYDIAN,
            151: NOMODAL,
            152: NOMODAL,
            153: NOMODAL,
            154: DORIAN,
            155: DORIAN,
            156: NOMODAL,
            157: NOMODAL,
            158: NOMODAL,
            159: NOMODAL,
            160: AMBIGUOUS,
            161: NOMODAL,
            162: DORIAN,
            163: NOMODAL,
            164: NOMODAL,
            165: NOMODAL,
            166: DORIAN,
            167: NOMODAL,
            168: DORIAN,
            169: NOMODAL,
            170: NOMODAL,
            171: DORIAN,
            172: NOMODAL,
            173: NOMODAL,
            174: DORIAN,
            175: NOMODAL,
            176: NOMODAL,
            177: NOMODAL,
            178: NOMODAL,
            179: NOMODAL,
            180: DORIAN,
            181: NOMODAL,
            182: NOMODAL,
            183: NOMODAL,
            184: NOMODAL,
            185: DORIAN,
            186: DORIAN,
            187: NOMODAL,
            188: NOMODAL,
            189: NOMODAL,
            190: NOMODAL,
            191: DORIAN,
            192: NOMODAL,
            193: NOMODAL,
            194: NOMODAL,
            195: NOMODAL,
            196: NOMODAL,
            197: DORIAN,
            198: NOMODAL,
            199: DORIAN,
            200: NOMODAL,
            201: NOMODAL,
            202: NOMODAL,
            203: DORIAN,
            204: NOMODAL,
            205: NOMODAL,
            206: DORIAN,
            207: NOMODAL,
            208: NOMODAL,
            209: NOMODAL,
            210: DORIAN,
            211: NOMODAL,
            212: NOMODAL,
            213: NOMODAL,
            214: NOMODAL,
            215: NOMODAL,
            216: NOMODAL,
            217: NOMODAL,
            218: DORIAN,
            219: NOMODAL,
            220: NOMODAL,
            221: NOMODAL,
            222: NOMODAL,
            223: NOMODAL,
            224: NOMODAL,
            225: NOMODAL,
            226: NOMODAL,
            227: DORIAN,
            228: NOMODAL,
            229: NOMODAL,
            230: NOMODAL,
            231: AMBIGUOUS,
            232: DORIAN,
            233: NOMODAL,
            234: NOMODAL,
            235: NOMODAL,
            236: NOMODAL,
            237: DORIAN,
            238: NOMODAL,
            239: NOMODAL,
            240: NOMODAL,
            241: NOMODAL,
            242: NOMODAL,
            243: NOMODAL,
            244: DORIAN,
            245: NOMODAL,
            246: NOMODAL,
            247: NOMODAL,
            248: NOMODAL,
            249: NOMODAL,
            250: NOMODAL,
            251: NOMODAL,
            252: NOMODAL,
            253: DORIAN,
            254: NOMODAL,
            255: NOMODAL,
            256: NOMODAL,
            257: NOMODAL,
            258: NOMODAL,
            259: NOMODAL,
            260: NOMODAL,
            261: NOMODAL,
            262: DORIAN,
            263: NOMODAL,
            264: NOMODAL,
            265: NOMODAL,
            266: NOMODAL,
            267: NOMODAL,
            268: NOMODAL,
            269: NOMODAL,
            270: NOMODAL,
            271: NOMODAL,
            272: NOMODAL,
            273: NOMODAL,
            274: NOMODAL,
            275: NOMODAL,
            276: NOMODAL,
            277: NOMODAL,
            278: NOMODAL,
            279: NOMODAL,
            280: NOMODAL,
            281: NOMODAL,
            282: NOMODAL,
            283: NOMODAL,
            284: NOMODAL,
            285: NOMODAL,
            286: NOMODAL,
            287: NOMODAL,
            288: NOMODAL,
            289: NOMODAL,
            290: NOMODAL,
            291: NOMODAL,
            292: NOMODAL,
            293: NOMODAL,
            294: NOMODAL,
            295: NOMODAL,
            296: NOMODAL,
            297: NOMODAL,
            298: NOMODAL,
            299: NOMODAL,
            300: NOMODAL,
            301: NOMODAL,
            302: DORIAN,
            303: NOMODAL,
            304: NOMODAL,
            305: NOMODAL,
            306: NOMODAL,
            307: NOMODAL,
            308: NOMODAL,
            309: NOMODAL,
            310: NOMODAL,
            311: NOMODAL,
            312: NOMODAL,
            313: NOMODAL,
            314: NOMODAL,
            315: NOMODAL,
            316: NOMODAL,
            317: NOMODAL,
            318: NOMODAL,
            319: NOMODAL,
            320: NOMODAL,
            321: DORIAN,
            322: NOMODAL,
            323: NOMODAL,
            324: NOMODAL,
            325: DORIAN,
            326: NOMODAL,
            327: NOMODAL,
            328: NOMODAL,
            329: NOMODAL,
            330: NOMODAL,
            331: NOMODAL,
            332: NOMODAL,
            333: NOMODAL,
            334: NOMODAL,
            335: NOMODAL,
            336: NOMODAL,
            337: NOMODAL,
            338: NOMODAL,
            339: NOMODAL,
            340: NOMODAL,
            341: NOMODAL,
            342: NOMODAL,
            343: NOMODAL,
            344: NOMODAL,
            345: NOMODAL,
            346: NOMODAL,
            347: NOMODAL,
            348: NOMODAL,
            349: NOMODAL,
            350: NOMODAL,
            351: NOMODAL,
            352: NOMODAL,
            353: NOMODAL,
            354: NOMODAL,
            355: NOMODAL,
            356: NOMODAL,
            357: NOMODAL,
            358: NOMODAL,
            359: NOMODAL,
            360: NOMODAL,
            361: NOMODAL,
            362: NOMODAL,
            363: NOMODAL,
            364: NOMODAL,
            365: NOMODAL,
            366: NOMODAL,
            367: NOMODAL,
            368: NOMODAL,
            369: DORIAN,
            370: NOMODAL,
            371: NOMODAL,
        }

        self._duplicate_list = [
                [5, 309],
                [9, 361],
                [23, 88],
                [53, 178],
                [64, 256],
                [86, 195, 305],
                [91, 259],
                [93, 257],
                [100, 126],
                [120, 349],
                [125, 326],
                [131, 328],
                [144, 318],
                [156, 308],
                [198, 307],
                [199, 302],
                [201, 306],
                [235, 319],
                [236, 295],
                [248, 354],
                [254, 282],
                [313, 353],
            ]

        self._invalid_measure_list = [130]

        self._train_dev_test_dict = {
            1: TRAIN,
            2: TRAIN,
            3: TRAIN,
            4: TRAIN,
            5: TEST,
            6: TRAIN,
            7: TRAIN,
            8: TRAIN,
            9: TRAIN,
            10: TRAIN,
            11: TEST,
            12: TEST,
            13: TRAIN,
            14: DEV,
            15: TRAIN,
            16: TRAIN,
            17: TEST,
            18: DEV,
            19: TRAIN,
            20: TEST,
            21: DEV,
            22: TRAIN,
            23: TEST,
            24: TRAIN,
            25: TRAIN,
            26: TRAIN,
            27: TRAIN,
            28: TRAIN,
            29: DEV,
            30: TRAIN,
            31: TRAIN,
            32: TEST,
            33: TRAIN,
            34: TEST,
            35: TRAIN,
            36: TRAIN,
            37: TRAIN,
            38: TRAIN,
            39: TRAIN,
            40: TRAIN,
            41: TRAIN,
            42: TRAIN,
            43: TRAIN,
            44: TRAIN,
            45: DEV,
            46: DEV,
            47: TRAIN,
            48: TRAIN,
            49: TRAIN,
            50: TRAIN,
            51: DEV,
            52: DEV,
            53: DEV,
            54: TRAIN,
            55: TRAIN,
            56: TRAIN,
            57: TRAIN,
            58: TRAIN,
            59: TRAIN,
            60: TRAIN,
            61: TRAIN,
            62: DEV,
            63: TRAIN,
            64: TEST,
            65: DEV,
            66: TRAIN,
            67: TRAIN,
            68: DEV,
            69: TRAIN,
            70: TEST,
            71: TRAIN,
            72: TRAIN,
            73: TEST,
            74: TRAIN,
            75: DEV,
            76: TEST,
            77: DEV,
            78: DEV,
            79: TRAIN,
            80: TRAIN,
            81: TRAIN,
            82: TEST,
            83: TRAIN,
            84: TRAIN,
            85: TRAIN,
            86: TRAIN,
            87: TRAIN,
            88: TEST,
            89: DEV,
            90: TRAIN,
            91: TRAIN,
            92: DEV,
            93: TRAIN,
            94: DEV,
            95: TEST,
            96: TRAIN,
            97: TRAIN,
            98: TRAIN,
            99: TRAIN,
            100: TRAIN,
            101: TRAIN,
            102: TRAIN,
            103: TRAIN,
            104: TRAIN,
            105: TEST,
            106: DEV,
            107: TRAIN,
            108: TRAIN,
            109: TRAIN,
            110: TRAIN,
            111: DEV,
            112: TRAIN,
            113: TRAIN,
            114: DEV,
            115: TRAIN,
            116: TRAIN,
            117: TEST,
            118: TEST,
            119: TRAIN,
            120: TRAIN,
            121: DEV,
            122: TEST,
            123: TEST,
            124: TRAIN,
            125: TRAIN,
            126: TRAIN,
            127: TRAIN,
            128: TRAIN,
            129: TRAIN,
            130: TRAIN,
            131: TEST,
            132: TRAIN,
            133: TRAIN,
            134: TEST,
            135: TRAIN,
            136: TEST,
            137: TRAIN,
            138: TRAIN,
            139: TRAIN,
            140: DEV,
            141: TRAIN,
            142: TEST,
            143: DEV,
            144: TRAIN,
            145: DEV,
            146: TRAIN,
            147: TRAIN,
            148: TRAIN,
            149: TRAIN,
            150: TEST,
            151: TRAIN,
            152: TRAIN,
            153: TRAIN,
            154: TRAIN,
            155: DEV,
            156: DEV,
            157: TRAIN,
            158: DEV,
            159: TRAIN,
            160: TRAIN,
            161: DEV,
            162: TRAIN,
            163: TRAIN,
            164: DEV,
            165: TRAIN,
            166: TEST,
            167: TRAIN,
            168: TRAIN,
            169: TRAIN,
            170: TRAIN,
            171: TRAIN,
            172: TRAIN,
            173: DEV,
            174: TRAIN,
            175: TRAIN,
            176: DEV,
            177: TRAIN,
            178: TRAIN,
            179: DEV,
            180: DEV,
            181: TRAIN,
            182: DEV,
            183: TEST,
            184: TRAIN,
            185: DEV,
            186: TRAIN,
            187: TRAIN,
            188: TRAIN,
            189: TRAIN,
            190: TRAIN,
            191: TRAIN,
            192: DEV,
            193: DEV,
            194: TEST,
            195: TEST,
            196: TRAIN,
            197: TRAIN,
            198: TRAIN,
            199: DEV,
            200: TRAIN,
            201: TEST,
            202: TRAIN,
            203: TEST,
            204: TRAIN,
            205: TEST,
            206: TRAIN,
            207: TRAIN,
            208: TRAIN,
            209: TRAIN,
            210: TRAIN,
            211: DEV,
            212: TRAIN,
            213: TRAIN,
            214: DEV,
            215: TRAIN,
            216: DEV,
            217: DEV,
            218: DEV,
            219: TRAIN,
            220: DEV,
            221: TRAIN,
            222: TRAIN,
            223: DEV,
            224: TRAIN,
            225: DEV,
            226: TRAIN,
            227: DEV,
            228: TRAIN,
            229: TRAIN,
            230: TRAIN,
            231: DEV,
            232: TRAIN,
            233: TRAIN,
            234: DEV,
            235: TRAIN,
            236: TEST,
            237: TEST,
            238: TRAIN,
            239: TRAIN,
            240: DEV,
            241: TRAIN,
            242: TRAIN,
            243: DEV,
            244: TRAIN,
            245: TRAIN,
            246: TRAIN,
            247: TRAIN,
            248: TRAIN,
            249: TRAIN,
            250: TEST,
            251: DEV,
            252: TEST,
            253: DEV,
            254: TRAIN,
            255: TRAIN,
            256: TRAIN,
            257: TRAIN,
            258: DEV,
            259: TRAIN,
            260: TRAIN,
            261: TRAIN,
            262: TEST,
            263: TRAIN,
            264: DEV,
            265: TRAIN,
            266: TRAIN,
            267: TRAIN,
            268: DEV,
            269: TEST,
            270: TEST,
            271: TEST,
            272: TRAIN,
            273: TRAIN,
            274: TRAIN,
            275: TEST,
            276: TRAIN,
            277: TRAIN,
            278: TRAIN,
            279: TRAIN,
            280: TEST,
            281: TEST,
            282: TEST,
            283: TRAIN,
            284: TEST,
            285: TRAIN,
            286: DEV,
            287: TRAIN,
            288: TEST,
            289: TRAIN,
            290: TRAIN,
            291: DEV,
            292: DEV,
            293: TRAIN,
            294: TRAIN,
            295: TEST,
            296: TRAIN,
            297: TRAIN,
            298: DEV,
            299: TEST,
            300: DEV,
            301: TEST,
            302: DEV,
            303: TRAIN,
            304: TRAIN,
            305: TRAIN,
            306: TEST,
            307: TEST,
            308: TEST,
            309: TRAIN,
            310: TEST,
            311: TEST,
            312: TRAIN,
            313: DEV,
            314: TEST,
            315: DEV,
            316: TRAIN,
            317: TEST,
            318: TRAIN,
            319: TEST,
            320: DEV,
            321: TEST,
            322: TEST,
            323: DEV,
            324: TRAIN,
            325: DEV,
            326: TRAIN,
            327: TRAIN,
            328: TRAIN,
            329: DEV,
            330: TEST,
            331: TEST,
            332: TRAIN,
            333: TRAIN,
            334: DEV,
            335: TEST,
            336: TEST,
            337: TRAIN,
            338: TRAIN,
            339: DEV,
            340: DEV,
            341: TRAIN,
            342: TRAIN,
            343: TEST,
            344: TRAIN,
            345: TEST,
            346: TEST,
            347: DEV,
            348: TEST,
            349: TEST,
            350: TEST,
            351: TRAIN,
            352: TEST,
            353: TEST,
            354: DEV,
            355: TRAIN,
            356: TRAIN,
            357: TEST,
            358: TRAIN,
            359: TEST,
            360: TRAIN,
            361: DEV,
            362: TRAIN,
            363: TRAIN,
            364: TRAIN,
            365: TRAIN,
            366: TEST,
            367: TRAIN,
            368: TRAIN,
            369: TRAIN,
            370: TEST,
            371: TRAIN,
        }

    def _get_ignore_duplicate_list(self):
        ignore_list = []
        for dpl in self._duplicate_list:
            for d in dpl[1:]:  # only allow most small riemann number
                ignore_list.append(d)
        ignore_list.extend(self._invalid_measure_list)
        return ignore_list

    def _get_scores(self, logger) -> Dict:
        bcl = ChoraleListRKBWV()
        ignore_list = self._get_ignore_duplicate_list()

        scores = {}
        cnt_allowed_timesig = 0
        cnt_not_duplicated = 0
        for k, v in bcl.byRiemenschneider.items():  # Riemenshneider contains 371 pieces
            riemenschneider = int(v['riemenschneider'])
            bwv_m21 = v['bwv']
            score = parse('bwv{}'.format(bwv_m21))
            parts = score.getElementsByClass(Part)
            timesig = score.parts[0].measure(2).getContextByClass('TimeSignature')

            if self._config.beat_type == BEAT_FOUR_FOUR:
                allowed_timesig = True if (timesig.denominator == 4 and timesig.numerator == 4) else False
            else:
                allowed_timesig = True

            if allowed_timesig:
                cnt_allowed_timesig += 1

            scale = self._major_minor_dict[riemenschneider]
            modal = self._modal_dict[riemenschneider]

            if riemenschneider not in ignore_list:
                cnt_not_duplicated += 1

            num_measures = None
            added = False
            if len(parts) == 4:
                if (parts[0].id == 'Soprano' and
                        parts[1].id == 'Alto' and
                        parts[2].id == 'Tenor' and
                        parts[3].id == 'Bass'):  # only allow four-part chorales
                    # check consistency of key signatures of parts
                    key_sigs = []
                    for part in parts:
                        key_sigs.append(tuple(part.getElementsByClass(Measure)[0].keySignature.alteredPitches))
                    key_sigs = list(set(key_sigs))
                    if len(key_sigs) == 1 and riemenschneider not in ignore_list and allowed_timesig:
                        added = True
                        scores[riemenschneider] = {
                            'riemenschneider': riemenschneider,
                            'bwv': bwv_m21,
                            'score': score,
                            'scale': scale,
                            'modal': modal,
                            'key_sigs': list(key_sigs[0])}
                    # check the num measures
                    num_measures = len(parts[0].getElementsByClass(Measure))
            logger.info('|riemenschneider{}|bwv{}|{}|added={}'.format(
                riemenschneider, bwv_m21, num_measures, added))
        logger.info('Allowed time signature={}/371'.format(cnt_allowed_timesig))
        logger.info('Not duplicated={}/371'.format(cnt_not_duplicated))
        logger.info('Passed riemenschneiders={}/371'.format(len(scores)))
        return scores

    def _get_transposed_scores(self, score_item: Dict):
        score = score_item['score']
        key_sigs = score_item['key_sigs']
        if key_sigs:
            n_sigs = len(key_sigs)
            if key_sigs[0].name[-1] == '#':
                key_pitch = Pitch(SHARP_CIRCLE[n_sigs])
            else:
                key_pitch = Pitch(FLAT_CIRCLE[n_sigs])
        else:
            key_pitch = Pitch('C')

        transposed_scores = []
        intervals = []
        if self._key_preprocessing == KEY_PREPROCESS_NORMALIZE:
            interval = Interval(key_pitch, Pitch('C'))
            transposed_score = score.transpose(interval)
            self._check_measure_alignment(transposed_score)
            transposed_scores.append(transposed_score)
            intervals.append(interval.directedSimpleName)
        else:
            raise NotImplementedError

        return transposed_scores, intervals

    @staticmethod
    def _check_measure_alignment(transposed_score):
        assert (
            len(transposed_score.parts[0].getElementsByClass(Measure)) ==
            len(transposed_score.parts[1].getElementsByClass(Measure)) ==
            len(transposed_score.parts[2].getElementsByClass(Measure)) ==
            len(transposed_score.parts[3].getElementsByClass(Measure))
        )

    def _get_fermatas(self, score, chordify=True):
        if chordify:
            chords = score.flat.chordify().notes
            fermatas = []
            for i, chord in enumerate(chords):
                for e in chord.expressions:
                    if e.name == 'fermata':
                        assert i not in fermatas
                        fermatas.append(i)
            return fermatas
        else:
            raise NotImplementedError

    @staticmethod
    def _align_offset(part_notes_raw, part_rests_raw, chord_offsets):
        part_notes = [part_notes_raw[0]]
        for pn in part_notes_raw[1:]:
            if pn.offset == part_notes[-1].offset:
                # if the number of notes in the part > 1, only one note is allowed
                pass
            else:
                part_notes.append(pn)

        if part_rests_raw:
            part_rests = {}
            for pr in part_rests_raw:
                assert pr.offset not in part_rests
                part_rests[pr.offset] = pr.duration
        else:
            part_rests = {}

        def isRest(_offset):
            is_rest = False
            for pr_offset, pr_duration in part_rests.items():
                if pr_offset <= _offset <= pr_offset + pr_duration.quarterLength:
                    is_rest = True
                    break
            return is_rest

        aligned_part_offset = {}
        cnt = 0
        for offset, chord in chord_offsets.items():
            if len(part_notes) <= cnt:
                if isRest(offset):
                    aligned_part_offset[offset] = Rest(offset=offset, duration=chord.duration)
                else:
                    aligned_part_offset[offset] = Note(part_notes[cnt - 1].pitch, offset=offset, duration=chord.duration)
            else:
                if offset == part_notes[cnt].offset:
                    assert offset not in aligned_part_offset
                    aligned_part_offset[offset] = Note(part_notes[cnt].pitch, offset=offset, duration=chord.duration)
                    cnt += 1
                else:
                    assert offset < part_notes[cnt].offset
                    if isRest(offset):
                        aligned_part_offset[offset] = Rest(offset=offset, duration=chord.duration)
                    else:
                        aligned_part_offset[offset] = Note(part_notes[cnt - 1].pitch, offset=offset, duration=chord.duration)
        assert len(aligned_part_offset) == len(chord_offsets)
        return aligned_part_offset

    def _get_aligned_satbs(self, score, fermatas):
        chords = score.flat.chordify().getElementsByClass(Chord)

        fermata_metrics = [c.offset + c.duration.quarterLength for i, c in enumerate(chords) if i in fermatas]
        chord_offsets = dict([(c.offset, c) for c in chords])
        soprano_notes = score.parts[0].flat.getElementsByClass(Note)
        soprano_rests = score.parts[0].flat.getElementsByClass(Rest)
        alto_notes = score.parts[1].flat.getElementsByClass(Note)
        alto_rests = score.parts[1].flat.getElementsByClass(Rest)
        tenor_notes = score.parts[2].flat.getElementsByClass(Note)
        tenor_rests = score.parts[2].flat.getElementsByClass(Rest)
        bass_notes = score.parts[3].flat.getElementsByClass(Note)
        bass_rests = score.parts[3].flat.getElementsByClass(Rest)

        aligned_soprano = self._align_offset(soprano_notes, soprano_rests, chord_offsets)
        aligned_alto = self._align_offset(alto_notes, alto_rests, chord_offsets)
        aligned_tenor = self._align_offset(tenor_notes, tenor_rests, chord_offsets)
        aligned_bass = self._align_offset(bass_notes, bass_rests, chord_offsets)

        # append entire rests
        soprano_rests = dict([(pr.offset, pr.duration) for pr in soprano_rests])
        alto_rests = dict([(pr.offset, pr.duration) for pr in alto_rests])
        tenor_rests = dict([(pr.offset, pr.duration) for pr in tenor_rests])
        bass_rests = dict([(pr.offset, pr.duration) for pr in bass_rests])
        entire_rests_keys = set(soprano_rests.keys()) & set(alto_rests.keys()) & set(tenor_rests.keys()) & set(
            bass_rests.keys())
        entire_rests_keys = list(entire_rests_keys)
        for er in entire_rests_keys:
            assert soprano_rests[er] == alto_rests[er] == tenor_rests[er] == bass_rests[er]
            assert (er not in aligned_soprano) and (er not in aligned_alto) and (er not in aligned_tenor) and (
                        er not in aligned_bass)
            aligned_soprano[er] = Rest(offset=er, duration=soprano_rests[er])
            aligned_alto[er] = Rest(offset=er, duration=alto_rests[er])
            aligned_tenor[er] = Rest(offset=er, duration=tenor_rests[er])
            aligned_bass[er] = Rest(offset=er, duration=bass_rests[er])
            chord_offsets[er] = Rest(offset=er, duration=soprano_rests[er])

        return fermata_metrics, entire_rests_keys, chord_offsets, aligned_soprano, aligned_alto, aligned_tenor, aligned_bass

    def _get_chords(self, score, fermatas):
        fermata_metrics, entire_rests_keys, chord_offsets, aligned_soprano, aligned_alto, aligned_tenor, aligned_bass = \
            self._get_aligned_satbs(score, fermatas)

        all_chords = []
        all_durations = []
        all_pitch_histograms = []
        section_chords = []
        section_durations = []
        section_pitch_histogram = np.zeros(12).astype(float)
        chord_offsets = sorted(chord_offsets.items())
        for m, chord in chord_offsets:
            if m in fermata_metrics:
                assert bool(section_chords)
                assert bool(section_durations)
                assert 0.0 < np.sum(section_pitch_histogram)
                assert len(section_chords) == len(section_durations)
                all_chords.append(section_chords)
                all_durations.append(section_durations)
                all_pitch_histograms.append(section_pitch_histogram)
                section_chords = []
                section_durations = []
                section_pitch_histogram = np.zeros(12).astype(float)

            duration = chord.duration.quarterLength
            s = aligned_soprano[m]
            a = aligned_alto[m]
            t = aligned_tenor[m]
            b = aligned_bass[m]
            satb_set = sorted(list(set([n.pitch for n in [s, a, t, b] if not n.isRest])))
            chord_set = sorted(list(set([n.pitch for n in chord.notes]))) if isinstance(chord, Chord) else []
            if satb_set != chord_set:
                # there are very few cases satb_set do not match chord_set
                print("Overlook the mismatch: ", satb_set, chord_set)

            section_chords.append([s, a, t, b])
            section_durations.append(duration)
            for n in [s, a, t, b]:
                if not n.isRest:
                    section_pitch_histogram[n.pitch.pitchClass] += float(duration)

        if section_chords:
            assert bool(section_durations)
            assert len(section_chords) == len(section_durations)
            if np.sum(section_pitch_histogram) <= 0.0:
                for sc in section_chords:
                    for scp in sc:
                        assert scp.isRest
                # if all items are rest, don't append
            else:
                all_chords.append(section_chords)
                all_durations.append(section_durations)
                all_pitch_histograms.append(section_pitch_histogram)

        return all_chords, all_durations, all_pitch_histograms

    def _get_chords_with_metrical_structure(self, score, fermatas):
        timesig = score.parts[0].measure(2).getContextByClass('TimeSignature')
        fermata_metrics, entire_rests_keys, chord_offsets, aligned_soprano, aligned_alto, aligned_tenor, aligned_bass = \
            self._get_aligned_satbs(score, fermatas)

        assert aligned_soprano.keys() == aligned_alto.keys() == aligned_tenor.keys() == aligned_bass.keys()
        aligned_keys = sorted(list(aligned_soprano.keys()))
        last_duration_max = max(
            [aligned_soprano[aligned_keys[-1]].duration.quarterLength,
             aligned_alto[aligned_keys[-1]].duration.quarterLength,
             aligned_tenor[aligned_keys[-1]].duration.quarterLength,
             aligned_bass[aligned_keys[-1]].duration.quarterLength]
        )
        metric_max = int(math.ceil(aligned_keys[-1] + last_duration_max))
        assert METRIC_BEAT_RATIO == 0.25
        if timesig.denominator == 2:
            metric_rate = METRIC_BEAT_RATIO * 2
        elif timesig.denominator == 4:
            metric_rate = METRIC_BEAT_RATIO
        else:
            raise NotImplementedError
        max_mc = int(1.0 / metric_rate)

        all_metrics = []
        all_chords = []
        all_beats = []
        all_pitch_histograms = []
        section_chords = []
        section_metrics = []
        section_beats = []
        section_pitch_histograms = np.zeros(12).astype(float)
        ak_count = 0
        for m in range(metric_max):
            for mc in range(max_mc):
                metric = m + metric_rate * mc
                if metric in fermata_metrics:
                    assert bool(section_chords)
                    assert bool(section_metrics)
                    assert bool(section_beats)
                    assert 0.0 < np.sum(section_pitch_histograms)
                    assert len(section_chords) == len(section_metrics) == len(section_beats)
                    all_chords.append(section_chords)
                    all_metrics.append(section_metrics)
                    all_beats.append(section_beats)
                    all_pitch_histograms.append(section_pitch_histograms)
                    section_chords = []
                    section_metrics = []
                    section_beats = []
                    section_pitch_histograms = np.zeros(12).astype(float)

                if ak_count < (len(aligned_keys) - 1) and aligned_keys[ak_count + 1] <= metric:
                    if ak_count < (len(aligned_keys) - 2) and aligned_keys[ak_count + 2] <= metric:
                        # There are few cases the chord has less duration than metric rate
                        print('Ignored chord the duration of which is less than metric_rate: ',
                              score.metadata.title,
                              aligned_keys[ak_count + 1],
                              chord_offsets[aligned_keys[ak_count + 1]])
                        if ak_count < (len(aligned_keys) - 3):
                            assert metric < aligned_keys[ak_count + 3]
                        ak_count += 2
                    else:
                        ak_count += 1

                k = aligned_keys[ak_count]
                satb = [
                    aligned_soprano[k],
                    aligned_alto[k],
                    aligned_tenor[k],
                    aligned_bass[k]
                ]
                # check the alignment
                if k not in entire_rests_keys:
                    satb_set = sorted(list(set([n.pitch for n in satb if not n.isRest])))
                    chord_set = sorted(list(set([n.pitch for n in chord_offsets[k].notes])))
                    if satb_set != chord_set:
                        # there are very few cases satb_set do not match chord_set
                        print("Overlook the mismatch: ", satb_set, chord_set)
                section_chords.append(satb)
                section_metrics.append(metric)
                section_beats.append(score.beatAndMeasureFromOffset(metric)[0])
                for n in satb:
                    if not n.isRest:
                        section_pitch_histograms[n.pitch.pitchClass] += metric_rate

        if section_chords:
            assert bool(section_metrics)
            assert bool(section_beats)
            assert len(section_chords) == len(section_metrics) == len(section_beats)
            if np.sum(section_pitch_histograms) <= 0.0:
                for sc in section_chords:
                    for scp in sc:
                        assert scp.isRest
                # if all items are rest, don't append
            else:
                all_chords.append(section_chords)
                all_metrics.append(section_metrics)
                all_beats.append(section_beats)
                all_pitch_histograms.append(section_pitch_histograms)
            if metric + metric_rate != fermata_metrics[-1]:
                # there are very few cases missing last fermata
                print('Missing the last fermata: ', score.metadata.title)

        assert max([max(sb) for sb in all_beats]) < (timesig.numerator + 1)
        assert ak_count == len(aligned_keys) - 1
        assert len(all_chords) == len(all_metrics) == len(all_beats) == len(all_pitch_histograms)
        return all_chords, all_metrics, all_beats, all_pitch_histograms, metric_rate, timesig

    def _convert_to_binary_chord(self, chord):
        bin_chord = [0] * 12
        for p in chord:
            assert bin_chord[p] == 0
            bin_chord[p] = 1
        return bin_chord

    @staticmethod
    def apply_vocab(instances_raw, vocab):
        instances = []
        for instance in instances_raw:
            bin_chord_sets = instance['observation_binary'].to_tensor().tolist()
            sequence_length = instance['sequence_length'].to_tensor().item()
            chord_observations = []
            for i, bin_chord in enumerate(bin_chord_sets):
                if i == 0:
                    chord_observations.append(vocab.pad_index)  # start = PAD
                elif i < sequence_length:
                    chord = tuple(np.where(0 < np.array(bin_chord))[0].tolist())
                    if chord in vocab.c2i:
                        chord_observations.append(vocab.c2i[chord])
                    else:
                        chord_observations.append(vocab.unk_index)
                else:
                    chord_observations.append(vocab.pad_index)
            instance['observation_index'] = ListInstance(chord_observations)
            instances.append(instance)
        return instances

    def create_hmm_instance_and_vocab(self, logger: Logger, vocab: VocabChord):
        config = self._config
        logger.info('Create instances')
        scores = self._get_scores(logger)
        instances_raw = []
        for riemen, item in scores.items():
            score = item['score']
            bwv = item['bwv']
            fermatas = self._get_fermatas(score)
            transposed_scores, intervals = self._get_transposed_scores(item)
            for transposed_score, interval in zip(transposed_scores, intervals):
                satbs, durations, pitch_histograms = self._get_chords(transposed_score, fermatas)

                # vocab update
                for section_satbs, section_durations in zip(satbs, durations):
                    for satb, duration in zip(section_satbs, section_durations):
                        sorted_pitches = sorted(list(set([n.pitch.pitchClass for n in satb if not n.isRest])))
                        vocab.update_counts(sorted_pitches, duration)
                num_sections = len(satbs)
                # add instance
                for section_id, (section_satbs, section_durations, section_histogram) in enumerate(
                        zip(satbs, durations, pitch_histograms)):
                    # start symbol (PAD)
                    section_binary_pitches = [vocab.pad_binary_pitch_class]
                    section_chord_durations = [DURATIONS_SPECIAL_SYMBOL]
                    section_midi_numbers = [[MIDI_SPECIAL_INDEX] * 4]
                    section_pitch_classes = [[-1]]  # -1 is dummy
                    for duration, satb in zip(section_durations, section_satbs):
                        pitch_classes = sorted(list(set([n.pitch.pitchClass for n in satb if not n.isRest])))
                        if pitch_classes == section_pitch_classes[-1]:
                            section_chord_durations[-1] += duration
                        else:
                            section_pitch_classes.append(pitch_classes)
                            section_binary_pitches.append(self._convert_to_binary_chord(pitch_classes))
                            section_chord_durations.append(duration)
                            section_midi_numbers.append([n.pitch.midi if not n.isRest else MIDI_REST_INDEX for n in satb])

                    sequence_length = len(section_binary_pitches)
                    assert (sequence_length ==
                            len(section_binary_pitches) ==
                            len(section_midi_numbers) ==
                            len(section_chord_durations)), (
                        len(section_binary_pitches),
                        len(section_midi_numbers),
                        len(section_chord_durations))
                    if sequence_length <= config.max_sequence_length:
                        section_binary_pitches += \
                            [vocab.pad_binary_pitch_class for _ in range(config.max_sequence_length - sequence_length)]
                        section_chord_durations += \
                            [DURATIONS_PAD for _ in range(config.max_sequence_length - sequence_length)]
                        section_midi_numbers += \
                            [[MIDI_PAD_INDEX] * 4 for _ in range(config.max_sequence_length - sequence_length)]
                    else:
                        section_binary_pitches = section_binary_pitches[:config.max_sequence_length]
                        section_chord_durations = section_chord_durations[:config.max_sequence_length]
                        section_midi_numbers = section_midi_numbers[:config.max_sequence_length]

                    instance = {
                        'observation_binary': ListInstance(list_instances=section_binary_pitches),
                        'observation_midi': ListInstance(list_instances=section_midi_numbers),
                        'duration': ListInstance(list_instances=section_chord_durations),
                        'pitch_histogram': ListInstance(list_instances=section_histogram.tolist()),
                        'sequence_length': ValueInstance(sequence_length),
                        META_DATA: {
                            'reader_name': self.__class__.__name__,
                            'bwv': bwv,
                            'riemenschneider': riemen,
                            'key_sigs': item['key_sigs'],
                            'interval': interval,
                            'scale': item['scale'],
                            'modal': item['modal'],
                            'section_id': section_id,
                            'num_sections': num_sections
                        }
                    }
                    instances_raw.append(instance)

        vocab.fix_index()
        instances = self.apply_vocab(instances_raw, vocab)

        train_instances, dev_instances, test_instances = [], [], []
        train_pieces = []
        dev_pieces = []
        test_pieces = []
        counted_riemens = []
        for instance in instances:
            riemen = instance[META_DATA]['riemenschneider']
            if self._train_dev_test_dict[riemen] == TRAIN:
                train_instances.append(instance)
                if riemen not in counted_riemens:
                    counted_riemens.append(riemen)
                    train_pieces.append(riemen)
            elif self._train_dev_test_dict[riemen] == DEV:
                dev_instances.append(instance)
                if riemen not in counted_riemens:
                    counted_riemens.append(riemen)
                    dev_pieces.append(riemen)
            else:
                assert self._train_dev_test_dict[riemen] == TEST
                test_instances.append(instance)
                if riemen not in counted_riemens:
                    counted_riemens.append(riemen)
                    test_pieces.append(riemen)
        logger.info("{}-pieces={}, {}-pieces={}, {}-pieces={}".format(
            TRAIN, len(train_pieces), DEV, len(dev_pieces), TEST, len(test_pieces)))
        logger.info("{}-phrases={}, {}-phrases={}, {}-phrases={}".format(
            TRAIN, len(train_instances), DEV, len(dev_instances), TEST, len(test_instances)))
        return train_instances, dev_instances, test_instances

    def create_hmm_instance_and_vocab_with_metric(self, logger: Logger, vocab: VocabChord):
        config = self._config
        logger.info('Create instances')
        scores = self._get_scores(logger)
        instances_raw = []
        for riemen, item in scores.items():
            score = item['score']
            bwv = item['bwv']
            fermatas = self._get_fermatas(score)
            transposed_scores, intervals = self._get_transposed_scores(item)
            for transposed_score, interval in zip(transposed_scores, intervals):
                satbs, metrics, beats, pitch_histograms, metric_rate, timesig = \
                    self._get_chords_with_metrical_structure(transposed_score, fermatas)
                # vocab update
                for section_satbs, section_metrics in zip(satbs, metrics):
                    for satb in section_satbs:
                        sorted_pitches = sorted(list(set([n.pitch.pitchClass for n in satb if not n.isRest])))
                        vocab.update_counts(sorted_pitches, metric_rate)
                num_sections = len(satbs)
                # add instance
                for section_id, (section_satbs, section_metrics, section_beats, section_histogram) in enumerate(
                        zip(satbs, metrics, beats, pitch_histograms)):
                    # start symbol (PAD)
                    section_binary_pitches = [vocab.pad_binary_pitch_class]
                    section_chord_beats = [BEAT_SPECIAL_SYMBOL]
                    section_midi_numbers = [[MIDI_SPECIAL_INDEX] * 4]
                    section_pitch_classes = \
                        [sorted(list(set([n.pitch.pitchClass for n in satb if not n.isRest]))) for satb in section_satbs]
                    section_binary_pitches.extend(
                        [self._convert_to_binary_chord(pitch_classes) for pitch_classes in section_pitch_classes])
                    section_chord_beats.extend(section_beats)
                    for satb in section_satbs:
                        midi_numbers = [n.pitch.midi if not n.isRest else MIDI_REST_INDEX for n in satb]
                        section_midi_numbers.append(midi_numbers)

                    sequence_length = len(section_binary_pitches)
                    assert (sequence_length ==
                            len(section_binary_pitches) ==
                            len(section_midi_numbers) ==
                            len(section_chord_beats)), (
                        len(section_binary_pitches),
                        len(section_midi_numbers),
                        len(section_chord_beats))
                    if sequence_length <= config.max_sequence_length:
                        section_binary_pitches += \
                            [vocab.pad_binary_pitch_class for _ in range(config.max_sequence_length - sequence_length)]
                        section_midi_numbers += \
                            [[MIDI_PAD_INDEX] * 4 for _ in range(config.max_sequence_length - sequence_length)]
                        section_chord_beats += \
                            [BEAT_SPECIAL_SYMBOL for _ in range(config.max_sequence_length - sequence_length)]
                    else:
                        section_binary_pitches = section_binary_pitches[:config.max_sequence_length]
                        section_midi_numbers = section_midi_numbers[:config.max_sequence_length]
                        section_chord_beats = section_chord_beats[:config.max_sequence_length]

                    instance = {
                        'observation_binary': ListInstance(list_instances=section_binary_pitches),
                        'observation_midi': ListInstance(list_instances=section_midi_numbers),
                        'beat': ListInstance(list_instances=section_chord_beats),
                        'pitch_histogram': ListInstance(list_instances=section_histogram.tolist()),
                        'sequence_length': ValueInstance(sequence_length),
                        'timesignature_numerator': ValueInstance(int(timesig.numerator)),
                        'timesignature_denominator': ValueInstance(int(timesig.denominator)),
                        META_DATA: {
                            'reader_name': self.__class__.__name__,
                            'bwv': bwv,
                            'riemenschneider': riemen,
                            'key_sigs': item['key_sigs'],
                            'interval': interval,
                            'scale': item['scale'],
                            'modal': item['modal'],
                            'section_id': section_id,
                            'num_sections': num_sections,
                            'metric_rate': metric_rate
                        }
                    }
                    instances_raw.append(instance)

        vocab.fix_index()
        instances = self.apply_vocab(instances_raw, vocab)

        train_instances, dev_instances, test_instances = [], [], []
        train_pieces = []
        dev_pieces = []
        test_pieces = []
        counted_riemens = []
        for instance in instances:
            riemen = instance[META_DATA]['riemenschneider']
            if self._train_dev_test_dict[riemen] == TRAIN:
                train_instances.append(instance)
                if riemen not in counted_riemens:
                    counted_riemens.append(riemen)
                    train_pieces.append(riemen)
            elif self._train_dev_test_dict[riemen] == DEV:
                dev_instances.append(instance)
                if riemen not in counted_riemens:
                    counted_riemens.append(riemen)
                    dev_pieces.append(riemen)
            else:
                assert self._train_dev_test_dict[riemen] == TEST
                test_instances.append(instance)
                if riemen not in counted_riemens:
                    counted_riemens.append(riemen)
                    test_pieces.append(riemen)
        logger.info("{}-pieces={}, {}-pieces={}, {}-pieces={}".format(
            TRAIN, len(train_pieces), DEV, len(dev_pieces), TEST, len(test_pieces)))
        logger.info("{}-phrases={}, {}-phrases={}, {}-phrases={}".format(
            TRAIN, len(train_instances), DEV, len(dev_instances), TEST, len(test_instances)))
        return train_instances, dev_instances, test_instances
