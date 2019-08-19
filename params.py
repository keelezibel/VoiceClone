# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:10:34 2019

@author: Admin
"""

DATASETS_ROOT = "./Input/Speech/" #Each speaker should be inside a directory, 
                                #e.g. <datasets_root>/Speech/spk_01/audio_01.wav

ENC_MODEL_FPATH = "encoder/saved_models/pretrained.pt"

SYN_MODEL_DIR = "synthesizer/saved_models/logs-pretrained/"

VOC_MODEL_FPATH = "vocoder/saved_models/pretrained/pretrained.pt"

LOW_MEM = False # "If True, the memory used by the synthesizer will be freed after each use. Adds large "
                # "overhead but allows to save some GPU memory for lower-end GPUs."

INPUT_TEXT = "./Input/script.txt"

OUTPUT_DIR = "./Input/Synth/"
