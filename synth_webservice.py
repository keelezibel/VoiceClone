# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 08:49:29 2019

API for webservice

eg. waitress-serve --listen 0.0.0.0:8000 synth_webservice:__hug_wsgi__

http://localhost:8000/synth?audio=./Input/Speech/&text=./Input/script.txt&output=./Input/Synth/

@author: BLT

"""

import hug
from bulk_synth import VoiceClone
import params

# Web service to synthesize bulk
@hug.cli()
@hug.get(examples="audio=./Input/Speech/&text=./Input/script.txt&output=./Input/Synth/")
@hug.post()
@hug.local()
def synth(audio: hug.types.text, text: hug.types.text, output: hug.types.text, hug_timer=3):
    # Synthesize wav based on target audio and text file        
    clone = VoiceClone(audio=audio,text=text,output_dir=output)    
    out_msg = clone.synt_speech()

    return {'out_msg': out_msg, 'took': float(hug_timer)}

# Web service to test configuration
@hug.cli()
@hug.get()
@hug.post()
@hug.local()
def test_config(hug_timer=3):
    clone = VoiceClone()
    out_msg = clone.test_config()
    
    return {'out_msg': out_msg, 'took': float(hug_timer)}



