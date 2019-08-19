from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import torch
import sys
import sounddevice as sd
import glob
import os
import traceback
import params

class VoiceClone:
    
    def __init__(self,
                 audio=params.DATASETS_ROOT,
                 text=params.INPUT_TEXT,
                 output_dir=params.OUTPUT_DIR):
        
        sys.excepthook = self.excepthook
        self.datasets_root = audio
        self.enc_model_fpath = Path(params.ENC_MODEL_FPATH)
        self.syn_model_dir = Path(params.SYN_MODEL_DIR)
        self.voc_model_fpath = Path(params.VOC_MODEL_FPATH)
        self.low_mem = params.LOW_MEM
        self.synthesizer = None # type: Synthesizer
        
        # Added to point directory of input and output directories
        self.input_text = text
        self.output_dir = output_dir
                
    
    def excepthook(self, exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)
        self.ui.log("Exception: %s" % exc_value)
    
    def load_models(self):
        if not torch.cuda.is_available():
                print("Your PyTorch installation is not configured to use CUDA. If you have a GPU ready "
                      "for deep learning, ensure that the drivers are properly installed, and that your "
                      "CUDA version matches your PyTorch installation. CPU-only inference is currently "
                      "not supported.", file=sys.stderr)
                quit(-1)
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
              "%.1fGb total memory.\n" % 
              (torch.cuda.device_count(),
               device_id,
               gpu_properties.name,
               gpu_properties.major,
               gpu_properties.minor,
               gpu_properties.total_memory / 1e9))
    
    
        ## Load the models one by one.
        print("Preparing the encoder, the synthesizer and the vocoder...")
        encoder.load_model(self.enc_model_fpath)
        print("Loaded Encoder")
        self.synthesizer = Synthesizer(self.syn_model_dir.joinpath("taco_pretrained"), low_mem=self.low_mem)
        print("Loaded Synth")
        vocoder.load_model(self.voc_model_fpath)
        print("Loaded Vocoder")
        
    
    def test_config(self):
        ## Print some environment information (for debugging purposes)
        print("Running a test of your configuration...\n")
        try:
            if not torch.cuda.is_available():
                print("Your PyTorch installation is not configured to use CUDA. If you have a GPU ready "
                      "for deep learning, ensure that the drivers are properly installed, and that your "
                      "CUDA version matches your PyTorch installation. CPU-only inference is currently "
                      "not supported.", file=sys.stderr)
                quit(-1)
            device_id = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device_id)
            print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
                  "%.1fGb total memory.\n" % 
                  (torch.cuda.device_count(),
                   device_id,
                   gpu_properties.name,
                   gpu_properties.major,
                   gpu_properties.minor,
                   gpu_properties.total_memory / 1e9))
        
        
            ## Load the models one by one.
            print("Preparing the encoder, the synthesizer and the vocoder...")
            encoder.load_model(self.enc_model_fpath)
            print("Loaded Encoder")
            self.synthesizer = Synthesizer(self.syn_model_dir.joinpath("taco_pretrained"), low_mem=self.low_mem)
            print("Loaded Synth")
            vocoder.load_model(self.voc_model_fpath)
            print("Loaded Vocoder")
            
            ## Run a test
            print("Testing your configuration with small inputs.")
            # Forward an audio waveform of zeroes that lasts 1 second. Notice how we can get the encoder's
            # sampling rate, which may differ.
            # If you're unfamiliar with digital audio, know that it is encoded as an array of floats 
            # (or sometimes integers, but mostly floats in this projects) ranging from -1 to 1.
            # The sampling rate is the number of values (samples) recorded per second, it is set to
            # 16000 for the encoder. Creating an array of length <sampling_rate> will always correspond 
            # to an audio of 1 second.
            print("\tTesting the encoder...")
            encoder.embed_utterance(np.zeros(encoder.sampling_rate))
            
            # Create a dummy embedding. You would normally use the embedding that encoder.embed_utterance
            # returns, but here we're going to make one ourselves just for the sake of showing that it's
            # possible.
            embed = np.random.rand(speaker_embedding_size)
            # Embeddings are L2-normalized (this isn't important here, but if you want to make your own 
            # embeddings it will be).
            embed /= np.linalg.norm(embed)
            # The synthesizer can handle multiple inputs with batching. Let's create another embedding to 
            # illustrate that
            embeds = [embed, np.zeros(speaker_embedding_size)]
            texts = ["test 1", "test 2"]
            print("\tTesting the synthesizer... (loading the model will output a lot of text)")
            mels = self.synthesizer.synthesize_spectrograms(texts, embeds)
            
            # The vocoder synthesizes one waveform at a time, but it's more efficient for long ones. We 
            # can concatenate the mel spectrograms to a single one.
            mel = np.concatenate(mels, axis=1)
            # The vocoder can take a callback function to display the generation. More on that later. For 
            # now we'll simply hide it like this:
            no_action = lambda *args: None
            print("\tTesting the vocoder...")
            # For the sake of making this test short, we'll pass a short target length. The target length 
            # is the length of the wav segments that are processed in parallel. E.g. for audio sampled 
            # at 16000 Hertz, a target length of 8000 means that the target audio will be cut in chunks of
            # 0.5 seconds which will all be generated together. The parameters here are absurdly short, and 
            # that has a detrimental effect on the quality of the audio. The default parameters are 
            # recommended in general.
            vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)
            
            print("\tAll test passed!")
            
            return("All test passed!")
            
        except Exception as e:
            return("Caught exception: %s" % repr(e))
        
    def compute_embedding(self, spk_file):
        in_fpath = spk_file
                    
        ## Computing the embedding
        # First, we load the wav using the function that the speaker encoder provides. This is 
        # important: there is preprocessing that must be applied.
        
        # The following two methods are equivalent:
        # - Directly load from the filepath:
        preprocessed_wav = encoder.preprocess_wav(in_fpath)
        # - If the wav is already loaded:
        original_wav, sampling_rate = librosa.load(in_fpath)
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        print("Loaded file succesfully")
        
        # Then we derive the embedding. There are many functions and parameters that the 
        # speaker encoder interfaces. These are mostly for in-depth research. You will typically
        # only use this function (with its default parameters):
        embed = encoder.embed_utterance(preprocessed_wav)
        print("Created the embedding\n")
        
        return embed
    
    def parse_text(self):
        lineList = [line.rstrip('\n') for line in open(self.input_text)]
        return lineList
    
    def gen_spect(self, embed, text):
        # The synthesizer works in batch, so you need to put your data in a list or numpy array
        embeds = np.stack([embed] * len(text))
        specs = self.synthesizer.synthesize_spectrograms(text, embeds)
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)
        
        print("Created the mel spectrogram\n")
        
        return spec, breaks
    
    def vocode(self, spec, breaks):
        ## Generating the waveform
        print("Synthesizing the waveform:")
        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
        # spectrogram, the more time-efficient the vocoder.
        wav = vocoder.infer_waveform(spec)
        
        # Add breaks
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])
        
        ## Post-generation
        # There's a bug with sounddevice that makes the audio cut one second earlier, so we
        # pad it.
        wav = np.pad(wav, (0, self.synthesizer.sample_rate), mode="constant")
        return wav
    
    def save_to_disk(self, generated_wav, spk):
        # Save it on the disk
        fpath = "output_%s.wav" % spk
        out_path = os.path.join(self.output_dir,fpath)
        librosa.output.write_wav(out_path, generated_wav.astype(np.float32), 
                                 self.synthesizer.sample_rate)
        
        print("\nSaved output as %s\n\n" % fpath)

    def synt_speech(self):        
        print("Starting web service")
        #num_generated = 0
        try:
            # Load encoder, synthesizer and vocoder models
            print("Loading models...\n")
            self.load_models()
            
            # Load script into a list
            text = self.parse_text()
        
            # Get the reference audio filepath
            spk_folders = os.listdir(self.datasets_root)
            
            for spk in spk_folders:
                print("Processing Speaker: {}".format(spk))
                spk_dir = os.path.join(self.datasets_root,spk)
                input_dir = os.path.join(spk_dir,"*.wav")
                spk_files_list = glob.glob(input_dir)
                print("Total number of audio files in directory: {}\n".format(len(spk_files_list)))
                print(spk_files_list)

                for spk_file in spk_files_list:
                    embed = self.compute_embedding(spk_file)
                    spec, breaks = self.gen_spect(embed, text)
                    generated_wav = self.vocode(spec, breaks)
                    self.save_to_disk(generated_wav, spk)
                    
        except Exception as e:
            print("Caught exception: %s" % repr(e))
        
        return("Done. Processed: {} speakers".format(len(spk_folders)))
        


if __name__ == '__main__':
    #clone = VoiceClone(**vars(args))
    clone = VoiceClone()
    clone.test_config()
    #clone.synt_speech()
    
        