## Usage
Clone voice of a particular speaker (American English)

## Sample endpoint
http://localhost:8000/synth?audio=./Input/Speech/&text=./Input/script.txt&output=./Input/Synth/ <br/>
Params:
1. audio: Directory for input speech. Different folders for different speakers
2. text: File for text-to-speech input
3. output: Directory for synthesized file

## Check config
```
python bulk_synth.py (by default runs test_config())
``` 
## Download models
https://drive.google.com/drive/folders/1y-TwiaSIFn1STJnAd1uzHwagYKyyK9PK?usp=sharing

<br/>
Download the models in the link above. Copy to the project folder in the three folders: encoder, synthesizer and vocoder.


## Steps to run on Docker
1. Build docker image
```
docker build -t <reponame> .
```
e.g. docker build -t audiosynth .

2. Run docker image
```
docker run -d -v <host_path>:/Input/Synth -p <port>:<port> --gpus all <reponame>
```
e.g. docker run -d -v /home/user/Downloads/VoiceClone/Input/Synth:/Input/Synth -p 8000:8000 --gpus all audiosynth


