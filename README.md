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

## Steps to run on Docker
1. Build docker image
```
docker build -t <reponame> .
```
e.g. docker build -t audiosynth .

2. Run docker image
```
docker run -p <port>:<port> --gpus all <reponame>
```
e.g. docker run -p 8000:8000 --gpus all audiosynth

