## Check config
```
python bulk_synth.py (by default runs test_config())
``` 

## Dependencies


## Steps to run on Docker
1. Build docker image
docker build -t <reponame> .
e.g. docker build -t audiosynth .

2. Export docker image
docker save -o <path for generated tar file> <image name>
e.g. docker save -o ./audiosynth.tar audiosynth

3. Load tar image
docker load < <path to image tar file>
e.g. docker load < ./audiosynth.tar

2. Run docker image
docker run -p <port>:<port> --gpus all <reponame>
e.g. docker run -p 8000:8000 --gpus all audiosynth

