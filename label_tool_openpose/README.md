### Requirements 

Pre req:
Anaconda installed

First, create a new conda env with python 3.9

```
conda create -n label_tool_py39 python=3.9
conda activate label_tool_py39
```

Then install dependencies to run the tool 

```
pip install -r requirements.txt
```

To run the tool 

```
python app.py --video-path <path to video>
```


To run splitter

```
python split_video_by_labels.py --video PATH_TO_VIDEO --annotations PATH_TO_JSON --outdir OUTPUT_DIRECTORY

```
