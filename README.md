

<img align="left" width="500"  src="figures/vrai-logo-b.png" style="margin-right:-230px"></br></br></br></br>
<h1> Code Computer & Geoscience paper</h1>
</br>


### Requirements

This project requires **Python 3.8** or alternatively Docker. Using Docker, the steps are:
Build the Docker image and launch the container:
```docker build -t gwl .docker run -it --gpus 'device=<GPU_ID>' --shm-size=64g  -v $(pwd)/<YOUR-PATH-FOLDER>:/gwl gwl-container ```
Then, to train the models, after connecting the Docker qwl-container, launch the *main.py* file. At the beginning of the file, specify the 
model to be trained and its configuration settings on which *box* to train and the depth. All other settings are specified in *models/model.json*.
```python main.py```