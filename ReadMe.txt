conda env list

conda activate u8what

export NSFW_MODEL_PATH=/Users/mini2018/projects/u8what-backend/models/nsfw_mobilenet2.224x224.h5
export NSFW_THRESHOLD=0.82   # tweak if you want

FLASK_APP=app.py flask run --port=5001
