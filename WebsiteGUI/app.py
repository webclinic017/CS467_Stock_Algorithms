from flask import Flask, render_template
import sys

sys.path.insert(1, '../Test-and-Training-Dataset-Creation')
sys.path.insert(1, '../Model-Training')
sys.path.insert(1, '../Model-Deployment')

import CreateTTData
import Training
import Deployment


app = Flask(__name__)


@app.route('/')
def home():  # put application's code here
    dataText = CreateTTData.createttdata()
    trainingText = Training.training()
    deploymentText = Deployment.deployment()
    return render_template("index.html",
                           dataText=dataText, trainingText=trainingText,
                           deploymentText=deploymentText)


if __name__ == '__main__':
    app.run()
