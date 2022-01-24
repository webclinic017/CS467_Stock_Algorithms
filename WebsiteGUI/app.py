from flask import Flask, render_template
import jyserver.Flask as jsf
import sys

sys.path.insert(1, '../Test-and-Training-Dataset-Creation')
sys.path.insert(1, '../Model-Training')
sys.path.insert(1, '../Model-Deployment')

import CreateTTData
import Training
import Deployment


app = Flask(__name__)


@jsf.use(app) # This is a module named jyserver that allows manipulation of DOM
class Jyserverapp:
    def __init__(self): # Create an __init__ method
        self.tr_status = ""
        self.tr_status_len = 0

    def status_update(self): # Increment method for DOM
        print("Updating")
        f = open("../Model-Training/training_status.txt")
        lines = f.readlines()
        print("tr_status: %d", len(self.tr_status))
        print("tr_status_len: %d", self.tr_status_len)
        print("Lines length is: %d", len(lines))
        # if it's new information then append it to the status
        if len(lines) > self.tr_status_len:
            for i in range(len(lines)):
                if i > self.tr_status_len:
                    print("HERE")
                    self.tr_status += lines[i]
                    self.js.document.getElementById("t_status").innerHTML =\
                        self.tr_status
            self.tr_status_len = len(lines)
            self.js.document.getElementById("t_status").scrollTop = \
                self.js.document.getElementById("t_status").scrollHeight

@app.route('/')
def index():  # put application's code here
    data_text = CreateTTData.createttdata()
    training_text = Training.training()
    deployment_text = Deployment.deployment()
    return Jyserverapp.render(render_template("index.html",
                           data_text=data_text, training_text=training_text,
                           deployment_text=deployment_text))


if __name__ == '__main__':
    app.run()
