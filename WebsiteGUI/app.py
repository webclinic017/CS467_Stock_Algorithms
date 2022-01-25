from flask import Flask, render_template
import jyserver.Flask as jsf
import sys
import os
import time

sys.path.insert(1, '../Test-and-Training-Dataset-Creation')
sys.path.insert(1, '../Model-Training')
sys.path.insert(1, '../Model-Deployment')

# Importing each python program and running them (this will change)
import CreateTTData
import Training
import Deployment

app = Flask(__name__)


@jsf.use(
    app)  # This is a module named jyserver that allows manipulation of DOM
class Jyserverapp:
    def __init__(self):  # Create an __init__ method
        self.status = ""
        self.status_len = 0
        self.filepath = ""
        self.loop = True

    def status_update(self, status_box):  # Increment method for DOM
        self.status = ""
        self.status_len = 0
        self.loop = True

        # This checks to see which status box is being updated
        if status_box == 't_status':
            self.filepath = "../Model-Training/training_status.txt"

        # This loop checks a given file for updates and puts contents to a
        # specific status text box on the webpage
        while self.loop:
            f = open(self.filepath, "r")
            lines = f.readlines()

            # If it's new information then append it to the status
            if len(lines) > self.status_len:
                for i in range(len(lines)):

                    # Check for file's change in number of lines
                    if i >= self.status_len:
                        self.status += lines[i]
                        self.status_len += 1

                        # Update the specified status through DOM manipulation
                        self.js.document.getElementById(status_box).innerHTML \
                            = self.status

                    # Stop looping if Done is sent and reset everything
                    if lines[i] == 'Done':
                        open(self.filepath, 'w').close()
                        self.status = ""
                        self.status_len = 0
                        self.loop = False

                # Make status scroll box scroll to bottom as it is updated
                self.js.document.getElementById(status_box).scrollTop = \
                    self.js.document.getElementById(status_box).scrollHeight
            f.close()


@app.route('/training')
def training():  # put application's code here
    data_text = CreateTTData.createttdata()
    training_text = Training.training()
    deployment_text = Deployment.deployment()
    return Jyserverapp.render(render_template("training.html",
                                              data_text=data_text,
                                              training_text=training_text,
                                              deployment_text=deployment_text))


if __name__ == '__main__':
    app.run()
