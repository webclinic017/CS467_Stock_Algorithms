from flask import Flask, render_template
import jyserver.Flask as jsf
import sys
import os
import time

sys.path.insert(1, '../Dataset-Creation')
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
        self.onPageReload = ""
        self.status_len = 0
        self.filepath = ""
        self.loop = True
        self.training_status = False
        self.dSelection = ""    # Storage variable for user dataset selection
        self.mNameInput = ""    # Storage variable for user Name for model


    # Updates text boxes with status and re-enables user input when 'Done' seen
    def training_status_box_update(self, status_box, t_ds_select,
                                   m_name_input, train_button):
        self.onPageReload = ""
        self.status = ""
        self.status_len = 0
        self.loop = True
        self.filepath = "../Model-Training/training_status.txt"


        # Check to see if training is already in progress and populate the
        # status box (in-case the user refreshed the browser) and disable
        # user input, if not clear the status box
        if self.training_status:
            print("IN IF STATMENT")
            f = open(self.filepath, "r")
            lines = f.readlines()
            for i in range(len(lines)):
                self.onPageReload += lines[i]
            # Re-Update the specified status through DOM manipulation
            self.js.document.getElementById(status_box).innerHTML \
                = self.onPageReload
            f.close()

            # Disable user input on the training page, if a refreshed happend
            # during training
            self.disable_training_user_input(t_ds_select,
                                            m_name_input, train_button)

            # Repopulate Dataset Selection on page reload
            self.js.document.getElementById(t_ds_select).value \
                = self.dSelection

            # Repopulate user defined Model Name on page reload
            self.js.document.getElementById(m_name_input).value \
                = self.mNameInput
        else:
            return None

        # This loop checks a given file for updates and puts contents to a
        # specific status text box on the webpage
        while self.loop:
            f = open(self.filepath, "r")
            lines = f.readlines()
            f.close()
            # If it's new information then append it to the status
            if len(lines) > self.status_len:
                for i in range(len(lines)):
                    # Check for file's change in number of lines
                    if i >= self.status_len:
                        self.status += lines[i]
                        self.status_len += 1

                    # Stop looping if Done is sent and reset everything
                    if lines[i] == 'Training Complete':
                        self.status += '\n.......End..........'

                        # This deactivates the training complete line by
                        # adding a new line, and also adds a
                        # separator to the training status file to separate
                        # training sessions
                        print('MATCH FOUND }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}')
                        open(self.filepath, "w").close()
                        self.loop = False
                        self.training_status = False
                        self.enable_training_user_input(t_ds_select,
                                   m_name_input, train_button)

                # Update the specified status through DOM manipulation
                self.js.document.getElementById(status_box).innerHTML \
                    = self.status

                # Make status scroll box scroll to bottom as it is updated
                self.js.document.getElementById(status_box).scrollTop = \
                    self.js.document.getElementById(status_box).scrollHeight

    # These are functions for the training page

    # Training Setters
    # Disables training user input and stores inputted user choices
    def disable_training_user_input(self, t_ds_select, m_name_input,
                                    train_button):
        self.training_status = True

        # Store Choices THESE CHOICES NEED TO BE WRITTEN TO A FILE BECAUSE
        # WHEN TRAIN IS CLICKED OR THE SCREEN REFRESHED ALL VARIABLES IN THE
        # CLASS ARE REINITIALIZED 
        if self.mNameInput == "" and self.dSelection == "":
            self.mNameInput = self.js.document.\
                getElementById(m_name_input).value
            self.dSelection = self.js.document.\
                getElementById(t_ds_select).value

        print("mNameInput------------->",self.mNameInput)
        print("dSelection------------->",self.dSelection)

        self.js.document.getElementById(t_ds_select).disabled = True
        self.js.document.getElementById(m_name_input).disabled = True
        self.js.document.getElementById(train_button).disabled = True

    # Enables training user input
    def enable_training_user_input(self, t_ds_select, m_name_input,
                                   train_button):
        self.training_status = False
        self.js.document.getElementById(t_ds_select).disabled = False
        self.js.document.getElementById(m_name_input).disabled = False
        self.js.document.getElementById(train_button).disabled = False

    # Training Getters
    def get_training_status(self):
        return self.training_status


@app.route('/dataset')
def dataset():  # put application's code here
    data_text = CreateTTData.createttdata()
    training_text = Training.training()
    deployment_text = Deployment.deployment()
    return Jyserverapp.render(render_template("dataset.html",
                                              data_text=data_text,
                                              training_text=training_text,
                                              deployment_text=deployment_text))


@app.route('/training')
def training():  # put application's code here
    # data_text = CreateTTData.createttdata()
    # training_text = Training.training()
    # deployment_text = Deployment.deployment()

    # Get the list of availible datasets
    dataset_filepath = "../Dataset-Creation/datasets_list.txt"
    dataset_list = open(dataset_filepath, 'r')
    lines = dataset_list.readlines()
    dataset_list.close()
    datasets = []
    for i in range(len(lines)):
        datasets.append(lines[i])
    return Jyserverapp.render(render_template("training.html",
                                              datasets=datasets))


@app.route('/backtest')
def backtest():  # put application's code here
    data_text = CreateTTData.createttdata()
    training_text = Training.training()
    deployment_text = Deployment.deployment()
    return Jyserverapp.render(render_template("backtest.html",
                                              data_text=data_text,
                                              training_text=training_text,
                                              deployment_text=deployment_text))


if __name__ == '__main__':
    app.run()
