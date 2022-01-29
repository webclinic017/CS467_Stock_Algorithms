from flask import Flask, render_template
import jyserver.Flask as jsf
import sys
import os
import time

sys.path.insert(1, '../Dataset-Creation')
sys.path.insert(1, '../Model-Training')
sys.path.insert(1, '../Model-Verification')

# Importing each python program and running them (this will change)
import CreateTTData
import Training
import Deployment

app = Flask(__name__)


@jsf.use(
    app)  # This is a module named jyserver that allows manipulation of DOM
class Jyserverapp:
    def __init__(self):  # Create an __init__ method
        self.training_box_status = ""
        self.training_status_len = 0
        self.on_training_page_reload = ""
        self.training_loop = True
        self.training_status = False
        self.training_dataset = ""  # Dataset User Selected to train on
        self.model_name = ""  # Storage variable for user Name for model

        self.dataset_box_status = ""
        self.dataset_status_len = ""
        self.on_dataset_page_reload = ""
        self.dataset_loop = True
        self.dataset_generation_status = False
        self.dataset_etf = ""  # ETF the user selected for dataset generation
        self.dataset_name = ""  # Storage variable for user Name for model

    # -------- The methods below are for the Training page ---------
    # Updates text boxes with status and re-enables user input when 'Done' seen
    def training_status_box_update(self, status_box, t_ds_select,
                                   m_name_input, train_button):

        self.on_training_page_reload = ""
        self.training_box_status = ""
        self.training_status_len = 0
        self.training_loop = True
        training_status_filepath = "../Model-Training/training_status.txt"

        # Check to see if training is already in progress and populate the
        # status box (in-case the user refreshed the browser) and disable
        # user input, if not clear the status box
        if self.training_status:
            f = open(training_status_filepath, "r")
            lines = f.readlines()
            for i in range(len(lines)):
                self.on_training_page_reload += lines[i]
            # Re-Update the specified status through DOM manipulation
            self.js.document.getElementById(status_box).innerHTML \
                = self.on_training_page_reload
            f.close()

            # Disable user input on the training page, if a refreshed happend
            # during training
            self.disable_training_user_input(t_ds_select,
                                             m_name_input, train_button)

            # Repopulate Dataset Selection on page reload
            self.js.document.getElementById(t_ds_select).value \
                = self.training_dataset

            # Repopulate user defined Model Name on page reload
            self.js.document.getElementById(m_name_input).value \
                = self.model_name
        else:
            return None

        # This loop checks a given file for updates and puts contents to a
        # specific status text box on the webpage

        while self.training_loop:
            f = open(training_status_filepath, "r")
            lines = f.readlines()
            f.close()
            # If it's new information then append it to the status
            if len(lines) > self.training_status_len:
                for i in range(len(lines)):
                    # Check for file's change in number of lines
                    if i >= self.training_status_len:
                        self.training_box_status += lines[i]
                        self.training_status_len += 1
                    # Stop looping if Done is sent and reset everything
                    if lines[i] == 'Training Complete':
                        self.training_box_status += '\n.......End..........'
                        print("MATCH FOUND ________________")
                        # This deactivates the training complete line by
                        # adding a new line, and also adds a
                        # separator to the training status file to separate
                        # training sessions
                        open(training_status_filepath, "w").close()
                        self.training_loop = False
                        self.training_status = False
                        self.enable_training_user_input(t_ds_select,
                                                        m_name_input,
                                                        train_button)

                # Update the specified status through DOM manipulation
                self.js.document.getElementById(status_box).innerHTML \
                    = self.training_box_status

                # Make status scroll box scroll to bottom as it is updated
                self.js.document.getElementById(status_box).scrollTop = \
                    self.js.document.getElementById(status_box).scrollHeight

    # Store User's choices for training in a text file
    def store_user_training_choices(self, status_box, t_ds_select,
                                    m_name_input, train_button):
        # Grab the data from the webpage
        if self.js.document.getElementById(t_ds_select).value != 0:
            self.training_dataset = str(self.js.document
                                        .getElementById(t_ds_select).value)
            self.model_name = str(self.js.document
                                  .getElementById(m_name_input).value)

        # Set path to the training config file
        training_config_filepath = "../Model-Training/training_config.txt"

        # Label each choice
        model_name = "Model Name: " + self.model_name
        training_dataset = "Training Dataset: " + self.training_dataset

        # Put user choices into a list
        lines_to_write = [model_name, training_dataset]

        # Erase the file
        open(training_config_filepath, 'w').close()

        # Write the list of variables to
        f = open(training_config_filepath, 'a')
        for i in range(len(lines_to_write)):
            f.write(lines_to_write[i] + "\n")
        f.close()

        # After user input has been stored called the update method
        self.training_status = True
        self.training_status_box_update(status_box, t_ds_select,
                                        m_name_input, train_button)

    # Training Setters
    # Disables training user input and stores inputted user choices
    def disable_training_user_input(self, t_ds_select, m_name_input,
                                    train_button):
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

    # -------- The methods below are for the Dataset Generation page ---------
    # Updates text boxes with status and re-enables user input when 'Done' seen
    def dataset_status_box_update(self, status_box, t_ds_select,
                                  m_name_input, train_button):
        self.on_dataset_page_reload = ""
        self.dataset_box_status = ""
        self.dataset_status_len = 0
        self.dataset_loop = True
        dataset_status_filepath = "../Dataset-Creation/dataset_status.txt"

        # Check to see if dataset generation is already in progress and
        # populate the status box (in-case the user refreshed the browser)
        # and disable user input, if not clear the status box
        if self.dataset_generation_status:
            f = open(dataset_status_filepath, "r")
            lines = f.readlines()
            for i in range(len(lines)):
                self.on_dataset_page_reload += lines[i]
            # Re-Update the specified status through DOM manipulation
            self.js.document.getElementById(status_box).innerHTML \
                = self.on_dataset_page_reload
            f.close()

            # Disable user input on the dataset page, if a refreshed happened
            # during dataset generation
            self.disable_dataset_user_input(t_ds_select,
                                            m_name_input, train_button)

            # Repopulate user defined Model Name on page reload
            self.js.document.getElementById(m_name_input).value \
                = self.dataset_name
        else:
            return None

        # This loop checks a given file for updates and puts contents to a
        # specific status text box on the webpage
        while self.dataset_loop:
            f = open(dataset_status_filepath, "r")
            lines = f.readlines()
            f.close()
            # If it's new information then append it to the status
            if len(lines) > self.dataset_status_len:
                for i in range(len(lines)):
                    # Check for file's change in number of lines
                    if i >= self.dataset_status_len:
                        self.dataset_box_status += lines[i]
                        self.dataset_status_len += 1

                    # Stop looping if Done is sent and reset everything
                    if lines[i] == 'Dataset Generation Complete':
                        self.dataset_box_status += '\n.......End..........'
                        print("Dataset Stop MATCH FOUND ________________")
                        # This deactivates dataset generation complete line by
                        # adding a new line, and also adds a
                        # separator to the dataset status file to separate
                        # dataset sessions
                        open(dataset_status_filepath, "w").close()
                        self.dataset_loop = False
                        self.dataset_generation_status = False
                        self.enable_dataset_user_input(t_ds_select,
                                                       m_name_input,
                                                       train_button)

                # Update the specified status through DOM manipulation
                self.js.document.getElementById(status_box).innerHTML \
                    = self.dataset_box_status

                # Make status scroll box scroll to bottom as it is updated
                self.js.document.getElementById(status_box).scrollTop = \
                    self.js.document.getElementById(status_box).scrollHeight

    # Store User's choices for dataset in a text file
    def store_user_dataset_choices(self, status_box, t_ds_select,
                                   m_name_input, train_button):

        # Grab the data from the webpage
        if self.js.document.getElementById(t_ds_select).value != 0:
            self.dataset_name = str(self.js.document
                                    .getElementById(m_name_input).value)

        # Set path to the dataset config file
        dataset_config_filepath = "../Dataset-Creation/dataset_config.txt"

        # Label each choice
        dataset_name = "Dataset Name: " + self.model_name

        # Put user choices into a list
        lines_to_write = [dataset_name]

        # Erase the file
        open(dataset_config_filepath, 'w').close()

        # Write the list of variables to
        f = open(dataset_config_filepath, 'a')
        for i in range(len(lines_to_write)):
            f.write(lines_to_write[i] + "\n")
        f.close()

        # After user input has been stored called the update method
        self.dataset_generation_status = True
        self.dataset_status_box_update(status_box, t_ds_select,
                                       m_name_input, train_button)

    # Dataset Setters
    # Disables dataset user input and stores inputted user choices
    def disable_dataset_user_input(self, t_ds_select, m_name_input,
                                   train_button):
        self.js.document.getElementById(t_ds_select).disabled = True
        self.js.document.getElementById(m_name_input).disabled = True
        self.js.document.getElementById(train_button).disabled = True

    # Enables dataset user input
    def enable_dataset_user_input(self, t_ds_select, m_name_input,
                                  train_button):
        self.dataset_generation_status = False
        self.js.document.getElementById(t_ds_select).disabled = False
        self.js.document.getElementById(m_name_input).disabled = False
        self.js.document.getElementById(train_button).disabled = False


@app.route('/dataset')
def dataset():
    # Get the list of availible datasets
    etf_filepath = "../Dataset-Creation/etf_list.txt"
    etf_list = open(etf_filepath, 'r')
    lines = etf_list.readlines()
    etf_list.close()
    etfs = []
    for i in range(len(lines)):
        etfs.append(lines[i])

    # Generate list of lag time values
    lag_time_values = [10, 30, 60, 90]

    return Jyserverapp.render(render_template("dataset.html",
                                              etfs=etfs,
                                              lag_time_values=lag_time_values))


@app.route('/training')
def training():
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
def verification():  # put application's code here
    pass


if __name__ == '__main__':
    app.run()
