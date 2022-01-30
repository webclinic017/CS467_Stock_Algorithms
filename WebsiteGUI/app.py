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
        # Training Variables
        self.training_box_status = ""
        self.training_dataset = ""  # Dataset User Selected to train on
        self.model_name = ""  # Storage variable for user Name for model
        self.training_neurons_per_layer_select = ""
        self.training_number_of_layers_select = ""
        self.training_output_estimation_span_select = ""
        self.training_status_len = 0
        self.on_training_page_reload = ""
        self.training_loop = True
        self.training_status = False

        # Dataset Variables
        self.dataset_d_status = ""
        self.dataset_d_name_input = ""  # Storage variable for database name
        self.dataset_d_etf_select = ""  # ETF the user selected for the dataset
        self.dataset_checkbox_putt_call = "False"
        self.dataset_checkbox_junk_bond_demand = "False"
        self.dataset_checkbox_mcclellan_summation_index = "False"
        self.dataset_checkbox_lag_time = "False"
        self.dataset_lag_time_select = ""  # This depends on checkbox lag time
        self.dataset_status_len = ""
        self.on_dataset_page_reload = ""
        self.dataset_loop = True
        self.dataset_generation_status = False

    # -------- The methods below are for the Training page ---------
    # Updates text boxes with status and re-enables user input when 'Done' seen
    def training_status_box_update(self, t_status,
                                   t_ds_select, m_name_input,
                                   train_button,
                                   neurons_per_layer_select,
                                   number_of_layers_select,
                                   output_estimation_span_select):

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
            self.js.document.getElementById(t_status).innerHTML \
                = self.on_training_page_reload
            f.close()

            # Disable user input on the training page, if a refreshed happened
            # during training
            self.disable_training_user_input(
                t_ds_select, m_name_input,
                train_button,
                neurons_per_layer_select,
                number_of_layers_select,
                output_estimation_span_select)

            # Repopulate Dataset Selection on page reload and options
            self.js.document.getElementById(t_ds_select).value \
                = self.training_dataset

            self.js.document.getElementById(neurons_per_layer_select).value \
                = self.training_neurons_per_layer_select

            self.js.document.getElementById(number_of_layers_select).value \
                = self.training_number_of_layers_select

            self.js.document.getElementById(output_estimation_span_select)\
                .value = self.training_output_estimation_span_select

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
                        self.enable_training_user_input(
                                    t_ds_select,
                                    m_name_input,
                                    train_button,
                                    neurons_per_layer_select,
                                    number_of_layers_select,
                                    output_estimation_span_select)

                # Update the specified status through DOM manipulation
                self.js.document.getElementById(t_status).innerHTML \
                    = self.training_box_status

                # Make status scroll box scroll to bottom as it is updated
                self.js.document.getElementById(t_status).scrollTop = \
                    self.js.document.getElementById(t_status).scrollHeight

    # Store User's choices for training in a text file
    def store_user_training_choices(self, t_status,
                                    t_ds_select, m_name_input, train_button,
                                    neurons_per_layer_select,
                                    number_of_layers_select,
                                    output_estimation_span_select):

        # Grab the data from the webpage
        if self.js.document.getElementById(t_ds_select).value != 0:
            self.training_dataset = str(self.js.document
                                        .getElementById(t_ds_select).value)
            self.model_name = str(self.js.document
                                  .getElementById(m_name_input).value)
            self.training_neurons_per_layer_select = str(self.js.document
                                                         .getElementById(
                neurons_per_layer_select).value)
            self.training_number_of_layers_select = str(self.js.document
                                                        .getElementById(
                number_of_layers_select).value)
            self.training_output_estimation_span_select = str(self.js.document
                                                              .getElementById(
                output_estimation_span_select).value)

        # Set path to the training config file
        training_config_filepath = "../Model-Training/training_config.txt"

        # Label each choice
        model_name = "Model Name: " + self.model_name
        training_dataset = "Training Dataset: " + self.training_dataset[:-1]
        neurons_per_layer = "Neurons Per Layer: " + \
                            self.training_neurons_per_layer_select
        number_of_layers = "Number of Layers: " + \
                           self.training_number_of_layers_select
        output_estimation_span = "Estimation Span: " + \
                                 self.training_output_estimation_span_select

        # Put user choices into a list
        lines_to_write = [model_name, training_dataset, neurons_per_layer,
                          number_of_layers, output_estimation_span]

        # Erase the file
        open(training_config_filepath, 'w').close()

        # Write the list of variables to
        f = open(training_config_filepath, 'a')
        for i in range(len(lines_to_write)):
            f.write(lines_to_write[i] + "\n")
        f.close()

        # After user input has been stored called the update method
        self.training_status = True
        self.training_status_box_update(t_status,
                                        t_ds_select, m_name_input,
                                        train_button,
                                        neurons_per_layer_select,
                                        number_of_layers_select,
                                        output_estimation_span_select)

    # Training Setters
    # Disables training user input and stores inputted user choices
    def disable_training_user_input(self,
                                    t_ds_select, m_name_input,
                                    train_button,
                                    neurons_per_layer_select,
                                    number_of_layers_select,
                                    output_estimation_span_select):

        self.js.document.getElementById(t_ds_select).disabled = True
        self.js.document.getElementById(m_name_input).disabled = True
        self.js.document.getElementById(train_button).disabled = True
        self.js.document.getElementById(neurons_per_layer_select).disabled = \
            True
        self.js.document.getElementById(number_of_layers_select).disabled = \
            True
        self.js.document.getElementById(output_estimation_span_select) \
            .disabled = True

    # Enables training user input
    def enable_training_user_input(self,
                                   t_ds_select, m_name_input,
                                   train_button,
                                   neurons_per_layer_select,
                                   number_of_layers_select,
                                   output_estimation_span_select):

        self.js.document.getElementById(t_ds_select).disabled = False
        self.js.document.getElementById(m_name_input).disabled = False
        self.js.document.getElementById(train_button).disabled = False
        self.js.document.getElementById(neurons_per_layer_select).disabled = \
            False
        self.js.document.getElementById(number_of_layers_select).disabled = \
            False
        self.js.document.getElementById(output_estimation_span_select) \
            .disabled = False

    # -------- The methods below are for the Dataset Generation page ---------
    # Updates text boxes with status and re-enables user input when 'Done' seen
    def dataset_status_box_update(self, d_status,
                                  d_name_input,
                                  d_etf_select,
                                  checkbox_putt_call,
                                  checkbox_junk_bond_demand,
                                  checkbox_mcclellan_summation_index,
                                  checkbox_lag_time,
                                  lag_time_select,
                                  gen_dataset_button):

        self.on_dataset_page_reload = ""
        self.dataset_d_status = ""
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
            self.js.document.getElementById(d_status).innerHTML \
                = self.on_dataset_page_reload
            f.close()

            # Repopulate user defined input on page reload
            # Database Name
            self.js.document.getElementById(d_name_input).value \
                = self.dataset_d_name_input

            # ETF the user selected
            self.js.document.getElementById(d_etf_select).value \
                = self.dataset_d_etf_select

            # The checkbox for the putt/call option
            if self.dataset_checkbox_putt_call:
                self.js.document.getElementById(checkbox_putt_call).checked \
                    = True
            else:
                self.js.document.getElementById(checkbox_putt_call).checked \
                    = False

            # The checkbox for the junk bond demand option
            if self.dataset_checkbox_junk_bond_demand:
                self.js.document.getElementById(checkbox_junk_bond_demand). \
                    checked = True
            else:
                self.js.document.getElementById(checkbox_junk_bond_demand). \
                    checked = False

            # The checkbox for the mcclellan summation index option
            if self.dataset_checkbox_mcclellan_summation_index:
                self.js.document. \
                    getElementById(checkbox_mcclellan_summation_index).checked \
                    = True
            else:
                self.js.document. \
                    getElementById(checkbox_mcclellan_summation_index).checked \
                    = False

            # The checkbox for the lag time option
            if self.dataset_checkbox_lag_time:
                self.js.document.getElementById(checkbox_lag_time).checked \
                    = True
                self.js.document.getElementById(lag_time_select).value = \
                    self.dataset_lag_time_select
                print("LAG TIME SELECT: ", self.dataset_lag_time_select)
            else:
                self.js.document.getElementById(checkbox_lag_time).checked \
                    = False

            # Disable user input on the dataset page, if a refreshed happened
            # during dataset generation
            self.disable_dataset_user_input(d_name_input,
                                            d_etf_select,
                                            checkbox_putt_call,
                                            checkbox_junk_bond_demand,
                                            checkbox_mcclellan_summation_index,
                                            checkbox_lag_time,
                                            lag_time_select,
                                            gen_dataset_button)

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
                        self.dataset_d_status += lines[i]
                        self.dataset_status_len += 1

                    # Stop looping if Done is sent and reset everything
                    if lines[i] == 'Dataset Generation Complete':
                        self.dataset_d_status += '\n.......End..........'
                        print("Dataset Stop MATCH FOUND ________________")
                        # This deactivates dataset generation complete line by
                        # adding a new line, and also adds a
                        # separator to the dataset status file to separate
                        # dataset sessions
                        open(dataset_status_filepath, "w").close()
                        self.dataset_loop = False
                        self.dataset_generation_status = False
                        self.enable_dataset_user_input(
                            d_name_input,
                            d_etf_select,
                            checkbox_putt_call,
                            checkbox_junk_bond_demand,
                            checkbox_mcclellan_summation_index,
                            checkbox_lag_time,
                            lag_time_select,
                            gen_dataset_button)

                # Update the specified status through DOM manipulation
                self.js.document.getElementById(d_status).innerHTML \
                    = self.dataset_d_status

                # Make status scroll box scroll to bottom as it is updated
                self.js.document.getElementById(d_status).scrollTop = \
                    self.js.document.getElementById(d_status).scrollHeight

    # Store User's choices for dataset in a text file
    def store_user_dataset_choices(self,
                                   d_status,
                                   d_name_input,
                                   d_etf_select,
                                   checkbox_putt_call,
                                   checkbox_junk_bond_demand,
                                   checkbox_mcclellan_summation_index,
                                   checkbox_lag_time,
                                   lag_time_select,
                                   gen_dataset_button):

        # Grab the data from the webpage

        # Get the name of the dataset
        self.dataset_d_name_input = str(self.js.document.
                                        getElementById(d_name_input).value)

        # Get the user selected ETF name
        self.dataset_d_etf_select = str(self.js.document.
                                        getElementById(d_etf_select).value)

        # User Selected options section
        # Putt/Call option
        if self.js.document.getElementById(checkbox_putt_call).value \
                == "True":
            self.dataset_checkbox_putt_call = True
        else:
            self.dataset_checkbox_putt_call = False

        # Junk bond demand option
        if self.js.document.getElementById(checkbox_junk_bond_demand).value \
                == "True":
            self.dataset_checkbox_junk_bond_demand = True
        else:
            self.dataset_checkbox_junk_bond_demand = False

        # Mcclellan Summation Index option
        if self.js.document.getElementById(checkbox_mcclellan_summation_index) \
                .value == "True":
            self.dataset_checkbox_mcclellan_summation_index = True
        else:
            self.dataset_checkbox_mcclellan_summation_index = False

        # Lag Time option - the user selected lag time will be saved
        if self.js.document.getElementById(checkbox_lag_time) \
                .value == "True":
            self.dataset_checkbox_lag_time = True
            self.dataset_lag_time_select = str(self.js.document \
                                               .getElementById(
                lag_time_select).value)
        else:
            self.dataset_checkbox_lag_time = False

        # Set path to the dataset config file
        dataset_config_filepath = "../Dataset-Creation/dataset_config.txt"

        # Label each choice
        dataset_name = "Dataset Name: " + self.dataset_d_name_input
        etf_selected = "Etf Selected: " + self.dataset_d_etf_select[:-1]
        putt_call_option = "Putt/Call: " + str(self.dataset_checkbox_putt_call)

        junk_bond_demand_option = "Junk Bond Demand: " + \
                                  str(self.dataset_checkbox_junk_bond_demand)

        mcclellan_summation_index_option = \
            "Mcclellan Summation Index: " + \
            str(self.dataset_checkbox_mcclellan_summation_index)

        if self.dataset_checkbox_lag_time:
            lag_time_option = "Lag Time: " + str(self.dataset_lag_time_select)
        else:
            lag_time_option = "Lag Time: " + "False"

        # Put user choices into a list
        lines_to_write = [dataset_name, etf_selected, putt_call_option,
                          junk_bond_demand_option,
                          mcclellan_summation_index_option,
                          lag_time_option]

        # Erase the file
        open(dataset_config_filepath, 'w').close()

        # Write the list of variables to
        f = open(dataset_config_filepath, 'a')
        for i in range(len(lines_to_write)):
            f.write(lines_to_write[i] + "\n")
        f.close()

        # After user input has been stored call the update method
        self.dataset_generation_status = True
        self.dataset_status_box_update(d_status,
                                       d_name_input,
                                       d_etf_select,
                                       checkbox_putt_call,
                                       checkbox_junk_bond_demand,
                                       checkbox_mcclellan_summation_index,
                                       checkbox_lag_time,
                                       lag_time_select,
                                       gen_dataset_button)

    # Dataset Setters
    # Disables dataset user input and stores inputted user choices
    def disable_dataset_user_input(self,
                                   d_name_input,
                                   d_etf_select,
                                   checkbox_putt_call,
                                   checkbox_junk_bond_demand,
                                   checkbox_mcclellan_summation_index,
                                   checkbox_lag_time,
                                   lag_time_select,
                                   gen_dataset_button):

        self.js.document.getElementById(d_name_input).disabled = True
        self.js.document.getElementById(d_etf_select).disabled = True
        self.js.document.getElementById(checkbox_putt_call).disabled = True
        self.js.document.getElementById(checkbox_junk_bond_demand).disabled = \
            True
        self.js.document.getElementById(checkbox_mcclellan_summation_index). \
            disabled = True
        self.js.document.getElementById(checkbox_lag_time).disabled = True
        self.js.document.getElementById(lag_time_select).disabled = True
        self.js.document.getElementById(gen_dataset_button).disabled = True

    # Enables dataset user input
    def enable_dataset_user_input(self,
                                  d_name_input,
                                  d_etf_select,
                                  checkbox_putt_call,
                                  checkbox_junk_bond_demand,
                                  checkbox_mcclellan_summation_index,
                                  checkbox_lag_time,
                                  lag_time_select,
                                  gen_dataset_button):

        self.js.document.getElementById(d_name_input).disabled = False
        self.js.document.getElementById(d_etf_select).disabled = False
        self.js.document.getElementById(checkbox_putt_call).disabled = False
        self.js.document.getElementById(checkbox_junk_bond_demand).disabled = \
            False
        self.js.document.getElementById(checkbox_mcclellan_summation_index). \
            disabled = False
        self.js.document.getElementById(checkbox_lag_time).disabled = False
        self.js.document.getElementById(lag_time_select).disabled = False
        self.js.document.getElementById(gen_dataset_button).disabled = False


@app.route('/dataset')
def dataset():
    # Get the list of 3x ETF's
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

    # Create list of possible neurons per layer
    neurons_per_layer_values = ['5', '10', '20', '50']

    # Create list of possible number of layers
    number_of_layers_values = ['5', '10', '20', '50']

    # Create list of possible output estimation span
    output_estimation_span_values = ['1', '10', '30', '60', '90', '120', '150',
                                     '180', '210', '240', '270', '300', '330',
                                     '360']

    return Jyserverapp.render(render_template("training.html",
                                              datasets=datasets,
                                              neurons_per_layer_values
                                              =neurons_per_layer_values,
                                              number_of_layers_values
                                              =number_of_layers_values,
                                              output_estimation_span_values
                                              =output_estimation_span_values))


@app.route('/backtest')
def verification():  # put application's code here
    pass


if __name__ == '__main__':
    app.run()
