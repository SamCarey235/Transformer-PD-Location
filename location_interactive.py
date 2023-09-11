import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import Symbol, nsolve
import numpy as np
from scipy.optimize import least_squares
from scipy import optimize
import autograd.numpy as np
from autograd import jacobian
from scipy.optimize import root
from scipy.optimize import fsolve
from pathlib import Path
import time
import vallenae as vae

# Title
st.title("Transformer PD Location Estimation")

# Function to read the Excel file and store it in a cache
@st.cache_data
def read_excel_file(uploaded_file):
    return pd.read_excel(uploaded_file)

# Flag to keep track of whether the file has been uploaded and read
file_uploaded = False

# Initialize session state
if 'section' not in st.session_state:
    st.session_state.section = 'upload'

# File Upload
if not file_uploaded:
    uploaded_file = st.file_uploader("Upload Excel Data File", type=["xlsx"])
    if uploaded_file is not None:
        tr_data = read_excel_file(uploaded_file)
        file_uploaded = True

# Ask the user for the starting time of the data in ms
start_val = st.number_input("Enter the starting time of the data (ms):", min_value=0)

# User input for x1, y1, z1, and c
x1 = st.number_input("Enter x-coordinate for sensor 1 (mm):", min_value=0)/1000
y1 = st.number_input("Enter y-coordinate for sensor 1 (mm):", min_value=0)/1000
z1 = st.number_input("Enter z-coordinate for sensor 1 (mm):", min_value=0)/1000
x2 = st.number_input("Enter x-coordinate for sensor 2 (mm):", min_value=0)/1000
y2 = st.number_input("Enter y-coordinate for sensor 2 (mm):", min_value=0)/1000
z2 = st.number_input("Enter z-coordinate for sensor 2 (mm):", min_value=0)/1000
x3 = st.number_input("Enter x-coordinate for sensor 3 (mm):", min_value=0)/1000
y3 = st.number_input("Enter y-coordinate for sensor 3 (mm):", min_value=0)/1000
z3 = st.number_input("Enter z-coordinate for sensor 3 (mm):", min_value=0)/1000
x4 = st.number_input("Enter x-coordinate for sensor 4 (mm):", min_value=0)/1000
y4 = st.number_input("Enter y-coordinate for sensor 4 (mm):", min_value=0)/1000
z4 = st.number_input("Enter z-coordinate for sensor 4 (mm):", min_value=0)/1000
c = st.number_input("Enter speed of sound in oil (m/s):", min_value=0)

# Define the bounds and the initial guess
x_dim = st.number_input("Enter the length of the transformer in the x direction (mm):", min_value=0)/1000
y_dim = st.number_input("Enter the length of the transformer in the y direction (mm):", min_value=0)/1000
z_dim = st.number_input("Enter the length of the transformer in the z direction (mm):", min_value=0)/1000

# Calculate PD Times
if st.button("Calculate PD Times"):
    # Convert to arrays
    Time = tr_data["time"].to_numpy()
    coupling_capacitor = tr_data['coupling capacitor'].to_numpy()
    sensor_2 = tr_data["sensor 1"].to_numpy()
    sensor_3 = tr_data["sensor 2"].to_numpy()
    sensor_4 = tr_data["sensor 3"].to_numpy()
    sensor_5 = tr_data["sensor 4"].to_numpy()

    # Setting up the timepicker plot
    def plot(t_wave, y_wave, y_picker, index_picker, name_picker, title):
        _, ax1 = plt.subplots(figsize=(8, 4), tight_layout=True)
        ax1.set_xlabel("Time [ms]")
        ax1.set_ylabel("Amplitude [mV]", color="g")
        ax1.plot(t_wave, y_wave, color="g")
        ax1.tick_params(axis="y", labelcolor="g")

        ax2 = ax1.twinx()
        ax2.set_ylabel(f"{name_picker}", color="r")
        ax2.plot(t_wave, y_picker, color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        plt.axvline(t_wave[index_picker], color="k", linestyle=":")
        plt.title(title)

        return plt

    # PD Time Calculation
    def calculate_pd_time(y_wave, aic_arr, start_val):
        aic_index = np.argmin(aic_arr)
        PD_time = (aic_index + 100000 * start_val) / 100000
        return PD_time

    # Calculate timepicker values
    er_arrcc, er_indexcc = vae.timepicker.energy_ratio(coupling_capacitor)
    aic_arr2, aic_index2 = vae.timepicker.aic(sensor_2)
    aic_arr3, aic_index3 = vae.timepicker.aic(sensor_3)
    aic_arr4, aic_index4 = vae.timepicker.aic(sensor_4)
    aic_arr5, aic_index5 = vae.timepicker.aic(sensor_5)

    # Plot the signals with the calculated PD inception times
    fig1 = plot(Time, coupling_capacitor, er_arrcc, er_indexcc, "Energy Ratio", "Coupling Capacitor")
    st.pyplot(fig1)

    fig2 = plot(Time, sensor_2, aic_arr2, aic_index2, "Akaike Information Criterion", "Sensor 2")
    st.pyplot(fig2)

    fig3 = plot(Time, sensor_3, aic_arr3, aic_index3, "Akaike Information Criterion", "Sensor 3")
    st.pyplot(fig3)

    fig4 = plot(Time, sensor_4, aic_arr4, aic_index4, "Akaike Information Criterion", "Sensor 4")
    st.pyplot(fig4)

    fig5 = plot(Time, sensor_5, aic_arr5, aic_index5, "Akaike Information Criterion", "Sensor 5")
    st.pyplot(fig5)

    PD_timecc = (er_indexcc + 300000)/100000
    st.write(f"PD Time Coupling Capacitor: {PD_timecc} ms")

    PD_time2 = (aic_index2 + 100000*start_val)/100000
    st.write(f"PD Time Sensor 2: {PD_time2} ms")

    PD_time3 = (aic_index3 + 100000*start_val)/100000
    st.write(f"PD Time Sensor 3: {PD_time3} ms")

    PD_time4 = (aic_index4 + 100000*start_val)/100000
    st.write(f"PD Time Sensor 4: {PD_time4} ms")

    PD_time5 = (aic_index5 + 100000*start_val)/100000
    st.write(f"PD Time Sensor 5: {PD_time5} ms")

# Calculate PD Times
if st.button("Estimate PD Location"):

    # Convert to arrays
    Time = tr_data["time"].to_numpy()
    coupling_capacitor = tr_data['coupling capacitor'].to_numpy()
    sensor_2 = tr_data["sensor 1"].to_numpy()
    sensor_3 = tr_data["sensor 2"].to_numpy()
    sensor_4 = tr_data["sensor 3"].to_numpy()
    sensor_5 = tr_data["sensor 4"].to_numpy()

    # PD Time Calculation
    def calculate_pd_time(y_wave, aic_arr, start_val):
        aic_index = np.argmin(aic_arr)
        PD_time = (aic_index + 100000 * start_val) / 100000
        return PD_time

    # Calculate timepicker values
    er_arrcc, er_indexcc = vae.timepicker.energy_ratio(coupling_capacitor)
    aic_arr2, aic_index2 = vae.timepicker.aic(sensor_2)
    aic_arr3, aic_index3 = vae.timepicker.aic(sensor_3)
    aic_arr4, aic_index4 = vae.timepicker.aic(sensor_4)
    aic_arr5, aic_index5 = vae.timepicker.aic(sensor_5)

    PD_timecc = (er_indexcc + 300000)/100000
    PD_time2 = (aic_index2 + 100000*start_val)/100000
    PD_time3 = (aic_index3 + 100000*start_val)/100000
    PD_time4 = (aic_index4 + 100000*start_val)/100000
    PD_time5 = (aic_index5 + 100000*start_val)/100000

    # Constants
    T1 = (PD_time2 - PD_timecc) / 1000
    T2 = (PD_time3 - PD_timecc) / 1000
    T3 = (PD_time4 - PD_timecc) / 1000
    T4 = (PD_time5 - PD_timecc) / 1000

    x_bounds = (0, x_dim)
    y_bounds = (0, y_dim)
    z_bounds = (0, z_dim)
    initial_guess = (x_dim / 2, y_dim / 2, z_dim / 2)

    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')

    f1 = (x - x1)**2 + (y - y1)**2 + (z - z1)**2 - (c * T1)**2
    f2 = (x - x2)**2 + (y - y2)**2 + (z - z2)**2 - (c * T2)**2
    f3 = (x - x3)**2 + (y - y3)**2 + (z - z3)**2 - (c * T3)**2
    f4 = (x - x4)**2 + (y - y4)**2 + (z - z4)**2 - (c * T4)**2

    #st.error("Error: Solution did not converge - the results below are not valid. Please test again with sensors in different positions and validate these results by comparing them to the newly collected data")
    #solution = nsolve((f1, f2, f3, f4), (x, y, z), initial_guess, verify=False, solver='bisect')
    #st.write("x:", solution[0])
    #st.write("y:", solution[1])
    #st.write("z:", solution[2])

    try:
        solution = nsolve((f1, f2, f3, f4), (x, y, z), initial_guess, solver='bisect')*1000
        st.write("Solution converged. Coordinates of PD location in mm:")
        st.write("x:", solution[0])
        st.write("y:", solution[1])
        st.write("z:", solution[2])
    except ValueError:
        st.error("Error: Solution did not converge - the results below are not valid. Please test again with sensors in different positions and validate these results by\
                comparing them to the newly collected data. Coordinates of estimated PD location:")
        solution = nsolve((f1, f2, f3, f4), (x, y, z), initial_guess, verify=False, solver='bisect')*1000
        st.write("x:", solution[0])
        st.write("y:", solution[1])
        st.write("z:", solution[2])
