import numpy as np
from tkinter import Frame, ttk
import tkinter as tk
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
import math
import matplotlib.pyplot as plt
import random
import seaborn as sns
from scipy.spatial import distance


"""
Algoritmo

1. Inicialice el peso de cada nodo w_ij a un valor aleatorio
2. Seleccione un vector de entrada aleatorio x_k
3. Repita los puntos 4 y 5 para todos los nodos del mapa:
4. Calcule la distancia euclidiana entre el vector de entrada x(t) y el vector de peso w_ij asociado con el primer nodo, donde t, i, j = 0.
5. Rastree el nodo que produce la distancia más pequeña t .
6. Encuentre la mejor unidad de coincidencia (BMU) general, es decir, el nodo con la distancia más pequeña de todos los calculados.
7. Determinar el vecindario topológico βij(t) su radio σ(t) de BMU en el mapa de Kohonen
8. Repita para todos los nodos en el vecindario de la BMU: actualice el vector de peso w_ij del primer nodo en el vecindario de la BMU agregando una fracción de la diferencia entre el vector de entrada x(t) y el peso w(t) de la neurona.
9. Repetir toda esta iteración hasta alcanzar el límite de iteración elegido t=n

"""


def MouseSelect():

    global xneg, yneg, xreal, yreal, x_data, y_data, x_data_Array, y_data_Array

    xneg = False
    yneg = False

    xreal = 0
    xreal = 0

    def presion_mouse(event):

        global xneg, yneg, xreal, yreal, x_data, y_data, x_data_Array, y_data_Array

        xreal = event.x - 30
        yreal = event.y - 280

        if xreal < 0:
            xreal = abs(xreal)
            xneg = False
        else:
            xneg = True

        if yreal < 0:
            yreal = abs(yreal)
            yneg = False
        else:
            yneg = True

        if (yreal < 250 | xreal < 290) & (yneg == False) & (xneg == True):
            canvas1.create_oval(event.x-5, event.y-5, event.x +
                                5, event.y+5, fill="#009EFF")

            x_data.append(xreal)
            y_data.append(yreal)

    def mover_mouse(event):

        xreal = event.x - 30
        yreal = event.y - 280

        if xreal < 0:
            xreal = abs(xreal)
            xneg = False
        else:
            xneg = True

        if yreal < 0:
            yreal = abs(yreal)
            yneg = False
        else:
            yneg = True

        if (yreal < 250 | xreal < 290) & (yneg == False) & (xneg == True):
            window_MouseSelect.title("X = "+str(xreal)+" Y = "+str(yreal))

    def inportData():

        global x_data, y_data, x_data_Array, y_data_Array, numberCities

        x_data_Array = np.array(x_data)
        y_data_Array = np.array(y_data)

        x1.set(x_data[len(x_data)-1])
        y1.set(y_data[len(y_data)-1])

        print("x", x_data_Array)
        print("y", y_data_Array)

        numberCities = len(x_data)

        window_MouseSelect.destroy()

    window_MouseSelect = tk.Toplevel()
    window_MouseSelect.title("Window Mouse Select")
    window_MouseSelect.config(width=350, height=350)
    canvas1 = tk.Canvas(window_MouseSelect, width=350,
                        height=350, background="#C0EEF2")
    canvas1.create_rectangle(30, 30, 320, 280, fill="white",)

    canvas1.bind("<Motion>", mover_mouse)
    canvas1.bind("<Button-1>", presion_mouse)
    canvas1.grid(column=0, row=1)

    btn_Input = tk.Button(window_MouseSelect, width=12, text="Ready", background="#16FF00", foreground='black', font=(
        'Calibri', 11, 'bold'), command=inportData).place(x=50, y=300)
    btn_Cancel = tk.Button(window_MouseSelect, width=12, text="Cancel", background="#FF1700", foreground='black', font=(
        'Calibri', 11, 'bold'), command=window_MouseSelect.destroy).place(x=200, y=300)
    LabelSelect = ttk.Label(window_MouseSelect, text="Select with mouse", background="#C0EEF2",
                            foreground='black', font=('Calibri', 13, 'bold')).place(x=100, y=5)


def generateData():
    global x_data, y_data, numberCities, x_data_Array, y_data_Array, x1, y1

    numberCities = tk.simpledialog.askinteger(
        "Number of cities", "Enter the number of cities", parent=main_window)

    for i in range(numberCities):
        x_data.append(random.randrange(0, numberCities, 1))
        y_data.append(random.randrange(0, numberCities, 1))

    x_data_Array = np.array(x_data)
    y_data_Array = np.array(y_data)

    x1.set(numberCities)
    y1.set(numberCities)


def draw():
    global x_data_Array, y_data_Array, cities, x0_aux, y0_aux, x1_aux, y1_aux

    x0_aux = float(x0.get())
    y0_aux = float(y0.get())
    x1_aux = float(x1.get())
    y1_aux = float(y1.get())

    axes_train.plot(x_data_Array, y_data_Array,  'bx')
    axes_train.plot(x0_aux, y0_aux,  'gx')
    axes_train.plot(x1_aux, y1_aux,  'rx')
    figure_train.canvas.draw()

    x_data_Array = np.append(x0_aux, x_data_Array)
    x_data_Array = np.append(x_data_Array, x1_aux)
    y_data_Array = np.append(y0_aux, y_data_Array)
    y_data_Array = np.append(y_data_Array, y1_aux)

    cities = np.transpose(np.array([x_data_Array, y_data_Array]))


def graphTrain(M, option):

    axes_train.clear()
    axes_train.plot(x_data_Array, y_data_Array,  'bx')
    axes_train.plot(x0_aux, y0_aux,  'gx')
    axes_train.plot(x1_aux, y1_aux,  'rx')

    x = np.squeeze(np.asarray(M[:, 0]))
    y = np.squeeze(np.asarray(M[:, 1]))
    area = (30 * 0.5)**2

    if option == 1:
        axes_train.scatter(x, y, s=area, c='b', alpha=0.15, edgecolor="black")
        axes_train.plot(x, y, c='b')
    elif option == 2:
        axes_train.scatter(x, y, s=area, c='g', alpha=0.15, edgecolor="black")
        axes_train.plot(x, y, c='g')
    figure_train.canvas.draw()
    figure_train.canvas.flush_events()


def graphLength(L):
    axes_length.clear()
    axes_length.plot(L, c='g', linestyle='dashed')
    figure_length.canvas.draw()
    figure_length.canvas.flush_events()


def graphAlpha(alpha):
    axes_epsilon.clear()
    axes_epsilon.plot(alpha, c='r')
    figure_epsilon.canvas.draw()
    figure_epsilon.canvas.flush_events()


def train():

    global M, cities, x0_aux, y0_aux, x1_aux, y1_aux
    global stopPress

    numberEpoch = tk.simpledialog.askinteger(
        "Number of epochs", "Enter the number of epochs", parent=main_window, initialvalue=10000)

    Mcleaned = np.array([x0_aux, y0_aux])
    lengthDistanceGraph = []
    alphaGraph = []

    "------------ Initialization weights -------"
    M = np.matrix([[random.uniform(0, numberCities)
                  for j in range(2)] for i in range(2*numberCities)])

    N = M.shape[0]
    ratio0 = N
    Log = math.log10(M.shape[0])
    alpha0 = 0.5
    AuxN = np.arange(0, N, 1)

    epoch = 0

    M[0, 0] = x0_aux
    M[0, 1] = y0_aux
    M[M.shape[0]-1, 0] = x1_aux
    M[M.shape[0]-1, 1] = y1_aux

    while (epoch < numberEpoch):

        if stopPress == True:
            break
        else:
            ratio = ratio0*math.exp(-((epoch*Log)/1000))
            alpha = alpha0*math.exp(-(epoch/1000))

            distanceEuclidean = distance.cdist(cities, M)

            for i in range(cities.shape[0]):

                nodeMin = np.argmin(distanceEuclidean[i])
                distanceNode = np.zeros(N, dtype=np.int32)
                distanceNode[:] = nodeMin - AuxN
                beta = np.exp(-((distanceNode**2) / (2 * ratio**2)))

                M += alpha * beta[:, np.newaxis] * \
                    np.squeeze(np.asarray((cities[i, :] - M)))

            alphaGraph.append(alpha)

            epoch = epoch+1
            M[0, 0] = x0_aux
            M[0, 1] = y0_aux
            M[M.shape[0]-1, 0] = x1_aux
            M[M.shape[0]-1, 1] = y1_aux

            for i in range(M.shape[0]):
                if i == M.shape[0]-1:
                    pass
                else:
                    lengthDistanceGraph.append(np.sqrt((np.squeeze(np.asarray(M[i+1, 0])) - np.squeeze(np.asarray(
                        M[i, 0])))**2 + (np.squeeze(np.asarray(M[i+1, 1])) - np.squeeze(np.asarray(M[i, 1])))**2))

            if epoch % 200 == 0:
                graphTrain(M, 2)
                graphLength(lengthDistanceGraph)
                graphAlpha(alphaGraph)

    for m in range(M.shape[0]):
        city = np.squeeze(np.asarray(M[m, :]))
        for i in range(cities.shape[0]):
            if abs(city[0] - np.squeeze(np.asarray(cities[i, :]))[0]) <= 0.1 and abs(city[1] - np.squeeze(np.asarray(cities[i, :]))[1]) <= 0.1:
                Mcleaned = np.vstack(
                    (Mcleaned, np.squeeze(np.asarray(cities[i, :]))))

    Mcleaned = np.delete(Mcleaned, (0), axis=0)
    graphTrain(Mcleaned, 1)


def stop():
    global stopPress
    stopPress = True


def clear():

    global x_data, y_data, x_data_Array, y_data_Array, M, Mcleaned, alphaGraph, lengthDistanceGraph

    numberCities = 1

    x0.set(0.0)
    y0.set(0.0)
    x1.set(0.0)
    y1.set(0.0)

    axes_train.clear()
    axes_length.clear()
    axes_epsilon.clear()

    x_data, y_data = [], []

    y_data_Array = np.empty([1, numberCities])
    x_data_Array = np.empty([1, numberCities])

    M = np.empty([numberCities*2, 2])
    Mcleaned = np.array([x0_aux, y0_aux])
    lengthDistanceGraph = []
    alphaGraph = []


matplotlib.use('TkAgg')


class Application(ttk.Frame):

    def __init__(self, main_window):
        super().__init__(main_window)

        global axes_train, axes_length, axes_epsilon
        global figure_train, figure_length, figure_epsilon
        global entryX0

        main_window.title("IA Kohonen Application")
        main_window.configure(height=500, width=700)
        main_window.eval('tk::PlaceWindow . center')
        self.place(relwidth=1, relheight=1)

        # Frames
        self.frm1 = Frame(self, height=500, width=700,
                          background="#009EFF").place(x=0, y=0)
        self.frm2 = Frame(self, height=100, width=680,
                          background="#00E7FF").place(x=10, y=40)
        self.frm3 = Frame(self, height=340, width=380,
                          background="#C0EEF2").place(x=10, y=150)
        self.frm4 = Frame(self, height=340, width=290,
                          background="#00FFF6").place(x=400, y=150)
        self.frm5 = Frame(self, height=90, width=200,
                          background="#088395").place(x=20, y=45)
        self.frm6 = Frame(self, height=90, width=200,
                          background="#05BFDB").place(x=230, y=45)

        # Labels
        self.Label1 = ttk.Label(self, text="Sergio Andres Cardenas", background="#009EFF",
                                foreground='white', font=('Calibri', 16, 'bold')).place(x=20, y=5)
        self.Label2 = ttk.Label(self, text="Kohonen", background="#009EFF", foreground='black', font=(
            'Calibri', 16, 'bold')).place(x=600, y=5)
        self.Label3 = ttk.Label(self, text="Initial", background="#05BFDB", foreground='black', font=(
            'Calibri', 12, 'bold')).place(x=250, y=70)
        self.Label4 = ttk.Label(self, text="Final", background="#05BFDB", foreground='black', font=(
            'Calibri', 12, 'bold')).place(x=255, y=100)
        self.Label5 = ttk.Label(self, text="X", background="#05BFDB", foreground='black', font=(
            'Calibri', 11, 'bold')).place(x=335, y=46)
        self.Label6 = ttk.Label(self, text="Y", background="#05BFDB", foreground='black', font=(
            'Calibri', 11, 'bold')).place(x=395, y=46)
        self.Label6 = ttk.Label(self, text="Location", background="#05BFDB", foreground='white', font=(
            'Calibri', 14, 'bold')).place(x=240, y=46)

        # Entries
        entryX0 = ttk.Entry(textvariable=x0, justify='center', width=4,
                            foreground='black', font=('Calibri', 12)).place(x=320, y=70)
        self.entryY0 = ttk.Entry(textvariable=y0, justify='center', width=4,
                                 foreground='black', font=('Calibri', 12)).place(x=380, y=70)
        self.entryX1 = ttk.Entry(textvariable=x1, justify='center', width=4,
                                 foreground='black', font=('Calibri', 12)).place(x=320, y=100)
        self.entryY1 = ttk.Entry(textvariable=y1, justify='center', width=4,
                                 foreground='black', font=('Calibri', 12)).place(x=380, y=100)

        # Buttons
        self.btn_MouseSelect = tk.Button(self, width=21, text="Mouse Select", background="#ACFCD9", font=(
            'Calibri', 12, 'bold'), command=MouseSelect).place(x=32, y=55)
        self.btn_generate = tk.Button(self, width=21, text="Generate", background="#55D6BE", font=(
            'Calibri', 12, 'bold'), command=generateData).place(x=32, y=95)
        self.btn_draw = tk.Button(self, width=25, text="Draw", background="#009EFF", foreground='white', font=(
            'Calibri', 12, 'bold'), command=draw).place(x=460, y=55)
        self.btn_train = tk.Button(self, width=12, text="Train", background="#16FF00", foreground='black', font=(
            'Calibri', 11, 'bold'), command=train).place(x=30, y=160)
        self.btn_stop = tk.Button(self, width=12, text="Stop", background="#FF1700", foreground='white', font=(
            'Calibri', 11, 'bold'), command=stop).place(x=150, y=160)
        self.btn_clear = tk.Button(self, width=12, text="Clear", background="#00FFF6", foreground='black', font=(
            'Calibri', 11, 'bold'), command=clear).place(x=270, y=160)

        # Radio buttons
        self.button_modKOHONEN = tk.Radiobutton(self, text="KOHONEN", value=1, foreground='black', background="#00E7FF", font=(
            'Calibri', 12, 'bold'), variable=selectMod).place(x=450, y=100)
        self.button_modGENETICO = tk.Radiobutton(self, text="GENETIC", value=2, foreground='black', background="#00E7FF", font=(
            'Calibri', 12, 'bold'), variable=selectMod).place(x=580, y=100)

        # Plots

        # Train
        sns.set_theme()
        figure_train = Figure(figsize=(6, 4.7), dpi=60)
        figure_train_canvas = FigureCanvasTkAgg(figure_train)
        axes_train = figure_train.add_subplot()
        figure_train_canvas.get_tk_widget().place(x=21, y=200)
        axes_train.set_title("Cities", fontsize=20)
        axes_train.grid(True)

        # Length
        figure_length = Figure(figsize=(5, 2.9), dpi=55)
        figure_length_canvas = FigureCanvasTkAgg(figure_length)
        axes_length = figure_length.add_subplot()
        figure_length_canvas.get_tk_widget().place(x=407, y=155)
        axes_length.set_title("Length", fontsize=16)
        axes_length.grid(True)

        # toolbar = NavigationToolbar2Tk(
        #     figure_length_canvas, main_window, pack_toolbar=False)
        # toolbar.update()

        # Epsilon
        figure_epsilon = Figure(figsize=(5, 2.9), dpi=55)
        figure_epsilon_canvas = FigureCanvasTkAgg(figure_epsilon)
        axes_epsilon = figure_epsilon.add_subplot()
        figure_epsilon_canvas.get_tk_widget().place(x=407, y=320)
        axes_epsilon.set_title("Epsilon", fontsize=14)
        axes_epsilon.grid(True)


"""--------------------------------------------------------------------Initialization-------------------------------------------------------------------"""
# Variables for interface creation
main_window = tk.Tk()
selectMod = tk.IntVar()
stopPress = False
x_data, y_data = [], []
x0 = tk.DoubleVar()
y0 = tk.DoubleVar()
x1 = tk.DoubleVar()
y1 = tk.DoubleVar()
app = Application(main_window)
app.mainloop()
