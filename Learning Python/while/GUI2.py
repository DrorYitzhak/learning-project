import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd

root = tk.Tk()

canvas1 = tk.Canvas(root, width=800, height=300)
canvas1.pack()

label1 = tk.Label(root, text='Data Analyser')
label1.config(font=('Arial', 20))
canvas1.create_window(400, 50, window=label1)


def getExcel():
    global df

    import_file_path = filedialog.askopenfilename()
    df = pd.read_excel(import_file_path)
    global bar1
    x = df['x']
    y = df['y']

    figure1 = Figure(figsize=(4, 3), dpi=100)
    subplot1 = figure1.add_subplot(111)
    subplot1.bar(x, y, color='lightsteelblue')
    bar1 = FigureCanvasTkAgg(figure1, root)
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=0)
    subplot1.plot(x, y, color='green', linestyle='dashed', linewidth=3, marker='o', markerfacecolor='blue',
                  markersize=12)


def clear_charts():
    bar1.get_tk_widget().pack_forget()


browseButton_Excel = tk.Button(text='Load File...', command=getExcel, bg='green', fg='white',
                               font=('helvetica', 12, 'bold'))
canvas1.create_window(400, 180, window=browseButton_Excel)

button2 = tk.Button(root, text='Clear Chart', command=clear_charts, bg='green', font=('helvetica', 11, 'bold'))
canvas1.create_window(400, 220, window=button2)

button3 = tk.Button(root, text='Exit!', command=root.destroy, bg='green', font=('helvetica', 11, 'bold'))
canvas1.create_window(400, 260, window=button3)

root.mainloop()