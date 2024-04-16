import tkinter as tk



window = tk.Tk()




def button_click(): #//יצירת פונק' שמתבצעת בעת הלחיצה
    num=entry.get() #//הפונקציה קוראת מספר מהמשתמש וממירה אותו לנתון שהבקר יודע לקרוא
    greeting["text"] = "The motor angle in degrees is: " + str(num)


button = tk.Button(
    text="Set [power,frequency]",
    width=20,
    height=2,
    bg="Blue",
    fg="white",
    command=button_click
)
# Label display ------------------------------------------------------------------------------------------------------
title = tk.Label( #//עיצוב של הממשק
    text="Loss Test For Components",
    foreground="black",  # Set the text color to white
    background="Light Blue",  # Set the background color to purple
    width=40,
    height=1,
    font=("Courier", 16)
)
SG_FreqStart = tk.Label( #//עיצוב של הממשק
    text="SG_FreqStart [GHz]",
    foreground="black",  # Set the text color to white
    background="light yellow",  # Set the background color to purple
    width=12,
    height=1,
    font=("Courier", 8) # גודל כתב
)
SG_FreqStop = tk.Label( #//עיצוב של הממשק
    text="SG_FreqStop [GHz]",
    foreground="black",  # Set the text color to white
    background="light yellow",  # Set the background color to purple
    width=12,
    height=1,
    font=("Courier", 8) # גודל כתב
)

SG_FreqStep = tk.Label( #//עיצוב של הממשק
    text="SG_FreqStep [GHz]",
    foreground="black",  # Set the text color to white
    background="light yellow",  # Set the background color to purple
    width=12,
    height=1,
    font=("Courier", 8) # גודל כתב
)

SG_powerStart = tk.Label( #//עיצוב של הממשק
    text="SG_powerStart [dBm]",
    foreground="black",  # Set the text color to white
    background="light yellow",  # Set the background color to purple
    width=12,
    height=1,
    font=("Courier", 8) # גודל כתב
)
SG_powerStop = tk.Label( #//עיצוב של הממשק
    text="SG_powerStop [dBm]",
    foreground="black",  # Set the text color to white
    background="light yellow",  # Set the background color to purple
    width=12,
    height=1,
    font=("Courier", 8) # גודל כתב
)
SG_powerStep = tk.Label( #//עיצוב של הממשק
    text="SG_powerStep [dBm]",
    foreground="black",  # Set the text color to white
    background="light yellow",  # Set the background color to purple
    width=12,
    height=1,
    font=("Courier", 8) # גודל כתב
)

spacing = tk.Label( #//עיצוב של הממשק
    text="   ",
    foreground="black",  # Set the text color to white
    # background="Light gray",  # Set the background color to purple
    width=12,
    height=1,
    font=("Courier", 8) # גודל כתב
)







entry = tk.Entry(fg="black", bg="white", width=20)
entry1 = tk.Entry(fg="black", bg="white", width=20)
entry2 = tk.Entry(fg="black", bg="white", width=20)

entry4 = tk.Entry(fg="black", bg="white", width=20)
entry5 = tk.Entry(fg="black", bg="white", width=20)
entry6 = tk.Entry(fg="black", bg="white", width=20)



# Arranging the GUI interface ------------------------------------------------------------------------------------------
title.grid(row=0, column=7, sticky='NESW', columnspan=5, rowspan=1, padx=1, pady=10)
SG_FreqStart.grid(row=2, column=0, sticky='NESW', columnspan=1, rowspan=1, padx=1, pady=1)
SG_FreqStop.grid(row=3, column=0, sticky='NESW', columnspan=1, rowspan=1, padx=1, pady=1)
SG_FreqStep.grid(row=4, column=0, sticky='NESW', columnspan=1, rowspan=1, padx=1, pady=1)

spacing.grid(row=5, column=0, sticky='NESW', columnspan=1, rowspan=1, padx=1, pady=1)


SG_powerStart.grid(row=6, column=0, sticky='NESW', columnspan=1, rowspan=1, padx=1, pady=1)
SG_powerStop.grid(row=7, column=0, sticky='NESW', columnspan=1, rowspan=1, padx=1, pady=1)
SG_powerStep.grid(row=8, column=0, sticky='NESW', columnspan=1, rowspan=1, padx=1, pady=1)





button.grid(row=1, column=0, sticky='NESW', columnspan=1, rowspan=1, padx=1, pady=1)


entry.grid(row=2, column=1, sticky='NESW', columnspan=1, rowspan=1, padx=1, pady=1)
entry1.grid(row=3, column=1, sticky='NESW', columnspan=1, rowspan=1, padx=1, pady=1)
entry2.grid(row=4, column=1, sticky='NESW', columnspan=1, rowspan=1, padx=1, pady=1)
entry4.grid(row=6, column=1, sticky='NESW', columnspan=1, rowspan=1, padx=1, pady=1)
entry5.grid(row=7, column=1, sticky='NESW', columnspan=1, rowspan=1, padx=1, pady=1)
entry6.grid(row=8, column=1, sticky='NESW', columnspan=1, rowspan=1, padx=1, pady=1)


window.mainloop()



