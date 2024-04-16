import tkinter as tk

window = tk.Tk()
def button_click(): #//יצירת פונק' שמתבצעת בעת הלחיצה
    num=entry.get() #//הפונקציה קוראת מספר מהמשתמש וממירה אותו לנתון שהבקר יודע לקרוא
    b = num.encode('utf-8')
    print(b)
    ser.write(b)
    resulting["text"]="The motor angle in degrees is: "+str(num)
def serfinish(): #//פונקציה שמתבצעת בעת לחיצה על לחצן הסיום
    ser.close()
    window.quit()
greeting = tk.Label( #//עיצוב של הממשק
    text="servo motor Arduino GUI",
    foreground="black",  # Set the text color to white
    background="purple",  # Set the background color to purple
    width=40,
    height=1,
    font = ("Courier", 16)
)
texting = tk.Label(
    text="enter the angle in degrees",
    foreground="black",  # Set the text color to black
    background="green",  # Set the background color to green
    width=40,
    height=1,
    font = ("Courier", 12)
)
resulting = tk.Label(
    text="The motor degrees is given here",
    foreground="black",  # Set the text color to black
    background="purple",  # Set the background color to purple
    width=60,
    height=1,
    font = ("Courier", 12)
)
button = tk.Button(
    text="Click to enter the rotation angle!",
    width=25,
    height=5,
    bg="blue",
    fg="yellow",
    command=button_click
)
entry = tk.Entry(fg="black", bg="white", width=50)
finish = tk.Button(
    text="finish!",
    width=25,
    height=5,
    bg="blue",
    fg="yellow",
    command=serfinish
)
greeting.pack()
texting.pack()
entry.pack()
button.pack()
resulting.pack()
finish.pack()
window.mainloop()
