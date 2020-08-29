from tkinter import *

window=Tk()

def kg_conversions():
    # Get user value from input box and multiply by 1000 to get kilograms
    gram = float(e1_value.get())*1000

    # Get user value from input box and multiply by 2.20462 to get pounds
    pound = float(e1_value.get())*2.20462

    # Get user value from input box and multiply by 35.274 to get ounces
    ounce = float(e1_value.get())*35.274

    # Empty the Text boxes if they had text from the previous use and fill them again
    # Deletes the content of the Text box from start to END
    t1.delete("1.0", END)
    # Fill in the text box with the value of gram variable
    t1.insert(END, gram)
    t2.delete("1.0", END)
    t2.insert(END, pound)
    t3.delete("1.0", END)
    t3.insert(END, ounce)


e2 = Label(window, text="Kg")
e2.grid(row=0, column=0)

e1_value=IntVar()
e1=Entry(window, textvariable=e1_value)
e1.grid(row=0,column=1)

b1=Button(window, text="Convert", command=kg_conversions)
b1.grid(row=0, column=2)

t1=Text(window, height=1, width=20)
t1.grid(row=1, column=0)

t2=Text(window, height=1, width=20)
t2.grid(row=1, column=1)

t3=Text(window, height=1, width=20)
t3.grid(row=1, column=2)

window.mainloop()
