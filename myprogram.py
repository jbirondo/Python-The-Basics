inputs = []
while True:
    user_input = input("Say somthing: ")
    if user_input == "\end":
        break
    else:
        words_array = user_input.split()
        string = ""
        #user_input.startswith(("who", "what", "where", "when", "why", "how"))
        if words_array[0].lower() in ["who", "what", "where", "when", "why", "how"]:
            #string = "{}?".format(user_input.capitalize())
            string = user_input.capitalize() + "?"
        else:
            string = user_input.capitalize() + "."

        inputs.append(string)
        continue

print(
    " ".join(inputs)
)