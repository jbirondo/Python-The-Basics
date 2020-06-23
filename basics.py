# import datetime
# print("The date and time is", datetime.datetime.now())

# mynow = datetime.datetime.now()
# print(mynow)

# mynumber = 10
# mytext = "Hello"

# print(mynumber, mytext)

# x = 10
# y = "10"
# z = 10.1

# sum1 = x + x
# sum2 = y + y

# print(sum1, sum2)
# print(type(x), type(y), type(z))

# student_grades = [9.1, 8.8, 7.5]
# student_grades = { "Marry": 9.1, "Sim": 8.8, "John": 7.5}

# mysum = sum(student_grades.values())
# length = len(student_grades)
# mean = mysum / length
# print(mean)

# monday_temperatures = (1, 4, 5) #immutable
# print(monday_temperatures)

# monday_temperatures = [9.1, 8.8, 7.5]
# monday_temperatures.append(8.1) append to the end
# monday_temperatures.clear() clears the list
# monday_temperatures.index(8.8) returns the first index of the value searched
# monday_temperatures.__getitem__(1) or monday_temperatures[1] get item at index
# monday_temperatures[1:2] or monday_temperature[:2] (from beginning) or monday_temperature[2:] (from index to the end) slice of list where first argument is the beginning and the last agrument is the last index not inclusive 
# monday_temperatures[-1] access the last index
# monday_temperatures[-2:] slice of the last two items of the list
# slicing works on strings as well

# can't slice dictionary

# student_grades = { "Marry": 9.1, "Sim": 8.8, "John": 7.5}


# def mean(value):
#     if isinstance(value, dict):
#         the_mean = sum(value.values()) / len(value)
#     else:
#         the_mean = sum(value) / len(value)

#     return the_mean

# print(mean([1, 4, 5]))
# print(mean(student_grades))

# def weather_condition(temperature):
#     if temperature > 7:
#         return "Warm"
#     else:
#         return "Cold"
    
# user_input = float(input("Enter temperature: "))
# print(weather_condition(user_input))

# name = input("Enter your name: ")
# surname = input("Enter your surname: ")
# # message = "Hello %s" % user_input
# message = f"Hello {name} {surname}!"
# print(message)

# def greet(name):
#     return "Hi %s" % name

# def greet_title(name):
#     return "Hi %s" % name.title()

monday_temperature = [9.1, 8.8, 7.6]

# print(round(monday_temperature[0])) 
for temperature in monday_temperature:
    print(round(temperature))

for letter in "hello":
    print(letter.title())