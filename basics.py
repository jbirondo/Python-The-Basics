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

# monday_temperature = [9.1, 8.8, 7.6]

# # print(round(monday_temperature[0])) 
# for temperature in monday_temperature:
#     print(round(temperature))
#     print("Done")

# for letter in "hello":
#     print(letter)

# student_grades = { "Marry": 9.1, "Sim": 8.8, "John": 7.5}

# # for grades in student_grades.items():
#     # print(grades) #will print a tuple

# for key, value in student_grades.items():
#     print("{} earned a grade of {}".format(key, value))

# phone_numbers = {"John Smith": "+37682929928", "Marry Simpons": "+423998200919"}

# for value in phone_numbers.values():
#     print(value.replace("+", "00"))
#     # print("00{}".format(value[1:]))

# username = ""

# while username != "pypy":
#     username = input("Enter username: ")

# while True:
#     userrname = input("Enter username: ")
#     if userrname == "pypy":
#         break
#     else:
#         continue

#Simple List Comprehension

# temps = [221, 234, 340, 230]
# # new_temps = []

# for temp in temps:
#     new_temps.append(temp/10)

# print(new_temps)

# new_temps = [temp / 10 for temp in temps]

# print(new_temps)

# temps = [221, 234, 340, -9999, 230]
# new_temps = [temp / 10 for temp in temps if temp != -9999]

# print(new_temps)

# def only_numbers(list):
#     new_list = [ele for ele in list if isinstance(ele, int)]
#     print(new_list)

# temps = [221, 234, 340, -9999, 230]

# new_temps = [temp / 10 if temp != -9999 else 0 for temp in temps]

# print(new_temps)

# Summary: List Comprehensions
# In this section you learned that:

# A list comprehension is an expression that creates a list by iterating over another container.

# A basic list comprehension:

# [i*2 for i in [1, 5, 10]]
# Output: [2, 10, 20]

# List comprehension with if condition:

# [i*2 for i in [1, -2, 10] if i>0]
# Output: [2, 20]

# List comprehension with an if and else condition:

# [i*2 if i>0 else 0 for i in [1, -2, 10]]
# Output: [2, 0, 20]

# def area(a, b):
# # def area(a, b = 6): # Default parameter, cannot be positioned ahead of non-default parameters
#     return a * b

# print(area(5 ,5)) #positional arguments
# print(area(a = 5,b = 5)) # keyword arguments

# def mean(*args):
#     # return args #returns tuple
#     return sum(args) / len(args)

# print(mean(5, 4, 6)) 

# def average(*args):
#     return sum(args) / len(args)

# def uppercase_and_sort(*args):
#     upper_list = [ele.upper() for ele in list(args)]
#     return sorted(upper_list)

# def mean(**kwargs):
#     return kwargs

# print(mean(a = 1, b = 2, c = 3))

# Summary: More on Functions
# In this section you learned that:

# Functions can have more than one parameter:

# def volume(a, b, c):
#     return a * b * c
# Functions can have default parameters (e.g. coefficient):

# def converter(feet, coefficient = 3.2808):
#     meters = feet / coefficient
#     return meters
 
# print(converter(10))
# Output: 3.0480370641306997

# Arguments can be passed as non-keyword (positional) arguments (e.g. a) or keyword arguments (e.g. b=2 and c=10):

# def volume(a, b, c):
#     return a * b * c
 
# print(volume(1, b=2, c=10))
# An *args parameter allows the  function to be called with an arbitrary number of non-keyword arguments:

# def find_max(*args):
#     return max(args)
# print(find_max(3, 99, 1001, 2, 8))
# Output: 1001

# An **kwargs parameter allows the function to be called with an arbitrary number of keyword arguments:

# def find_winner(**kwargs):
#     return max(kwargs, key = kwargs.get)
 
# print(find_winner(Andy = 17, Marry = 19, Sim = 45, Kae = 34))
# Output: Sim

# Here's a summary of function elements:

# myfile = open("files/file.txt")
# content = myfile.read()
# print(content.split())
# myfile.close()

with open("files/vegetables.txt", "w") as myfile:
    # content = myfile.read()
    myfile.write("tomato\ncucumber\nonion\n")
    myfile.write('garlic')
# print(content)


# file = open("bear.txt")

# print(file.read())

# with open("bear.txt") as myfile:
#     content = myfile.read()[:90]
    
# print(content)

# def find_chars(str, file):
#     with open(file) as myfile:
#         content = myfile.read()
#         count = content.count(str)
        
#     return count
    
# with open("file.txt", "w") as myfile:
# #     myfile.write("snail")

# with open("files/vegetables.txt", "w") as myfile:
#     # content = myfile.read()
#     myfile.write("tomato\ncucumber\nonion\n")
#     myfile.write('garlic')

# with open("files/vegetables.txt", "a") as myfile:
#     myfile.write("\nOkra")
#     myfile.seek(0)


# with open("bear1.txt") as first:
#     content = first.read()
    
# with open("bear2.txt", "a+") as second:
#     second.write(content)

# with open("data.txt", "a+") as file:
#     file.seek(0)
#     content = file.read()
#     file.seek(0)
#     file.write(content)
#     file.write(content)

# Summary: File Processing
# In this section you learned that:

# You can read an existing file with Python:

# with open("file.txt") as file:
#     content = file.read()
# You can create a new file with Python and write some text on it:

# with open("file.txt", "w") as file:
#     content = file.write("Sample text")
# You can append text to an existing file without overwriting it:

# with open("file.txt", "a") as file:
#     content = file.write("More sample text")
# You can both append and read a file with:

# with open("file.txt", "a+") as file:
#     content = file.write("Even more sample text")
#     file.seek(0)
#     content = file.read()