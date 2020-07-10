# Some numbers have funny properties. For example:

# 89 --> 8¹ + 9² = 89 * 1

# 695 --> 6² + 9³ + 5⁴= 1390 = 695 * 2

# 46288 --> 4³ + 6⁴+ 2⁵ + 8⁶ + 8⁷ = 2360688 = 46288 * 51

# Given a positive integer n written as abcd... (a, b, c, d... being digits) and a positive integer p

# we want to find a positive integer k, if it exists, such as the sum of the digits of n taken to the successive powers of p is equal to k * n.
# In other words:

# Is there an integer k such as : (a ^ p + b ^ (p+1) + c ^(p+2) + d ^ (p+3) + ...) = n * k

# If it is the case we will return k, if not return -1.

# Note: n and p will always be given as strictly positive integers.

# dig_pow(89, 1) should return 1 since 8¹ + 9² = 89 = 89 * 1
# dig_pow(92, 1) should return -1 since there is no k such as 9¹ + 2² equals 92 * k
# dig_pow(695, 2) should return 2 since 6² + 9³ + 5⁴= 1390 = 695 * 2
# dig_pow(46288, 3) should return 51 since 4³ + 6⁴+ 2⁵ + 8⁶ + 8⁷ = 2360688 = 46288 * 51

# def dig_pow(n, p):
#     nums = int_to_list(n)
#     total = []
#     for num in nums:
#         total.append(num ** p)
#         p = p + 1
    
#     if sum(total) % n == 0:
#         return int(sum(total) / n)
#     else:
#         return -1

# def int_to_list(num):
#     s = str(num)
#     return [int(c) for c in s]

# print(int_to_list(123123))
# print(dig_pow(46288, 3))
# print(isinstance(int_to_list(123123), str)) == True

# Given the triangle of consecutive odd numbers:

#              1
#           3     5
#        7     9    11
#    13    15    17    19
# 21    23    25    27    29
# ...
# Calculate the row sums of this triangle from the row index (starting at index 1) e.g.:

# row_sum_odd_numbers(1); # 1
# row_sum_odd_numbers(2); # 3 + 5 = 8

# def row_sum_odd_numbers(num):
#     if num == 1:
#         return 1
    
#     i = 1
#     j = [[1]]
#     k = 3
#     while i < num:
#         l = []
#         while len(l) < len(j[-1]) + 1:
#             if k % 2 != 0:
#                 l.append(k)
#             k = k + 1
#         j.append(l)
#         i = i + 1
    
#     return sum(j[-1])

# def row_sum_odd_numbers(n):
#     #your code here
#     return n ** 3




# print(row_sum_odd_numbers(41))

# Very simple, given a number, find its opposite.

# Examples:

# 1: -1
# 14: -14
# -34: 34

# def opposite(number):
#     return number * -1

# Given an array of integers, return a new array with each value doubled.

# For example:

# [1, 2, 3] --> [2, 4, 6]

# For the beginner, try to use the map method - it comes in very handy quite a lot so is a good one to know.

# def maps(a):
#     return [ele * 2 for ele in a]

# print(maps([1,2,3,4,5,6,7,8,9]))

# The Western Suburbs Croquet Club has two categories of membership, Senior and Open. They would like your help with an application form that will tell prospective members which category they will be placed.

# To be a senior, a member must be at least 55 years old and have a handicap greater than 7. In this croquet club, handicaps range from -2 to +26; the better the player the lower the handicap.

# Input
# Input will consist of a list of lists containing two items each. Each list contains information for a single potential member. Information consists of an integer for the person's age and an integer for the person's handicap.

# Note for F#: The input will be of (int list list) which is a List<List>

# Example Input
# [[18, 20],[45, 2],[61, 12],[37, 6],[21, 21],[78, 9]]
# Output
# Output will consist of a list of string values (in Haskell: Open or Senior) stating whether the respective member is to be placed in the senior or open category.

# Example Output
# ["Open", "Open", "Senior", "Open", "Open", "Senior"]

# def open_or_senior(data):
#     result = []
#     for datum in data:
#         if datum[0] > 55 and datum[1] > 7:
#             result.append("Senior")
#         else:
#             result.append("Open")
#     return result

# def openOrSenior(data):
#   return ["Senior" if age >= 55 and handicap >= 8 else "Open" for (age, handicap) in data]

# You are going to be given an array of integers. Your job is to take that array and find an index N where the sum of the integers to the left of N is equal to the sum of the integers to the right of N. If there is no index that would make this happen, return -1.

# For example:

# Let's say you are given the array {1,2,3,4,3,2,1}: Your function will return the index 3, because at the 3rd position of the array, the sum of left side of the index ({1,2,3}) and the sum of the right side of the index ({3,2,1}) both equal 6.

# Let's look at another one.
# You are given the array {1,100,50,-51,1,1}: Your function will return the index 1, because at the 1st position of the array, the sum of left side of the index ({1}) and the sum of the right side of the index ({50,-51,1,1}) both equal 1.

# Last one:
# You are given the array {20,10,-80,10,10,15,35}
# At index 0 the left side is {}
# The right side is {10,-80,10,10,15,35}
# They both are equal to 0 when added. (Empty arrays are equal to 0 in this problem)
# Index 0 is the place where the left side and right side are equal.

# Note: Please remember that in most programming/scripting languages the index of an array starts at 0.

# Input:
# An integer array of length 0 < arr < 1000. The numbers in the array can be any integer positive or negative.

# Output:
# The lowest index N where the side to the left of N is equal to the side to the right of N. If you do not find an index that fits these rules, then you will return -1.

# Note:
# If you are given an array with multiple answers, return the lowest correct index.

# def find_even_index(arr):
#     num = 0
#     while num < len(arr):
#         if sum(arr[:num]) == sum(arr[num + 1:]):
#             return num
#         num = num + 1
#     return -1

# You get an array of numbers, return the sum of all of the positives ones.

# Example [1,-4,7,12] => 1 + 7 + 12 = 20

# Note: if there is nothing to sum, the sum is default to 0.

# def positive_sum(arr):
#     new_arr = [i if i > 0 else 0 for i in arr]
#     return sum(new_arr)

# def positive_sum(arr):
#     return sum([i if i > 0 else 0 for i in arr])

# Write Number in Expanded Form
# You will be given a number and you will need to return it as a string in Expanded Form. For example:

# expanded_form(12) # Should return '10 + 2'
# expanded_form(42) # Should return '40 + 2'
# expanded_form(70304) # Should return '70000 + 300 + 4'
# NOTE: All numbers will be whole numbers greater than 0.

def expanded_form(num):
    result = []
    while num > 0:
        string = str(num)
        new_num = int(string[0]) * (10 ** (len(string) - 1))
        result.append("+ %s " % new_num)
        num = num - new_num
    return "".join(result)[2:-1]
print(expanded_form(7000120003))