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

# def expanded_form(num):
#     result = []
#     while num > 0:
#         string = str(num)
#         new_num = int(string[0]) * (10 ** (len(string) - 1))
#         result.append(str(new_num))
#         num = num - new_num
#     return " + ".join(result)
# print(expanded_form(7000120003))

# Your goal in this kata is to implement a difference function, which subtracts one list from another and returns the result.

# It should remove all values from list a, which are present in list b.

# array_diff([1,2],[1]) == [2]
# If a value is present in b, all of its occurrences must be removed from the other:

# array_diff([1,2,2,2,3],[2]) == [1,3]

# def array_diff(a, b):
#     result = []
#     for i in a:
#         if i not in b:
#             result.append(i)
#     return result

# def array_diff(a, b):
#     return [x for x in a if x not in b]

# Jaden Smith, the son of Will Smith, is the star of films such as The Karate Kid (2010) and After Earth (2013). Jaden is also known for some of his philosophy that he delivers via Twitter. When writing on Twitter, he is known for almost always capitalizing every word. For simplicity, you'll have to capitalize each word, check out how contractions are expected to be in the example below.

# Your task is to convert strings to how they would be written by Jaden Smith. The strings are actual quotes from Jaden Smith, but they are not capitalized in the same way he originally typed them.

# Example:

# Not Jaden-Cased: "How can mirrors be real if our eyes aren't real"
# Jaden-Cased:     "How Can Mirrors Be Real If Our Eyes Aren't Real"

# def to_jaden_case(string):
#     array = string.split()
#     return " ".join([ele.capitalize() for ele in array])

# Story
# Ben has a very simple idea to make some profit: he buys something and sells it again. Of course, this wouldn't give him any profit at all if he was simply to buy and sell it at the same price. Instead, he's going to buy it for the lowest possible price and sell it at the highest.

# Task
# Write a function that returns both the minimum and maximum number of the given list/array.

# Examples
# min_max([1,2,3,4,5])   == [1,5]
# min_max([2334454,5])   == [5, 2334454]
# min_max([1])           == [1, 1]
# Remarks
# All arrays or lists will always have at least one element, so you don't need to check the length. Also, your function will always get an array or a list, you don't have to check for null, undefined or similar.

# def min_max(lst):
#     return [min(lst), max(lst)]

# print(min_max([5,6,7,3,1,324,5]))

# Your task is to sort a given string. Each word in the string will contain a single number. This number is the position the word should have in the result.

# Note: Numbers can be from 1 to 9. So 1 will be the first word (not 0).

# If the input string is empty, return an empty string. The words in the input String will only contain valid consecutive numbers.

# Examples
# "is2 Thi1s T4est 3a"  -->  "Thi1s is2 3a T4est"
# "4of Fo1r pe6ople g3ood th5e the2"  -->  "Fo1r the2 g3ood 4of th5e pe6ople"
# ""  -->  ""

# def order(sentence):
#     array = sentence.split()
#     new_array = []
#     i = 1
#     while i < (len(array) + 1):
#         for ele in array:
#             if ele.find("{}".format(i)) != -1:
#                 new_array.append(ele)
#                 i = i + 1
#     return " ".join(new_array)

# def order(s):
#     z = []
#     for i in range(1,10):
#         for j in list(s.split()):
#             if str(i) in j:
#                z.append(j)
#     return " ".join(z)

# You are given an array(list) strarr of strings and an integer k. Your task is to return the first longest string consisting of k consecutive strings taken in the array.

# Example:
# longest_consec(["zone", "abigail", "theta", "form", "libe", "zas", "theta", "abigail"], 2) --> "abigailtheta"

# n being the length of the string array, if n = 0 or k > n or k <= 0 return "".

# def longest_consec(strarr, k):
#     index = 0
#     longest = ""
#     if k < 0:
#         return ""
#     while (index + k) <= len(strarr):
#         if len("".join(strarr[index: index + k])) > len(longest):
#             longest = "".join(strarr[index: index + k])
#         index = index + 1
#     return longest

# print(longest_consec(["zone", "abigail", "theta", "form", "libe", "zas"], 2))

# def longest_consec(strarr, k):
#     result = ""
    
#     if k > 0 and len(strarr) >= k:
#         for index in range(len(strarr) - k + 1):
#             s = ''.join(strarr[index:index+k])
#             if len(s) > len(result):
#                 result = s
            
#     return result

# Write a function that takes an array of numbers (integers for the tests) and a target number. It should find two different items in the array that, when added together, give the target value. The indices of these items should then be returned in a tuple like so: (index1, index2).

# For the purposes of this kata, some tests may have multiple answers; any valid solutions will be accepted.

# The input will always be valid (numbers will be an array of length 2 or greater, and all of the items will be numbers; target will always be the sum of two different items from that array).

# Based on: http://oj.leetcode.com/problems/two-sum/

# twoSum [1, 2, 3] 4 === (0, 2)

def two_sum(numbers, target):
    for x in range(0, len(numbers) - 1):
        for y in range(0, len(numbers) - 1):
            if numbers[x] + numbers[y] == target and x != y:
                return (x, y)

print(two_sum([2,2,3], 4))