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

# def two_sum(numbers, target):
#     for x in range(0, len(numbers) - 1) or []:
#         for y in range(0, len(numbers) - 1) or []:
#             if numbers[x] + numbers[y] == target and x != y:
#                 return (x, y)

# Your job is to write a function which increments a string, to create a new string.

# If the string already ends with a number, the number should be incremented by 1.
# If the string does not end with a number. the number 1 should be appended to the new string.
# Examples:

# foo -> foo1

# foobar23 -> foobar24

# foo0042 -> foo0043

# foo9 -> foo10

# foo099 -> foo100

# Attention: If the number has leading zeros the amount of digits should be considered.

# def increment_string(strng):
#     w1 = ""
#     w2 = ""
#     if len(strng) == 0:
#         return "1"
#     for x in range(0, len(strng)):
#         if strng[x].isnumeric():
#             w2 = strng[x:]
#             w1 = strng[:x]
#             break
#         else:
#             w2 = "0"
#             w1 = strng
#     pad = len(w2)
#     mod_w2 = str((int(w2) + 1)).zfill(pad)
#     return w1 + mod_w2



# print(increment_string("hello world 005"))

# Given a positive number n > 1 find the prime factor decomposition of n. The result will be a string with the following form :

#  "(p1**n1)(p2**n2)...(pk**nk)"
# with the p(i) in increasing order and n(i) empty if n(i) is 1.

# Example: n = 86240 should return "(2**5)(5)(7**2)(11)"

# def primeFactors(n):
#     start = 2
#     result = []
#     final_string = ""
#     while n > 1:
#         if n % start == 0 and is_prime(start):
#             result.append(start)
#             n = n / start
#         else:
#             start = start + 1
#     s = sorted(set(result))
#     for x in s:
#         if result.count(x) > 1:
#             final_string = final_string + "({}**{})".format(x, result.count(x))
#         else:
#             final_string = final_string + "({})".format(x)
    
#     return final_string
        

# def is_prime(num):
#     if num < 2:
#         return False
#     for x in range(2, int(num/2) + 1):
#         if num % x == 0:
#             return False    
#     return True

# print(primeFactors(1129019283))

# def primeFactors(n):
#     ret = ''
#     for i in xrange(2, n + 1):
#         num = 0
#         while(n % i == 0):
#             num += 1
#             n /= i
#         if num > 0:
#             ret += '({}{})'.format(i, '**%d' % num if num > 1 else '')
#         if n == 1:
#             return ret

# def primeFactors(n):
#     i, j, p = 2, 0, []
#     while n > 1:
#         while n % i == 0: n, j = n / i, j + 1
#         if j > 0: p.append([i,j])
#         i, j = i + 1, 0
#     return ''.join('(%d' %q[0] + ('**%d' %q[1]) * (q[1] > 1) + ')' for q in p)

# def primeFactors(n):
#       result = ''
#   fac = 2
#   while fac <= n:
#     count = 0
#     while n % fac == 0:
#       n /= fac
#       count += 1
#     if count:
#       result += '(%d%s)' % (fac, '**%d' % count if count > 1 else '')
#     fac += 1
#   return result

# In this little assignment you are given a string of space separated numbers, and have to return the highest and lowest number.

# Example:

# high_and_low("1 2 -3 4 5") # return "5 -3"
# high_and_low("1 9 3 4 -5") # return "9 -5"
# Notes:

# All numbers are valid Int32, no need to validate them.
# There will always be at least one number in the input string.
# Output string must be two numbers separated by a single space, and highest number is first.

# def high_and_low(numbers):
#     array = [int(x) for x in numbers.split(" ")]
#     return "{} {}".format(max(array), min(array))

# print(high_and_low("1 2 3 4 5"))  # return "5 1"

# Given an array of ones and zeroes, convert the equivalent binary value to an integer.

# Eg: [0, 0, 0, 1] is treated as 0001 which is the binary representation of 1.

# Examples:

# Testing: [0, 0, 0, 1] ==> 1
# Testing: [0, 0, 1, 0] ==> 2
# Testing: [0, 1, 0, 1] ==> 5
# Testing: [1, 0, 0, 1] ==> 9
# Testing: [0, 0, 1, 0] ==> 2
# Testing: [0, 1, 1, 0] ==> 6
# Testing: [1, 1, 1, 1] ==> 15
# Testing: [1, 0, 1, 1] ==> 11
# However, the arrays can have varying lengths, not just limited to 4.

# def binary_array_to_number(arr):
#     # print("".join([str(x) for x in arr]))
#     return int("".join([str(x) for x in arr]), 2)

# # print(binary_array_to_number([0,0,0,1]))

# def binary_array_to_number(arr):
#   return int("".join(map(str, arr)), 2)

# def binary_array_to_number(arr):
#     s = 0
#     for digit in arr:
#         s = s * 2 + digit

#     return s

# Description:
# This time we want to write calculations using functions and get the results. Let's have a look at some examples:

# seven(times(five())) # must return 35
# four(plus(nine())) # must return 13
# eight(minus(three())) # must return 5
# six(divided_by(two())) # must return 3
# Requirements:

# There must be a function for each number from 0 ("zero") to 9 ("nine")
# There must be a function for each of the following mathematical operations: plus, minus, times, dividedBy (divided_by in Ruby and Python)
# Each calculation consist of exactly one operation and two numbers
# The most outer function represents the left operand, the most inner function represents the right operand
# Divison should be integer division. For example, this should return 2, not 2.666666...:
# eight(divided_by(three()))

# def zero(f = None): return 0 if not f else f(0)
# def one(f = None): return 1 if not f else f(1)
# def two(f = None): return 2 if not f else f(2)
# def three(f = None): return 3 if not f else f(3)
# def four(f = None): return 4 if not f else f(4)
# def five(f = None): return 5 if not f else f(5)
# def six(f = None): return 6 if not f else f(6)
# def seven(f = None): return 7 if not f else f(7)
# def eight(f = None): return 8 if not f else f(8)
# def nine(f = None): return 9 if not f else f(9)

# def plus(y): return lambda x: x+y
# def minus(y): return lambda x: x-y
# def times(y): return lambda  x: x*y
# def divided_by(y): return lambda  x: x/y

# def zero(arg=""):  return eval("0" + arg)
# def one(arg=""):   return eval("1" + arg)
# def two(arg=""):   return eval("2" + arg)
# def three(arg=""): return eval("3" + arg)
# def four(arg=""):  return eval("4" + arg)
# def five(arg=""):  return eval("5" + arg)
# def six(arg=""):   return eval("6" + arg)
# def seven(arg=""): return eval("7" + arg)
# def eight(arg=""): return eval("8" + arg)
# def nine(arg=""):  return eval("9" + arg)

# def plus(n):       return "+%s" % n
# def minus(n):      return "-%s" % n
# def times(n):      return "*%s" % n
# def divided_by(n): return "/%s" % n

# def zero(cb=None): return cb(0) if cb else 0
# def one(cb=None): return cb(1) if cb else 1
# def two(cb=None): return cb(2) if cb else 2
# def three(cb=None): return cb(3) if cb else 3
# def four(cb=None): return cb(4) if cb else 4
# def five(cb=None): return cb(5) if cb else 5
# def six(cb=None): return cb(6) if cb else 6
# def seven(cb=None): return cb(7) if cb else 7
# def eight(cb=None): return cb(8) if cb else 8
# def nine(cb=None): return cb(9) if cb else 9

# def plus(n): return lambda x: x+n
# def minus(n): return lambda x: x-n
# def times(n): return lambda x: x*n
# def divided_by(n): return lambda x: x//n

# The goal of this exercise is to convert a string to a new string where each character in the new string is "(" if that character appears only once in the original string, or ")" if that character appears more than once in the original string. Ignore capitalization when determining if a character is a duplicate.

# Examples
# "din"      =>  "((("
# "recede"   =>  "()()()"
# "Success"  =>  ")())())"
# "(( @"     =>  "))((" 
# Notes

# Assertion messages may be unclear about what they display in some languages. If you read "...It Should encode XXX", the "XXX" is the expected result, not the input!

# def duplicate_encode(word):
#     return "".join(["(" if word.count(x) == 1 else ")" for x in list(word)])

# If you have completed the Tribonacci sequence kata, you would know by now that mister Fibonacci has at least a bigger brother. If not, give it a quick look to get how things work.

# Well, time to expand the family a little more: think of a Quadribonacci starting with a signature of 4 elements and each following element is the sum of the 4 previous, a Pentabonacci (well Cinquebonacci would probably sound a bit more italian, but it would also sound really awful) with a signature of 5 elements and each following element is the sum of the 5 previous, and so on.

# Well, guess what? You have to build a Xbonacci function that takes a signature of X elements - and remember each next element is the sum of the last X elements - and returns the first n elements of the so seeded sequence.

# xbonacci {1,1,1,1} 10 = {1,1,1,1,4,7,13,25,49,94}
# xbonacci {0,0,0,0,1} 10 = {0,0,0,0,1,1,2,4,8,16}
# xbonacci {1,0,0,0,0,0,1} 10 = {1,0,0,0,0,0,1,2,3,6}
# xbonacci {1,1} produces the Fibonacci sequence

# def Xbonacci(signature,n):
#     length = len(signature)
#     if n < len(signature):
#         return signature[:n]
#     while len(signature) < n:
#         signature.append(sum(signature[-length:]))
#     return signature

# print(Xbonacci([1,1,1,1], 10))

# John and Mary want to travel between a few towns A, B, C ... Mary has on a sheet of paper a list of distances between these towns. ls = [50, 55, 57, 58, 60]. John is tired of driving and he says to Mary that he doesn't want to drive more than t = 174 miles and he will visit only 3 towns.

# Which distances, hence which towns, they will choose so that the sum of the distances is the biggest possible to please Mary and John?

# Example:

# With list ls and 3 towns to visit they can make a choice between: [50,55,57],[50,55,58],[50,55,60],[50,57,58],[50,57,60],[50,58,60],[55,57,58],[55,57,60],[55,58,60],[57,58,60].

# The sums of distances are then: 162, 163, 165, 165, 167, 168, 170, 172, 173, 175.

# The biggest possible sum taking a limit of 174 into account is then 173 and the distances of the 3 corresponding towns is [55, 58, 60].

# The function chooseBestSum (or choose_best_sum or ... depending on the language) will take as parameters t (maximum sum of distances, integer >= 0), k (number of towns to visit, k >= 1) and ls (list of distances, all distances are positive or null integers and this list has at least one element). The function returns the "best" sum ie the biggest possible sum of k distances less than or equal to the given limit t, if that sum exists, or otherwise nil, null, None, Nothing, depending on the language. With C++, C, Rust, Swift, Go, Kotlin return -1.

# Examples:

# ts = [50, 55, 56, 57, 58] choose_best_sum(163, 3, ts) -> 163

# xs = [50] choose_best_sum(163, 3, xs) -> nil (or null or ... or -1 (C++, C, Rust, Swift, Go)

# ys = [91, 74, 73, 85, 73, 81, 87] choose_best_sum(230, 3, ys) -> 228

# import itertools as it

# def choose_best_sum(t, k, ls):
#     subsets = it.combinations(ls, k)
#     l = list(subsets)
#     # if len(l) == 0:
#     #     return None
#     # else:
#     #     return max(sum(x) for x in l if sum(x) <= t)
#     dists = [sum(x) for x in l if sum(x) <= t]
#     return dists

# xs = [100, 76, 56, 44, 89, 73, 68, 56, 64, 123, 2333, 144, 50, 132, 123, 34, 89]
# print(choose_best_sum(480, 9, xs))

# import itertools
# def choose_best_sum(t, k, ls):
#     try: 
#         return max(sum(i) for i in itertools.combinations(ls,k) if sum(i)<=t)
#     except:
#         return None
        
# The prime numbers are not regularly spaced. For example from 2 to 3 the gap is 1. From 3 to 5 the gap is 2. From 7 to 11 it is 4. Between 2 and 50 we have the following pairs of 2-gaps primes: 3-5, 5-7, 11-13, 17-19, 29-31, 41-43

# A prime gap of length n is a run of n-1 consecutive composite numbers between two successive primes (see: http://mathworld.wolfram.com/PrimeGaps.html).

# We will write a function gap with parameters:

# g (integer >= 2) which indicates the gap we are looking for

# m (integer > 2) which gives the start of the search (m inclusive)

# n (integer >= m) which gives the end of the search (n inclusive)

# In the example above gap(2, 3, 50) will return [3, 5] or (3, 5) or {3, 5} which is the first pair between 3 and 50 with a 2-gap.

# So this function should return the first pair of two prime numbers spaced with a gap of g between the limits m, n if these numbers exist otherwise nil or null or None or Nothing (depending on the language).

# In C++ return in such a case {0, 0}. In F# return [||]. In Kotlin return []

# #Examples: gap(2, 5, 7) --> [5, 7] or (5, 7) or {5, 7}

# gap(2, 5, 5) --> nil. In C++ {0, 0}. In F# [||]. In Kotlin return[]`

# gap(4, 130, 200) --> [163, 167] or (163, 167) or {163, 167}

# ([193, 197] is also such a 4-gap primes between 130 and 200 but it's not the first pair)

# gap(6,100,110) --> nil or {0, 0} : between 100 and 110 we have 101, 103, 107, 109 but 101-107is not a 6-gap because there is 103in between and 103-109is not a 6-gap because there is 107in between.

# Note for Go
# For Go: nil slice is expected when there are no gap between m and n. Example: gap(11,30000,100000) --> nil

# #Ref https://en.wikipedia.org/wiki/Prime_gap

# def gap(g, m, n):
#     l = [x for x in range(m, n) if is_prime(x)]
#     i = 0
#     while i < len(l) - 1:
#         if l[i] + g == l[i + 1]:
#             return [l[i], l[i + 1]]
#         i = i + 1
#     return None

# def is_prime(num):
#     if num < 2:
#         return False
#     for x in range(2, int(num / 2)):
#         if num % x == 0:
#             return False
#     return True
# 
# A traveling salesman has to visit clients. He got each client's address e.g. "432 Main Long Road St. Louisville OH 43071" as a list.

# The basic zipcode format usually consists of two capital letters followed by a white space and five digits. The list of clients to visit was given as a string of all addresses, each separated from the others by a comma, e.g. :

# "123 Main Street St. Louisville OH 43071,432 Main Long Road St. Louisville OH 43071,786 High Street Pollocksville NY 56432".

# To ease his travel he wants to group the list by zipcode.

# Task
# The function travel will take two parameters r (addresses' list of all clients' as a string) and zipcode and returns a string in the following format:

# zipcode:street and town,street and town,.../house number,house number,...

# The street numbers must be in the same order as the streets where they belong.

# If a given zipcode doesn't exist in the list of clients' addresses return "zipcode:/"

# Examples
# r = "123 Main Street St. Louisville OH 43071,432 Main Long Road St. Louisville OH 43071,786 High Street Pollocksville NY 56432"

# travel(r, "OH 43071") --> "OH 43071:Main Street St. Louisville,Main Long Road St. Louisville/123,432"

# travel(r, "NY 56432") --> "NY 56432:High Street Pollocksville/786"

# travel(r, "NY 5643") --> "NY 5643:/"
# Note for Elixir:
# In Elixir the empty addresses' input is an empty list, not an empty string.

# Note:
# You can see a few addresses and zipcodes in the test cases.

# def travel(r, zipcode):
#     l = r.split(",")
#     zips = [" ".join(x.split()[-2:]) for x in l]
#     d = {}
#     for x in zips:
#         d.update({x: []})

#     for y in l:
#         a = " ".join(y.split()[-2:])
#         d[a].append(y)
    
#     if zipcode not in d:
#         return zipcode + ":/"

#     ads = [" ".join(x.split()[1:-2]) for x in d[zipcode]]
#     nums = [x.split()[0] for x in d[zipcode]]
#     return zipcode + ":" + ",".join(ads) + "/" + ",".join(nums)

    
    
    


# ad = ("123 Main Street St. Louisville OH 43071,432 Main Long Road St. Louisville OH 43071,786 High Street Pollocksville NY 56432,"
# "54 Holy Grail Street Niagara Town ZP 32908,3200 Main Rd. Bern AE 56210,1 Gordon St. Atlanta RE 13000,"
# "10 Pussy Cat Rd. Chicago EX 34342,10 Gordon St. Atlanta RE 13000,58 Gordon Road Atlanta RE 13000,"
# "22 Tokyo Av. Tedmondville SW 43098,674 Paris bd. Abbeville AA 45521,10 Surta Alley Goodtown GG 30654,"
# "45 Holy Grail Al. Niagara Town ZP 32908,320 Main Al. Bern AE 56210,14 Gordon Park Atlanta RE 13000,"
# "100 Pussy Cat Rd. Chicago EX 34342,2 Gordon St. Atlanta RE 13000,5 Gordon Road Atlanta RE 13000,"
# "2200 Tokyo Av. Tedmondville SW 43098,67 Paris St. Abbeville AA 45521,11 Surta Avenue Goodtown GG 30654,"
# "45 Holy Grail Al. Niagara Town ZP 32918,320 Main Al. Bern AE 56215,14 Gordon Park Atlanta RE 13200,"
# "100 Pussy Cat Rd. Chicago EX 34345,2 Gordon St. Atlanta RE 13222,5 Gordon Road Atlanta RE 13001,"
# "2200 Tokyo Av. Tedmondville SW 43198,67 Paris St. Abbeville AA 45522,11 Surta Avenue Goodville GG 30655,"
# "2222 Tokyo Av. Tedmondville SW 43198,670 Paris St. Abbeville AA 45522,114 Surta Avenue Goodville GG 30655,"
# "2 Holy Grail Street Niagara Town ZP 32908,3 Main Rd. Bern AE 56210,77 Gordon St. Atlanta RE 13000")

# print(travel(ad, 'ZP 32908'))

# You are given a secret message you need to decipher. Here are the things you need to know to decipher it:

# For each word:

# the second and the last letter is switched (e.g. Hello becomes Holle)
# the first letter is replaced by its character code (e.g. H becomes 72)
# Note: there are no special characters used, only letters and spaces

# Examples

# decipherThis('72olle 103doo 100ya'); // 'Hello good day'
# decipherThis('82yade 115te 103o'); // 'Ready set go'
import re

def decipher_this(string):
    s = string.split()
    for x in s:
        temp = re.compile("([a-zA-Z]+)([0-9]+)")
        res = temp.match(x).groups() 
        x = res[0][0]
        y = res[0][-1]
        print(chr(int(res[1])) + y + res[0][1:-2] + x)


print(decipher_this("hello72 world77"))