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

def row_sum_odd_numbers(num):
    if num == 1:
        return 1
    
    i = 1
    j = [[1]]
    k = 3
    while i < num:
        l = []
        while k % 2 == 1 & len(l) < len(i[-1] + 2):
            l.append(k)
            k = k + 1
        j.append(k)
        i = i + 1
    
    return j




print(row_sum_odd_numbers(1))
print(row_sum_odd_numbers(2))
