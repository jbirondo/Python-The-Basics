class Solution:
    def productExceptSelf(self, nums):
        product = 1
        zero_count = 0
        #calculate the product of the array
        for i in range(len(nums)):
            if(nums[i] != 0):
                product = product*nums[i]
            else:
                #keep track of 0s
                zero_count+=1
        #if > 1 zeros exist, the product will be 0 for all elements regardless
        if(zero_count > 1):
            return [0]*len(nums)
        #if there are no zeros, the product except itself will be product//i
        #if there is 1 zero, the product will be 0 for non-zero elments and the product for the zero element
        return [product//i for i in nums] if zero_count == 0 else [product if i == 0 else 0 for i in nums] 

    def containsDuplicate(self, nums: List[int]) -> bool:
        dict = {}
        for x in range(0, len(nums)):
            if nums[x] not in dict:
                dict[nums[x]] = True
            else:
                return dict[nums[x]]
        return False

    def maxProfit(self, prices: List[int]) -> int:
        buy = prices[0]
        mx_profit = 0

        for i in range(1,len(prices)):
            profit = prices[i]-buy

            if profit>mx_profit:
                mx_profit = profit

            if buy>prices[i]:
                buy = prices[i]

        return mx_profit

    def maxSubArray(self, nums):
        sumSoFar = max(nums)
        currentSum = 0
        for i in nums:
            currentSum += i
            if currentSum < 0:
                currentSum = 0
            else:
                sumSoFar = max(sumSoFar, currentSum)
        return sumSoFar

    def findMin(self, nums: List[int]) -> int:
        self.nums = nums
        if len(nums) is 0:
            return None
        if len(nums) is 1:
            return nums[0]
        if nums[0] < nums[-1]:
            return nums[0]
        return self.binarySearch(0, len(nums)-1)
    
    def binarySearch(self, start, end):
        if start > end:
            return None
        else:
            mid = (start + end) // 2
            if self.nums[mid] > self.nums[mid+1]:
                return self.nums[mid+1]
            if self.nums[mid] < self.nums[mid-1]:
                return self.nums[mid]
            else:
                if self.nums[start] < self.nums[mid]:
                    return self.binarySearch(mid + 1, end)
                else:
                    return self.binarySearch(start, mid - 1)
                    
    def search(self, nums: List[int], target: int) -> int:
        low = 0
        high = len(nums) - 1
        while low <= high:
            mid = (low + high)//2
            if nums[mid] == target:
                return mid
            elif nums[mid] >= nums[low]:
                if target < nums[mid] and target >= nums[low]:
                    high = mid - 1
                else:
                    low = mid + 1
            else:
                if target > nums[mid] and target <= nums[high]:
                    low = mid + 1
                else:
                    high = mid - 1
        return -1

class Solution:
    arr = {}
    def climbStairs(self, n: int) -> int:
        if n == 1: return 1
        if n == 2: return 2
        if not self.arr.get(n-1): self.arr[n-1] = self.climbStairs(n-1)
        if not self.arr.get(n-2): self.arr[n-2] = self.climbStairs(n-2)
        return self.arr[n-1] + self.arr[n-2]

    def climbStairs(self, n: int) -> int:
        dp = [0]*(n+1)
        self.helper(n,dp)
        return dp[n]
    
    
    def helper(self,n,dp):
        if n<0:
            return 0
        if n==0:
            return 1
        if dp[n]!=0:
            return dp[n]
        left = self.helper(n-1,dp)
        right = self.helper(n-2,dp)
        dp[n] = left+right
        return left+right

        # It is classic dp problem based on fibonacci number. It can be easily done using tabulations method as well. Thanks!!

class Solution:
    def climbStairs(self, n: int) -> int:
        if n<=3: return n
        first = 2
        second = 3
        
        for i in range(3,n):
            ans = first+second
            first, second = second, ans
        return ans

class Solution:
    def climbStairs(self, n: int) -> int:
    
        if n<2:
            return n
        
        # ways={}
        # ways[n-1]=1
        # ways[n-2]=2
        # for i in range(n-3,-1,-1):
        #     ways[i] = ways[i+1]+ways[i+2]
        # return ways[0]
    
        # Without the dictionary
        ways=2
        a,b=1,2
        for i in range(n-3,-1,-1):
            ways = a+b
            a,b = b,ways
        return ways 

from typing import Optional


class Solution:
    """
    n = len(coins)
    m = amount
    
    Solution #1 (brute-force - without memo):
    Time -> O(n^m * m)
    Space -> O(m)
    
    Solution #2 (memo):
    Time -> O(n*m^2)
    Space -> O(m)
    
    """
    def coinChange(self, coins: List[int], amount: int) -> int:
        depth = self.seekDepth(coins, amount, {})
        if depth is None:
            return -1
        return depth
        
    def seekDepth(self, coins: List[int], amount: int, memo: dict) -> Optional[int]:
        if amount in memo:
            return memo[amount]
        elif amount == 0:
            return 0
        elif amount < 0:
            return None
        else:
            min_depth = None
            for coin in coins:
                remaining_amount = amount - coin
                depth = self.seekDepth(coins, remaining_amount, memo)
                if depth is not None:
                    depth += 1
                    if min_depth is None or depth < min_depth:
                        min_depth = depth
            memo[amount] = min_depth
            return min_depth

# BFS Solution

from  collections import deque

class Solution:
	def coinChange(self, coins: List[int], amount: int) -> int:
		qset = set()
		q = deque()
		qset.add(amount)
		q.append((0, amount)) # (num coins,  amount)
		while q:
			(num_coins, rest)  = q.popleft()
			if rest == 0:
				return num_coins
			if rest < 0:
				continue
			for coin in coins:
				x = rest - coin
				if  x not in qset:
					q.append((num_coins + 1, x))
					qset.add(x)
		return -1

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # Example: coins = [1 2 5], amount = 5
        
        # amount   =  0  1   2   3   4   5
        # mincoins = [0  inf inf inf inf inf]
        mincoins = [float('inf') for _ in range(amount + 1)]
        # Only 1 way to make `0` change: return 0 coin
        mincoins[0] = 0
        
        # 1st pass: coin = 1 -> mincoins = [0 1 2 3 4 5]
        # 2nd pass: coin = 2 -> mincoins = [0 1 1 2 2 3]
        # 3rd pass: coin = 5 -> mincoins = [0 1 1 2 2 1]
        for coin in coins:
            for target in range(1, len(mincoins)):
                # If coin can be used to make up part of the amount
                if coin <= target:
                    # Try use it and check what the min number of coins to make up 
                    # the rest `mincoins[target-coin]` and add 1 (rest + current coin)
                    mincoins[target] = min(mincoins[target], mincoins[target-coin] + 1)
					
        # if mincoins[amount] couldn't be used then no 
		# combination of coins could make up target amount
        return mincoins[amount] if mincoins[amount] != float('inf') else -1

class Solution:
    def coinChange(self, coins, amount):
        @lru_cache(None)
        def dp(i):
            if i == 0: return 0
            if i < 0: return float("inf")
            return min(dp(i - coin) + 1 for coin in coins)
        
        return dp(amount) if dp(amount) != float("inf") else -1

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:

    # bisect_left(list, target, beg, end) 
    d=[nums[0]]
    for i in range(1,len(nums)):
        a=bisect.bisect_left(d,nums[i])
        if a==len(d):
            d.append(nums[i])
        else:
            d[a]=nums[i]
    return len(d)

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
		# do this shift because dp[:num]
        m = min(nums)
        if(m <= 0):
            for i in range(len(nums)):
                nums[i] += -m + 1

        dp = [0]*(max(nums)+1)
        
        for num in nums:
            dp[num] = max(dp[:num])+1
        
        return max(dp)

class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # develop a dp table with 1 for all elements
        dp = [1] * len(nums)
        
        # for each element, look back and find the longest subsequence for all elements behind it
        # then choose the one with largest subsequence and add 1 to it
        for i in range(1, len(nums)): # O(n)
            dp[i] = 1 + max([dp[index] for index in range(i) if nums[index] < nums[i]] or [0]) #O(n) in the loop --> O(n^2)
        return max(dp)

from collections import defaultdict
class Solution:
    def lengthOfLIS(self, nums):

        dp = defaultdict(list)
        dp[nums[0]] = 1
        
        i = 1
        
        while i < len(nums):
            if nums[i] < min(dp.keys()):
                dp[nums[i]] = 1
            else:
                max_length = max([l for v, l in dp.items() if nums[i] > v]+[0])
                dp[nums[i]] = max_length+1

            i += 1
        
        
        return max([l for n, l in dp.items()])

# Idea
# In array tails the value tails[idx] is the smallest number that would be the end of a sequence of length i+1. We iterate over the array nums while making sure to uphold this constraint. Each new number will be either inserted to array tails if it is larger than the largest tail we have encountered so far, or it will update the existing tail with a value that is less than or equal to it.

# Complexity
# Time: O(NlogN), because for each number we do a binary search in an array tails.
# Memory: O(N), since we may store up to N numbers in our array tails.

def lengthOfLIS(self, nums: List[int]) -> int:
	tails = []
	for n in nums:
		idx = bisect.bisect_left(tails, n)
		tails[idx:idx+1] = [n]
	return len(tails)

# bfs solution

visited = set() # set of strings that we've already visited and determined not to be useful
q = deque([s]) # the magical q that will allow us to BFS

while q:
    curString = q.popleft() # get the currentstring that we test
    
    if curString == "": # if it is nothing then we can make the word break
        return True
    
    for word in wordDict: # go through each word in word dictionary
    # first we check if the current word actually is a part of the string
    # by slicing the currentString from 0 to the len of the word, we can check if the prefix of the string and the word equals
    # Then we check if the rest of the string has already been previously checked. If it isn't then we add to queue and mark it visited
        if curString[0:len(word)] == word and curString[len(word)::] not in visited:
            q.append(curString[len(word)::])
            visited.add(curString[len(word)::])
return False # we couldn't make a word break

# Recursion with memoization

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        hash=defaultdict(list)
        for word in wordDict:
            hash[len(word)].append(word)
        
        def helper(i,dp):
            if i==len(s):
                return True
            if dp[i]!= -1:
                return dp[i]
            for key,value in hash.items():
                word=s[i:i+key]
                if word in value and helper(i+key,dp):
                    dp[i]=True    
                    return True
            dp[i]=False
            return False
        
        dp=[-1]*len(s)
        return helper(0,dp)

# Dynamic Programming

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        words=set(wordDict)
        dp=[False]*(len(s)+1)
        dp[0]=True
        for i in range(1,len(s)+1):
            for j in range(i):
                if dp[j] and s[j:i] in words:
                    dp[i]=True
                    break
        return dp[len(s)]

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        wordDict = set(wordDict)
        dp = [True] + [False] * len(s)
        for i in range(len(s)):
            if dp[i]:
                for j in range(i, len(dp)):
                    if s[i:j] in wordDict:
                        dp[j] = True
        return dp[-1]

import functools


class Solution:
    def __init__(self):
        self.words = None
        self.s = None
    
    @functools.cache
    def check(self, start) -> bool:
        s = self.s
        wordDict = self.words
        
        if start == len(s):
            return True
        if start > len(s):
            return False
        
        checked_starts = set()

        for word in wordDict:
            if s.startswith(word, start):
                new_start = start+len(word)
                if self.check(new_start):
                    return True
        return False
    
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        self.s = s
        self.words = wordDict
        return self.check(start=0)

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = [False for i in range(len(s) + 1)]
        dp[0] = True
        for i in range(1, len(s) + 1):
            for word in wordDict:
                length = len(word)
                if length <= i and word == s[i - length: i]:
                    dp[i] = dp[i] or dp[i - length]
        return dp[-1]

# You probably saw a lot of different solutions for this problem, but I can guess, that you did not saw this one. The idea is here is to use generating functions. Imagine, that we have coins 1, 2, 5. Let us create polynomial (x^1 + x^2 + x^5). Now, let us consider powers of this polynom. For example:

# (x^1 + x^2 + x^5)^ 1 = (x^1 + x^2 + x^5), nothing very intersting here.
# (x^1 + x^2 + x^5)^2 = x^2 + x^3 + x^6 + x^3 + x^4 + x^8 + x^6 + x^7 + x^10. What we have here. For example we can see that there is only one way to get target 2, because coefficient will be equal to 1 before x^2, there is 2 ways to get 6 and so on.
# In general if we have (x^1 + x^2 + x^5)^k, then coefficients will show us how many ways we can get one or another target, using exactly k coins.
# Now, we can use problem restrictions: that is that answer will always be less than 2^32, so if we use x = 2^32 and will work with operations in 2^32 base system, everything will be fine. Also we use python long numbers, which is very helpful here.

# Create T is our polynomial with x = 2^32.
# Define S = 1 and ans = 0.
# Iterate over powers of T. Each time we multiply S by T. However we do not want to have too big numbers, so we use & (1<<(32*t+32)) - 1, trick, which will give us last t + 1 digits in 2^32 base system.
# Update our answer: this is value S >> (32*t), that is value of t-th digit in our 2^32 base system.
# Complexity
# It is a bit difficult to estimate it like this, but on each operation and we have t of them we will work with number which have no more than O(32m) bits, where m is the biggest number among nums. So, total time complexity will be O(32*m*t), space complexity is O(32*m). However in practice it works quite fast.

# Code
class Solution:
    def combinationSum4(self, nums, t):
        T = sum([1<<(32*n) for n in nums])
        S, ans = 1, 0
        for i in range(t):
            S = (S*T) & (1<<(32*t+32)) - 1
            ans += S >> (32*t)
            
        return ans
# Remark
# If you will write it as oneliner, I will be happy.

# If you have any questions, feel free to ask. If you like solution and explanations, please Upvote!

class Solution(object):
    
    def combinationSum4(self, nums, target):
        dp = [0] * target
        nums = sorted(nums)
        for i in range(target):
            for j in nums:
                if j >i+1:
                    break
                elif j == i+1:
                    dp[i]+=1
                elif j < i+1:
                    dp[i]+=dp[i-j]
        return dp[-1]

# I've been trying to finish the four problems in in this series - so far I've only finished LC 39: Combination Sum. This recursive is kinda messy, but I think it is a natural extension of LC 39, and it's what I first thought of when I saw this problem.

# LC 39, asks for to return any combinations of the nums array (with any multiplicity of that taken number). If you've solve LC 39 the naive backtracking solution comes to mind. And we can extend that framework to this problem.

# Idea:
# * whenever we take a number, we decrement the target by that taken number
# * but we need we have two cases on when and now to recurse
# * case 1: if when we take that number we are still > 0, well we want to keep recursing on that number!
# * case 2: otherwise go on to the next number and recurse
# * we actually don't build the paths, but just increment
# * and of course lets hash with a physical memo instead of decorating with @lru

class Solution(object):
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        memo  = {}
        def rec(idx,target):
            count = 0
            #careful not to keep going after target diminishes
            if target <= 0:
                if target == 0:
                    #valid path
                    return 1
                return 0
            if (idx,target) in memo:
                return memo[(idx,target)]
            for i in range(idx,len(nums)):
                candidate = nums[i]
                if target - candidate > 0:
                    #stay on the index
                    count += rec(idx,target-candidate)
                else:
                    #move up the index
                    count += rec(idx+1,target-candidate)
            memo[(idx,target)] = count
            return count
        return rec(0,target)

class Solution(object):
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        f = [0 for _ in range(target + 1)]
        f[0] = 1
        for i in range(target + 1):
            for num in sorted(nums):
                if num > i :
                    break
                f[i] += f[i - num]

        return f[target]

# Implementation by DFS in Python:

class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
    
        # look-up table
        combination_count = {}
        
        # --------------------------------
        
        def dfs(wanted):
            
            ## base cases:
            
            if wanted < 0:
                # stop condition for negative number
                return 0
            
            elif wanted == 0:
                # stop condition for perfect match
                return 1
            
            if wanted in combination_count:
                # quick resposne by look-up table
                return combination_count[wanted]
            
            ## general cases
            count = 0
            
            for number in nums:
                
                next_wanted = wanted - number

                count += dfs(next_wanted)
            
            combination_count[wanted] = count
            return count
        
        # --------------------------------
        
        return dfs(wanted=target)

        # Version 1
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        memo = {}
        
        def dfs(t: int) -> int:
            if t <= 0: return int(t == 0)
            if t in memo: return memo[t]
            memo[t] = sum( dfs(t - num) for num in nums )
            return memo[t]
        
        return dfs(target)

# Version 2

class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        
        @lru_cache(None)
        def dfs(t: int) -> int:
            if t <= 0: return int(t == 0)
            return sum( dfs(t - num) for num in nums )
        
        return dfs(target)

# Version 3

class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        
        @cache
        def dfs(t: int) -> int:
            if t <= 0: return int(t == 0)
            return sum( dfs(t - num) for num in nums )
        
        return dfs(target)

def combinationSum4(self, nums: List[int], target: int) -> int:
    	count = [0] * (target + 1)
	count[0] = 1
	for countPerIndex in range(1, target + 1):
		for num in nums:
			if num <= countPerIndex:
				count[countPerIndex] += count[countPerIndex - num]

	return count[target]

# 213. House Robber II
# Medium

# 2804

# 65

# Add to List

# Share
# You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police if two adjacent houses were broken into on the same night.

# Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

 

# Example 1:

# Input: nums = [2,3,2]
# Output: 3
# Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.
# Example 2:

# Input: nums = [1,2,3,1]
# Output: 4
# Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
# Total amount you can rob = 1 + 3 = 4.
# Example 3:

# Input: nums = [0]
# Output: 0
 

# Constraints:

# 1 <= nums.length <= 100
# 0 <= nums[i] <= 1000

def rob(self, nums: List[int]) -> int:
    if len(nums) <= 2:
        return self.rob1(nums)
    return max(nums[0] + self.rob1(nums[2:len(nums)-1]), self.rob1(nums[1:]))
    
# The following is any solution to Robber I problem.
def rob1(self, nums: List[int]) -> int:
    if len(nums) == 0:
        return 0
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums[0], nums[1])
    m_2 = nums[0]
    m_1 = max(nums[0], nums[1])
    for n in range(2, len(nums)):
        m_n = max(nums[n] + m_2, m_1)
        m_2 = m_1
        m_1 = m_n
    return m_n

class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) < 3:
            return max(nums)
        a, b = nums[0], max(nums[0], nums[1])
        c, d = nums[1], max(nums[1], nums[2])

        for i in range(2, len(nums) - 1):
            j = i + 1
            a, b = b, max(nums[i] + a, b)
            c, d = d, max(nums[j] + c, d)
            
        return max(b,d)

class Solution:
    def rob(self, nums: List[int]) -> int:
        l = len(nums)
        if l==1:
            return nums[0]
        l-=1
        rob1 = 0
        rob2 = 0
        for i in range(0,l):
            temp = max(rob1+nums[i], rob2)
            rob1,rob2 = rob2,temp
        ans1 = rob2
        rob1, rob2 = 0,0
        l+=1
        for i in range(1,l):
            temp = max(rob1+nums[i], rob2)
            rob1,rob2 = rob2,temp
        ans2 = rob2
            
        return max(ans1,ans2)

# Rob the house twice, under different constraints. In one scenario, you can look at everything except the last house. In the other scenario, you cannot look at the first house.

class Solution:
    def rob(self, nums: List[int]) -> int:
      if len(nums) < 2: return nums[0]
      house1 = [0] * len(nums)
      house2 = [0] * len(nums)
      
      house1[0] = nums[0]
      house1[1] = max(nums[1], house[0])
      
      house2[0] = 0
      house2[1] = nums[1]
      
      for i in range(2, len(nums)):
        house1[i] = max(house1[i - 1], nums[i] + house1[i - 2])
        house2[i] = max(house2[i - 1], nums[i] + house2[i - 2])
      
      return max(house2[-1], house1[-2])

class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums : return 0
        if len(nums) < 3: 
            return max(nums)
        
        nums1 = nums[:-1]
        if len(nums1) < 3 : return max(nums1)
        a = nums1[0]
        b = max(nums1[0],nums1[1])
        for i in range(2,len(nums1)) : 
            Curr_max = max(a+nums1[i], b)
            a = b 
            b = Curr_max 
        
        nums2 = nums[1:]
        # if not nums2 : return 0
        if len(nums2) < 3 : return max(nums2)
        c = nums2[0]
        d = max(nums2[0],nums2[1])
        for i in range(2,len(nums2)) : 
            Curr_max = max(c+nums2[i], d)
            c = d 
            d = Curr_max 
        return max(b,d)

# 91. Decode Ways
# Medium

# 4292

# 3443

# Add to List

# Share
# A message containing letters from A-Z can be encoded into numbers using the following mapping:

# 'A' -> "1"
# 'B' -> "2"
# ...
# 'Z' -> "26"
# To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above (there may be multiple ways). For example, "11106" can be mapped into:

# "AAJF" with the grouping (1 1 10 6)
# "KJF" with the grouping (11 10 6)
# Note that the grouping (1 11 06) is invalid because "06" cannot be mapped into 'F' since "6" is different from "06".

# Given a string s containing only digits, return the number of ways to decode it.

# The answer is guaranteed to fit in a 32-bit integer.

 

# Example 1:

# Input: s = "12"
# Output: 2
# Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).
# Example 2:

# Input: s = "226"
# Output: 3
# Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
# Example 3:

# Input: s = "0"
# Output: 0
# Explanation: There is no character that is mapped to a number starting with 0.
# The only valid mappings with 0 are 'J' -> "10" and 'T' -> "20", neither of which start with 0.
# Hence, there are no valid ways to decode this since all digits need to be mapped.
# Example 4:

# Input: s = "06"
# Output: 0
# Explanation: "06" cannot be mapped to "F" because of the leading zero ("6" is different from "06").
 

# Constraints:

# 1 <= s.length <= 100
# s contains only digits and may contain leading zero(s).

def numDecodings(self, s: str) -> int:
    mp = collections.defaultdict(int)
    def dfs(s):
        if s in mp:
            return mp[s]
        if not s :
            return 1
        if s[0] == "0":
            return 0
        if len(s)==1:
            return 1
        ways = dfs(s[1:])
        if s[0] == "1"  or (s[0]=="2" and s[1] <"7"):
            ways += dfs(s[2:])
        mp[s] = ways
        return ways
    return  dfs(s)

# Approach

# At each step you either take 1 or 2 digits. You take 1 digit if it is not zero and you take 2 digit if it doesn't start with zero and less than or equalt to 26
# We'll recursively call from 1st index in one call and 2nd index in another call
# We need base conditions
# i. if the length of the string is 1, return 1 if it is not equal to zero
# ii. similarly, for length 2, you check if it starts with 0 and if it is less than or equal to 26. Again for values less than or equal to 26, there are edge cases at 10 and 20. You handle them and return the values
# Add memoization
# Simple! Vote if you liked this solution

class Solution:
    def numDecodings_recur(self, s: str, cache: dict) -> int:
        if s in cache:
            return cache[s]
        if len(s) == 1 and s != '0':
            return 1
        if len(s) == 2:
            if s.startswith('0'):
                return 0
            if int(s) <= 26:
                if int(s) == 10:
                    return 1
                if int(s) == 20:
                    return 1
                return 2
        count = 0
        x = s[:1]
        y = s[:2]
        if x != "" and int(x) != 0:
            count += self.numDecodings_recur(s[1:], cache)
        if y != "" and not y.startswith('0') and int(y) <= 26:
            count += self.numDecodings_recur(s[2:], cache)
        cache[s] = count
        return count
    
    def numDecodings(self, s: str) -> int:
        return self.numDecodings_recur(s, {})

class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        # if the array is empty or start with 0 we return 0
        
        if not s or s[0] == '0':
            return 0
        
        # initialize a dp table with length len(s) + 1
        # index i in dp correlates to index i-1 in the string
        # first element of dp is a dummy 1 ( to capture first two leading digits if they exist)
        # second element corresponds to the first element
        dp = (len(s) + 1) * [0]
        dp[:2] = [1,1]
        
        for i in range(2, len(dp)): # updating dp
            # if the current element is a valid number, the total number of decodes is the same as 
            # the number of decodes right before it
            if 0 < int(s[i-1]) <= 9:
                dp[i] += dp[i-1]
            # now if current element forms a valid number with the previous element too, we will add 
            # the same number as two previous elements (because we need to remove element before this)
            if 10 <= int(s[i-2:i]) <= 26:
                dp[i] += dp[i-2]
            
        return dp[-1]
		
		```

class Solution:
    def __init__(self):
        self.hashTable = {}
		
    def numDecodings(self, s: str) -> int:
        if s in self.hashTable: # Getting Cached Value If Any for string
            return self.hashTable[s]
        
        if s=='0': #base case if string is actually zero
            return 0
        
        if len(s)<=1: #base case if s = '' or a num between 1-9
            return 1
                  
        ways = 0
        if s[0]!='0': # dont add ways if string starts with zero
            ways += self.numDecodings(s[1:]) #take one character
            if int(s[:2])<=26: # take two only if <=26
                ways += self.numDecodings(s[2:])
        self.hashTable[s] = ways #cache values
        return ways
	```

Everything related to the letters can be totally skipped here, cause at the end the problem can be summarized like:
Number of ways to combine string numbers in groups of 1 or 2, where substrings that have leading zeros, are zero or bigger than 26 are not allowed.

Example: 226

We have 3 options: [2, 2, 6], [22, 6], or [2,26]

We can imagine this as a tree where every node we pick from 1-2 chars and the rest is the remaining.

                 226
			/            \                 
		(2,26)          (22,6)               
      /	       \
   (22,6)    (226, X)
  /             
(226, X)
When we iterated over the whole string, it's over.

Problem is that there are times that we are repeating the subproblems, for example we end up twice to a node (22,6), that means that we have 22 selected and we can combinate it with the remaining 6.

In order to avoid twice the calculation (even if in this example is really not a problem), we can cache the substrings.

class Solution:
    def helper(self, s, position, memo):
        if position >= len(s):
            return 1
        
        portion = s[position:]
        
        if portion in memo:
            return memo[portion]
        
        result = 0
        for index in range(1,3):
            if position + index > len(s):
                continue
                
            number = s[position:position+index]
            
            if  number[0] == "0" or int(number) > 26 or int(number) == 0:
                continue
                
            result += self.helper(s, position+index, memo)
            
        memo[portion] = result
        
        return memo[portion]
        
    def numDecodings(self, s: str) -> int: 
        memo = {}
        return self.helper(s, 0, memo)
O(N) space complexity, because we build a memo that at least will be as long as the string size. Also the recursive stack is the height of the tree, which is related to string length too.
O(N) time, because we will visit at least N nodes for each string char. With memoization we prune the rest of the tree calls.

Hope it helps !

62. Unique Paths
Medium

5003

244

Add to List

Share
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?

 

Example 1:


Input: m = 3, n = 7
Output: 28
Example 2:

Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Down -> Down
2. Down -> Down -> Right
3. Down -> Right -> Down
Example 3:

Input: m = 7, n = 3
Output: 28
Example 4:

Input: m = 3, n = 3
Output: 6
 

Constraints:

1 <= m, n <= 100
It's guaranteed that the answer will be less than or equal to 2 * 109.

Remember the recursive solutions will always be slow, even if they are memoized!
This solution just to learn.

class Solution:
    def uniquePaths(self, m: int, n: int, memo = None) -> int:
        if memo == None: memo = {}
        key = str(m) + "," + str(n)
        if key in memo: return memo[key]
        if m == 1 or n == 1: return 1
        
        memo[key] = self.uniquePaths(m-1, n, memo) + self.uniquePaths(m, n-1, memo)
        
        return memo[key]

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        table = [[0]*(n+1) for _ in range(m+1)]
        table[1][1] = 1
        
        for i in range(m+1):
            for j in range(n+1):
                current = table[i][j]
                if j+1 <= n: table[i][j+1] += current
                if i+1 <= m: table[i+1][j] += current
                    
        return table[-1][-1]

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        table = [[0]*(n+1) for _ in range(m+1)]
        table[1][1] = 1
        
        for i in range(m+1):
            for j in range(n+1):
                current = table[i][j]
                if j+1 <= n: table[i][j+1] += current
                if i+1 <= m: table[i+1][j] += current
                    
        return table[-1][-1]

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        count = [[0 for x in range(n)] for y in range(m)]
        for i in range(0,m):
            count[i][0] = 1
        for j in range(0,n):
            count[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):            
                count[i][j] = count[i-1][j] + count[i][j-1]
        return count[m-1][n-1]

class Solution:
    
    def uniquePaths(self, m: int, n: int) -> int:
        # populate the blank bottom up dp table
        dp = []
        for i in range(m):
            columns = [""] * n
            dp.append(columns)

        # all 0th row and columns are 1, there will always only be 1 way
        for row in dp:
            row[0] = 1
        for index in range(n):
            dp[0][index] = 1

        # populate the entire dp table
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]

        # return the lower right corner of dp table
        # this takes theta(n x m)
        return dp[m-1][n-1]

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        @lru_cache(None)
        def explore(i, j):
            if i == m and j == n:
                return 1
            if i == m:
                return explore(i, j+1)
            if j == n:
                return explore(i+1, j)
            
            return explore(i+1, j) + explore(i, j+1)
        
        return explore(1, 1)

The maximum we can (and will) go to the right (+x axis) direction is n because the total number of columns is n, and the maximum we can (and will) go to the down (-y axis) direction is m because the total number of rows is m.
To reach the target, we have to go n RIGHTs and m DOWNs, now we can create multiple permutations with repetitions in O(1) time (if factorials are already in cache) to reach there, i.e. (m+n)!/(m!n!) because RIGHT is repeated n times and DOWN is repeated m times. e.g. DDR, DDR, DRD, DRD, RDD, RDD are permutations with repetitions and we can remove the redundant ones to get the final permutations as DDR, DRD, RDD (2+1)!/(2!1!)=3
We keep factorial (to be reused as cache) if needed in future
Since both m, n and n, m will return the same result, we can reduce computation of factorial by taking m as the maximum and n as the minimum of the two numbers
class Solution:
    factorial = [1, 1]

    def fact(self, number: int) -> int:
        if number > len(self.factorial) - 1:
            self.factorial.append(number * self.fact(number - 1))
        return self.factorial[number]

    def uniquePaths(self, m: int, n: int) -> int:
        m, n = m - 1, n - 1
        m, n = max(m, n), min(m, n)
        numerator = 1
        for i in range(m + 1, m + n + 1):
            numerator *= i
        denominator = self.fact(n)
        unique_paths = numerator // denominator
        return unique_paths

def uniquePaths(self, m: int, n: int) -> int:
    ups = [[0] * n for _ in range(m)]
    for i in range(n):
        ups[0][i] = 1
    for i in range(m):
        ups[i][0] = 1
    self.uniquePathHelper(m-1, n-1, ups)
    return ups[m-1][n-1]
    
def uniquePathHelper(self, m,n,ups):
    if (ups[m][n] != 0):
        return True
    if (self.uniquePathHelper(m-1, n, ups) and self.uniquePathHelper(m, n-1, ups)):
        ups[m][n] = ups[m-1][n] + ups[m][n-1]
        return True

55. Jump Game
Medium

6183

423

Add to List

Share
Given an array of non-negative integers nums, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.

 

Example 1:

Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
Example 2:

Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.
 

Constraints:

1 <= nums.length <= 3 * 104
0 <= nums[i] <= 105

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if len(nums) <= 1:
            return True
        if nums[0] ==0:
            return False

        maxReach = nums[0]
        steps = maxReach
        for i in range(1,len(nums)-1):
            steps -=1
            maxReach = max(maxReach,nums[i]+i)
            if steps == 0:
                steps = maxReach - i
                if steps == 0:
                    return False
        return True

 class Solution:
        def canJump(self, nums: List[int]) -> bool:
        reach = False
        index = []
        if len(set(nums)) == 1 and nums[0] == 1:
            return True
        n = len(nums)
        for i in range(0,n):
            if nums[i]+i >= n-1:
                index.append(1)
            elif i != n-1 and nums[i] == 0:
                index.append(0)
            else:
                index.append(0)
        flag = 0
        for i in index:
            if i == 1:
                flag  = 1
                break
        if flag == 0:
            return reach
        for j in reversed(range(0,n)):
            if index[j] == 1:
                tmp = []
                for k in range(0,j):
                    if nums[k] + k >= j:
                        tmp.append(1)
                    else:
                        tmp.append(0)
                for l in range(0,len(tmp)):
                    if tmp[l] == 1:
                        index[l] = 1
        if index[0] == 1:
            reach = True
        return reach

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_index_reached =0
        if len(nums)<=1:
            return 1
        for i in range(len(nums)):
            max_index_reached = max(max_index_reached , i+nums[i])
            if max_index_reached <= i:
                return 0
            if max_index_reached >= len(nums)-1:
                return 1
        return 0

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if len(nums)==1: return True
        L=[0]*(len(nums)-1)
        L[0]=nums[0]
        for i in range(1,len(L)):
            if L[i-1]==0:return False
            L[i]=max(L[i-1]-1,nums[i])
        return True if L[-1]>=1 else False

class Solution:
    def canJump(self, nums: List[int]) -> bool:

        # set last element as target
        target = len(nums)-1
        
        for i in range(target-1,-1,-1):
            # checking either we can cross our target or not from the indeces before it.
            if i + nums[i] >= target:
                # setting new target.
                target = i
        
        return target == 0 # checking we are at 0th index or not.

def canJump(self, nums: List[int]) -> bool:
    goal = len(nums)-1
    for i in range(len(nums)-1,-1,-1):
        if i + nums[i] >= goal:
            goal = i
    return True if goal == 0 else False

class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        for index, num in enumerate(nums):
            if num == 0 and index != len(nums)-1:
                canMakeJump = False
                for i in range(index):
                    if nums[i] > index - i:
                        canMakeJump = True
                if not canMakeJump:
                    return False
        return True

206. Reverse Linked List
Easy

6860

129

Add to List

Share
Given the head of a singly linked list, reverse the list, and return the reversed list.

 

Example 1:


Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]
Example 2:


Input: head = [1,2]
Output: [2,1]
Example 3:

Input: head = []
Output: []
 

Constraints:

The number of nodes in the list is the range [0, 5000].
-5000 <= Node.val <= 5000
 

Follow up: A linked list can be reversed either iteratively or recursively. Could you implement both?


# Iterative
def reverseList(self, head):
        
    if head == None:
        return None
    
    node = head
    previous_node = None
    
    while(node != None):
        next_node = node.next
        node.next = previous_node
        previous_node = node
        node = next_node
    return previous_node

def recur(self,curHeadNode,reverseThisNode,nextNode):
      reverseThisNode.next = curHeadNode  # reverse the middle node to point to the first node
  if nextNode is None:  # when no more node to proceed, the whole list is reversed, return the new head
    return reverseThisNode
  return self.recur(reverseThisNode,nextNode,nextNode.next)  # proceed recursion by moving the 3-node window forward 

def reverseList(self, head: ListNode) -> ListNode:
  if not head or not head.next:   #house keeping
    return head 
  firstToBeReversedNode = head.next
  head.next=None  # make head to point to null
  return self.recur(head,firstToBeReversedNode,firstToBeReversedNode.next)

Not the fastest, but i think is pretty straightforward.
Basically, the recur function takes 3 consecutive nodes at a time, and reverse the middle node to link to the first node, and place the third node as the next "to be reversed" node during the next recur function call. The recursion stops when no more node to proceed, which means the whole list is reversed, return the new head.

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        temp_list = self.fromListNodeToList(head)
        print(temp_list)
        temp_list.reverse()
        temp_list_node = self.from_list_listNode(temp_list)
        return temp_list_node


    def from_list_listNode(self, list_numbers: List[int]) -> List[ListNode]:
        # define the scenario where the list is None
        # define the scenario where the list is empty
        if list_numbers is None or list_numbers == []:
            return ListNode("")
        # the list is not empty
        # loop through the list
        temp = ListNode(list_numbers[0])
        current = temp
        for i in range(len(list_numbers) - 1):
            # insert list_numbers entry to the ListNode.val
            current.next = ListNode(list_numbers[i + 1])
            # set the next
            current = current.next
        return temp

    def fromListNodeToList(self, list_node: ListNode) -> List:
        temp = list()
        try:
            list_node.val
        except:
            return []
        if list_node.val == "" or list_node.val == None:
            return []
        else:
            while list_node:
                try:
                    temp.append(list_node.val)
                except:
                    temp.append(list_node)
                    break
                list_node = list_node.next
        return temp

class Solution:
    def reverseList(self, head):
        if head is None or head.next is None:
            return head
        result = self.reverseList(head.next)
        head.next.next,head.next = head,None
        return result

141. Linked List Cycle
Easy

4434

606

Add to List

Share
Given head, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

Return true if there is a cycle in the linked list. Otherwise, return false.

 

Example 1:


Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).
Example 2:


Input: head = [1,2], pos = 0
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 0th node.
Example 3:


Input: head = [1], pos = -1
Output: false
Explanation: There is no cycle in the linked list.
 

Constraints:

The number of the nodes in the list is in the range [0, 104].
-105 <= Node.val <= 105
pos is -1 or a valid index in the linked-list.
 

Follow up: Can you solve it using O(1) (i.e. constant) memory?

def hasCycle(self, head: ListNode) -> bool:
    p1=head
    p2=head
    while p2 and p2.next:
        p1=p1.next
        p2=p2.next.next
        if p1==p2:#if loop exists, this condition will be true at some point of time
            return True
        
    
    return False

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        
        if head is None:
            return False
        
        head.found = True
        curr = head.next
        
        while curr:
            if hasattr(curr, 'found'):
                return True
            curr.found = True
            curr = curr.next
            
        return False

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        start = head
        counter = {}
        while start:
            if start in counter:
                counter[start] += 1
                return True
            else:
                counter[start] = 1
            
            start = start.next
            
        for key in counter:
            if counter[key] > 1:
                return True
            else:
                False

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                return True
        return False

def hasCycle(self, head: ListNode) -> bool:
        if not head:                                             # Check if our head is None to return false immediately 
            return False
        sp, fp = head, head.next                                 # Set slow pointer to head and fast pointer to head.next 
        while fp is not sp:                                      # while our sp != fp
            if fp is None or fp.next is None:                    # We can skip checking the slow pointer here 
                return False                                     # if there's an end return False, no cycle
            sp = sp.next                                         # increment our slow pointer
            fp = fp.next.next                                    # increment our fast pointer twice
        return True                                              # if we broke the while loop conditions without returning False we can return True

# In this problem we can check for a cycle by seeing if we ever get directed via next to a ListNode we have already seen. We will keep track of the ListNodes as we iterate through the Linked List and if we ever come across a ListNode we have seen before then we know there is a cycle. If we ever reach None, then we know there was not a cycle.

def hasCycle(self, head: ListNode) -> bool:
	nodes = set()
    current = head
    while current:
		if current in nodes:
			return True
        nodes.add(current)
        current = current.next
    return False
# If we wanted to avoid using memory storing the nodes there is another approach we could take. We will
# use the constaint of 10^4 possible nodes to our advantage for this. The idea is to keep looping through
# nodes and if we come across a None at any point we do not have a cycle. If we go through 10^4 +1
# nodes and we still have not found a None, then we know we must have a cycle because there cannot
# be that many nodes in the Linked List.

def hasCycle(self, head: ListNode) -> bool:
	for i in range(10**4 + 1):
		if not head:
			return False
        head = head.next
return True

21. Merge Two Sorted Lists
Easy

6669

765

Add to List

Share
Merge two sorted linked lists and return it as a sorted list. The list should be made by splicing together the nodes of the first two lists.

 

Example 1:


Input: l1 = [1,2,4], l2 = [1,3,4]
Output: [1,1,2,3,4,4]
Example 2:

Input: l1 = [], l2 = []
Output: []
Example 3:

Input: l1 = [], l2 = [0]
Output: [0]
 

Constraints:

The number of nodes in both lists is in the range [0, 50].
-100 <= Node.val <= 100
Both l1 and l2 are sorted in non-decreasing order.

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        cur = root = ListNode()
    
        while l1 and l2:
            if l1.val <= l2.val:
                cur.next = cur = ListNode(l1.val)
                l1 = l1.next
            else:
                cur.next = cur = ListNode(l2.val)
                l2 = l2.next

        cur.next = l1 or l2
        return root.next

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        head_node = ListNode(0)
        root = head_node
        
        while l1 and l2:
            
            v1 = l1.val
            v2 = l2.val
            
            if v1<v2:
                head_node.next = l1
                l1 = l1.next
            else:
                head_node.next = l2
                l2 = l2.next
                
            head_node = head_node.next
        
        if l1:
            head_node.next = l1
        if l2:
            head_node.next = l2
        
        return root.next

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        l = ListNode()
        head = l
        if not l1 and not l2:
            return None
        elif not l1:
            return l2
        elif not l2:
            return l1
        while(l1 and l2):
            if l1.val <= l2.val:
                s = l1
                l1 = l1.next
            else:
                s = l2
                l2 = l2.next
            s.next = None
            l.next = s
            l = s
        while(l1):
            s = l1
            l1 = l1.next
            s.next = None
            l.next = s
            l = s
        while(l2):
            s = l2
            l2 = l2.next
            s.next = None
            l.next = s
            l = s
        return head.next

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        
        l3=root=ListNode(0)
        
        while(l1 or l2):
            
            if(l1 and l2):
                if(l1.val<=l2.val):
                    l3.next=ListNode(l1.val)
                    l3=l3.next
                    l1=l1.next
                
                else:
                    l3.next=ListNode(l2.val)
                    l3=l3.next
                    l2=l2.next
                    
            elif(l1):
                l3.next=ListNode(l1.val)
                l3=l3.next
                l1=l1.next
                
            elif(l2):
                l3.next=ListNode(l2.val)
                l3=l3.next
                l2=l2.next
        
        return root.next

23. Merge k Sorted Lists
Hard

7094

356

Add to List

Share
You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.

 

Example 1:

Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6
Example 2:

Input: lists = []
Output: []
Example 3:

Input: lists = [[]]
Output: []
 

Constraints:

k == lists.length
0 <= k <= 10^4
0 <= lists[i].length <= 500
-10^4 <= lists[i][j] <= 10^4
lists[i] is sorted in ascending order.
The sum of lists[i].length won't exceed 10^4.

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        '''
        Divide and Conquer: Nlogk
        
                       [[1,4,5],[1,3,4],[2,6]]
                       /                    \
                [[1,4,5],[1,3,4]]           [[2,6]]
                /       \
        [[1,4,5]]       [[1,3,4]]
        
                            k
                        /       \
                    k/2         k/2
                /       \
            k/4         k/4
            
        levels = logk at each level number of nodes are N
        So, its Nlogk solution
                
        '''
        
        def merge(a,b):
            if a is None and b is None: return
            if a is None: return b
            if b is None: return a
            result = None
            if a.val<b.val:
                result = a
                result.next = merge(a.next,b)
            else:
                result = b
                result.next = merge(a,b.next)
            return result
        
        def mergeUtil(lists,low,high):
            if low == high: return lists[low]
            mid = low + (high-low)//2
            left = mergeUtil(lists,low,mid)
            right = mergeUtil(lists,mid+1,high)
            return merge(left,right)
        
        if not lists or len(lists)==0: return
        return mergeUtil(lists,0,len(lists)-1)

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        lis=[]
        for i in lists:
            head = i
            while head:
                lis.append(head.val)
                head = head.next
        lis= sorted(lis)
        if len(lis)!=0:
            k = ListNode()
            k.val = lis[0]
            del lis[0]
            head = k
            out = head
            for i in lis:
                k = ListNode()
                k.val = i
                head.next = k
                head = head.next
            return out

from heapq import heappush, heappop


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:

        # initiate a dummy head for the linked list to be used to append next elements
        head = ListNode(-1)
        current = head
        # initiate an early heap that gets the head of all linked lists and move them all forward
        heap = [] # this heap will have the size of len(lists) => k
        for i, node in enumerate(lists):
            if node:
                heappush(heap, [node.val,i])
                lists[i] = node.next # move them one forward
        # now, for each element that we drop, we will add an element from the same list
        while heap:
            current_min, current_index = heappop(heap) 
            if lists[current_index]:
                heappush(heap, [lists[current_index].val,current_index])
                lists[current_index] = lists[current_index].next
            # traverse the tree
            current.next = ListNode(current_min)
            current = current.next
            
        return head.next

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class MyListNode(ListNode):
    def __init__(self, listnode):
        super().__init__(listnode.val, listnode.next)
        
    def __lt__(self, other):
        while True:
            if not self and not other:
                return 0
            if not self:
                return -1
            if not other:
                return 1
            if self.val < other.val:
                return -1
            elif self.val > other.val:
                return 1
            else:
                self = self.next
                other = other.next

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        heap = []
        
        for lst in lists:
            if lst:
                heapq.heappush(heap, (lst.val, MyListNode(lst)))
            
        dummy = ListNode()
        result = dummy
        while heap:
            value, lst = heapq.heappop(heap)
            result.next = ListNode(value)
            result = result.next
            lst = lst.next
            if lst:
                heapq.heappush(heap, (lst.val, MyListNode(lst)))
        
        return dummy.next

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:

    # Create dummy node to start
    pre_head = ListNode()
    start_node = pre_head
    heap = []
    
    # Iterate through each node and append all node values into a heap, heap will sort it 
    for node in lists:
        curr = node
        while curr:
            heapq.heappush(heap, curr.val)
            curr = curr.next
    
    # We have the dummy node, we pop the val from min heap, and create listNode.
    # Essentially we are creating a new linked lists instead of sorting it in place
    while heap:
        start_node.next = ListNode(heapq.heappop(heap))
        start_node = start_node.next
        
        
    return pre_head.next # Dummy node is not the real start head
"""
This method, we don't need compare every elements in the lists with each other, although it would be a more optime solution.
time: O(n + nlogn)
space: O(n)

"""

19. Remove Nth Node From End of List
Medium

5357

309

Add to List

Share
Given the head of a linked list, remove the nth node from the end of the list and return its head.

Follow up: Could you do this in one pass?

 

Example 1:


Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]
Example 2:

Input: head = [1], n = 1
Output: []
Example 3:

Input: head = [1,2], n = 1
Output: [1]
 

Constraints:

The number of nodes in the list is sz.
1 <= sz <= 30
0 <= Node.val <= 100
1 <= n <= sz

class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        d = {}
        curr = head
        i = 1
        while(curr):
            d[i] = curr
            curr = curr.next
            i+=1
        d[i] = None
        
        list_length = max(list(d.keys()))
        
        to_remove = list_length-n
        
        if(to_remove-1 in d):
            d[to_remove-1].next = d[to_remove+1]
        else:
            return d[to_remove+1]
        
        return head

class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        p = d =head
        c=0
        while p!=None:
            c+=1
            if c>n+1:
                d=d.next 
            p=p.next
        if c==n:
            return head.next
        d.next = d.next.next
        return head

def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
    counter = 0
    current = head
    if current.next == None:
        return None
    dic = {}
    while current:
        counter+=1
        dic[counter] = current        
        current = current.next
    el = counter - n +1
    node = dic[el]
    if node.next != None:
        node.val = node.next.val
        node.next = node.next.next
    else:
        node = dic[el-1]
        node.next = None 
    return head

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        length=0
        itr=head
        while(itr!=None):
            length+=1
            itr=itr.next
        if length==1:
            return None
        st=length-n+1
        if st==1:
            return head.next
        ct=1
        itr=head
        while(itr!=None):
            if ct+1==st:
                ne=itr.next.next
                itr.next=ne
                break
            itr=itr.next
            ct+=1
        return head

def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        stack = []
        dh = ListNode(0, head)
        dn = dh
        while dn:
            stack.append(dn)
            dn = dn.next

        for _ in range(n):
            last_node = stack.pop()
        
        current_node = stack.pop()
        current_node.next = last_node.next
        return dh.next

143. Reorder List
Medium

3128

146

Add to List

Share
You are given the head of a singly linked-list. The list can be represented as:

L0 → L1 → … → Ln - 1 → Ln
Reorder the list to be on the following form:

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
You may not modify the values in the list's nodes. Only nodes themselves may be changed.

 

Example 1:


Input: head = [1,2,3,4]
Output: [1,4,2,3]
Example 2:


Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]
 

Constraints:

The number of nodes in the list is in the range [1, 5 * 104].
1 <= Node.val <= 1000

class Solution:
    def reorderList(self, head: ListNode) -> None:
        NL = []
        node = head
        while node:
            NL.append(node)
            node = node.next
        C = 1
        while C < len(NL) / 2:
            NL[C-1].next = NL[-C]
            NL[-C].next = NL[C]
            C += 1
        NL[len(NL) // 2].next = None

class Solution:
    def reorderList(self, head: ListNode) -> None:
        if not head:
            return head
        # the key insight is that we need to interleave two halves of the linked list
        # thus we need to (1) find where the mid point is located (using find mid which has a pretty simple implementation)
        # (2) reverse the right half and (3) merge the two
        def find_mid(head):
            fast = slow = head
            while fast and fast.next:
                fast = fast.next.next
                slow = slow.next
            return slow
         
        def reverse(head):
            prev, current = None, head
            while current:
                tmp = current.next
                current.next = prev
                prev = current
                current = tmp
            return prev
        
        # we get an instance of the head so that we can manipulate as we go
        current = head 
        # get the head to the right half of the linked list that is reserved
        right = reverse(find_mid(head))
        
        # we will stop when righ half is done (since the current pointer traverse over the whole linked list and righ is the one that once finished we are done)
        while right.next: # if this is None it means we have exhausted the list of right elements and the far right element is now last element of the right side
            # we start with the first element of current stay as is and we keep its next pointer to be used later
            tmp = current.next
            current.next = right # update the next element of current
            current = tmp # traversing to the next element that we skipped
            
            # a similiar implementation, we keep track of right pointer 
            tmp = right.next # we know there exists a next node due to the while loop condition
            right.next = current # update right element's next element with current next element
            right = tmp # traverse to the right element. at this point we might have reached the end 
        return head 

class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        # can reverse seond half instead stack
        
        slow = fast = head
        
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next 
        
        end = slow.next
        slow.next = None
        startOfReversed = None

        while end:
            nextNode = end.next
            end.next = startOfReversed
            startOfReversed, end = end, nextNode
        
        cur = head
        while cur and startOfReversed:
            
            nextNodeStart = cur.next
            nextNodeEnd = startOfReversed.next
            
            cur.next = startOfReversed
            cur.next.next = nextNodeStart
            
            startOfReversed = nextNodeEnd
            cur = nextNodeStart
        
        return head

First, let's just rearrange a list: [0,1,2,3,4,5,6] becomes [0,6,1,5,2,4,3]. The pattern = here is: for i in range(0, len(nums)//2) we point nums[i] to nums[n-1] and nums[n-i] to nums[i+1]. This took me a bit of thinking, and writing out, so try it yourself.
Then take care of the middle element (if len(nums) is odd, it's i, otherwise it's i-1
Ok, simple enough. But we've got a linked list. So we need to attach the nth node to the nth element in the list. use a dict for that. So nodeDict[i] = nodeDict[i-1].next

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if head.next ==None:
            return head
        nodeDict = {}
        i=0
        curr = head
        while curr:
            nodeDict[i] = curr
            curr = curr.next
            i+=1
        n=i-1
        i=0
        
        while i <= n//2:
            nodeDict[i].next = nodeDict[n-i]
            nodeDict[n-i].next = nodeDict[i+1]
            print(i, nodeDict[i].val, nodeDict[n-i].val, nodeDict[i].next.val)
            i=i+1
        if n%2 ==1:
            nodeDict[i].next = None
        else:
            nodeDict[i-1].next = None
And that's it. Simple, inelegant, slow; but it works!

133. Clone Graph
Medium

3025

1704

Add to List

Share
Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.

class Node {
    public int val;
    public List<Node> neighbors;
}
 

Test case format:

For simplicity, each node's value is the same as the node's index (1-indexed). For example, the first node with val == 1, the second node with val == 2, and so on. The graph is represented in the test case using an adjacency list.

An adjacency list is a collection of unordered lists used to represent a finite graph. Each list describes the set of neighbors of a node in the graph.

The given node will always be the first node with val = 1. You must return the copy of the given node as a reference to the cloned graph.

 

Example 1:


Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
Explanation: There are 4 nodes in the graph.
1st node (val = 1)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
2nd node (val = 2)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
3rd node (val = 3)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
4th node (val = 4)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
Example 2:


Input: adjList = [[]]
Output: [[]]
Explanation: Note that the input contains one empty list. The graph consists of only one node with val = 1 and it does not have any neighbors.
Example 3:

Input: adjList = []
Output: []
Explanation: This an empty graph, it does not have any nodes.
Example 4:


Input: adjList = [[2],[1]]
Output: [[2],[1]]
 

Constraints:

The number of nodes in the graph is in the range [0, 100].
1 <= Node.val <= 100
Node.val is unique for each node.
There are no repeated edges and no self-loops in the graph.
The Graph is connected and all nodes can be visited starting from the given node.