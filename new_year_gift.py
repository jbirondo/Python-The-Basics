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

class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        if node is None:
            return None
        
        visited = {node : Node(node.val,[])}
        queue = [node]
        
        while queue:
            v = queue.pop(0)
            for n in v.neighbors:
                if n not in visited:
                    visited[n] = Node(n.val,[])
                    queue.append(n)
                visited[v].neighbors.append(visited[n])

        return visited[node]

Simple one-liner
# from copy import deepcopy
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        return deepcopy(node)
Harder variant
def clone(node, visited):
    if node.val in visited:
        return visited[node.val]
    visited[node.val] = node.val
    visited[node.val] = Node(node.val, [clone(neighbor, visited) for neighbor in node.neighbors])
    return visited[node.val]

class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if node is None:
            return

        visited = {
            node.val: node.val
        }
        visited[node.val] = Node(node.val, [clone(neighbor, visited) for neighbor in node.neighbors])
        for node in visited.values():
            node.neighbors = [visited.get(neighbor, neighbor) for neighbor in node.neighbors]
        return visited[1]

if (not node): return None
queue = [node]
newNode = Node(node.val)
seenMap = { node: newNode}
while (queue):
    curr = queue.pop()
    newCurr = seenMap[curr]
    for next in curr.neighbors:
        if (not next in seenMap):
            newNext = Node(next.val)
            newCurr.neighbors.append(newNext)
            seenMap[next] = newNext
            queue.insert(0, next)
        else:                    
            newCurr.neighbors.append(seenMap[next])
return newNode

class Solution:
    	def cloneGraph(self, node: 'Node') -> 'Node':
		if not node:
			return 
		v = helper(node, {}, [])
		return v
    
    
def helper(node, hashMap, inProgress):
	if node.val in inProgress:
		return

	copy = Node(node.val)
	hashMap[node.val] = copy
	inProgress.append(node.val)

	for child in node.neighbors:
		if child.val in hashMap:
			if child.val not in [item.val for item in hashMap[node.val].neighbors]:
				hashMap[node.val].neighbors.append(hashMap[child.val])
			if node.val not in [item.val for item in hashMap[child.val].neighbors]:
				hashMap[child.val].neighbors.append(copy)
        
		helper(child, hashMap, inProgress)
    
	return hashMap[node.val]

class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        seen = set() # To keep track of visited nodes
        queue = [] # Queue to perform BFS on the graph node
        queue.append(node)
        nodes = defaultdict(int) 
		""" Defaultdictionary will help to create  only 1 instance of object 
			any node with values node.val, which we can use for mapping edges"""
		
        while queue:
            top = queue.pop(0)
            if top and top.val not in seen:
                if top.val not in nodes: 
				""" Check if we have mapping between top.val  to actual node 
					if not present then create node object with top.val """
                    nodes[top.val] = Node(top.val)
                seen.add(top.val)
                for neigh in top.neighbors:
                    if neigh.val not in nodes: 
					""" Check if we have mapping between neigh.val to actual node 
						if not present then create node object with neigh.val """
                        nodes[neigh.val] = Node(neigh.val)
                    nodes[top.val].neighbors.append(nodes[neigh.val])
                    queue.append(neigh)
        return nodes[1] if len(nodes) > 0 else None """ to check the edge case """

207. Course Schedule
Medium

5822

245

Add to List

Share
There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return true if you can finish all courses. Otherwise, return false.

 

Example 1:

Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0. So it is possible.
Example 2:

Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.
 

Constraints:

1 <= numCourses <= 105
0 <= prerequisites.length <= 5000
prerequisites[i].length == 2
0 <= ai, bi < numCourses
All the pairs prerequisites[i] are unique.

class Solution:
    def kahn(self, adj, inDegree):
        counter = 0
        processedQueue = []
        for i in range(len(adj)):
            if not inDegree[i]:
                processedQueue.append(i)
        
        while processedQueue:
            vertex = processedQueue.pop(0)
            counter += 1
            
            for j in range(len(adj)):
                if adj[vertex][j]:
                    inDegree[j] -= 1
                    if not inDegree[j]:
                        processedQueue.append(j)
        
        return counter == len(adj)
        
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        adj = [[0 for j in range(numCourses)] for i in range(numCourses)]
        inDegree = [0 for i in range(len(adj))]
        
        for next_sub, prev_sub in prerequisites:
            adj[prev_sub][next_sub] = 1
            inDegree[next_sub] += 1
            
        return self.kahn(adj, inDegree)

class Solution:
    def canFinish(self, nc: int, prerequisites: List[List[int]]) -> bool:
        v=set();gr=defaultdict(list);
        for p in prerequisites:gr[p[0]].append(p[1])
        def _(n,u):
            if n in v:return n not in u
            u.add(n)
            v.add(n)
            for nn in gr[n]:
                if not _(nn,u):return False
            u.remove(n)
            return True
        for n in range(nc):
            if not _(n,set()):return False
        return True

Using Kahn's Algorithm, essentially:

by performing BFS starting with the nodes which have indegree 0
when we traverse to its adjacent nodes we decrease the indegree counter of the adjacent node by 1
if that adjacent node is now indegree 0, then we can add it to the queue of bfs
if you can count the same amount of nodes through the topological sort, then you have the right order and no cycles.
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        adj_list = defaultdict(list)
        indegree = [0]*(numCourses)
        
        num=0
        
        for course,prereq in prerequisites:
            adj_list[prereq].append(course)
            indegree[course]+=1
            
        q = deque()
        for course in range(numCourses):
            if indegree[course] == 0:
                q.append(course)

        while q:
            n = q.popleft()
            num+=1

            for nei in adj_list[n]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
        return num == numCourses

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        adj_list=collections.defaultdict(list)
        
        for course,preq in prerequisites:
            adj_list[preq].append(course)
        color = {}
        for u in range(numCourses):
            color[u] = 'W'
        print(color)
        def dfs(u, color):
            color[u] = 'G'
            for v in adj_list[u]:
                if color[v] == 'W':
                    cycle = dfs(v, color)
                    if cycle == True:
                        return True
                elif color[v] == "G":
                    return True
            color[u] = "B"
            return False

        is_cyclic = False
        for u in range(numCourses):
            if color[u] == 'W':
                is_cyclic = dfs(u, color)
                if is_cyclic == True:
                    return False
        return True

417. Pacific Atlantic Water Flow
Medium

2181

558

Add to List

Share
You are given an m x n integer matrix heights representing the height of each unit cell in a continent. The Pacific ocean touches the continent's left and top edges, and the Atlantic ocean touches the continent's right and bottom edges.

Water can only flow in four directions: up, down, left, and right. Water flows from a cell to an adjacent one with an equal or lower height.

Return a list of grid coordinates where water can flow to both the Pacific and Atlantic oceans.

 

Example 1:


Input: heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
Example 2:

Input: heights = [[2,1],[1,2]]
Output: [[0,0],[0,1],[1,0],[1,1]]

While everyone is coming up with solutions that involves going from ocean to the continent. I wanted to solve it using the regular BFS way. It's slow but works

class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights:
            return []
        res_coords = []
        visited = set()
        queue=collections.deque()
        rows, cols = len(heights), len(heights[0])
        for i in range(rows):
            for j in range(cols):
                if (i,j) not in visited:
                    visited_1 = set()
                    visited_1.add((i,j))
                    visited.add((i,j))
                    queue.append((i,j))
                    flag_1 = False
                    flag_2 = False
                    while queue:
                        node = queue.popleft()
                        r,c = node
                        if r == rows-1 or c == cols-1:
                            flag_1 = True
                        if r == 0 or c == 0:
                            flag_2 = True
                        nbrs = self.getNbrs(node, heights, flag_1, flag_2)
                        for nbr in nbrs:
                            if nbr not in visited_1:
                                visited_1.add(nbr)
                                queue.append(nbr)
                    if flag_1 and flag_2:
                        res_coords.append((i,j))
        return res_coords
        
    def getNbrs(self, node, heights, flag_1, flag_2):
        result = []
        rows,cols = len(heights), len(heights[0])
        r,c = node
        if r+1 < rows and heights[r+1][c] <= heights[r][c]:
            result.append((r+1,c))
        if r-1 >= 0 and heights[r-1][c] <= heights[r][c]:
            result.append((r-1,c))
        if c+1 < cols and heights[r][c+1] <= heights[r][c]:
            result.append((r,c+1))
        if c-1 >= 0 and heights[r][c-1] <= heights[r][c]:
            result.append((r,c-1))
        return result

Implementation:

class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights or not heights[0]:
            return []
		
		row, col = len(heights), len(heights[0])
        
        pacific_queue = deque([(r, c) for r in range(row) for c in range(col) if (r == 0) or (c == 0)])
        atlantic_queue = deque([(r, c) for r in range(row) for c in range(col) if (r == row-1) or (c == col-1)])
		
		def _bfs_helper(queue):
			nonlocal row, col
			
            visited, directions = set(), [(-1, 0), (1, 0), (0, 1), (0, -1)]
            
            while queue:
                x, y = queue.popleft()
                visited.add((x, y))

                for d in directions:
                    dx, dy = d[0] + x, d[1] + y

                    # check bounds
                    if 0 <= dx < row and 0 <= dy < col and (dx, dy) not in visited and heights[dx][dy] >= heights[x][y]:
                        queue.append((dx, dy))
                        visited.add((dx, dy))
            return visited
		
		pacific_visited = _bfs_helper(pacific_queue)
        atlantic_visited = _bfs_helper(atlantic_queue)
		
		return list(pacific_visited.intersection(atlantic_visited))
Time/Space: O(MN) where M is the num of rows and N is the num of cols

class Solution:
    def pacificAtlantic(self, nums: List[List[int]]) -> List[List[int]]:
        
        m,n = len(nums), len(nums[0])
        
        p_q = []
        a_q =[]
        
        for i in range(m):
            p_q.append((i, 0))
            a_q.append((i, n-1))
        
        for j in range(n): 
            p_q.append((0, j))
            a_q.append((m-1, j))
            
            
        def get_flow_nodes(q):
            seen = set(q)
            while q:
                
                new_q =[]
                for i,j in q:
                    for x,y in [(i+1, j), (i-1, j), (i, j-1), (i, j+1)]:
                        if (x,y) in seen or x>m-1 or y>n-1 or x<0 or y<0: continue

                        if nums[x][y]>= nums[i][j]:
                            seen.add((x,y))
                            new_q.append((x,y))
                            
                q= new_q
            
            return seen
            

        p_nodes = get_flow_nodes(p_q)
        
        #atlantic nodes
        a_nodes = get_flow_nodes(a_q)
        
        return list(a_nodes.intersection(p_nodes))

	def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        m, n = len(heights), len(heights[0])
        pacific = deque([[0,j] for j in range(n)] + [[i,0] for i in range(m)])
        atlantic = deque([[i,n-1] for i in range(m)] + [[m-1, i] for i in range(n)])
                
        def bfs(queue):
            visited = set()
            while queue:
                x,y = queue.popleft()
                visited.add((x,y))
                for dx,dy in [[1,0],[0,1],[-1,0],[0,-1]]:
                    if 0 <= x+dx < m and 0 <= y+dy < n:
                        if (x+dx, y+dy) not in visited:
                            if heights[x+dx][y+dy] >= heights[x][y]:
                                queue.append((x+dx, y+dy))
            return visited
        
        p, a = bfs(pacific), bfs(atlantic)
                        
        return p.intersection(a)

from collections import deque
class Solution(object):
    def pacificAtlantic(self, heights):
        """
        :type heights: List[List[int]]
        :rtype: List[List[int]]
        """
        if not heights:
            return []
        m , n = len(heights), len(heights[0]) # get the coordinate structure
        # initiate a list of coordinate for each ocean


        def bfs_traverse(edge_vetices):
            # we will perform a BFS ==> intiating a queue
            queue = deque()
            # we will need a set simply because it is simple to get the intersect of two lists (check the return statement)
            # also it works as the "visited" list to make sure we won't check the same entry twice
            reached = set()
            # first initate the set and queue with the edge vertices
            for edge_vert in edge_vetices:
                queue.append((edge_vert, heights[edge_vert[0]][edge_vert[1]])) # I keep track of the parent height, you dont need to though
                reached.add(edge_vert) # all the edge vertices are already connected to the ocean so we add them to its corresponding set
            
            while queue: # perform a BFS
                current, current_height = queue.popleft() # pop the queue 
                x, y = current
                for next_x, next_y in [(x-1,y), (x+1,y),(x,y-1),(x,y+1)]: # update the queue with neighboring candidates
                    # these candidates should be (1) within the matrix boundary, (2) have higher heights than their parents
                    # (3) and have not been seen before
                    if 0<=next_x<m and 0<=next_y<n and heights[next_x][next_y] >= current_height and (next_x, next_y) not in reached:
                        queue.append(((next_x,next_y), heights[next_x][next_y] )) # update the queue
                        reached.add((next_x, next_y)) # update the reached set
            return reached
        
        # initiate the ocean edges
        pacific = [(0,i) for i in range(n)] + [(j,0) for j in range(1,m)]
        atlantic = [(m-1,i) for i in range(n)] + [(j,n-1) for j in range(m)]
        # return the reached set for each ocean
        pacific_set = bfs_traverse(pacific)
        atlantic_set = bfs_traverse(atlantic)
        
        return list(atlantic_set.intersection(pacific_set))  # we want to get a list of coordinates that are reached from both edges

200. Number of Islands
Medium

8433

243

Add to List

Share
Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

 

Example 1:

Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
Example 2:

Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
 

Constraints:

m == grid.length
n == grid[i].length
1 <= m, n <= 300
grid[i][j] is '0' or '1'.

DFS
Time complexity Big O (nm)
Space Complexity (nm) in the worst case when all elements == '1'


def numIslands(self, grid):
        
        
        # 1) Check IF there A GRID or Has any ROWS (length is > 0)
        if grid == None or len(grid) == 0:
            return 0
        
        
        # 2) Initial the islands
        islands = 0
        
        rows = len(grid)
        cols = len(grid[0])
        
        # 3) If we have [0] , we do not care
       
        #if we have the "1" # we got the land ! 
        # then we need to visit ALL the neighbours if they are Also +1 
        # we can FLIP the visited ELEMENT form to 1 -> to 0, so we do not double count
        # at THIS grid, we MAKE DFS or BFS, to calculate, visit ,the nodes and return the output
        
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == "1":
                    islands += 1
                    self.dfs(grid, row, col)
                    
        return islands
    
    def dfs(self, grid, row, col):
        #check the boundries
        #base case
        
        if (row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or grid[row][col] == "0"):
            return       
        
        #recursive call within 4 neighbours
        else:
            grid[row][col] = "0" # MARK the current element - so it is VISITED
            self.dfs(grid, row+1, col)
            self.dfs(grid, row-1, col)
            self.dfs(grid, row, col+1)
            self.dfs(grid, row, col-1)

BFS
Time complexity Big O (nm)
Maximum siblings in queue will be min(M, N)
So space complexity is min(M,N)'

    def numIslands(self, grid):
        
        if len(grid) == 0 or len(grid[0]) == 0:
            return grid
        
        rows = len(grid)
        cols = len(grid[0])
        islands = 0
        
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == "1":
                    islands += 1
                    # when we see the Island "1" we can perform the BFS to update the land around
                    self.bfs(grid, row, col)
                    
        return islands
    
    
    def bfs(self, grid, row, col):
        
        def isValid(grid, row, col):
            if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]):
                return False
            else:
                return True
            
        queue = collections.deque()
        visited = set()
        
        directions = [[1,0],[-1,0], [0,1],[0,-1]]
        
        #add the element to the queue
        queue.append((row, col))
        #mark as visited
        grid[row][col] = "0"
        visited.add((row, col))
        
        while (len(queue) > 0):
            
            level = len(queue)
            current = queue.popleft()
            #print(current)
            currrent_row = current[0]
            current_col = current[1]
          
            
            #for i in range(level):
                
            for direction in directions:
                next_row = currrent_row + direction[0]
                next_col = current_col + direction[1]

                    
                if isValid(grid, next_row, next_col) and grid[next_row][next_col] == "1":
                    queue.append((next_row, next_col))
                    grid[next_row][next_col] = "0" ```

Approach 1: DFS

class Solution:
    
    def DFS(self, grid: List[List[str]], row: int, col:int):
        
        row_size = len(grid)
        col_size = len(grid[0])
        
        dirs = [(0,1), (1,0), (0,-1), (-1,0)]
        
        for dx, dy in dirs:
            new_row = row + dx
            new_col = col + dy
            
            if 0 <= new_row < row_size and 0 <= new_col < col_size and grid[new_row][new_col] == '1':
                grid[new_row][new_col] = '*' # visited
                self.DFS(grid, new_row, new_col)
    
    def numIslands(self, grid: List[List[str]]) -> int:
        
        if not grid:
            return 0
        
        islands = 0
        self.row_size = len(grid)
        self.col_size = len(grid[0])
        
        for row in range(self.row_size):
            for col in range(self.col_size):
                if grid[row][col] == '1': # island
                    islands += 1
                    self.DFS(grid, row, col)
        
        return islands

Approach 2: BFS


class Solution:
    
    def BFS(self, grid: List[List[str]], x: int, y: int):
        
        dirs = [(0,1),(1,0),(0,-1),(-1,0)]
        
        q = collections.deque()
        q.append((x,y))
        
        while q:
            x, y = q.popleft()
            
            for dx, dy in dirs:
                new_x = x+dx
                new_y = y+dy
                
                if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]) and grid[new_x][new_y] == '1':
                    q.append((new_x, new_y))
                    grid[new_x][new_y] = '*'
        
    def numIslands(self, grid: List[List[str]]) -> int:
        
        if not grid:
            return 0
        
        islands = 0
        row_size = len(grid)
        col_size = len(grid[0])
        
        for row in range(row_size):
            for col in range(col_size):
                if grid[row][col] == '1': # island
                    islands += 1
                    self.BFS(grid, row, col)
        
        
        return islands

class Solution:
    def dfs(self,grid,i,j): 
        if i == len(grid) or i < 0 or j < 0 or j == len(grid[i]) or grid[i][j] == "0": 
            return 
        
        grid[i][j] = "0"
        
        self.dfs(grid,i+1,j)
        self.dfs(grid,i-1,j)
        self.dfs(grid,i,j+1)
        self.dfs(grid,i,j-1)
        
    def numIslands(self, grid: List[List[str]]) -> int:
        count = 0
        for i in range(len(grid)): 
            for j in range(len(grid[i])): 
                if grid[i][j] == "1": 
                    count +=1
                    self.dfs(grid, i, j)
        return count 

128. Longest Consecutive Sequence
Hard

5112

251

Add to List

Share
Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

 

Example 1:

Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
Example 2:

Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9
 

Constraints:

0 <= nums.length <= 104
-109 <= nums[i] <= 109

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return len(nums)
        nums = sorted(set(nums))
        maxi = 1
        for start in range(len(nums)):
            curr = start + maxi
            if curr < len(nums):
                while curr < len(nums) and nums[curr] == (nums[start] + (curr - start)):
                    curr += 1
                maxi = max(maxi, curr - start)
        return maxi

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums:
            return 0
        nums.sort()
        maxy,c=1,1
        for i in range(1,len(nums)):
            diff=nums[i]-nums[i-1]
            if diff==1:
                c+=1
            elif diff>1:
                c=1
            maxy=max(maxy,c)
        return maxy

Instead of the standard array for parent (list(range(n))), this time use a map since the range of numbers is very large, and not consecutive
Unions is a map of parentId -> elements. This helps in tracking the sets and the max_union_size
For every element in array, we have a few options. More explanations in the code below

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        
        maxunionsize = 0
        parent = {}
        unions = defaultdict(list)

        for n in nums:
            cont = 0               # this variable tells if n-1 or n+1 has already been seen (read as continue)
            if n in parent:
                continue
            if n-1 in parent:      # add the curr num to the already present union
                cont = 1
                parent[n] = parent[n-1]
                unions[parent[n-1]].append(n)
                maxunionsize = max(maxunionsize, len(unions[parent[n-1]]))

            if n+1 in parent:
                cur_parent = parent[n+1]
                cur_union = unions[cur_parent]

                if cont==1:        # n-1 was also present, merge parents (eg case: 2,4,3)
                    prev_parent = parent[n-1]
                    prev_union = unions[prev_parent]
                    for x in prev_union:         # merging unions (can be sped up by picking up shorter union to traverse)
                        parent[x] = cur_parent
                    unions[cur_parent].extend(prev_union)
                    maxunionsize = max(maxunionsize, len(unions[cur_parent]))
                    del unions[prev_parent]
                    
                else:                # only n+1 is present (similar steps as first if case)
                    cont = 1
                    parent[n] = cur_parent
                    unions[cur_parent].append(n)
                    maxunionsize = max(maxunionsize, len(unions[cur_parent]))

            # no relation present, standalone number
            if cont==0:
                parent[n] = n
                unions[n] = [n]
                maxunionsize = max(maxunionsize, 1)
        
        return maxunionsize

# Create a node DS
# Create a node for every num in the array and set it as next of a n+1 valued node if present
# also set num - 1 valued node as the parent of current node if present 
# if the node has parent set its top property as False to maintain O(n) (more info below)
# iterate through the node of top nodes and calculate the max length of each graph

class Node:
    def __init__(self, val):
        self.val = val
        self.count = 1
        self.next = None
        self.top = True

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        node_map = {}
        max_count = 0
        
        for num in nums:
            if num not in node_map:
                num_node = Node(num)
                
                if (num+1) in node_map:
                    num_node.next = node_map[num+1]
                    node_map[num+1].top = False
                    
                if (num-1) in node_map:
                    node_map[num-1].next = num_node
                    num_node.top = False
                
                node_map[num] = num_node
                    
        for num in node_map:
            curr = node_map[num]
            
            if not curr.top: # causes the time complexity to remain O(n) (O(2n) maybe max)
                continue
                
            count = 1
            while curr.next:
                count += 1
                curr = curr.next
            max_count = max(count, max_count)
        
        return max_count

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        largest_sequence = 0
		nums_visit = {}
		
        for num in nums:
            nums_visit[num] = False
        for num in nums:
            nums_visit[num] = True
            sequence_count = 1

            next_num = num + 1
            while next_num in nums_visit and not nums_visit[next_num]:
                nums_visit[next_num] = True
                sequence_count += 1
                next_num = next_num + 1

            prev_num = num - 1
            while prev_num in nums_visit and not nums_visit[prev_num]:
                nums_visit[prev_num] = True
                sequence_count += 1
                prev_num = prev_num - 1

            if sequence_count > largest_sequence:
                largest_sequence = sequence_count

        return largest_sequence


73. Set Matrix Zeroes
Medium

3463

367

Add to List

Share
Given an m x n matrix. If an element is 0, set its entire row and column to 0. Do it in-place.

Follow up:

A straight forward solution using O(mn) space is probably a bad idea.
A simple improvement uses O(m + n) space, but still not the best solution.
Could you devise a constant space solution?
 

Example 1:


Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]
Example 2:


Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
 

Constraints:

m == matrix.length
n == matrix[0].length
1 <= m, n <= 200
-231 <= matrix[i][j] <= 231 - 1

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n, m = len(matrix), len(matrix[0])
        # we traverse over the matrix, every time we see a zero, we set its columns and rows to None unless it is zero already
        for row in range(n):
            for col in range(m):
                if matrix[row][col] !=  None:
                    if matrix[row][col] == 0:
                        print(row, col)
                        for i in range(m):
                            matrix[row][i] = 0 if matrix[row][i] == 0 else  None
                        for j in range(n):
                            matrix[j][col] = 0 if matrix[j][col] == 0 else  None
                            
        for row in range(n):
            for col in range(m):
                if matrix[row][col] is None:
                    matrix[row][col] = 0
        return matrix
The only issue with this is that it touches each element twice. But regrdless it is O(n) O(1) and it is much more intuitive and easier to understand, (I don't like too many flags)

The time complexity is O(mnm + mnn + mn) = O(mn*(m+n+1)) = O(mn)

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        def initialize(matrix,x,y,n,m):
            for i in range(0,m):
                matrix[x][i]=0
            for i in range(0,n):
                matrix[i][y]=0
        def f(matrix):
            n=len(matrix)
            m=len(matrix[0])
            d=set()
            for i in range(n):
                for j in range(m):
                    if(matrix[i][j]==0):
                        d.add((i,j))
            for i in d:
                initialize(matrix,i[0],i[1],n,m)
            for i in matrix:
                print(i)
        f(matrix)

	# 1. Traverse Matrix and look out for any 0s
	# 2. if you find a zero, call function to populate it's neighboring cells (cells above, below & sides)
	
    def populate_sides(row, col): # helper function that will populate neighboring cells
        # left and right
        matrix[row][col] = 'M' 
		# we use an identifier "M" to mark cells where we know we will need to add zeros
        for col2 in range(COLS):
            if col != col2:
                if matrix[row][col2] == 0: # if we see another zero we need to populate its sides
                    populate_sides(row, col2)
                else:
                    matrix[row][col2] = 'M'
        # below and above
        for row2 in range(ROWS):
            if row != row2:
                if matrix[row2][col] == 0:
                    populate_sides(row2, col)
                else:
                    matrix[row2][col] = 'M'
                    
    COLS = len(matrix[0])
    ROWS = len(matrix)
    
    for row in range(ROWS): 
        for col in range(COLS): # standard iteration through 2d array
            if matrix[row][col] == 0:
                populate_sides(row, col)
                
    # add zeros
    for row in range(ROWS):
        for col in range(COLS):
            if matrix[row][col] == 'M':
                matrix[row][col] = 0

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Use the first elt in each row/each col as marker for whether or not that 
        row or col should be zeroed out.
		
		First cells marked X (a.k.a. all cells in matrix where matrix[r][0] or matrix[c][0]):
		[
			[x, x, x, x]
			[x, 1, 2, 3]
			[x, 1, 2, 3]
		]
		
        ALGORITHM:
        First iteration. 
        If a cell is 0:
        1) we set the first cell in its row to 0. (matrix[row][0] = 0)
        2) We set the first cell in its col to 0. (matrix[0][col] = 0)
        
        Second iteration, go through all first cell in rows and first cell in cols.
        If a first cell in row is 0, set entire row to 0.
        If a first cell in col is 0, set entire col to 0.
        
        THERE IS A CORNER CASE.
        Note, first cell in row zero and first cell in col zero is the same.
        In second iteration, if we see matrix[0][0] set to 0.
        
        - Was it triggered by a zero in col zero => Set all entries in col zero to 0.
        - Was it triggered by a zero in row zero => Set all entries in row zero to 0.
        - Was matrix[0][0] originally 0? => Set all entries in row zero and col zero to 0.
        
        Seeing matrix[0][0] set to 0 after the second iteration doesn't help us differentiate
        between the 3 cases.
        
        So that needs to be handled separately.
        """
        """
        n_rows = len(matrix)
        n_cols = len(matrix[0])
        
        # Corner case.
        row_0_should_clear = False
        col_0_should_clear = False
        
        for r in range(n_rows):
            if matrix[r][0] == 0:
                col_0_should_clear = True
                break
                
        for c in range(n_cols):
            if matrix[0][c] == 0:
                row_0_should_clear = True
                
        # General case.
        for r in range(1, n_rows):
            for c in range(1, n_cols):
                if matrix[r][c] == 0:
                    matrix[r][0] = 0    # Set first elt in row to zero.
                    matrix[0][c] = 0    # Set first elt in col to zero.
                    
        for r in range(1, n_rows):
            for c in range(1, n_cols):
                if matrix[r][0] == 0 or matrix[0][c] == 0:
                    matrix[r][c] = 0
                    
        # Corner case (cont.)        
        if row_0_should_clear:
            for c in range(n_cols):
                matrix[0][c] = 0
        
        if col_0_should_clear:
            for r in range(n_rows):
                matrix[r][0] = 0

54. Spiral Matrix
Medium

3873

671

Add to List

Share
Given an m x n matrix, return all elements of the matrix in spiral order.

 

Example 1:


Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]
Example 2:


Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]
 

Constraints:

m == matrix.length
n == matrix[i].length
1 <= m, n <= 10
-100 <= matrix[i][j] <= 100

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m = len(matrix)
        n = len(matrix[0])
        
        i = j = 0
        
        right = True
        left = False
        down = False
        up = False
        
        res = []
        cnt = 0
        
        while cnt < m*n:
            
            ##STOP RIGHT GO DOWN
            if right and (j == n-1):
                right = False
                down = True
            elif right and matrix[i][j+1] == None:
                right = False
                down = True
                
            ##STOP DOWN GO LEFT    
            elif down and (i == m-1):
                down = False
                left = True
            elif down and matrix[i+1][j] == None:
                down = False
                left = True
                
            ##STOP LEFT GO UP    
            elif left and (j == 0):
                left = False
                up = True
            elif left and matrix[i][j-1] == None:
                left = False
                up = True
                
            #STOP UP GO RIGHT    
            elif up and (i == 0):
                up = False
                right = True
            elif up and matrix[i-1][j] == None:
                up = False
                right = True    
            
            #store the result
            res.append(matrix[i][j])
            cnt += 1
            
            #Block the place
            matrix[i][j] = None
            
            #NEXT MOVES
            if right: j += 1
            elif left: j -= 1
            elif down : i += 1
            elif up: i -= 1
            
        return res

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        t=l=0
        b=len(matrix)-1
        r=len(matrix[0])-1
        res=[]
        while t<=b and l<=r: 
            
            for i in range(l,r+1):
                res.append(matrix[t][i])
            
            
            for i in range(t+1,b):
                res.append(matrix[i][r])
            
            if t!=b:
                for i in range(r,l-1,-1):
                    res.append(matrix[b][i])

            if l!=r:
                for i in range(b-1,t,-1):
                    res.append(matrix[i][l])
            
            l+=1
            t+=1
            r-=1
            b-=1
            
        return  res

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        top = left = 0
        bottom = len(matrix) - 1
        right = len(matrix[0]) - 1
        op = []
    
        while True:
            if left > right:
                break
            
            for i in range(left,right+1):
                op.append(matrix[top][i])
            
            top += 1
            
            if top > bottom:
                break
                
            for i in range(top,bottom+1):
                op.append(matrix[i][right])

            right -= 1
            
            if left > right:
                break
        
            #print(m,n,i)
            for i in range(right,left-1,-1):
                op.append(matrix[bottom][i])
                
            bottom -= 1
            
            if top > bottom:
                break
            
            for i in range(bottom,top-1,-1):
                op.append(matrix[i][left])
            
            left += 1
            

        return op
        
    '''
    [1,2,3,4]
    [5,6,7,8]
    [9,10,11,12]
    [13,14,15,16]
    
    '''

	def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m, n, result = len(matrix), len(matrix[0]), []
        num_spirals = min(m//2, n//2) # the number of times we spiral (right, down, left, up) through the matrix
        
		# if the matrix is a single row or column, we simply return the elements in order
        if not(m//2 and n//2):
            for i,j in product(range(m), range(n)):
                result.append(matrix[i][j])
            return result
        
		# otherwise, we traverse through the matrix num_spirals times in order
        for i in range(num_spirals):
            for j in range(i,n-i): # traverse right on the top side
                result.append(matrix[i][j])
            for j in range(i+1,m-i-1): # traverse down on the right side
                result.append(matrix[j][n-i-1])
            for j in range(n-i-1,i-1,-1): # traverse left on the bottom side
                result.append(matrix[m-i-1][j])
            for j in range(m-i-2,i,-1): # traverse up on the left side
                result.append(matrix[j][i])
        
		# if the matrix isn't an nxn square, we still need to add elements from the middle
        for i in range(num_spirals, m-num_spirals):
            for j in range(num_spirals, n-num_spirals):
                result.append(matrix[i][j])
                
        return result

48. Rotate Image
Medium

5063

345

Add to List

Share
You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

 

Example 1:


Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]
Example 2:


Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
Example 3:

Input: matrix = [[1]]
Output: [[1]]
Example 4:

Input: matrix = [[1,2],[3,4]]
Output: [[3,1],[4,2]]

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        #Transpose Matrix
        
        n = len(matrix)
        for i in range(n):
            for j in range(i,n):
                matrix[i][j], matrix[j][i] = matrix[j][i],matrix[i][j]
        
        #Reverse Matrix
        for i in range(n):
            for j in range(n//2):
                matrix[i][j], matrix[i][n-1-j] = matrix[i][n-1-j], matrix[i][j]

class Solution(object):
    def rotate(self, matrix):

    #matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    for m in range(len(matrix)):
        for n in range(m, len(matrix[m])):
            matrix[m][n], matrix[n][m] = matrix[n][m], matrix[m][n]

    for i in range(len(matrix)):
        matrix[i].reverse()

    return matrix

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        start = 0
        while n - 2*start > 1:
            for row in range(start,n-start-1):
                matrix[start][row],matrix[row][n-start-1],matrix[n-start-1][n-row-1],matrix[n-row-1][start] = matrix[n-row-1][start],matrix[start][row],matrix[row][n-start-1],matrix[n-start-1][n-row-1]
            start+=1

class Solution:
    def transposeMatrix(self, matrix):
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix[0])):
                temp = matrix[i][j]
                matrix[i][j] = matrix[j][i]
                matrix[j][i] = temp
                
    
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        self.transposeMatrix(matrix)
        for j in range(len(matrix[0]) // 2):
            for i in range(len(matrix)):
                temp = matrix[i][j]
                matrix[i][j] = matrix[i][len(matrix[0]) - j - 1]
                matrix[i][len(matrix[0]) - j - 1] = temp

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        for row in range(len(matrix)):
            for col in range(row,len(matrix[0])):
                matrix[row][col],matrix[col][row]=matrix[col][row],matrix[row][col]
        print(matrix)
        for row in range(len(matrix)):
            for col in range(len(matrix)//2):
                matrix[row][col],matrix[row][-col-1]=matrix[row][-col-1],matrix[row][col]

79. Word Search
Medium

5736

244

Add to List

Share
Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

 

Example 1:


Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true
Example 2:


Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
Output: true
Example 3:


Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
Output: false
 

Constraints:

m == board.length
n = board[i].length
1 <= m, n <= 6
1 <= word.length <= 15
board and word consists of only lowercase and uppercase English letters.
 

Follow up: Could you use search pruning to make your solution faster with a larger board?

Time and space complxity should be roughly the same as the accepted recursive solution. However, I had to copy the 'visited' hashset for each backtracking state because I couldn't figure out a way to iteratively use existing memory to maintain the 'visited' state like the recursive solution did.

class State():
    def __init__(self, point, word_index, visited):
        self.point = point
        self.word_index = word_index
        self.visited = visited

class Solution(object):
    def exist(self, board, word):
        self.rows = len(board)
        self.columns = len(board[0])
        self.board = board
        
        for i in range(0, self.rows):
            for j in range(0, self.columns):
                if self.board[i][j] == word[0]:
                    word_found = self.dfs(i, j, word)
                    if word_found:
                        return True
        return False
        
    def dfs(self, row, column, word):
        if len(word) == 1:
            if self.board[row][column] == word:
                return True
            else:
                return False
            
        visited = set()
        initial_state = State((row,column), 0, visited)
        stack = []
        stack.append(initial_state)
        
        while (len(stack) > 0):
            current_state = stack[-1]
            stack.pop()
            
            next_letter_index = current_state.word_index + 1
            if next_letter_index == len(word):
                #At this point, we know there's a path that contains the entire word
                return True
            
            if current_state.point not in current_state.visited:
                current_state.visited.add(current_state.point)
            
            expected_letter = word[next_letter_index]
            
            for row_offset, column_offset in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                offset_row = current_state.point[0] + row_offset
                offset_column = current_state.point[1] + column_offset
                if (offset_row >= 0 and offset_row < self.rows) and (offset_column >= 0 and offset_column < self.columns):
                        if (offset_row, offset_column) not in current_state.visited:
                            if self.board[offset_row][offset_column] == expected_letter:
                                #Make sure to create a copy the 'visited' hashset. If we don't, all states will share the same hashset in memory. 
                                new_state = State((offset_row, offset_column), next_letter_index, current_state.visited.copy())
                                stack.append(new_state)
        return False

# This is an advanced problem. It took me some time to see other solutions and come up with this approach after seeing other solution.
# This problem uses DFS approach along with backtracking. (The Fact that the visited is marked as True before getting into recursion inside the function and again marking it as False if we have not gotten the answer is the reason why this problem is a backtracking problem.) 

# Algo
    # You have to try each location in the board is as a strating point. So 2d loop at the start is necessary.
    # We start from each character, and try to move in each possible direction until we can match the complete word. We use DFS for this.
        # When we know that we have seen the complete word, we mark found in global variable, and return True from main function.
        # Note that you cannot mark same location more than once so, we have to use visited 2d array to mark it as visited.
        # At times we may have gone in a wrong path and due to this we would have mark some locations useless as visited, we have unvisit them, if we see that id haven't lead to the answer. This is why this problem is backtracking + DFS.
    # Space - O(m * n), because in worst case, the whold word length stack space would be used.
    # Time - O(N * 3 ^ L). N is total number of cells in the board. L is the length of the word. 3 power because, we will have three sides to expand each time.


class Solution:
    
    def __init__(self):
        self.found = False
    
    def helper(self, board, i, j, word, word_index, visited):
        
        if board[i][j] == word[word_index] and word_index == len(word) - 1:
            self.found = True
            return
        
        if board[i][j] != word[word_index]:
            return
        
        visited[i][j] = True
        # print(i, j)
        if i - 1 >= 0 and not visited[i-1][j]:
            self.helper(board, i - 1, j, word, word_index + 1, visited)
        if j + 1 < len(board[i]) and not visited[i][j+1]:
            self.helper(board, i, j + 1, word, word_index + 1, visited)
        if i + 1 < len(board) and not visited[i+1][j]:
            self.helper(board, i + 1, j, word, word_index + 1, visited)
        if j - 1 >= 0 and not visited[i][j-1]:
            self.helper(board, i, j - 1, word, word_index + 1, visited)
        visited[i][j] = False
    
    def exist(self, board: List[List[str]], word: str) -> bool:
        
        for i in range(len(board)):
            for j in range(len(board[i])):
                visited = [[False for _ in range(len(board[i]))] for _ in range(len(board))]
                self.helper(board, i, j, word, 0, visited)
                if self.found:
                    return True
                # print('-------------')
                
        return False

This is a pretty standard backtrack + pruning, with the addition of a bit of code to handle the more time consuming corner cases. The code returns the matrix to its original state before returning.

        if set(word) - set(x for y in board for x in y):
            return False
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        def find(row, col, remain, direction):
            if board[row][col] == remain[-1]:
                temp = remain.pop()
                board[row][col] = None

                res = (
                    not remain or
                    direction != 'down' and row > 0 and find(row - 1, col, remain, 'up') or
                    direction != 'up' and row < len(board) - 1 and find(row + 1, col, remain, 'down') or
                    direction != 'right' and col > 0 and find(row, col - 1, remain, 'left') or 
                    direction != 'left' and col < len(board[0]) - 1 and find(row, col + 1, remain, 'right')
                )

                board[row][col] = temp
                remain.append(temp)
                return res
            return False

        if set(word) - set(x for y in board for x in y):
            return False

        remain = list(word[::-1])
        for i in range(len(board)):
            for j in range(len(board[0])):
                if find(i, j, remain, None):
                    return True
        return False

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        l, rows, cols = len(word), len(board), len(board[0])
        
        def dfs(x, y, d): # d is the depth of recursion
            
            if d == l: return True
            else:
                if 0 <= x < cols and 0 <= y < rows and board[y][x] == word[d]:
                    board[y][x], tmp = "", board[y][x]
                    for dx, dy in ((-1, 0), (1, 0), (0, 1), (0,-1)):
                        if dfs(x + dx, y + dy, d + 1): return True
                    board[y][x] = tmp
        counter, starts = Counter(word), []
        for row in range(rows):
            for col in range(cols):
                counter[board[row][col]] -= 1
                if board[row][col] == word[0]: starts.append((row, col))
        if max(counter.values()) > 0: return False
        for row, col in starts:
            if dfs(col, row, 0): return True

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        def recr(i,j,w,b,idx):
            # If index of string exausted we completed the search so return true
            if idx == len(w)-1:
                return True
            
            # Mark the visited index to avoid revisiting
            # b[i][j] = chr(ord(b[i][j]) - 65) # OR use below
            temp = b[i][j]
            b[i][j] = '-1'
            
            # Check for 4 conditions
            if i>0 and b[i-1][j] == w[idx+1] and recr(i-1, j, w, b, idx+1):
                return True
            elif i<len(b)-1 and b[i+1][j] == w[idx+1] and recr(i+1, j, w, b, idx+1):
                return True
            elif j>0 and b[i][j-1] == w[idx+1] and recr(i, j-1, w, b, idx+1):
                return True 
            elif j<len(b[0])-1 and b[i][j+1] == w[idx+1] and recr(i, j+1, w, b, idx+1):
                return True
            
            # b[i][j] = chr(ord(b[i][j]) + 65) # OR use below
            b[i][j] = temp
            
            return False
        
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == word[0] and recr(i,j,word,board,0):
                    return True
        return False

104. Maximum Depth of Binary Tree
Easy

4002

97

Add to List

Share
Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

 

Example 1:


Input: root = [3,9,20,null,null,15,7]
Output: 3
Example 2:

Input: root = [1,null,2]
Output: 2
Example 3:

Input: root = []
Output: 0
Example 4:

Input: root = [0]
Output: 1

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        queue = []
        queue = [root]
        height = 0
        while(queue):
            cnt = len(queue)
            height+=1
            while(cnt):
                node = queue.pop(0)
                cnt-=1
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return height

class Solution:
    def maxDepth(self, root):
        if not root: return 0
        
        res , st , depth = [] , [] , []
        curr_depth , max_depth = 1 , 1
        
        cycle = True
        while cycle:
            if root.left != None:
                st.append(root)
                depth.append(curr_depth)
                root = root.left
                root.val += 1
                curr_depth += 1
                if curr_depth > max_depth: max_depth = curr_depth
            else:
                res.append(root.val)
                if root.right != None:
                    root = root.right
                    curr_depth += 1
                    if curr_depth > max_depth: max_depth = curr_depth
                else:
                    if st:
                        root = st[-1]
                        curr_depth = depth[-1]
                        st.pop()
                        depth.pop()
                        root.left = None
                    else: cycle = False
        return max_depth

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        
        def findMaxDepth(node,maxDepth=0):
            if not node:
                return maxDepth
            else:
                return max(maxDepth,findMaxDepth(node.left,maxDepth+1), findMaxDepth(node.right,maxDepth+1))
        
        return findMaxDepth(root)

def maxDepth(self, root: TreeNode) -> int:
    	def traverse(root, level):
		if not root:
			return level
		return max(traverse(root.left, level + 1), traverse(root.right, level + 1))
	return traverse(root, 0)

    def maxDepth(self, root: TreeNode) -> int:
        return 0 if not root else 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

100. Same Tree
Easy

3310

88

Add to List

Share
Given the roots of two binary trees p and q, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

 

Example 1:


Input: p = [1,2,3], q = [1,2,3]
Output: true
Example 2:


Input: p = [1,2], q = [1,null,2]
Output: false
Example 3:


Input: p = [1,2,1], q = [1,1,2]
Output: false
 

Constraints:

The number of nodes in both trees is in the range [0, 100].
-104 <= Node.val <= 104

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if p  == None or q == None:
            if p  == None and  q == None:
                return True
            else:
                return False
        elif p.val ==  q.val:
            return (self.isSameTree(p.left, q.left) and True and self.isSameTree(p.right, q.right))
        else:
            return False
   Practically if the nodes match, look at the offshoot nodes, If they both are nothing, return true becacuse you have reached the end of the branch. Compare all results so that you are sure you have received all trues.

class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if p is None or q is None:
            return p is q
        elif p.val==q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right):
            return True
        else:
            return False

class Solution:
    def isSameTree_(self, p, q):
        
        # recursive dfs
        # T: O(N)
        # S: O(N)
        
        if p and q:
            return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return p is q

    
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        
        # recursive pre-order traversal
        # T: O(N)
        # S: O(N)
        
        p1 = self.pre_order(p, [])
        q1 = self.pre_order(q, [])
		
        return p1 == q1
    
    def pre_order(self, node, lst):
        if not node:
            lst.append(None)
        else:
            lst.append(node.val)
            self.pre_order(node.right, lst)
            self.pre_order(node.left, lst)
        return lst

class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        def twoTreeTraversal(p: TreeNode, q:TreeNode) -> bool:
            if p and q and p.val == q.val:
                    return twoTreeTraversal(p.left, q.left) and twoTreeTraversal(p.right, q.right)
            elif p or q:
                return False
            else:
                return True
        
        return twoTreeTraversal(p,q)

226. Invert Binary Tree
Easy

5282

78

Add to List

Share
Given the root of a binary tree, invert the tree, and return its root.

 

Example 1:


Input: root = [4,2,7,1,3,6,9]
Output: [4,7,2,9,6,3,1]
Example 2:


Input: root = [2,1,3]
Output: [2,3,1]
Example 3:

Input: root = []
Output: []
 

Constraints:

The number of nodes in the tree is in the range [0, 100].
-100 <= Node.val <= 100

class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        def tree(node):
            if not node:
                return None
            tp=TreeNode(node.val)
            tp.left=tree(node.right)
            tp.right=tree(node.left)
            return tp
        return tree(root)

def invertTree(self, root: TreeNode) -> TreeNode:
        def invert(node):
            if not node or (not node.left and not node.right):
                return node
            node.right, node.left  = invert(node.left), invert(node.right)
            return node
        invert(root)
        return root

class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        def do_invert(node):
            if not node:
                return
            
            node.left, node.right = node.right, node.left
            do_invert(node.left)
            do_invert(node.right)
        
        do_invert(root)
        return root

class Solution(object):
    	def invertTree(self, root):
		self.invert(root)
		return root

	def invert(self,root):
		if(root):
			temp=root.right
			root.right=root.left
			root.left=temp
			self.invert(root.left)
			self.invert(root.right)

# O(n) time/ O(n) space
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        queue = [root]
        while queue:
            node = queue.pop(0)
            if not node:
                continue
            if not node.left and not node.right:
                continue
            node.left, node.right = node.right, node.left
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return root

# O(n) time / O(h) space where h is the height of the tree
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
		 self.invert(root)
		 return root
        
    def invert(self, root):
        if not root:
            return
        if not root.left and not root.right:
            return
        root.left, root.right = root.right, root.left
        self.invert(root.left)
        self.invert(root.right)

124. Binary Tree Maximum Path Sum
Hard

5808

402

Add to List

Share
A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the root of a binary tree, return the maximum path sum of any path.

 

Example 1:


Input: root = [1,2,3]
Output: 6
Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.
Example 2:


Input: root = [-10,9,20,null,null,15,7]
Output: 42
Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.
 

Constraints:

The number of nodes in the tree is in the range [1, 3 * 104].
-1000 <= Node.val <= 1000

*A node can only appear in the sequence at most once, and the path does not need to pass through the root.
*So if we choose to go both left and right at a certain node, that node would be the "root" node of our path
*For each node, we consider three different cases
*1. We stop at current node, which means we neither go left nor go right
*2. We choose either left or right
*3. We choose both left and right

class Solution:

def maxPathSum(self, root: TreeNode) -> int:
    ans = [float("-inf")]
    def search(node, curr_max):
        if not node:
            return 0
        left_max = search(node.left, curr_max)
        right_max = search(node.right,curr_max)

        single_max = max(0,left_max, right_max)+node.val # case1 and case 2
        double_max = max(single_max, left_max+right_max+node.val) # case 3

        ans[0] = max(ans[0], single_max, double_max)
        # remember to return the value of case 1 and case 2, since if we go up, current node can't be the "root" node
        return single_max 
    search(root, 0)
    return ans[0]

Hello, my train of thought was : print the inorderTraversal of the tree and use kadane Algorithm to find the maximumSubArray, this would give you the maximumPath.

But i can't manage to pass all the tests could somebody look at the code and explain me why ?
I think this have to do with the fact i used InOrderTraversal instead of preOrder but i'm not sure...

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        def inOrderTraversal(root,array):
            if root is None:
                return
            inOrderTraversal(root.left,array)
            array.append(root.val)
            inOrderTraversal(root.right,array)
        inOrder = []
        inOrderTraversal(root,inOrder)
        maxSum = inOrder[0]
        currentSum = inOrder[0]
        if len(inOrder) <= 1:
            return inOrder[0]
        for i in range(1,len(inOrder)):
            currentSum = max(inOrder[i],currentSum + inOrder[i])
            print(currentSum)
            maxSum = max(currentSum,maxSum)
            print(maxSum)
        return maxSum

class Solution:
    
    def __init__(self):
        self.ans=float("-inf")
    
    def maxPathSumU(self,root):
        
        if root==None:
            return 0
        
        ls=self.maxPathSumU(root.left)
        rs=self.maxPathSumU(root.right)
        
        self.ans=max(self.ans,ls+rs+root.val)
        self.ans=max(self.ans,max(ls,rs)+root.val)
        self.ans=max(self.ans,root.val)
        return max(max(ls,rs)+root.val,root.val)
        
    
    def maxPathSum(self, root: TreeNode) -> int:
        self.maxPathSumU(root)   
        return self.ans

class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        #
        # stack = []
        res = float('-inf')
        
        def dfs(root,res):
            
            if not (root.left or root.right):
                cur = root.val
                res = max(res,cur)
                return cur,res
            
            cur = root.val
            res = max(res,cur)
            
            if root.left:
                left,tmp = dfs(root.left,res)
                res = max(res,tmp)
                if left > 0:
                    cur += left
                    res = max(cur,res)
            if root.right:
                right,tmp = dfs(root.right,res)
                res = max(res,tmp)
                if right > 0:
                    cur += right
                    res = max(cur,res)
            
            if root.left and root.right and cur == root.val+left+right:
                res = max(res,cur)
                cur = root.val+max(left,right)
            return cur,res
        
        cur,res = dfs(root,res)
        
        return max(cur,res)

We are going to approach this problem by asking ourselves 'For each node, what is the maximum path sum if we include this node in the path?' We have 4 cases to consider. 1.) We only use this node in the path. 2.) We use this node and the best path we found in the left subtree that leads to this node. 3.) We use this node and the best path we found in the right subtree that leads to this node. 4.) The path formed by connecting the best path in the left subtree with the current node and the best path in the right subtree. We initialize the maximum path sum to the root val and dfs through the tree. If any of the paths from the 4 cases stated above is greater than what we have found so far we update the maximum path sum.

class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        ans = root.val
        def dfs(node):
            nonlocal ans
            if not node:
                return 0
            else:
                left = dfs(node.left)
                right = dfs(node.right)
                if node.val + left + right > ans:
                    ans = node.val + left + right
                if node.val + left > ans:
                    ans = node.val + left
                if node.val + right > ans:
                    ans = node.val + right
                if node.val > ans:
                    ans = node.val
                return max(node.val, node.val + left, node.val + right)
            
        max_path_using_root = dfs(root)
        if max_path_using_root > ans:
            return max_path_using_root
        else:
            return ans

102. Binary Tree Level Order Traversal
Medium

4845

108

Add to List

Share
Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

 

Example 1:


Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]
Example 2:

Input: root = [1]
Output: [[1]]
Example 3:

Input: root = []
Output: []
 

Constraints:

The number of nodes in the tree is in the range [0, 2000].
-1000 <= Node.val <= 1000

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        result = []
        self.treeTraversal(root, result, 0)
        return result
        
    def treeTraversal(self, node, result, level):
        if node == None:
            return
        if len(result) < level + 1:
            result.append([])
        result[level].append(node.val)
        self.treeTraversal(node.left, result, level + 1)
        self.treeTraversal(node.right, result, level + 1)


def level_order_traversal(self,root,level_number,levels_values):
        if not root:return 
        if(level_number in levels_values):
            levels_values[level_number].append(root.val)
        else:
            levels_values[level_number]=[root.val]
        if(root.left):self.level_order_traversal(root.left,level_number+1,levels_values)
        if(root.right):self.level_order_traversal(root.right,level_number+1,levels_values)
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root: return []
        hashmap={}
        self.level_order_traversal(root,0,hashmap)
        ans=[]
        for i in sorted(hashmap.keys()):
            ans.append(hashmap[i])
        return ans

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        answer = []
        if root is None:
            return []
        queue = [(root,0)]
        while len(queue) > 0 :
            pos,rank = queue.pop(0)
            if len(answer) <= rank :
                answer.append([pos.val])
            else :
                answer[rank].append(pos.val)
            if pos.left is not None:
                queue.append((pos.left,rank+1))
            if pos.right is not None:
                queue.append((pos.right,rank+1))
        return answer

What is level in our binary tree? It is set of nodes, for which distance between root and these nodes are constant. And if we talk about distances, it can be a good idea to use bfs.

We put our root into queue, now we have level 0 in our queue.
On each step extract all nodes from queue and put their children to to opposite end of queue. In this way we will have full level in the end of each step and our queue will be filled with nodes from the next level.
In the end we just return result.
Complexity
Time complexity is O(n): we perform one bfs on our tree. Space complexity is also O(n), because we have answer of this size.

Code
class Solution:
    def levelOrder(self, root):
        if not root: return []
        queue, result = deque([root]), []
        
        while queue:
            level = []
            for i in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:  queue.append(node.left)
                if node.right: queue.append(node.right)
            result.append(level)
        return result

def levelOrder(self, root: TreeNode) -> List[List[int]]:
    
    d=defaultdict(list)
    
    
    def dfs(node,level):
        
        if node is None: return
        
        d[level].append(node.val)
        
        dfs(node.left,level+1)
        dfs(node.right,level+1)
    
    
    dfs(root,0)
    
    
    return d.values()

BFS Implementation:
Move level wise by maintaining a queue that stores all the elements. We check the length of the queue before each iteration in order to know how many elements are present in the particular level. Then for each node in the queue we append its left and/or right to the queue if they arenot none.

def levelOrder(self, root: TreeNode) -> List[List[int]]:
      queue, list0 = [root], [] # Queue is used to store all the nodes in level order, list0 is a list of lists with each list representing a level in the binary tree
      while queue and root:
          list1 = []                  # Used to store nodes of a particular level
          for i in range(len(queue)): # The current length of the queue represents the number of nodes in the current level
              node = queue.pop(0)
              queue += filter(None, (node.left, node.right)) # Add left and right of 'node' to queue if they are not null
              list1.append(node.val)  
          list0.append(list1)
      return list0
DFS Implementation:
Move down the tree instead of across it. Recursively check left and right subtrees of each node while passing the current depth as a parameter to the method depth_first_search

    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        ret = []
        def depth_first_search(root, depth): # Nested method to compute depth of each node and append it to list
            if not root: return              # Stopping condition when we reach None
            if len(ret) == depth: ret.append([])  # len(ret) represents the number of levels already in ret
            ret[depth].append(root.val)
            depth_first_search(root.left, depth + 1) # Recursively move down left subtree
            depth_first_search(root.right, depth + 1) # Recursively move down right subtree
        depth_first_search(root, 0) # Calling the method depth_first_search and initializing depth of root to 0
        return ret

297. Serialize and Deserialize Binary Tree
Hard

4374

200

Add to List

Share
Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

Clarification: The input/output format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.

 

Example 1:


Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]
Example 2:

Input: root = []
Output: []
Example 3:

Input: root = [1]
Output: [1]
Example 4:

Input: root = [1,2]
Output: [1,2]
 

Constraints:

The number of nodes in the tree is in the range [0, 104].
-1000 <= Node.val <= 1000

class Codec:
    def serialize(self, root):
        if not root: return ""
        queue, res = deque([root]), []
        
        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.val == 10000: res.append("*")
                else: 
                    res.append(str(node.val))
                    if node.left: queue.append(node.left)
                    else: queue.append(TreeNode(10000))
                    if node.right: queue.append(node.right)   
                    else: queue.append(TreeNode(10000))
                        
        return '='.join(res)

    def deserialize(self, data):
        if not data: return None
        data_l = data.split('=')
        root = TreeNode(data_l[0])
        queue = deque([root])
        idx = 1
                            
        while queue:
            length = len(queue)
            for _ in range(length):
                node = queue.popleft()
                node.left = None if data_l[idx] == '*' else TreeNode(data_l[idx])
                if node.left: queue.append(node.left)
                idx += 1
                node.right = None if data_l[idx] == '*' else TreeNode(data_l[idx])
                if node.right: queue.append(node.right)
                idx += 1

        return root

class Codec:
    
    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        
        BFS solution
        if it is Null it puts A in a list
        else it converts the int to string
        adn appends it in list
        """
        
        if not root:
            return ''
        
        stri = []
        lst = [root]
        
        while lst:
            
            
            y = lst.pop(0)
            stri.append(str(y.val) if not (y is None) else 'A')
            
            if y:
                lst.append(y.left)
                lst.append(y.right)
        #print(stri)       
        return stri
                
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        
        we recreate the string into a tree
        iteratively create the tree 
        
        """
        
        if data == '':
            return []
        data = [ s for s in data]
        tree = TreeNode(int(data.pop(0)))
        tmp = tree
        nodes = [tmp]
                        
        while data:
            
            x = nodes.pop(0)
            if x:
                
                leftval = 'A'
                rightval = 'A'
                if data:
                    leftval = data.pop(0)
                if data:
                     rightval = data.pop(0)
                left = TreeNode(int(leftval)) if leftval != 'A' else None
                right = TreeNode(int(rightval)) if rightval != 'A' else None
                if left:
                    nodes.append(left)
                    x.left = left

                if right:
                    nodes.append(right)
                    x.right = right
                    
        return tree

Implemented serialize and deserize using both recursive and iterative approaches. O(n) time and O(H) space where H is the height of the binary tree.

class Codec:
    def serialize(self, root):
        def serialize_REC(root,result=[]):
            if not root:
                result.append('-1001')
                return result
            result.append(str(root.val))
            serialize_REC(root.left)
            serialize_REC(root.right)
            return result
        
        def serialize_ITR(root,result=[]):
            stack=[root] if root else [None]
            result=[]
            while stack:
                node=stack.pop()
                result.append(str(node.val) if node else '-1001')
                if node:
                    stack.append(node.right) 
                    stack.append(node.left) 
            return result
        if random.randrange(0,2,1):
            return '#'.join(serialize_REC(root))
        return '#'.join(serialize_ITR(root)) 

    def deserialize(self, data):
        arr=list(map(lambda x: int(x) if x!='-1001' else None,data.split('#')))
        r=random.randrange(0,3,1) 
        if  r==0:
            return self.deserialize_ITR(arr)
        elif r==1:
            self.i=0
            return self.deserialize_REC_globalvar(arr)
        else:
            return self.deserialize_REC_localvar(arr)[0]
        
    def deserialize_ITR(self, data):
        stack=[]
        for i in range(len(data)):
            child=TreeNode(data[i]) if data[i]  is not None else None  #big bug: .... if arr[i] else None : this caused child==None if arr[i]==0
            if i==0:
                root=child
            if stack:
                stack[-1][1]+=1
                if stack[-1][1]==1:
                    stack[-1][0].left=child
                elif stack[-1][1]==2:
                    stack[-1][0].right=child
                    while stack and stack[-1][1]==2 : # we can pop ANY node as long as its both children are known
                        stack.pop()
            stack.append([child,0]) if child else 0
        return root
    
    def deserialize_REC_globalvar(self, data,):
        if data[self.i] is None : #node is None
            self.i+=1
            return None
        node=TreeNode(data[self.i]) 
        self.i+=1
        node.left=self.deserialize_REC_globalvar(data)
        node.right=self.deserialize_REC_globalvar(data,)
        return node
    def deserialize_REC_localvar(self, data,i=0):
        if data[i] is None : #node is None
            i+=1
            return None,i
        node=TreeNode(data[i]) 
        node.left,i=self.deserialize_REC_localvar(data,i+1)
        node.right,i=self.deserialize_REC_localvar(data,i)
        return node,i

class Codec:
    
    def serialize(self, root):
        
        ret = [root]
        
        for u in filter(None, ret):
            
            ret.append(u.left)
            ret.append(u.right)
        
        return ",".join("null" if u is None else str(u.val) for u in ret)
            
        

    def deserialize(self, data):
        
        nodes = deque(None if x == "null" else TreeNode(int(x)) for x in data.split(","))
        
        it = iter(nodes)
        next(it)
        
        for u in filter(None, nodes):
            
            u.left = next(it)
            u.right = next(it)
        
        return nodes[0]

572. Subtree of Another Tree
Easy

3524

175

Add to List

Share
Given the roots of two binary trees root and subRoot, return true if there is a subtree of root with the same structure and node values of subRoot and false otherwise.

A subtree of a binary tree tree is a tree that consists of a node in tree and all of this node's descendants. The tree tree could also be considered as a subtree of itself.

 

Example 1:


Input: root = [3,4,5,1,2], subRoot = [4,1,2]
Output: true
Example 2:


Input: root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2]
Output: false
 

Constraints:

The number of nodes in the root tree is in the range [1, 2000].
The number of nodes in the subRoot tree is in the range [1, 1000].
-104 <= root.val <= 104
-104 <= subRoot.val <= 104

class Solution:
    def isSubtree(self, root: TreeNode, subroot: TreeNode) -> bool:
        
        def is_same(a,b):
            if not a and not b:
                return True
            if not a or not b:
                return False
            return a.val == b.val and is_same(a.left,b.left) and is_same(a.right,b.right)
        
        def is_sub(root,sub):
            if not root and not sub:
                return True
            if not root or not sub:
                return False
            return is_same(root,sub) or is_sub(root.left,sub) or is_sub(root.right,sub) 
        
        return is_sub(root,subroot)

def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:
    	stack = [root]

	while stack:
		node = stack.pop()

		if node.val==subRoot.val:
			if self.check(node, subRoot):
				return True
		if node.left:
			stack.append(node.left)
		if node.right:
			stack.append(node.right)
	return False

def check(self, r:TreeNode, s:TreeNode) -> bool:
	stack = [(r, s)]

	while stack:
		node1, node2 = stack.pop()
		#print(node1, node2)
		if node1.val!=node2.val:
			return False
		if node1.left or node2.left:
			if (node1.left and not node2.left) or (not node1.left and node2.left):
				return False    
			else:
				stack.append((node1.left, node2.left))
		if node1.right or node2.right:
			if (node1.right and not node2.right) or (not node1.right and node2.right):
				return False
			else:
				stack.append((node1.right, node2.right))
	return True

    def isSubtree(self, root, subRoot):
        if not root or not subRoot:
            return False

        if root.val == subRoot.val:
            if self.isSame(root, subRoot):
                return True
            
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
        
    def isSame(self, p, q):
        if not p or not q:
            return p == q

        if p.val != q.val:
            return False

        return self.isSame(p.left, q.left) and self.isSame(p.right, q.right)

class Solution:
    def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:
        '''
        Avoiding String Comparison:
        
        Idea is similar to find duplicate subtrees problem.
        1. Serialize subroot postorder manner
        2. Find if root has that serialization in same postorder manner. 
        
        Time and Space complexity: O(m+n)
        
        '''

        def seri_subtree(subRoot):
            if subRoot is None:
                return '#'
            left = seri_subtree(subRoot.left)
            right = seri_subtree(subRoot.right)
            temp = 'l'+left+str(subRoot.val)+'r'+right
            return temp
        
        table = {}
        seri = seri_subtree(subRoot)
        table[seri] = True
        ans = False
        
        def check_sub(root):
            nonlocal ans
            if root is None:
                return '#'
            left = check_sub(root.left)
            right = check_sub(root.right)
            temp = 'l'+left+str(root.val)+'r'+right
            if temp in table:
                ans = True
            return temp
        
        check_sub(root)
        return ans

class Solution:
    
    def __init__(self):
        self.flag = False
        
    def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:
        
        def check(node1,node2):
            
            if (not node1 and not node2):
                return 
            
            if  (not node1 and node2) or (not node2 and node1) or node1.val != node2.val:
                self.flag = False 
                return 
            
            check(node1.left,node2.left)
            
            check(node1.right,node2.right)
            
        
        def inorder(node):
            if not node or self.flag == True:
                return 
            
            if node.val == subRoot.val:
                self.flag = True 
                check(node,subRoot)
                if self.flag == True:
                    return 
            
            inorder(node.left)
            
            inorder(node.right) 
        
        inorder(root)
        
        
        return self.flag 

105. Construct Binary Tree from Preorder and Inorder Traversal
Medium

5374

133

Add to List

Share
Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.

 

Example 1:


Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
Example 2:

Input: preorder = [-1], inorder = [-1]
Output: [-1]
 

Constraints:

1 <= preorder.length <= 3000
inorder.length == preorder.length
-3000 <= preorder[i], inorder[i] <= 3000
preorder and inorder consist of unique values.
Each value of inorder also appears in preorder.
preorder is guaranteed to be the preorder traversal of the tree.
inorder is guaranteed to be the inorder traversal of the tree.

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if(len(inorder) == 0):
            return None
        
        root_element = preorder.pop(0)
        root = TreeNode(root_element)
        
        inorder_index = inorder.index(root_element)
        
        left_inorder = inorder[:inorder_index]
        right_inorder = inorder[inorder_index+1:]
        
        root.left = self.buildTree(preorder, left_inorder)
        root.right = self.buildTree(preorder, right_inorder)
        
        return root

class Solution(object):
    def buildTree(self,preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        global preorder_ind
        preorder_ind = 0
        head = TreeNode()
        self.buildTreeHelper(head, preorder, inorder)

        return head


    def buildTreeHelper(self, p, preorder, inorder):
        global preorder_ind
        root_val = preorder[preorder_ind]
        p.val = root_val

        if root_val in inorder:
            ind = inorder.index(root_val)
        else:
            return

        leftSub = inorder[0:ind]
        rightSub = inorder[ind+1:]

        if len(leftSub) != 0 and preorder_ind + 1 < len(preorder):
            preorder_ind += 1
            p.left = TreeNode()
            self.buildTreeHelper(p.left, preorder, leftSub)
        if len(rightSub) != 0 and preorder_ind + 1 < len(preorder):
            preorder_ind += 1
            p.right = TreeNode()
            self.buildTreeHelper(p.right, preorder, rightSub)


The preorder traversal has each root in the tree, followed by its left and then right subtree while the inorder traversal has a left subtree followed by the current root then the right subtree. So we can use the preorder array to get each root. Then we find that value in the inorder array, and the left subtree will be all the values in the inorder array to the left of that root value and the right subtree consists of the values to the right. Then we recursively build the trees for the left and right subtrees and increment the preorder index to get the next root for these subtrees.

The idea here is to that
preOrder[0] = root ,
if we find the root value in Inorder then
the half below the inorder postion of the root is the left tree
and the other half contains the right subtree
You can either search the root value in the inorder list or you can simply create a hash map/dictionary to store the map inorder list values to indexes ( As every element is unique)

class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        inOrderHash = dict()
        n = len(preorder)
        m = len(inorder)
        for i in range(m):
            inOrderHash[inorder[i]] = i

        def buildTreeHelper(preStart, inStart, inEnd, preEnd):
            if preStart>=preEnd or inStart >=inEnd:
                 return None
            root = TreeNode(preorder[preStart])
            InOIndex = inOrderHash[root.val]
            root.left = buildTreeHelper(preStart+1, inStart, InOIndex, preEnd)
            root.right = buildTreeHelper(preStart+InOIndex-inStart+1, InOIndex+1, inEnd, preEnd)
            return root
        return buildTreeHelper(0,0,m,n)

    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        root = None
        
        pq = deque(preorder)
        iq = deque(inorder)
        seen = set()
        
        def rec():
            node = TreeNode(pq.popleft())
            seen.add(node.val)
            
            if node.val != iq[0]:
                #find left
                node.left = rec()
            
            #pop self from inorder
            iq.popleft()
               
            # if inorder still has unseen items, find right
            if iq and iq[0] not in seen:
                node.right = rec()
            
            return node
            
        
        return rec()
        
        #preorder = [3,9,1,20,15,7], inorder = [1,9,3,15,20,7]

def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
    		if len(preorder)==0:
            return None
        rootData = preorder[0]
        root = TreeNode(rootData)
        inOrderIndex = inorder.index(rootData)
        if inOrderIndex == -1:
            return None
        leftInorder = inorder[0:inOrderIndex]
        rightInorder = inorder[inOrderIndex +1:]
        
        lenLeftSubtree = len(leftInorder)
        
        leftPreorder = preorder[1:lenLeftSubtree+1]
        rightPreorder = preorder[lenLeftSubtree+1:]
        
        root.left = self.buildTree(leftPreorder,leftInorder)
        root.right = self.buildTree(rightPreorder,rightInorder)
        return root

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not inorder:
            return None
        
        root = TreeNode(preorder[0])
        inorder_idx = inorder.index(preorder[0])
        preorder.pop(0)
        
        left_nodes = inorder[:inorder_idx]
        right_nodes = inorder[inorder_idx + 1:]
        root.left = self.buildTree(preorder, left_nodes)
        root.right = self.buildTree(preorder, right_nodes)
        
        return root
In preorder traversal, the nodes are marked visited in the order Root, Left, Right. In inorder traversal, the nodes are marked visited in the order Left, Root, Right.
With this information we can determine that the root of the tree is the first element in the preorder list.
The nodes in the left subtree are the nodes to the left of the root value's index in the inorder list and the nodes in the right subtree are the nodes to the right of the root value's index in the inorder list.

98. Validate Binary Search Tree
Medium

6238

699

Add to List

Share
Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.
 

Example 1:


Input: root = [2,1,3]
Output: true
Example 2:


Input: root = [5,1,4,null,null,3,6]
Output: false
Explanation: The root node's value is 5 but its right child's value is 4.
 

Constraints:

The number of nodes in the tree is in the range [1, 104].
-231 <= Node.val <= 231 - 1

class Solution(object):
    def isValidBST(self, root, minumum = float('-inf'), maximum = float('inf')):
        if not root:
            return True
        if root.val >= maximum or root.val <= minumum:
            return False
        return self.isValidBST(root.left, minumum, root.val) and \
               self.isValidBST(root.right, root.val, maximum)

The idea behind the approach is to track the upper and lower limit of a node.

Upper limit of left child node will be equal to it's parent value.
Lower limit of the left child node will be equal to lower limit of it's parent node.
Upper limit of right child node will be equal to upper limit of it's parent node.
Lower limit of right child node will be equal to it's parent value.
Run inorder traversal over the tree and validate the upper and lower limits for each node.

class Solution:
    def isValidBST(self, root, upper_limit=float('inf'), lower_limit=float('-inf')):
        return True if root is None else ( 
            root.val > lower_limit
            and root.val < upper_limit
            and self.isValidBST(root.left, root.val, lower_limit)
            and self.isValidBST(root.right, upper_limit, root.val))
Recursive call diagram for input [5,1,14,-1,2,3,16]:
image

Link to diagram to edit or play with more test cases:

https://docs.google.com/drawings/d/1sOhhFcJcbcWa8V2NP1gd3HmTPUYq-WFid4vmLP3Vh3w/edit?usp=sharing

def visit(node):
    yield node.val

def in_order(node):
    if node is not None:
        yield from in_order(node.left)
        yield from visit(node)
        yield from in_order(node.right)

class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        last = None
        for node in in_order(root):
            if last is not None:
                if last >= node:
                    return False
            last = node
        return True

    def isValidBST(self, root: TreeNode, lower=float('-inf'), upper=float('inf')) -> bool:
    		#part A
		if not root:
            return True
        
        val = root.val
		# part B
        if val <= lower or val >= upper:
            return False
        # part C
        if not self.isValidBST(root.right, val, upper):
            return False
        if not self.isValidBST(root.left, lower, val):
            return False
        return True
        
To our method we added the lower and upper as input parameters with default infinite values to avoid modifying the signature of it. In the first if, we have a base case (see Part A in code)which validates empty trees, If node is None, True is returned from the method. If not, we continue and assign the value of the root element to a variable.

Next, we check if val is less or equal to lower or if val is greater or equal to upper (see Part B in code). We do this because the value of the current node should be greater than all the values of the children in the left subtree, and it should be less than all the values of the children in the right subtree. If you remember, we set as default with infinite values, so this will not be triggered yet, because we are on the root node.

Now that we have checked the current node, it’s time to check it for the subtrees (see Part C in code). For this, we make a recursive call to the right subtree of the current node. The right node is passed as node, val is passed as lower while upper stays the same.

Lower is now the lower bound for the right subtree as all the children in the right subtree have to be greater than the value of the current node. If the recursive call returns False, we're done and that's it.

For the other hand, the left subtree is evaluated through a recursive too. Out val variable is passed as upper for the recursive call as all the children in the left subtree have to be less than the value of the current node.

If none of these calls returns a False, we send a True and that means that the BST is satisfied.

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        
        return self.isValidHelper(root, -100000000000, 10000000000)
    
    def isValidHelper(self, root, mn, mx):
        if root == None:
            return True
        
        if root.val <= mn or root.val >= mx:
            return False
        
        return self.isValidHelper(root.left, mn, root.val) and self.isValidHelper(root.right, root.val, mx)
		
For a binary tree to be valid, everything in the left subtree must be less than the current value of the root and everything in the right subtree must be greater than the root. Then to check if the tree is valid we can check if the current node value is less than or equal to the minimum value or greater than or equal to the maximum value. The maximum value for the left subtree is the current root value because everything to the left must be less than the root and the minimum value for the right subtree is the current root value because everything to the right must be greater than the root. We can then validate the tree by recursing on the left and right subtrees with these new maximum and minimum values.

class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        if root==None:
            return 1
        if root.left==None and root.right==None:
            return 1

        list=[]
        self.Inorder(root,list)
        s=set(list)
        if list==sorted(list) and len(list)==len(s):
            return 1
        return 0
        
    
    
    def Inorder(self,root,list):
        if root==None:
            return
        self.Inorder(root.left,list)
        list.append(root.val)
        self.Inorder(root.right,list)

230. Kth Smallest Element in a BST
Medium

4014

89

Add to List

Share
Given the root of a binary search tree, and an integer k, return the kth (1-indexed) smallest element in the tree.

 

Example 1:


Input: root = [3,1,4,null,2], k = 1
Output: 1
Example 2:


Input: root = [5,3,6,2,4,null,null,1], k = 3
Output: 3
 

Constraints:

The number of nodes in the tree is n.
1 <= k <= n <= 104
0 <= Node.val <= 104
 

Follow up: If the BST is modified often (i.e., we can do insert and delete operations) and you need to find the kth smallest frequently, how would you optimize?

def kthSmallest(self, root: TreeNode, k: int) -> int:
        """
        DFS lmr
        
        """
        
        nums =[]
        
        def dfs(node):
            if len(nums) == k:
                return
            
            if not node:
                return
            
            dfs(node.left)
            if len(nums) == k:
                return
            nums.append(node.val)
            if len(nums) == k:
                return
            dfs(node.right)
            
        dfs(root)    
        return nums[-1]
		```

class Solution:
    def _inorder(self, node, out, k):
        if node:
            self._inorder(node.left, out, k)
            if out[1] < k:
                out[1] += 1
                out[0] = node.val
            self._inorder(node.right, out, k)
        
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        out = [0, 0]
        self._inorder(root, out, k)
        return out[0]

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
	def kthSmallest(self, root, k):
		"""
		:type root: TreeNode
		:type k: int
		:rtype: int
		"""
		value = self.reverseInorder(root, k, [0], [-1])
		return value

	def reverseInorder(self, root, k, numberOfVisits, lastValue):
		if root is None or numberOfVisits[0] >= k:
			return
		self.reverseInorder(root.left, k, numberOfVisits, lastValue)
		if numberOfVisits[0] < k:
			numberOfVisits[0] += 1
			lastValue[0] = root.val
			self.reverseInorder(root.right, k, numberOfVisits, lastValue)
		return lastValue[0]

class Solution(object):
    def kthSmallest(self, root, k):
        global count, output
        count = k
        def inorder(node):
            global count, output
            if(count>0 and node):
                inorder(node.left)
                if(count > 0):
                    count -= 1
                    if(count == 0):
                        output = node.val
                    else:
                        inorder(node.right)
        inorder(root)
        return output

Here is my optimized code using the find approach. Here the condition for find is when count equals 0.

class Solution:
    def kthSmallest(self,root,k):
        self.count=k
        node=self.inorder(root)
        return node.val
       
    def inorder(self,root):
        if not root:
            return None
        left=self.inorder(root.left)
        if left:
            return left
        self.count-=1
        if self.count==0:
            return root
        return self.inorder(root.right)
The code below is inefficient although we will get the right answer. We have not terminated the recursion. Placing a return statement when count is 0, will not terminate the recursion. It will only stop the right subtree traversal of that particular node. Other nodes' left and right subtree will still be explored which is unnecessary.

class Solution:
    def kthSmallest(self,root,k):
        self.count,self.ksm=k,0
		self.inorder(root)
        return self.ksm
       
    def inorder(self,root):
        if not root:
            return
        self.inorder(root.left)
        self.count-=1
        if self.count==0:
			self.ksm=root.val
            return
        self.inorder(root.right)
Its necessary to understand that we must use the find or search approach in such kind of problems. We need to completely terminate recursion when we reach our desirable state.

class Solution:
    # follow-up: we store the number of nodes in the left subtree as an attribute on each node.
    # this helps us decide whether we want to go left or right for the k-th smallest element.
    # we should then, of course, increment or decrement this counter as we insert or delete nodes.

    def kthSmallest(self, root: TreeNode, k: int) -> int:  # O(n) time and O(n) space
        # recursive in-order traversal, then select element k - 1 from the array
        def traverse(node: TreeNode) -> None:
            if not node:
                return
            traverse(node.left)
            array.append(node.val)
            traverse(node.right)

        array = []
        return traverse(root) or array[k - 1]  # neat little trick

    def kthSmallest(self, root: TreeNode, k: int) -> int:  # O(n) time and O(n) space
        # recursive in-order traversal, then select element k - 1 from the array - condensed
        return (dfs := lambda node: dfs(node.left) + [node] + dfs(node.right) if node else [])(root)[k - 1].val

    def kthSmallest(self, root: TreeNode, k: int) -> int:  # O(k) time and O(k) space
        # recursive in-order traversal with early stopping
        def traverse(node: TreeNode) -> None:
            if not node:
                return
            traverse(node.left)
            if len(array) == k:
                return
            array.append(node.val)
            traverse(node.right)

        array = []
        return traverse(root) or array[-1]

    def kthSmallest(self, root: TreeNode, k: int) -> int:  # O(k) time and O(k) space
        # iterative in-order traversal with early stopping
        node, stack = root, []
        while node or stack:
            if node:
                stack.append(node)
                node = node.left
            else:
                node = stack.pop()
                k -= 1
                if k == 0:
                    return node.val
                node = node.right

    def kthSmallest(self, root: TreeNode, k: int) -> int:  # O(k) time and O(k) space
        # iterative in-order traversal with early stopping - condensed
        node, arr, val = root, [], None
        while k:
            val, node, arr, k = (0, node.left, arr + [node], k) if node else (arr[-1].val, arr.pop().right, arr, k - 1)
        return val

    def kthSmallest(self, root: TreeNode, k: int) -> int:  # O(k) time and O(1) space
        # lazy in order traversal using an iterator with early stopping
        def traverse(node: TreeNode):
            if not node:
                return
            yield from traverse(node.left)
            yield node.val
            yield from traverse(node.right)

        for i, val in enumerate(traverse(root)):
            if k - i == 1:
                return val

    def kthSmallest(self, root: TreeNode, k: int) -> int:  # O(k) time and O(1) space
        # lazy in order traversal using an iterator with early stopping - condensed
        def traverse(node: TreeNode):
            yield from (*traverse(node.left), node.val, *traverse(node.right)) if node else ()

        return next(val for i, val in enumerate(traverse(root), 1) if not k - i)

    def kthSmallest(self, root: TreeNode, k: int) -> int:  # O(k) time and O(1) space
        # cannot do a one-liner using the Walrus operator here it seems:
        # SyntaxError: assignment expression cannot be used in a comprehension iterable expression
        gen = (dfs := lambda node: (yield from (*dfs(node.left), node.val, *dfs(node.right)) if node else ()))(root)
        return next(val for i, val in enumerate(gen) if k - i == 1)

    def kthSmallest(self, root, k):  # O(h) amortized time if balanced, O(n) worst case time, and O(h) space
        # recursive binary search
        def count_nodes(node: TreeNode) -> int:
            if not node:
                return 0
            return 1 + count_nodes(node.left) + count_nodes(node.right)

        def traverse(node: TreeNode, n: int) -> int:
            count = count_nodes(node.left)
            if count < n:
                return traverse(node.right, n - count - 1)
            elif count == n:
                return node.val
            return traverse(node.left, n)

        return traverse(root, k - 1)

    def kthSmallest(self, root, k):  # O(h) amortized time if balanced, O(n) worst case time, and O(h) space
        # recursive binary search - condensed

        def traverse(node: TreeNode, n: int) -> int:
            ctr = (dfs := lambda node: 1 + dfs(node.left) + dfs(node.right) if node else 0)(node.left)
            return traverse(node.right, n - ctr - 1) if ctr < n else node.val if ctr == n else traverse(node.left, n)

        return traverse(root, k - 1)

    def kthSmallest(self, root: TreeNode, k: int) -> int:  # amortized O(h) time (if balanced), worst case O(n) time
        # iterative binary search
        def count_nodes(node: TreeNode) -> int:
            if not node:
                return 0
            return 1 + count_nodes(node.left) + count_nodes(node.right)

        node, count = root, count_nodes(root.left)
        while count != k - 1:
            if count < k - 1:
                node = node.right
                k -= count + 1
            else:
                node = node.left
            count = count_nodes(node.left)
        return node.val

    def kthSmallest(self, root: TreeNode, k: int) -> int:  # amortized O(h) time (if balanced), worst case O(n) time
        # iterative binary search - condensed
        def count_nodes(node: TreeNode) -> int:
            return 1 + count_nodes(node.left) + count_nodes(node.right) if node else 0

        node, count = root, count_nodes(root.left)
        while count != k - 1:
            node, k = (node.right, k - count - 1) if count < k - 1 else (node.left, k)
            count = count_nodes(node.left)
        return node.val

    def kthSmallest(self, root: TreeNode, k: int) -> int:  # O(k) time and O(1) space
        # Morris traversal
        node = root
        while node:
            if not node.left:
                k -= 1
                if k == 0:
                    return node.val
                node = node.right
            else:
                rightmost = node
                node = temp = node.left
                while temp.right:
                    temp = temp.right
                rightmost.left = None
                temp.right = rightmost

    def kthSmallest(self, root: TreeNode, k: int) -> int:  # O(k) time and O(1) space
        # Morris traversal - condensed
        node, val = root, None
        while k:
            if not node.left:
                val, node, k = node.val, node.right, k - 1
            else:
                rightmost, node, temp = node, node.left, node.left
                while temp.right:
                    temp = temp.right
                rightmost.left, temp.right = None, rightmost
        return val

235. Lowest Common Ancestor of a Binary Search Tree
Easy

3243

136

Add to List

Share
Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

 

Example 1:


Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
Output: 6
Explanation: The LCA of nodes 2 and 8 is 6.
Example 2:


Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
Output: 2
Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.
Example 3:

Input: root = [2,1], p = 2, q = 1
Output: 2
 

Constraints:

The number of nodes in the tree is in the range [2, 105].
-109 <= Node.val <= 109
All Node.val are unique.
p != q
p and q will exist in the BST.

'''
Time Complexity O(log(n)) where n is number of nodes in Tree.
Process:
Since Tree is BST, we will follow BST property which is left node<root<right node.
For example for tree root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
p(2)<root(6)<q(8) so root 6 is answer.

'''

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        node1=p if p.val<q.val else q
        node2=q if q.val>p.val else p
        res= self.lcautil(root,node1,node2)
        return res
    
    def lcautil(self,root,node1,node2):
        if node1.val<=root.val and root.val<=node2.val:
            return root
        elif root.val>node1.val and root.val>node2.val:
            return self.lcautil(root.left,node1,node2)
        else:
            return self.lcautil(root.right,node1,node2)

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root: return
        
        if root == p or root == q: return root
        
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        if left and right: return root
        else: return left or right

class Solution(object):
    	def lowestCommonAncestor(self, root, p, q):
		if(root):
			if(p.val<root.val and q.val<root.val):
				return self.lowestCommonAncestor(root.left,p,q)
			elif(p.val>root.val and q.val>root.val):
				return self.lowestCommonAncestor(root.right,p,q)
			else:
				return root

def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    	if p.val>q.val:
		p, q = q, p
	return self.find_lca(root, p, q)

def find_lca(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
	if (root.val>=p.val and root.val<=q.val):
		return root
	elif (root.val>p.val and root.val>q.val):
		return self.find_lca(root.left, p, q)
	elif (root.val<p.val and root.val<q.val):
		return self.find_lca(root.right, p, q)

208. Implement Trie (Prefix Tree)
Medium

4684

71

Add to List

Share
A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:

Trie() Initializes the trie object.
void insert(String word) Inserts the string word into the trie.
boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.
 

Example 1:

Input
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
Output
[null, null, true, false, true, null, true]

Explanation
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // return True
trie.search("app");     // return False
trie.startsWith("app"); // return True
trie.insert("app");
trie.search("app");     // return True
 

Constraints:

1 <= word.length, prefix.length <= 2000
word and prefix consist only of lowercase English letters.
At most 3 * 104 calls in total will be made to insert, search, and startsWith.

class Trie:
    class Node:
        def __init__(self):
            self.children=[None]*26
            self.isword=False
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root=Trie.Node()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        self._traverse(word,toadd=True).isword=True
        
    def _traverse(self,word,toadd=False) -> any:
        root=self.root
        p=0
        while p<len(word):
            c=ord(word[p])-0x61
            if not root.children[c]:
                if not toadd:
                    return None
                else:
                    root.children[c]=self.Node()
            root=root.children[c]
            p+=1
        return root

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        return r.isword if (r:=self._traverse(word) ) else False

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        return True if (r:=self._traverse(prefix) ) else False

from functools import reduce
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        T = lambda : defaultdict(T)
        self.root = T()
        self.END = '#'

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        reduce(dict.__getitem__, word, self.root)[self.END] = word
        
        # equivalent to the following code:
        #curr = self.root
        #for ch in word:
        #    curr = curr[ch]
        #curr[self.END] = word

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        curr = self.root
        for ch in word:
            if ch not in curr:
                return False
            curr = curr[ch]
        return self.END in curr
        

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        curr = self.root
        for ch in prefix:
            if ch not in curr:
                return False
            curr = curr[ch]
        return True
        


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

from collections import defaultdict

class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.node = defaultdict(dict)
        
    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node = self.node 
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['__end__'] = True     
        
    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        node = self.node 
        for char in word:
            if char not in node: return False       
            node = node[char]       
        return True if '__end__' in node else False
        
    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self.node
        for p in prefix:
            if p not in node: return False 
            node = node[p] 
        return True 
        
        


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

class Trie:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie = {}

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        # O(n) time complexity
        cur = self.trie
        for char in word:
            if char in cur:
                cur = cur[char]
            else:
                cur[char] = {}
                cur = cur[char]
        cur["#"] = 0 # Signifies the end of the word
        return

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        # O(n) time complexity
        cur = self.trie
        for char in word:
            if char in cur:
                cur = cur[char]
            else:
                return False
        return "#" in cur

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        # O(n) time complexity
        cur = self.trie
        for char in prefix:
            if char in cur:
                cur = cur[char]
            else:
                return False
        return True

class Trie:
    
    trie = None
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie = {}

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        tempTrie = self.trie
        for c in word:
            if c not in tempTrie.keys():
                tempTrie[c] = {}
            tempTrie = tempTrie[c]
        tempTrie['#'] = 1
        
    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        tempTrie = self.trie
        for c in word:
            if c in tempTrie.keys():
                tempTrie = tempTrie[c]
            else:
                return False
        return '#' in tempTrie

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        tempTrie = self.trie
        for c in prefix:
            if c in tempTrie.keys():
                tempTrie = tempTrie[c]
            else:
                return False
        return True

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

211. Design Add and Search Words Data Structure
Medium

3084

130

Add to List

Share
Design a data structure that supports adding new words and finding if a string matches any previously added string.

Implement the WordDictionary class:

WordDictionary() Initializes the object.
void addWord(word) Adds word to the data structure, it can be matched later.
bool search(word) Returns true if there is any string in the data structure that matches word or false otherwise. word may contain dots '.' where dots can be matched with any letter.
 

Example:

Input
["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output
[null,null,null,null,false,true,true,true]

Explanation
WordDictionary wordDictionary = new WordDictionary();
wordDictionary.addWord("bad");
wordDictionary.addWord("dad");
wordDictionary.addWord("mad");
wordDictionary.search("pad"); // return False
wordDictionary.search("bad"); // return True
wordDictionary.search(".ad"); // return True
wordDictionary.search("b.."); // return True
 

Constraints:

1 <= word.length <= 500
word in addWord consists lower-case English letters.
word in search consist of  '.' or lower-case English letters.
At most 50000 calls will be made to addWord and search.

class WordDictionary:
    def __init__(self):
        self.map = collections.defaultdict(list)

    def addWord(self, word):
        self.map[len(word)].append(word)       
    

    def search(self, word):
        indexes = [ i for i,v in enumerate(word) if v !='.']
        
        for item in self.map[len(word)]:
            ismatch = True
            for i in indexes:
                if item[i] != word[i]:
                    ismatch = False
                    continue
            if ismatch:
                return True
        return False

class WordDictionary:
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.value = ''
        self.children = dict()

    def addWord(self, word: str) -> None:
        self._addWord(word, word)

    def _addWord(self, word: str, remaining: str) -> None:
        """
        Inserts a word into the trie.
        """
        if not remaining:
            self.value = word
            return
        c = remaining[0]
        if c in self.children:
            self.children[c]._addWord(word, remaining[1:])
        else:
            new_node = self.children[c] = WordDictionary()
            new_node._addWord(word, remaining[1:])

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        if not word:
            return bool(self.value)
        c = word[0]
        if c == '.':
            return any(n.search(word[1:]) for n in self.children.values())
        if c in self.children:
            return self.children[c].search(word[1:])
        return False

class TrieNode:
    def __init__(self):
        self.is_word = False
        self.children = {}

class WordDictionary:

    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_word = True

    def search(self, word: str) -> bool:
        nodes = [self.root]
        for ch in word:
            next_nodes = []
            for node in nodes:
                if ch in node.children:
                    next_nodes.append(node.children[ch])
                if ch == '.':
                    next_nodes.extend(node.children.values())
            nodes = next_nodes
        
        for node in nodes:
            if node.is_word:
                return True
        
        return False

class WordDictionary:
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.len_to_words = defaultdict(set)
        self.letter_to_words = defaultdict(set)
        self.n_stored = 0

    def addWord(self, word: str) -> None:
        self.n_stored += 1
        self.len_to_words[len(word)].add(word)
        letters = set(c for c in word)
        for letter in letters:
            self.letter_to_words[letter].add(word)
        # print(f"{self.len_to_words=}; {self.letter_to_words=}")

    def search(self, word: str) -> bool:
        # Iteratively generate a candidate list, keep intersecting
        # print(f"Searching for {word}")
        candidates = self.len_to_words[len(word)]
        # print(f"\t{candidates=}")
        letters = set(c for c in word if c != ".")
        for letter in letters:
            # print(f"\t{letter=}")
            # print(f"\tIntersecting with {self.letter_to_words[letter]}")
            candidates = candidates.intersection(self.letter_to_words[letter])
            # print(f"\t{candidates=}")
        
        # Now this list is short enough that we can brute force it
        return any(self._matches(word, candidate) for candidate in candidates)

    def _matches(self, word, candidate) -> bool:    
        for w, c in zip(word, candidate):
            if w != c and w != ".":
                return False
        return True

# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)

class WordDictionary:
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie = {}
        

    def addWord(self, word: str) -> None:
        node = self.trie

        for ch in word:
            if not ch in node:
                node[ch] = {}
            node = node[ch]
        node['$'] = True

    def search(self, word: str) -> bool:
        q = [self.trie]
        
        for ch in word:            
            ln = len(q) 
            for i in range(ln): # only pop what you put in last iteration 
                t = q.pop(0)
                if ch not in t:
                    if ch == ".":
                        for node in t:
                            if t[node] != True: # Append all nodes that are hashtables
                                q.append(t[node])                    
                else: # if character is there append its hashtable
                    q.append(t[ch])
        
        if not q: return False # If its empty, it's false
        
        for nod in q:
            if '$' in nod: # Check if there is any word 
                return True

# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)

class TrieNode:
    def __init__(self, letter=None):
        # initialize TrieNode
        # it should have link to the downstream neighbirs
        # also a signal that checks whether current node is end of
        # a word
        self.neighbors = {} # this allows us to access both the letter and its node instance 
        self.last_letter = False
        
class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie = TrieNode()
        

    def addWord(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        head = self.trie # get a pointer to the root of the trie
        for letter in word:
            # if we already dont have it in the neighbors add it
            if letter not in head.neighbors:
                head.neighbors[letter] = TrieNode(letter)
            # traverse to the correct neighbor
            head = head.neighbors[letter]
        head.last_letter = True # last letter of the word is marked so that search method knows whether it reached the end
            
    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        head = self.trie
        def search_recursive(head, index):
            # getting to the end of the trie and last head matched
            if index == len(word):
                return head.last_letter
            if word[index] in head.neighbors:
                return search_recursive(head.neighbors[word[index]], index + 1)
            if word[index] == '.':
                return any([search_recursive(neighbor, index + 1) for neighbor in head.neighbors.values()])
            return False
        return search_recursive(head,0)

212. Word Search II
Hard

3861

148

Add to List

Share
Given an m x n board of characters and a list of strings words, return all words on the board.

Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.

 

Example 1:


Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]
Example 2:


Input: board = [["a","b"],["c","d"]], words = ["abcb"]
Output: []
 

Constraints:

m == board.length
n == board[i].length
1 <= m, n <= 12
board[i][j] is a lowercase English letter.
1 <= words.length <= 3 * 104
1 <= words[i].length <= 10
words[i] consists of lowercase English letters.
All the strings of words are unique.

class Solution:
    class TrieNode:
        def __init__(self):
            self.next=[None]*26
            self.word=None
            self.children=0
        def probe(self,word,add=False)->bool:
            root,p=self,0
            while p<len(word):
                c=ord(word[p])-0x61
                if not root.next[c]:
                    if add:
                        root.next[c]=Solution.TrieNode()
                        root.children+=1
                    else:
                        return None
                root=root.next[c]
                p+=1
            return root
        def add(self,word)->None:
            self.probe(word,True).word=word
            
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        def dfs(r,c,root):
            if not (newroot:=root.probe(board[r][c]) ):
                return
            saved,board[r][c]=board[r][c],'*'
            if newroot.word :
                res.append(newroot.word)
                newroot.word=None
            for dr,dc in dirs:
                nr,nc=r+dr,c+dc
                if nr>=0 and nr<m and nc>=0 and nc<n and board[nr][nc]!='*':
                        dfs(nr,nc,newroot)
            board[r][c]=saved            
            if newroot.children==0:
                root.next[ord(saved)-0x61]=None
                root.children-=1
                
        m,n=len(board),len(board[0])
        dirs=((-1,0),(1,0),(0,-1),(0,1))
        root=self.TrieNode()
        for word in words:
            root.add(word)
        res=[]
        for r,row in enumerate(board):
            for c,col in enumerate(row):
                dfs(r,c,root)
        return res

class Solution: # Pre-optimized version. Use Trie to speed up pruning but without any other optimizations.
    class TrieNode:
        def __init__(self):
            self.next=[None]*26
            self.isword=False
        def search(self,word)->bool:
            # BUG 3: None.isword error: return self._probe(word).isword 
            r=self._probe(word)
            return r.isword if r else False
        def _probe(self,word,add=False):
            root=self
            p=0
            while p<len(word):
                c=ord(word[p])-0x61
                if not root.next[c]:
                    if add:
                        root.next[c]=Solution.TrieNode()
                    else:
                        return None
                root=root.next[c]
                p+=1
            return root
        def add(self,word)->None:
            newnode=self._probe(word,True)
            newnode.isword=True
        def startswith(self,word)->bool:
            return self._probe(word) 
        def printtrie(self,sofar='',istop=True):
            if istop:
                print("Trie:",end=" ")
            if self.isword:
                            print(sofar,end=" ")
            for i,next in enumerate(self.next):
                # BUG 2. should be self.isword:  if next.isword: print(sofar,end=" ")
                if next:
                    next.printtrie(sofar+chr(0x61+i),False)
            if istop:
                print()
            
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root=self.TrieNode()
        recursions=0
        def dfs(r,c,sofar,current,pruning,root=root):
            nonlocal recursions
            recursions+=1
            newroot=root.startswith(current) if root else None
            if pruning and not newroot:
                return
            visited.add(n*r+c)
            sofar+=current
            if newroot and newroot.isword :
                res.add(sofar)
            for dr,dc in dirs:
                nr,nc=r+dr,c+dc
                if nr>=0 and nr<m and nc>=0 and nc<n and nr*n+nc not in visited :
                        dfs(nr,nc,sofar,board[nr][nc],pruning,newroot)
            visited.remove(n*r+c)
                
        for word in words:
            root.add(word)
        dirs=((-1,0),(1,0),(0,-1),(0,1))
        m,n=len(board),len(board[0])
        # BUG 4. there could be duplicates eg two eat starting from different cell. need to use set to dedup: res=[]
        # BUG 5. later on, I remove leafnodes gradually from Trie every time a matched word is found, this guarantees no duplicate and no need for a set
        res=set()
        visited=set()
        pruning=True
        for r,row in enumerate(board):
            for c,col in enumerate(row):
                # BUG 1. should have use prefix to early terminate an unqualified starting node, because single char might be a valid word like 'a'
                # Later, I refactored and moved the prefix check into dfs() function
                dfs(r,c,'',col,pruning,root)
        print("Total recursions for matrix of size ",'{}*{} = '.format(m,n),recursions)
        return list(res)

class Node:
    def __init__(self, end = 0):
        self.end = end
        self.kids = {}

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        res, root, m, n = set(), Node(0), len(board), len(board[0])
        
        def setTrie(word):
            cur = root
            for w in word:
                if w not in cur.kids:
                    cur.kids[w] = Node()
                cur = cur.kids[w]
            cur.end = 1
            return
        
        def helper(i, j, root, visited, word):
            if root.end == 1: res.add(word)
            visited.add((i, j)) 

            for dx, dy in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                x, y = i + dx, j + dy
                if x < 0 or x >= m or y < 0 or y >= n or (x, y) in visited or board[x][y] not in root.kids: continue
                helper(x, y, root.kids[board[x][y]], visited, word + board[x][y])
            visited.remove((i, j))

            return        
        
        for word in words: setTrie(word)

        for i in range(m):
            for j in range(n):
                if board[i][j] in root.kids: helper(i, j, root.kids[board[i][j]], set(), board[i][j])         
                
        return list(res)

class TrieNode:
    def __init__(self):
        self.child = {}
    
    def insert(self,word):   #insert word in a trie
        current = self.child
        for l in word:
            if l not in current:
                current[l] = {}
            current =current[l]
        current['#'] = True
        current['!'] = word
    
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        
        trie = TrieNode()
        for word in words:
            trie.insert(word)   #insert all words in trie
            
        output = []
        
        def dfs(board,i,j,nextletter):
            if i < 0 or j < 0 or i >= len(board) or j >= len(board[0]) or nextletter.get(board[i][j]) == None:
                return   #make sure within bounds
 
            c = board[i][j]   #saving the characters for other dfs calls
    
            if '#' in nextletter[board[i][j]] and nextletter[board[i][j]]['#'] == True:
                output.append(nextletter[board[i][j]]['!'])   #adding the word in output
                nextletter[board[i][j]]['#'] = False  #making the word unavailable in trie so that it can be used as prefix for other words
            
            nextletter = nextletter[board[i][j]]        
            
            board[i][j] = '*'
            
            dfs(board,i-1,j,nextletter)
            dfs(board,i+1,j,nextletter)
            dfs(board,i,j-1,nextletter)
            dfs(board,i,j+1,nextletter)
            
            board[i][j] = c
            
        for i in range(len(board)):
            for j in range(len(board[0])):
                dfs(board,i,j,trie.child)
        
        return output 

class Solution:
    
    def __init__(self):
        self.trie = {}
        self.direction = [(1,0), (-1, 0), (0, 1), (0, -1)]
        self.valid_words = set()
		
     # inserting a word into trie
    def insert_word(self, word):
        node = self.trie
        
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        
        node["end"] = True

    def dfs(self, grid, r, c, node, word):
        
        if "end" in node:
            self.valid_words.add(word)
        
        char = grid[r][c]
        grid[r][c] = "#"  

        for row, col in self.direction:
            nr, nc = r+row, c +col 
             
            if not (0<=nr< self.rows) or not (0<=nc<self.cols):
                continue 
 
            if grid[nr][nc] not in node:
                continue
 
            self.dfs(grid, nr, nc, node[grid[nr][nc]], word+grid[nr][nc])
        
        grid[r][c] = char 
        return False
    
    def findWords(self, grid, dictonary):
        
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        for word in dictonary:
            self.insert_word(word)
        
        
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r][c] not in self.trie:
                    continue
                self.dfs(grid, r, c, self.trie[grid[r][c]], grid[r][c])

                    
        return self.valid_words

57. Insert Interval
Medium

3055

256

Add to List

Share
Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).

You may assume that the intervals were initially sorted according to their start times.

 

Example 1:

Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
Example 2:

Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
Example 3:

Input: intervals = [], newInterval = [5,7]
Output: [[5,7]]
Example 4:

Input: intervals = [[1,5]], newInterval = [2,3]
Output: [[1,5]]
Example 5:

Input: intervals = [[1,5]], newInterval = [2,7]
Output: [[1,7]]
 

Constraints:

0 <= intervals.length <= 104
intervals[i].length == 2
0 <= intervals[i][0] <= intervals[i][1] <= 105
intervals is sorted by intervals[i][0] in ascending order.
newInterval.length == 2
0 <= newInterval[0] <= newInterval[1] <= 105

class Solution:
    def insert(self, intervals: List[List[int]], insert: List[int]) -> List[List[int]]:
        result = []
        if not len(intervals):
            return [insert]
        chk = False
        if insert[0] <= intervals[0][0]:
            result.append(insert)
            chk = True
        i = 0
        while i < len(intervals):
            n = len(result)
            if n == 0 or (result[n-1][1]<intervals[i][0] and (chk or intervals[i][0]<=insert[0])) :
                result.append(intervals[i])

            elif not chk:
                if result[n-1][1]<insert[0]:
                    result.append(insert)
                else:
                    result[n-1][1] = max(result[n-1][1],insert[1])
                    result[n-1][0] = min(result[n-1][0],insert[0])
                chk = True
                i-=1
            else:
                result[n-1][1] = max(result[n-1][1],intervals[i][1])
                result[n-1][0] = min(result[n-1][0],intervals[i][0])
            i+=1
        if not chk:
            n = len(result)
            if result[n-1][1]<insert[0]:
                result.append(insert)
            else:
                result[n - 1][1] = max(result[n - 1][1], insert[1])
                result[n - 1][0] = min(result[n - 1][0], insert[0])
        return result

from bisect import insort
from typing import List
class Solution:
  def insert(self,intervals:List[List[int]],newInterval:List[int]) -> List[List[int]]:
    intervals.sort(key = lambda x:x[0])
    insort(intervals,newInterval)
    start = 0
    end = len(intervals) - 1
    while (start< end):
      if((intervals[start][1] >= intervals[start+1][0]) and (intervals[start][0] <=intervals[start][1])):
        intervals[start][1] = max(intervals[start][1],intervals[start+1][1])
        del intervals[start + 1]
        end -= 1
      else:
        start += 1
    return intervals

def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    t=0
    for i in range(len(intervals)):
        if intervals[i][0]>newInterval[0]:
            intervals.insert(i,newInterval)
            t=1
            break
    if t==0:
        intervals.append(newInterval)
    fin=[intervals[0]]
    for i in intervals[1:]:
        if i[1]>=fin[-1][1] and i[0]<=fin[-1][1]:
            fin[-1]=[fin[-1][0],i[1]]
        elif i[1]>=fin[-1][1] and i[0]>=fin[-1][1]:
            fin.append(i)
    return fin

class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:     
        res = []
        i = 0
        while i < len(intervals):
            interval = intervals[i]
            if interval[1] < newInterval[0]:
                res.append(interval)
            elif interval[0] > newInterval[1]:
                break
            else:
                newInterval[0] = min(newInterval[0], interval[0])
                newInterval[1] = max(newInterval[1], interval[1])
        
            i += 1
        
        res.append(newInterval)
        for j in range(i, len(intervals)):
            res.append(intervals[j])
                    
        return res  

435. Non-overlapping Intervals
Medium

2185

66

Add to List

Share
Given an array of intervals intervals where intervals[i] = [starti, endi], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

 

Example 1:

Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: [1,3] can be removed and the rest of the intervals are non-overlapping.
Example 2:

Input: intervals = [[1,2],[1,2],[1,2]]
Output: 2
Explanation: You need to remove two [1,2] to make the rest of the intervals non-overlapping.
Example 3:

Input: intervals = [[1,2],[2,3]]
Output: 0
Explanation: You don't need to remove any of the intervals since they're already non-overlapping.
 

Constraints:

1 <= intervals.length <= 2 * 104
intervals[i].length == 2
-2 * 104 <= starti < endi <= 2 * 104

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        cnt = 0
        stack = [[-inf, -inf]]
        for iv in intervals:
            if iv[0] >= stack[-1][1]:
                stack.append(iv)
                continue
            else:
                if iv[1] >= stack[-1][1]:
                    continue
                else:
                    stack.pop()
                    stack.append(iv)
		#print(stack)
        return len(intervals)-len(stack)+1

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: x[1])
        last, ans = 0, 0
        for i in range(1, len(intervals)):
            if intervals[last][1] > intervals[i][0]: ans += 1
            else: last = i
        return ans
    def sol(self, intervals) -> int:
        end = float('-inf')
        erased = 0
        # Sort intervals by end time
        intervals.sort(key=lambda a: a[1])
        # Traverse intervals
        for interval in intervals:
            if interval[0] >= end:
                end = interval[1]
            # Track Overlap (Need not track intervals themselves)
            else:
                erased += 1
        return erased

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        # whenever there is overlap between two elements, one must be removed
        # then the follow up would be: which one should be removed and which one should stay
        # at this step we use greedy method: the one having less influence on the following elements stays which is expressed by last = intervals[i] if intervals[i][1] < last[1] else last
        
        intervals.sort()
        last = intervals[0]
        count = 0
        for i in range(1, len(intervals)):
            if intervals[i][0] < last[1]:
                count += 1
                last = intervals[i] if intervals[i][1] < last[1] else last
            else:
                last = intervals[i]
        return count

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key = lambda x: x[0])
        pend = intervals[0][1]
        remove = 0
        for i in range(1,len(intervals)):
            if pend<=intervals[i][0]:
                pend = intervals[i][1]
            else:
                pend = min(pend,intervals[i][1])
                remove+=1
        return remove

def eraseOverlapIntervals(self, intervals):
        
        if len(intervals) == 1:
            return 0
        
        intervals.sort(key=lambda x: x[1])
        
        #Sort intervals BY THE END TIME, THAN compare OTHER interval, ITS START TIME, WITH THE END TIME of THE
        #previous (LAST) interval
        
        #If we have OVERLAP, that means, IT must be removed
        
        #greedy algorithm, we want to accumulate THE MOST meetings possible in the list 
        
        # [[1,2],[2,3],[3,4],[1,3]]
        # [[1, 2], [2, 3], [1, 3], [3, 4]] (SORTED by END TIME)
        
        
        lastInterval = intervals[0]
        count = 0
        for i in range(1, len(intervals), 1):
            if intervals[i][0] >= lastInterval[1]:
                lastInterval = intervals[i]
            else:
                count += 1
                
        return count

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x:x[0])
        n=len(intervals)
        i=0
        j=1
        count=0
        while j<n:
            if intervals[i][1]<=intervals[j][0]: #Non overlapping
                i=j
                j+=1
            elif intervals[i][1]<=intervals[j][1]: #partial overlapping
                j+=1
                count+=1
            elif intervals[i][1]>=intervals[j][1]: #full overalapping
                i=j
                count+=1
                j+=1
        return count

371. Sum of Two Integers
Medium

1763

2834

Add to List

Share
Given two integers a and b, return the sum of the two integers without using the operators + and -.

 

Example 1:

Input: a = 1, b = 2
Output: 3
Example 2:

Input: a = 2, b = 3
Output: 5
 

Constraints:

-1000 <= a, b <= 1000

class Solution:
    def getSum(self, a: int, b: int) -> int:
        return int(math.log10((10**a) *(10**b)))

class Solution:
    def getSum(self, a: int, b: int) -> int:
        a &= 65535
        b &= 65535
        res = carry = 0
        bit = 1
        for _ in range(16):
            d1 = bool(a&bit)
            d2 = bool(b&bit)
            if (d1^d2^carry):
                res |= bit
            carry = d1&d2 or d1&carry or d2&carry
            bit <<= 1
        if res&32768:
            res |= (~65535)
        return res

class Solution:
    def getSum(self, a: int, b: int) -> int:
        aa = abs(a)
        ab = abs(b)
        
        if aa < ab:
            return self.getSum(b, a)
        
        if a == 0: return b
        if b == 0: return a
        
        la = [None] * aa
        lb = [None] * ab
        
        sign = 1 if a > 0 else -1
        
        if a * b > 0:
            la.extend(lb)
            
        if a * b < 0:
            for e in lb:
                la.remove(e)
        
        return len(la) * sign

class Solution:
    def getSum(self, a: int, b: int) -> int:
        return sum([a, b])

191. Number of 1 Bits
Easy

1595

651

Add to List

Share
Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).

Note:

Note that in some languages, such as Java, there is no unsigned integer type. In this case, the input will be given as a signed integer type. It should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 3, the input represents the signed integer. -3.
 

Example 1:

Input: n = 00000000000000000000000000001011
Output: 3
Explanation: The input binary string 00000000000000000000000000001011 has a total of three '1' bits.
Example 2:

Input: n = 00000000000000000000000010000000
Output: 1
Explanation: The input binary string 00000000000000000000000010000000 has a total of one '1' bit.
Example 3:

Input: n = 11111111111111111111111111111101
Output: 31
Explanation: The input binary string 11111111111111111111111111111101 has a total of thirty one '1' bits.
 

Constraints:

The input must be a binary string of length 32.

class Solution:
    def hammingWeight(self, n: int) -> int:
        num = 0
        while n:
            if n & 1:
                num += 1
            n >>= 1
        return num

class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n != 0:
            if(n%2 == 1):
                count += 1
            n = n//2
        return count  

class Solution:
    def hammingWeight(self, n: int) -> int:
        result = bin(n).count("1")
        return result 

class Solution:
    def hammingWeight(self, n: int) -> int:
        n = bin(n)[2:]
        n = sorted(n)
        x, j = len(n), 0
        for i in range(1,x+1):
            if n[-i] == '1':
                j += 1
            else:
                return j
        return j

338. Counting Bits
Easy

4204

221

Add to List

Share
Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.

 

Example 1:

Input: n = 2
Output: [0,1,1]
Explanation:
0 --> 0
1 --> 1
2 --> 10
Example 2:

Input: n = 5
Output: [0,1,1,2,1,2]
Explanation:
0 --> 0
1 --> 1
2 --> 10
3 --> 11
4 --> 100
5 --> 101
 

Constraints:

0 <= n <= 105
 

Follow up:

It is very easy to come up with a solution with a runtime of O(n log n). Can you do it in linear time O(n) and possibly in a single pass?
Can you do it without using any built-in function (i.e., like __builtin_popcount in C++)?

class Solution:
    def countBits(self, num: int) -> List[int]:
        c=[]
        for i in range(0,num+1):
            t=bin(i).replace("0b","")
            x=t.count('1')
            c.append(x)
        return c

class Solution:
    def countBits(self, n: int) -> List[int]:
        result = []

        for i in range(n+1):
            result.append(str(bin(i).replace("0b","")).count("1"))

        return result

class Solution:
    def countBits(self, n: int) -> List[int]:
        dp = [0,1,1,2]
        while len(dp)<n+1:
            dp.extend([num+1 for num in dp])
        return dp[:n+1]

class Solution:
    def countBits(self, n: int) -> List[int]:
        dp=[0]*(n+1)
        for i in range(1,n+1):
            dp[i]=dp[i>>1]+i%2
        return dp

class Solution:
    def countBits(self, n: int) -> List[int]:
        ans=[0]*(n+1)
        for i in range(1,n+1):
            ans[i]=(i&1)+ans[i>>1]
        return ans

Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.

Follow up: Could you implement a solution using only O(1) extra space complexity and O(n) runtime complexity?

 

Example 1:

Input: nums = [3,0,1]
Output: 2
Explanation: n = 3 since there are 3 numbers, so all numbers are in the range [0,3]. 2 is the missing number in the range since it does not appear in nums.
Example 2:

Input: nums = [0,1]
Output: 2
Explanation: n = 2 since there are 2 numbers, so all numbers are in the range [0,2]. 2 is the missing number in the range since it does not appear in nums.
Example 3:

Input: nums = [9,6,4,2,3,5,7,0,1]
Output: 8
Explanation: n = 9 since there are 9 numbers, so all numbers are in the range [0,9]. 8 is the missing number in the range since it does not appear in nums.
Example 4:

Input: nums = [0]
Output: 1
Explanation: n = 1 since there is 1 number, so all numbers are in the range [0,1]. 1 is the missing number in the range since it does not appear in nums.
 

Constraints:

n == nums.length
1 <= n <= 104
0 <= nums[i] <= n
All the numbers of nums are unique

class Solution:
    	def missingNumber(self, nums):
		"""
		Runtime: 124 ms, faster than 90.27% of Python3 online submissions for Missing Number.
		Memory Usage: 15.5 MB, less than 48.94% of Python3 online submissions for Missing Number.
		"""
		return int((len(nums)*(len(nums)+1))/2)-sum(nums)

class Solution(object):
    def missingNumber(self, nums):
        
        noZero = True
        mx = -1
        sm = 0
        
        for num in nums:
            if num == 0:
                noZero = False
            
            mx = max(mx, num)
            sm += num
            
            
        if noZero:
            return 0
        
        t = mx * (mx + 1) // 2
        
        return mx + 1 if t == sm else t - sm

class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
        return (n*(n+1))//2 - sum(nums)

    class Solution:
        def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
		#since the list only has n-1 space, we create a variable just for the last index
        final_elem = [10000]
		#we store the position of zero in the array to check later for 0 case
        pos_zero = 99999
		#iterate through the array, and mulitply each index with -1. Take speacial care for the nth element.
        for i in range(n):
            if(abs(nums[i])==n):
                final_elem[0]*=-1
            elif(abs(nums[i])==0):
                pos_zero=i
                nums[abs(nums[i])]*=-1
            else:
                nums[abs(nums[i])]*=-1
		# if throughout the iterations, our zero position variable doesn't changem then it means we never encountered a zero, and so our answer is zero
        if(pos_zero==99999):
            return 0
		# we can iterate through the array to find the element that is still greater than zero
        for i in range(n):
            if(nums[i]>0):
                return i
		# just tot check the nth element
        if(final_elem[0]>0):
            return n
		#if we dont find answer till now, then it means that our missing element had a zero in the array at its index, so we just return the postioin of zero in the array
        return pos_zero

class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        nums_b = [False for i in range(0, len(nums) + 1)]
        for i in range(len(nums)):
            nums_b[nums[i]] = True
        for i in range(len(nums) + 1):
            if nums_b[i] == False:
                return i

190. Reverse Bits
Easy

1867

592

Add to List

Share
Reverse bits of a given 32 bits unsigned integer.

Note:

Note that in some languages such as Java, there is no unsigned integer type. In this case, both input and output will be given as a signed integer type. They should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 2 above, the input represents the signed integer -3 and the output represents the signed integer -1073741825.
Follow up:

If this function is called many times, how would you optimize it?

 

Example 1:

Input: n = 00000010100101000001111010011100
Output:    964176192 (00111001011110000010100101000000)
Explanation: The input binary string 00000010100101000001111010011100 represents the unsigned integer 43261596, so return 964176192 which its binary representation is 00111001011110000010100101000000.
Example 2:

Input: n = 11111111111111111111111111111101
Output:   3221225471 (10111111111111111111111111111111)
Explanation: The input binary string 11111111111111111111111111111101 represents the unsigned integer 4294967293, so return 3221225471 which its binary representation is 10111111111111111111111111111111.
 

Constraints:

The input must be a binary string of length 32

The basic idea here is that I wanted to simulate a 2 pointer approach where we have a left pointer and a right pointer. The right pointer scans the bits for 1's and the left pointer sets the bits (on an initially empty array of bits aka zero).

I check right most bits using a check bits formula (that I picked up from the CTCI book) where i is in range 0-32:

n & (1 << i) != 0
For the left "pointer" I use the hexdecimal 0x80000000 to represent the binary 10000000000000000000000000000000
Or you could also use 1 << 31 - i

As the right "pointer" moves from right to left (by left shifting <<) to check for significant bits, the left "pointer" is moving left to right (by right shifting) setting bits to 1 by using the OR | operand.

def reverseBits(self, n: int) -> int:
	left = 0x80000000
	res = 0

	for i in range(32):
		if n & (1 << i) != 0:
			# can also use res |= 1 << 31 - i 
			res |= (left >> i) 

	return res

class Solution:
    def reverseBits(self, n: int) -> int:
        def bf_v1(n):
            x=0
            for i in range(32):
                x=x<<1|n&1
                n>>=1
            return x
        def bf_with_early_termination_v2(n):
            x=0
            pow=31
            while n:
                x|=(n&1)<<pow
                #x+=(n&1)<<pow #equiv
                n>>=1
                pow-=1
            return x
        def cache_v3():
            cache=[0]*4
            def reverse_byte(b):
                x=0
                pow=7
                while b:
                    x+=(b&1)<<pow
                    b>>=1
                    pow-=1
                return x 
            x=0
            for i in range(4):
                cache[i]=reverse_byte(n>>8*i&0xff)
                x|=cache[i]<<8*(3-i)
            return x    
        def merge_swap_v4(n,i,j):
            if i==j:
                x= (n>>i&1)<<i
                return x
            halfwidth=(j-i)//2
            lo=merge_swap(n,i,i+halfwidth)
            hi=merge_swap(n,i+halfwidth+1,j)
            return lo<<halfwidth+1|hi>>halfwidth+1
        def merge_swap_hard_coded_v5(n):
            n=(0xaaaaaaaa&n)>>1|(n&0x55555555)<<1
            n=(0xcccccccc&n)>>2|(n&0x33333333)<<2
            n=(0xf0f0f0f0&n)>>4|(n&0x0f0f0f0f)<<4
            n=(0xff00ff00&n)>>8|(n&0x00ff00ff)<<8
            n=(0xffff0000&n)>>16|(n&0x0000ffff)<<16 # equiv: n= n>>16|n<<16
            return n
        
        #return bf_v0(n)
        #return bf_with_early_termination_v1(n)
        #return cache_v2()
        #return merge_swap(n,0,31)
        return merge_swap_hard_coded_v5(n)

def reverseBits(n):
    orgbits = list(str(n))
    for j in range(0,16):
        orgbits[j], orgbits[31-j] = orgbits[31-j], orgbits[j]
    return int("".join(orgbits),2)

class Solution:
    def reverseBits(self, n: int) -> int:
        l = 32
        rb = 0
        while n > 0:
            m = n % 2
            n = n // 2
            if m == 1:
                rb = rb + 1 * 2 ** (l - 1)
            l -= 1
        return rb

def reverseBits_1(self, n:int) -> int:
    mask = 2**31
    ans = 0
    for i in range(32):
        if n & mask : ans += 2**i
        mask >>= 1
    return ans

1. Two Sum
Easy

21871

743

Add to List

Share
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

 

Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Output: Because nums[0] + nums[1] == 9, we return [0, 1].
Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
 

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
 

Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity?

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        mp = {}
        
        for i in range(len(nums)):
            n = target - nums[i]
            if n in mp:
                return(mp[n], i)
            mp[nums[i]] = i  

class Solution(object):
    def twoSum(self, nums, target):
        self.nums = nums
        self.target = target
        dict = {}
        length = len(nums)
        for i in range(0, length):
            reste = target - nums[i] 
            if dict.__contains__(reste) and dict.get(reste) != i:
                return [dict.get(reste), i]
            dict[nums[i]] = i

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i, number in enumerate(nums[:-1]):
            complementary = target - number
            if complementary in (nums[i+1:]):
                return nums.index(number), nums.index(complementary)

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        numIndex = dict()
        for i in range(len(nums)):
            n = nums[i]
            n2 = target - n
            if n2 in numIndex:
                return [i,numIndex[n2]]
            numIndex[nums[i]] = i
        return None

121. Best Time to Buy and Sell Stock
Easy

9143

379

Add to List

Share
You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

 

Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
Example 2:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.
 

Constraints:

1 <= prices.length <= 105
0 <= prices[i] <= 104

def maxProfit(self, prices: List[int]) -> int:
    low= prices[0]
    max_profit= 0

    if prices == sorted(prices)[::-1]:
        return 0

    for i in prices[1::]:
        pr= i - low
        if pr > max_profit:
            max_profit= pr

        elif low > i: 
            low= i

    return max_profit

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        
        max_profit = 0
        best_price = prices[0] #Assume best_price to buy as first day price initially
        
        for i in the range(i,n):
            diff = prices[i] - best_price #price diff b/w today's & assumed best price
            
            if diff > 0 and diff > max_profit:  #Update max_profit
                max_profit = diff

            if prices[i] < best_price:  #Update best_price to buy based on today's price
                best_price = prices[i]
                
            i=i+1
            
        return max_profit

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
      maxValueFromRight = prices[-1]
      maxProfit = 0
      
      for i in prices[::-1]:
        if i<maxValueFromRight:
          maxProfit = max(maxProfit, maxValueFromRight - i)
        else:
          maxValueFromRight = i
      
      return maxProfit

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price = None
        max_price = None
        max_profit = 0
        for price in prices:
            if min_price is None or price < min_price:
                min_price = max_price = price
            elif price > max_price:
                max_price = price
                max_profit = max(max_profit, max_price - min_price)

        return max_profit


class Solution:
    def maxProfit(self, prices: List[int]):
        minPrice = prices[0]
        maxPrice = prices[-1]
        currProfit = maxPrice - minPrice
        maxProfit = max(0, currProfit)
        
        for price in prices[1:-1]:
            if price < minPrice:
                minPrice = price
                maxPrice = prices[-1] #Invalidate any previously updated maxPrice.
                currProfit = maxPrice - minPrice
                maxProfit = max(currProfit, maxProfit)
            
            if price > maxPrice:
                maxPrice = price
                currProfit = maxPrice - minPrice
                maxProfit = max(currProfit, maxProfit)
        
        return maxProfit

217. Contains Duplicate
Easy

1868

854

Add to List

Share
Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.

 

Example 1:

Input: nums = [1,2,3,1]
Output: true
Example 2:

Input: nums = [1,2,3,4]
Output: false
Example 3:

Input: nums = [1,1,1,3,3,4,3,2,4,2]
Output: true
 

Constraints:

1 <= nums.length <= 105
-109 <= nums[i] <= 109

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        dict = {}
        for x in range(0, len(nums)):
            if nums[x] not in dict:
                dict[nums[x]] = True
            else:
                return False
        return True

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return  not len(set(nums)) == len(nums)

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        
        s = set(nums)
        if(len(nums) == len(s)):
            return False
        else:
            return True

class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        d={}
        for i in nums:
            if i in d:
                return True
            d[i]=True
        return False

238. Product of Array Except Self
Medium

7865

577

Add to List

Share
Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.

 

Example 1:

Input: nums = [1,2,3,4]
Output: [24,12,8,6]
Example 2:

Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]
 

Constraints:

2 <= nums.length <= 105
-30 <= nums[i] <= 30
The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
 

Follow up: Can you solve the problem in O(1) extra space complexity? (The output array does not count as extra space for space complexity analysis.)

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        result = [1 for _ in nums] # result array not included in space complexity
        
        # The problem can essentially be thought of as computing the products
        # of the elements to the left of self and products of elements to the
        # right of self
        
        # With nums = [1, 2, 3, 4], if i = 1 , then left product is 1 and right is 12
        # Combine them to get the correct product for i = 1
        
        # Start by computing the left hand product for each i
        for i in range(1, len(nums)):
            # result[i-1] does not include nums[i-1] in its product,
            # so it needs to be included
            result[i] = result[i-1] * nums[i-1] 
        
        
        # Now we work the other way and read from right to left
        Rproduct = 1
        for i in range(len(nums) - 1, -1, -1):
            result[i] *= Rproduct
            Rproduct *= nums[i]
            
        return result

class Solution(object):
    def productExceptSelf(self, nums):
        N = len(nums)
        
        left = [1]*N
        for i in xrange(1, len(nums)):
            left[i] = left[i-1] * nums[i-1]
        
        right = [1]*N
        for i in xrange(N-2, -1, -1):
            right[i] = right[i+1] * nums[i+1]
        
        ans = []
        for i in xrange(N):
            ans.append(left[i]*right[i])
        return ans

class Solution(object):
    def productExceptSelf(self, nums):
        N = len(nums)
        
        opt = [1]*N
        for i in xrange(1, len(nums)):
            opt[i] = opt[i-1] * nums[i-1]
        
        t = 1
        for i in xrange(N-2, -1, -1):
            t *= nums[i+1]
            opt[i] *= t
        
        return opt

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # if there is are more than one zero, everything return 0 array
        # if there is one zero, everything will be zero except the zero
        
        ans = len(nums) * [0]
        total = 1
        
        # traverse the array look for zeros
        zero_count = 0
        zero_index = -1
        for i in range(len(nums)):
            if nums[i] == 0:
                zero_index = i
                zero_count += 1
            else:
                total *= nums[i] 
                
        print(zero_index)
        
        if zero_count > 1:
            return ans
        
        if zero_count == 1:
            ans[zero_index] = total
            return ans
        
        
        
        for i in range(len(nums)):
            ans[i] = total//nums[i]
            
        return ans

53. Maximum Subarray
Easy

12729

609

Add to List

Share
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

 

Example 1:

Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
Example 2:

Input: nums = [1]
Output: 1
Example 3:

Input: nums = [5,4,-1,7,8]
Output: 23
 

Constraints:

1 <= nums.length <= 3 * 104
-105 <= nums[i] <= 105
 

Follow up: If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        maxSum = nums[0]
        maxEnding = nums[0]
        
        for i in range(1, len(nums)):
            maxEnding = max(maxEnding + nums[i], nums[i])
            maxSum = max(maxSum, maxEnding)     
                            
        return maxSum

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        maxs = nums[0]
        curr = nums[0]
        for i in range(1,len(nums)):
            curr = max(nums[i],curr+nums[i])
            maxs = max(maxs,curr)

        return maxs

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        maxSub=nums[0]
        cur=0
        
        for i in nums:
            if cur<0:
                cur=0
            cur+=i
            maxSub=max(maxSub,cur)
        return maxSub

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        local_max = 0
        global_max = -10 ** 5
        for num in nums:
            local_max = max(num, num + local_max)
            if local_max > global_max:
                global_max = local_max

        return global_max

152. Maximum Product Subarray
Medium

7312

238

Add to List

Share
Given an integer array nums, find a contiguous non-empty subarray within the array that has the largest product, and return the product.

It is guaranteed that the answer will fit in a 32-bit integer.

A subarray is a contiguous subsequence of the array.

 

Example 1:

Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
Example 2:

Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
 

Constraints:

1 <= nums.length <= 2 * 104
-10 <= nums[i] <= 10
The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        maxprod = maxcurr = maxneg = 0
        if nums[0] > 0:
            maxprod = maxcurr = nums[0]
        else:
            maxprod = maxneg = nums[0]
        for n in nums[1:]:
            if n == 0:
                maxcurr = maxneg = 0
            elif n > 0:
                maxcurr = maxcurr * n if maxcurr > 0 else n
                maxneg = maxneg * n if maxneg < 0 else n
            else:
                temp = maxcurr
                maxcurr = maxneg * n if maxneg < 0 else 0
                maxneg = temp * n if temp > 0 else n
            maxprod = max(maxprod, maxcurr)
        return maxprod

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        prevMin = prevMax = 1
        maxProduct = nums[0]
        for num in nums:
            prevMin, prevMax = min(num,num*prevMin,num*prevMax), max(num,num*prevMin,num*prevMax)
            maxProduct = max(maxProduct,prevMax)
        return maxProduct

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        best = nums[0]
        
        longestSubPos = [nums[0]]*n 
        longestSubNeg =  [nums[0]]*n 
        
        
        for i in range(1,n):
            
            longestSubPos[i] = max(nums[i], nums[i]*longestSubPos[i-1], nums[i] * longestSubNeg[i-1])
            longestSubNeg[i] = min(nums[i], nums[i]*longestSubPos[i-1], longestSubNeg[i-1]*nums[i])
            best = max(best, longestSubPos[i])
        
        return best

    def maxProduct(self, nums: List[int]) -> int:
        ans = -math.inf
        n = len(nums)
        mi,ma = [1]*(n+1),[1]*(n+1)
        for i in range(1,n+1):
            v = nums[i-1]
            mi[i] = min(v,mi[i-1]*v,ma[i-1]*v)
            ma[i] = max(v,ma[i-1]*v,mi[i-1]*v)
            ans = max(ans,mi[i],ma[i])
        return ans

def maxProduct(self, nums: List[int]) -> int:        
	ans = -math.inf
	n = len(nums)
	mi,ma = 1,1
	for i in range(1,n+1):
		v = nums[i-1]
		new_mi = min(v,mi*v,ma*v)
		ma = max(v,ma*v,mi*v)
		mi = new_mi
		ans = max(ans,mi,ma)
	return ans

153. Find Minimum in Rotated Sorted Array
Medium

3769

312

Add to List

Share
Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

[4,5,6,7,0,1,2] if it was rotated 4 times.
[0,1,2,4,5,6,7] if it was rotated 7 times.
Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

Given the sorted rotated array nums of unique elements, return the minimum element of this array.

You must write an algorithm that runs in O(log n) time.

 

Example 1:

Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.
Example 2:

Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.
Example 3:

Input: nums = [11,13,15,17]
Output: 11
Explanation: The original array was [11,13,15,17] and it was rotated 4 times. 

class Solution:
    def findMin(self, nums: List[int]) -> int:
        start = 0
        end = len(nums) - 1
        while start <= end:
            mid = start + (end - start)//2        
            if nums[mid] > nums[end]:
                start = mid + 1
            elif mid == 0 or nums[mid - 1] > nums[mid]:
                return nums[mid]
            else:
                end = mid - 1

class Solution:
    def findMin(self, nums: List[int]) -> int:
        low=0
        high=len(nums)-1
        while low<high:
            mid=(low+high)>>1
            if nums[mid]>nums[high]:
                low=mid+1
            else:
                high=mid
        return nums[high]

class Solution(object):
    def findMin(self, nums):
        return min(nums)

class Solution(object):
    def findMin(self, nums):
        for i in range(1,len(nums)):
            if nums[i] < nums[i-1]:
                return nums[i]
        return nums[0]

33. Search in Rotated Sorted Array
Medium

8411

701

Add to List

Share
There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is rotated at an unknown pivot index k (0 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

Given the array nums after the rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

 

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
Example 2:

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
Example 3:

Input: nums = [1], target = 0
Output: -1

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        def util(nums, target, start, end):
            if start > end:
                return -1
            mid = (start+end)//2
            if nums[mid] == target:
                return mid
            if nums[mid] >= nums[start]:
                if target>=nums[start] and target <= nums[mid]:
                    return util(nums, target, start, mid)
                else:
                    return util(nums, target, mid+1, end)
            else:
                if target <= nums[end] and target >= nums[mid]:
                    return util(nums, target, mid+1, end)
                return util(nums, target, start, mid)
            return -1
        start = 0
        end = len(nums) - 1
        idx = util(nums, target, start,end)
        return idx

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        start = 0
        end = len(nums) - 1
        
        # Revised Binary Search
        while start <= end:
            mid = start + (end - start) // 2
            
            if nums[mid] == target:
                return mid
            
            # Sorted Array is in left half
            elif nums[mid] >= nums[start]:
                # Check if target is within the range of sorted array
                if target >= nums[start] and target < nums[mid]:
                    end = mid - 1
                else:
                    start = mid + 1
            
            # Sorted Array is in right half
            else:
                # Check if target is within the range of sorted array
                if target <= nums[end] and target > nums[mid]:
                    start = mid + 1
                else:
                    end = mid - 1
                    
        return -1

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)-1
        
        while l <= r:
            m = l + (r-l)//2
            if nums[m] == target:
                return m
				
			# 0 index falls in right region and target falls in left region
            if nums[l] <= target and target < nums[m]:
                r = m
				
			# 0 index falls in left region and target falls in left region
            elif nums[l] > nums[m] and (target < nums[m] or nums[r] < target):
                r = m
				
			# other cases
            else:
                l = m + 1
				
        return -1

def search(self, nums: List[int], target: int) -> int:
    l=0
    r=len(nums)-1
    
    while l<=r:
        mid=(l+r)//2
        if nums[mid]==target:
            return mid
        if nums[mid]>=nums[0]:#left sorted region
            if nums[mid]<target or target<nums[0]: #if we either need to increase num or decrease it when target is lower than left boundary of left sorted region that is nums[0]
                l=mid+1
            else:
                r=mid-1 #decrease num when target is greater than left boundary of left sorted region that is nums[0]
        else:#right sorted region
            if nums[mid]>target or target>nums[-1]: #if we either need to decrease num or increase it when target is larger than right boundary of right sorted region that is nums[-1]
                r=mid-1
            else:
                l=mid+1 #increase num when target is lesser than right boundary of right sorted region that is nums[-1]
    return -1 


15. 3Sum
Medium

11551

1147

Add to List

Share
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

 

Example 1:

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Example 2:

Input: nums = []
Output: []
Example 3:

Input: nums = [0]
Output: []

"""
Hash Table
Time: O(N^2)
Space: O(N)

memo := {num:[index1, index2,...]}
Iterate through every i and j, see if the required number exist in memo.
"""
import collections
class Solution(object):
    def threeSum(self, nums):
        memo = collections.defaultdict(list)
        N = len(nums)
        ans = set()
        
        for k in xrange(N):
            memo[nums[k]].append(k)
        
        for i in xrange(N):
            for j in xrange(i+1, N):
                t = (nums[i]+nums[j])*-1
                
                for k in memo[t]:
                    if k!=i and k!=j:
                        ans.add(tuple(sorted([nums[i], nums[j], nums[k]])))
                        break
        return ans

"""
Two Pointers
Time: O(N^2)
Space: O(1)

Sort the nums.
Iterate through i.
j = i+1 and k = N-1, see if the sum s equals to 0.
if s>0, means that we need to reduce the s and it can only be done by decreasing k.
if s<0, means that we need to increase the s and it can only be done by increasing j.
"""
class Solution(object):
    def threeSum(self, nums):
        nums.sort()
        ans = set()
        N = len(nums)
        
        for i in xrange(N):
            j = i+1
            k = N-1
            
            while j<k:
                s = nums[i]+nums[j]+nums[k]
                if s>0:
                    k -= 1
                elif s<0:
                    j += 1
                else:
                    ans.add(tuple(sorted([nums[i], nums[j], nums[k]])))
                    k -= 1
                    j += 1
                    
        return ans

"""
Two Pointers
Time: O(N^2)
Space: O(1)

Same as above. Move pointers to avoid repeation. Faster.
"""
class Solution(object):
    def threeSum(self, nums):
        nums.sort()
        ans = []
        N = len(nums)
        
        for i in xrange(N):
            j = i+1
            k = N-1
            
            if i>0 and nums[i]==nums[i-1]: continue #[1]
            
            while j<k:
                s = nums[i]+nums[j]+nums[k]
                
                if s>0:
                    k -= 1
                elif s<0:
                    j += 1
                else:
                    ans.append([nums[i], nums[j], nums[k]])
                    
                    while j<k and nums[k]==nums[k-1]: k -= 1 #[2]
                    while j<k and nums[j]==nums[j+1]: j += 1 #[3]
                    k -= 1
                    j += 1
                    
        return ans

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums)<3:
            return []
        dic = collections.defaultdict(int)
        for num in nums:
            dic[num] += 1
            
        res = set()
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                v1 = nums[i]
                v2 = nums[j]
                dic[v1]-=1
                dic[v2]-=1
            
                v3 = 0-(v1+v2)
                if dic.get(v3, 0) > 0:
                    tpl = tuple(sorted([v1, v2, v3]))
                    if tpl not in res:
                        res.add(tpl)

                dic[v1]+=1
                dic[v2]+=1
        
        return [list(pair) for pair in res]

11. Container With Most Water
Medium

10390

742

Add to List

Share
Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of the line i is at (i, ai) and (i, 0). Find two lines, which, together with the x-axis forms a container, such that the container contains the most water.

Notice that you may not slant the container.

 

Example 1:


Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
Example 2:

Input: height = [1,1]
Output: 1
Example 3:

Input: height = [4,3,2,1,4]
Output: 16
Example 4:

Input: height = [1,2,1]
Output: 2

class Solution:
    def maxArea(self, height: List[int]) -> int:        
        l=0
        r=len(height)-1
        mx_ = 0
        
        while l != r:
            mx_ = max(
                mx_,
                min(height[l], height[r])*(r-l)
            )
            if height[l] <= height[r]:
                l += 1
            else:
                r -= 1
        return mx_

class Solution:
    def maxArea(self, height: List[int]) -> int:
        
        max_volume = 0
        left, right = 0, len(height) - 1
        
        while left < right:
            left_height, right_height = height[left], height[right]
            curr_volume = (right - left) * min(left_height, right_height)
            max_volume = max(max_volume, curr_volume)
            
            if left_height <= right_height:
                left += 1
            if right_height <= left_height:
                right -= 1
        
        return max_volume

def maxArea(self, height: List[int]) -> int:
    	l, r, maxArea = 0, len(height)-1, 0
	hmax = max(height)
	while (l<r and (r-l)*hmax > maxArea):
		maxArea = max(maxArea, min(height[l], height[r])*(r-l))
		if (height[l] < height[r]):
			l+=1
		elif (height[l] > height[r]):
			r-=1
		else:
			l+=1
			r-=1
	return maxArea

class Solution:
    def maxArea(self, height: List[int]) -> int:

    max = 0
    index_1 = 0 
    index_2 = len(height)-1
    
    while index_1 < index_2:
        
        min_index_12 = min(height[index_1],height[index_2])
        area = min_index_12*(index_2-index_1)
        
        if max < area:
            max = area

        if height[index_1] < height[index_2]:
            i = index_1+1
            while height[i]< height[index_1]:
                i+=1
            index_1 = i            
            
        elif height[index_1] > height[index_2]:
            j = index_2-1
            while height[j]< height[index_2]:
                j-=1
            index_2 = j
            print(index_2)
            
        elif height[index_1]==height[index_2]:
            index_1 +=1
            index_2 -=1
            
    return max

70. Climbing Stairs
Easy

7175

223

Add to List

Share
You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

 

Example 1:

Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
Example 2:

Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step

from functools import lru_cache
class Solution: 

def climbStairs(self, n: int) -> int:
    x = self.fib(n + 1)
    return x
	
@lru_cache(None)
def fib(self, n):
    if (n <= 1):
        return n
    else:
        return self.fib(n-1) + self.fib(n-2)

class Solution:
    def climbStairs(self, n: int) -> int:
        if n < 3:
            return n
			
        data = {1: 1, 2: 2}
        for i in range(3, n + 1):
            data[i] = data[i - 1] + data[i - 2]
            
        return data[n]

class Solution:
    def climbStairs(self, n: int) -> int:
        prev, result = 0, 1
        for i in range(n):
            prev, result = result, result + prev
        return result

class Solution:
    def climbStairs(self, n: int) -> int:
        
        # 0: Make a function for combinations
        
        def factorial(x):
            y = 1
            if x == 0:
                return y
            else:
                for i in range(1, x):
                    y *= i + 1
            return y
        
        
        def combinations(n, k):
            return factorial(n) / (factorial(k) * factorial(n - k))     
        
        # 1: Count the number of 2 in the number of stairs
        
        nb_two = n // 2 
        nb_one = n % 2
        
        # 2: Sum the combinations
        
        sum_out = 0
        
        for i in range(nb_two, 0, -1):
            # Debug
            # print(nb_two, nb_one, combinations(nb_two + nb_one, nb_two))
            
            # Get the sum
            sum_out += combinations(nb_two + nb_one, nb_two) 
            
            # Update
            nb_two -= 1
            nb_one += 2
            
            
        sum_out += 1 # Add the last config when all is 1
            
        return int(sum_out)

class Solution:
    def climbStairs(self, n: int) -> int:
        if (n == 1):
            return 1
        if (n == 2):
            return 2
        res = [1,2]
        for i in range(2,n):
            res.append(res[i-1] + res[i-2])
        return res[-1]

class Solution:
    def __init__(self):
        self.d={1:1,2:2}
    def climbStairs(self, n: int) -> int:
        if n not in self.d:
            self.d[n]= self.climbStairs(n-1) + self.climbStairs(n-2)
        return self.d[n]

322. Coin Change
Medium

7514

207

Add to List

Share
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

 

Example 1:

Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
Example 2:

Input: coins = [2], amount = 3
Output: -1
Example 3:

Input: coins = [1], amount = 0
Output: 0
Example 4:

Input: coins = [1], amount = 1
Output: 1
Example 5:

Input: coins = [1], amount = 2
Output: 2

class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        ways = [float('inf') for _ in range( amount + 1)]
        ways[0] = 0 # base case
        
        for coin in coins:
            for amt in range(1, amount + 1):
                if coin <= amt:
                    ways[amt] = min(ways[amt], 1 + ways[amt-coin])
        return ways[amount] if ways[amount] != float('inf') else -1

def coinChange(coins, amount):
    
	if amount == 0:
		return 0
	from collections import deque
	q = deque()
	q.append(amount) # amt, depth
	visited = set()
	depth = 0
	while q:
		for i in range(len(q)):
			amt = q.popleft()

			if amt < 0: # skip prune branches that yeild -ve nodes
				continue

			elif amt == 0:
				return depth

			if amt not in visited: # skip nodes seen before - see explanantion above
				visited.add(amt)

				# move down a level
				for c in coins: 
					q.append(amt-c)

		depth += 1

	return -1

def coinChange(coins, amount):
    
	if amount == 0:
		return 0
	from collections import deque
	q = deque()
	q.append(amount) # amt, depth
	visited = set()
	depth = 0
	while q:
		depth += 1
		for i in range(len(q)):
			amt = q.popleft()

			if amt not in visited:
				visited.add(amt)

				# move down a level
				for c in coins:
					if amt-c < 0:
						continue
					elif amt-c in visited:
						continue
					elif amt-c == 0:
						return depth
					else:
						q.append(amt-c)
	return -1

def coinChange(coins, amount):
    	if amount == 0:
            return 0
        visited = set()
        minPath = float('inf')
        stack = [(amount, 0)] # amount, lenght
        while stack:
            amt, path = stack.pop()

            # skip -ve
            if amt < 0:
                continue
            elif amt == 0:
                minPath = min(minPath, path)
            
            if amt not in visited:
                visited.add(amt)
            
                for c in coins:
                    stack.append((amt-c, path+1))
                    
        if minPath > 0:
            return minPath
        return -1

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        if len(coins) == 1:
            if amount % coins[0] == 0:
                return amount // coins[0]
            else:
                return -1
        else:
            arr = []
            for i in range(len(coins) + 1):
                if i == 0:
                    arr.append([float('inf') - 1] * (amount + 1))
                else:
                    arr.append([0] + [None] * (amount))
            j = 1
            while j < (amount + 1):
                if j % coins[0] == 0:
                    arr[1][j] = (j // coins[0])
                else:
                    arr[1][j] = float('inf') - 1
                j += 1
            for i in range(2,len(arr)):
                for j in range(1,len(arr[0])):
                    if coins[i - 1] <= j:
                        arr[i][j] = min(arr[i - 1][j] ,1 + arr[i][j - coins[i - 1]])
                    else:
                        arr[i][j] = arr[i - 1][j]
            return arr[-1][-1] if arr[-1][-1] < float('inf') else -1

300. Longest Increasing Subsequence
Medium

8091

174

Add to List

Share
Given an integer array nums, return the length of the longest strictly increasing subsequence.

A subsequence is a sequence that can be derived from an array by deleting some or no elements without changing the order of the remaining elements. For example, [3,6,2,7] is a subsequence of the array [0,3,1,6,2,2,7].

 

Example 1:

Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.
Example 2:

Input: nums = [0,1,0,3,2,3]
Output: 4
Example 3:

Input: nums = [7,7,7,7,7,7,7]
Output: 1

class Solution:  # 2516 ms, faster than 64.96%
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n
        for i in range(n):
            for j in range(i):
                if nums[i] > nums[j] and dp[i] < dp[j] + 1:
                    dp[i] = dp[j] + 1
        return max(dp)

class Solution:  # 68 ms, faster than 93.92%
    def lengthOfLIS(self, nums: List[int]) -> int:
        sub = []
        for x in nums:
            if len(sub) == 0 or sub[-1] < x:
                sub.append(x)
            else:
                idx = bisect_left(sub, x)  # Find the index of the smallest number >= x
                sub[idx] = x  # Replace that number with x
        return len(sub)

class Solution:
    def pathOfLIS(self, nums: List[int]):
        sub = []
        subIndex = []  # Store index instead of value for tracing path purpose
        path = [-1] * len(nums)  # path[i] point to the index of previous number in LIS
        for i, x in enumerate(nums):
            if len(sub) == 0 or sub[-1] < x:
                path[i] = -1 if len(subIndex) == 0 else subIndex[-1]
                sub.append(x)
                subIndex.append(i)
            else:
                idx = bisect_left(sub, x)  # Find the index of the smallest number >= x, replace that number with x
                path[i] = -1 if idx == 0 else subIndex[idx - 1]
                sub[idx] = x
                subIndex[idx] = i

        ans = []
        t = subIndex[-1]
        while t >= 0:
            ans.append(nums[t])
            t = path[t]
        return ans[::-1]

class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        LIS_dic = {}
        N = len(nums)
        LIS_dic[N-1] = 1
        for i in range(2, N+1):
            cur_index = N - i
            tmp_max = 1
            for j in range(cur_index+1, N):
                #accounting for possible gaps
                if nums[cur_index] < nums[j]:
                    if ((1+LIS_dic[j]) > tmp_max):
                        tmp_max = 1+LIS_dic[j]
            LIS_dic[cur_index] = tmp_max
        max_val = float("-inf")
        for key in LIS_dic:
            if LIS_dic[key] > max_val:
                max_val = LIS_dic[key]
        return max_val 

class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 1:
            return 1
        arr = sorted(list(set(nums)))
        n = len(arr) + 1
        k = len(nums) + 1
        lst = [[0] * k for i in range(n)]
        for i in range(1,len(lst)):
            for j in range(1,len(lst[0])):
                if arr[i - 1] == nums[j - 1]:
                    lst[i][j] = 1 + lst[i - 1][j - 1]
                else:
                    lst[i][j] = max(lst[i - 1][j],lst[i][j - 1])
        return lst[-1][-1]

139. Word Break
Medium

7243

340

Add to List

Share
Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

 

Example 1:

Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
Example 2:

Input: s = "applepenapple", wordDict = ["apple","pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.
Example 3:

Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: false
 

Constraints:

1 <= s.length <= 300
1 <= wordDict.length <= 1000
1 <= wordDict[i].length <= 20
s and wordDict[i] consist of only lowercase English letters.
All the strings of wordDict are unique.

def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    mem = {}
    return Solution.wordBreakHelper(s, wordDict, mem)

def wordBreakHelper(s, wordDict, mem):
    if not s:
        return True
    
    if s in mem:
        return mem[s]

    for i in range(len(s) + 1):
        currWord = s[:i]
        if currWord in wordDict:
            if Solution.wordBreakHelper(s[i:], wordDict, mem):
                mem[s] = True
                return True
            
    mem[s] = False
    return False

def wordBreak(self, s: str, wordDict: List[str]):
    q = deque([0])
    visited = set()
    while q:
        left = q.popleft()
        if left in visited:
            continue
        visited.add(left)
        for w in wordDict:
            right = left+len(w)
            if s[left:right]==w:
                if right==len(s):
                    return True
                q.append(right)
    return False

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        return self.backtrack(s, wordDict, 0, dict())
        
        
        
    
    def backtrack(self, s: str, wordDict: list, index: int, used: dict):
        if index in used:
            return used[index]
        
        current = ""
        valid = False
        for i in range(index, len(s)):
            current = current + s[i]
            if current in wordDict:
                valid = self.backtrack(s, wordDict, i+1, used)
                if valid:
                    break
                
        used[index] = current in wordDict or valid
        return used[index]

def wordBreak(s, wordDict):
        from collections import deque
        q = deque()
        q.append(0) # startIndx
        visited = set()
        dictSet = set(wordDict)
        while q:
            for i in range(len(q)):
                startIndx = q.popleft()
                if startIndx == len(s): # - NOTE [1]
                    return True
                
                if startIndx not in visited:
                    visited.add(startIndx)
                    
                    for endIndx in range(startIndx+1, len(s)+1): # NOTE [2]
                        sub = s[startIndx: endIndx]
                        if sub in dictSet:
                            q.append(endIndx) # endIndx is the new startIndx
                            
        return False   

377. Combination Sum IV
Medium

2477

275

Add to List

Share
Given an array of distinct integers nums and a target integer target, return the number of possible combinations that add up to target.

The answer is guaranteed to fit in a 32-bit integer.

 

Example 1:

Input: nums = [1,2,3], target = 4
Output: 7
Explanation:
The possible combination ways are:
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
Note that different sequences are counted as different combinations.
Example 2:

Input: nums = [9], target = 3
Output: 0
 

Constraints:

1 <= nums.length <= 200
1 <= nums[i] <= 1000
All the elements of nums are unique.
1 <= target <= 1000
 

Follow up: What if negative numbers are allowed in the given array? How does it change the problem? What limitation we need to add to the question to allow negative numbers?

class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        nums.sort()
        n = len(nums)
        if n == 0:
            return 0
        
        if target<nums[0]:
            return 0
        
        ans =[0]*(target+1)
        for i in range(0, n):
            if nums[i]<=target:
                ans[nums[i]] = 1
            
        ans[0] = 1
        
        for i in range(1, target+1):
            for j in range(n):
                if nums[j]<=i:
                    ans[i] += ans[i-nums[j]]
        
        return int(ans[target]/2)

def combinationSum4(self, nums: List[int], target: int) -> int:
    
    nums.sort()
    if nums[0]>target:
        return 0
    
    dp={0:1}
    for total in range(1,1+target):
        dp[total]=0
        for num in nums:
            if num>total:
                break
            dp[total]+=dp[total-num]    
    
    return dp[target]

class Solution:
    
    def combinationSum4(self, nums: List[int], target: int) -> int:

        nums.sort()
        mem=[None]*(1+target)

        def combinationSumHelper(target):
            if target==0:
                return 1
            if target<0:
                return 0
            if mem[target] is not None:
                return mem[target]
            result=0
            for num in nums:  
                if num>target:
                    break
                else:
                    result+=combinationSumHelper(target-num)
            mem[target]=result
            return result          
        
        return combinationSumHelper(target)

class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        nums.sort()
        memory = [1]
        for i in range(1, target + 1):
            temp, j = 0, len(memory) - 1
            for n in nums:
                while j > 0 and n + j > i:
                    j -= 1
                if n + j == i:
                    temp += memory[j]
            memory.append(temp)
        return memory[-1]

198. House Robber
Medium

7638

203

Add to List

Share
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

 

Example 1:

Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.
Example 2:

Input: nums = [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
Total amount you can rob = 2 + 9 + 1 = 12.

class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums)==1:
            return nums[0]
        if len(nums)==0:
            return 0
        dp=[0]*len(nums)
        if len(nums)==2:
            return max(nums[0],nums[1])
        dp[0]=nums[0]
        dp[1]=max(nums[0],nums[1])
        
        
        for i in range(2,len(nums)):
            dp[i]=max(dp[i-2]+nums[i],dp[i-1])
            
        print(dp)
        return dp[len(nums)-1]

class Solution:
    def rob(self, nums: List[int]) -> int:
        dp = [0, 0]
        for i, num in enumerate(nums):
                dp.append(max(dp[i] + num, dp[i+1]))
        return dp[-1]

class Solution:
    def rob(self, nums: List[int]) -> int:
        
        self.n = len(nums)
        self.nums = nums 
        return self.rob_house(0)
        
    def rob_house(self, i):
        if i > (self.n-1): return 0
        
        include_current_house = self.nums[i]+self.rob_house(i+2)
        exclude_current_house = self.rob_house(i+1)
        return max(include_current_house, exclude_current_house)

class Solution:
    def rob(self, nums: List[int]) -> int:
        
        n = len(nums)
        dp = [0]*(n+1)
        
        dp[0] = 0
        dp[1] = nums[0] # for house 0, dp index starts from 1
        
        for i in range(1,n):
            
			include_cur_house =  nums[i]+dp[i-1]
			exclude_cur_house =  dp[i]
            dp[i+1] = max(include_cur_house, exclude_cur_house)
            
        return dp[n]

class Solution:
    def rob(self, nums: List[int]) -> int:
        
        n = len(nums)
        dp = [0]*(n+1)
        
        prev = 0 # profit after robbing previous house
        prev2prev = 0 # profit after robbing previous 2 previous house
        
        for i in range(n):
            
            temp = prev
            prev = max(nums[i]+prev2prev, prev)
            prev2prev = temp
            
        return prev

213. House Robber II
Medium

3158

68

Add to List

Share
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

 

Example 1:

Input: nums = [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.
Example 2:

Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.
Example 3:

Input: nums = [0]
Output: 0

class Solution:
    def rob(self, nums: List[int]) -> int:
        
        def robHelper(nums: List[int]) -> int:
            dp = [0, 0]
            for i, num in enumerate(nums):
                    dp.append(max(dp[i] + num, dp[i+1]))
            return dp[-1]
        
        if len(nums) == 1:
            return nums[0]
        return max(robHelper(nums[:-1]), robHelper(nums[1:]))

91. Decode Ways
Medium

4848

3526

Add to List

Share
A message containing letters from A-Z can be encoded into numbers using the following mapping:

'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above (there may be multiple ways). For example, "11106" can be mapped into:

"AAJF" with the grouping (1 1 10 6)
"KJF" with the grouping (11 10 6)
Note that the grouping (1 11 06) is invalid because "06" cannot be mapped into 'F' since "6" is different from "06".

Given a string s containing only digits, return the number of ways to decode it.

The answer is guaranteed to fit in a 32-bit integer.

 

Example 1:

Input: s = "12"
Output: 2
Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).
Example 2:

Input: s = "226"
Output: 3
Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
Example 3:

Input: s = "0"
Output: 0
Explanation: There is no character that is mapped to a number starting with 0.
The only valid mappings with 0 are 'J' -> "10" and 'T' -> "20", neither of which start with 0.
Hence, there are no valid ways to decode this since all digits need to be mapped.
Example 4:

Input: s = "06"
Output: 0
Explanation: "06" cannot be mapped to "F" because of the leading zero ("6" is different from "06").
 

Constraints:

1 <= s.length <= 100
s contains only digits and may contain leading zero(s).

class Solution(object):
    dp = {}
    def numDecodings(self, s):
        # Dynamic Pogramming Optimizer
        if s in self.dp:
            return self.dp[s]

        # Basic cases test
        if len(s) == 0:
            return 1

        if s[0] == '0':
            return 0

        if len(s) == 1:
            return 1
 
        # Recursive calls: 
		# 1. The number of combinations fixing the first char as an isolated
        #   word times the number of combinations of the rest of the string
        # 2. The number of combinations fixing the first two chars
        #  as an isolated word times the number of combinations of the rest of the string
        count = self.numDecodings(s[0]) * self.numDecodings(s[1:])
        if int(s[0:2]) < 27 and int(s[0:2]) > 0:   
            count = count + self.numDecodings(s[2:])  
        self.dp[s] = count
        return count  

class Solution:
    def numDecodings(self, s: str) -> int:
        """
        time: O(2^N)
        space: O(1)
        """
        memo = {}
        
        def dfs(start, memo) -> int:
            if start == len(s):
                return 1
            
            if start in memo:
                return memo[start]
            
            res = 0
            if s[start] != '0':
                if start < len(s)-1 and int(s[start: start+2]) <= 26:
                    res += dfs(start + 2, memo)

                res += dfs(start+1, memo)
            
            memo[start] = res
            
            return memo[start]

        return dfs(0, memo)

ZERO = "0"
class Solution:
    def numDecodings(self, s: str) -> int:
        dp = [None for _ in range(len(s))]
        def decode(ss):
            if not ss:
                return 1  # we have decoded the complete string, so one way
            if dp[len(s)-len(ss)] is not None:
                return dp[len(s)-len(ss)]
            one = decode(ss[1:]) if ss[0] != ZERO else 0
            two = decode(ss[2:]) if 9 < int(ss[:2]) < 27 else 0
            dp[len(s)-len(ss)] = one+two
            return one+two
        return decode(s)

class Solution:
    def numDecodings(self, s: str):
        if s[0] == '0': return 0
        if len(s) == 1: return 1
        codeset = {str(i) for i in range(1, 27)}
        
        dp = [0]*(len(s)+1)
		#adding additional element at the end of dp list to make dp[-1] == 1 while
		#computing dp[1-2] to avoid additional if else statements
        dp[0], dp[-1] = 1, 1
        
        for i in range(1, len(s)):
            if s[i] in codeset:
                dp[i] += dp[i-1]
            if s[i-1:i+1] in codeset:
                dp[i] += dp[i-2]
            if not dp[i]:
                return 0
            
        return dp[-2]

class Solution:
    def numDecodings(self, s: str) -> int:
        codeset = {str(i) for i in range(1, 27)}
        memoizeArr = [0]*len(s)
        
        def possibleCodes(i):            
            if i == len(s):
                return 1
            
            if memoizeArr[i]:
                return memoizeArr[i]
            
            total = 0
            if s[i] in codeset:
                total += possibleCodes(i+1)
            if i+1 < len(s) and s[i:i+2] in codeset:
                total += possibleCodes(i+2)
            
            memoizeArr[i] = total
            return total
        
        return possibleCodes(0)

62. Unique Paths
Medium

5753

251

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

def uniquePaths(self, m: int, n: int) -> int:
    	dp=[0 for x in range(m)]
	for i in range(m):
		dp[i]=1
	for i in range(1,n):
		for j in range(1,m):
			dp[j]=dp[j]+dp[j-1]
	return dp[m-1]

def uniquePaths(self, m: int, n: int) -> int:
        dic={}
        def unique(m,n):
            if (m,n) in dic:
                return dic[(m,n)]
            if m==1 or n==1:
                dic[(m,n)]=1
                return 1
            dic[(m,n)]=unique(m-1,n)+unique(m,n-1)
            return dic[(m,n)]
        
        return unique(m,n)

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        return self.cmb(m + n - 2, n - 1)
    
    def cmb(self, n, r):
        if n - r < r:
            r = n - r
        if r == 0:
            return 1
        if r == 1:
            return n

        numerator = [n - r + k + 1 for k in range(r)]
        denominator = [k + 1 for k in range(r)]

        for p in range(2, r + 1):
            pivot = denominator[p - 1]
            if pivot > 1:
                offset = (n - r) % p
                for k in range(p - 1, r, p):
                    numerator[k - offset] /= pivot
                    denominator[k] /= pivot

        result = 1
        for k in range(r):
            if numerator[k] > 1:
                result *= int(numerator[k])

        return result

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        
        dp = [[0 for j in range(n)] for i in range(m)]
        for i in range(m):
            dp[i][0] = 1
            
        for j in range(n):
            dp[0][j] = 1
            
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i][j-1] + dp[i-1][j]
                
        return dp[m-1][n-1]

55. Jump Game
Medium

7080

447

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

def canJump(self, nums: List[int]) -> bool:
        # Greedy: Remember which is the maximum position you can move to.
        # This means that somehow you can get at least that far. 
        # Move up to there and while you do that, keep track if there's a better
        # combination. If there is, update the maximum distance you can go to.
        # Time: O(n). Space: O(1)
        
        pos = maxPos = 0
        
        while pos <= len(nums) -1 and pos <= maxPos:
            maxPos = max(maxPos, pos + nums[pos]) # current position + max jump length
            pos += 1
        
        return pos == len(nums) # If pos > maxPos, we can't get to the end

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if len(nums) == 1: return True
        if nums[0] == 0: return False
        for i in range(len(nums)-1):
            if nums[i] == 0:
                for j in range(1, i+1):
                    if nums[i-j] > j: break
                else:
                    return False
        
        return True

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        
		# GREEDY
        n = len(nums)
        goal = n-1
        
        # Simply start backwards and keep changing the
        # GOAL you want to reach. if it reachs starting point, VOILA!
        for i in range(n-1,-1,-1):
            if i+nums[i] >= goal:
                goal = i
        return goal==0
        
        
        
        # DFS with DP
        if n==1: return True
        def inner(idx,dp):
            # base case 
            if idx>=n:
                return False
            
            # memoization
            if dp[idx]!=-1:
                return dp[idx]
            
            # We got to the end! Found our answer
            if idx==n-1:
                return True
            
            # If its 0 and not the last value then just repeats same index.
            if nums[idx]==0:
                return False
            
            # Keep looping and choosing all values from (1,num) to add to
            lst = []
            for i in range(1,nums[idx]+1):
                lst += [inner(idx+i,dp)]
            
            # If any of the list is True then we found a path!
            dp[idx]=any(lst)
            return dp[idx]

        return inner(0, [-1 for _ in range(n)])

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        maxJump = 1
        for idx in range(len(nums)):
            maxJump = max(nums[idx], maxJump - 1)
            if maxJump == 0 and idx != len(nums) - 1:
                return False
        return True

104. Maximum Depth of Binary Tree
Easy

4395

102

Add to List

Share
Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

 

Example 1:


Input: root = [3,9,20,null,null,15,7]
Output: 3
Example 2:

Input: root = [1,null,2]
Output: 2
Example 3:

Input: root = []
Output: 0
Example 4:

Input: root = [0]
Output: 1

class Solution:
    def maxDepth(self, root: TreeNode) -> int: 
        if not root:
            return 0
        queue = collections.deque([(root, 1)])        
        while queue:
            node, level = queue.popleft()
            if node.left:
                queue.append((node.left, level +1))
            if node.right:
                queue.append((node.right, level +1))
        
        return level

class Solution:
    def maxDepth(self, root: TreeNode) -> int: 
        if root is None:
            return 0        
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        l = self.maxDepth(root.left)
        r = self.maxDepth(root.right)
        return 1 + max(l, r)


nathancy's avatar
nathancy
1
3 days ago

30 VIEWS

For BFS, we can do a modified level order traversal where we group the children in the same level into a temporary queue. We know the level is done when the queue is empty so we increment the depth and replace the queue.

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        # BFS (level order traversal)
        def BFS(root):
            depth = 1
            next_level = []
            queue = [root]
            while queue:
                root = queue.pop()
                if root:
                    # find adjacent children and put it into temp queue
                    if root.left:
                        next_level.append(root.left)
                    if root.right:
                        next_level.append(root.right)
                # if level is done
                if not queue and next_level:
                    depth += 1
                    queue = next_level
                    next_level = []
            return depth

        return BFS(root)

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        # base case
        if not root:
            0
        self.depth = 0
        
        # preorder traversal (root, left, right)
        def DFS(root, current_depth):
            if root:
                current_depth += 1
                self.depth = max(current_depth, self.depth)
                # find adjacent nodes and recursively call DFS
                DFS(root.left, current_depth)
                DFS(root.right, current_depth)
            else:
                current_depth -= 1
        DFS(root, 0)
        return self.depth

100. Same Tree
Easy

3659

94

Add to List

Share
Given the roots of two binary trees p and q, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

 

Example 1:


Input: p = [1,2,3], q = [1,2,3]
Output: true
Example 2:


Input: p = [1,2], q = [1,null,2]
Output: false
Example 3:


Input: p = [1,2,1], q = [1,1,2]
Output: false
 

Constraints:

The number of nodes in both trees is in the range [0, 100].
-104 <= Node.val <= 104

def recr_v1(p,q) -> bool:
    if p==q==None:
        return True
    elif p==None or q is None or p.val!=q.val:
        return False
    return recr_v1(p.left,q.left) and recr_v1(p.right,q.right)

def recr_v2(p,q):
    if p and q:
        return p.val==q.val and recr_v2(p.left,q.left) and recr_v2(p.right, q.right)
    return p is q # equiv  p==q and  this is only possible when  p==q==None

def recr_oneliner_v3(p,q):
    return p and q and p.val==q.val and recr_oneliner_v3(p.left,q.left) and recr_oneliner_3(p.right, q.right) or p is q

def iter_dfs_inorder_traversal_v0a(p) -> bool:
    st=[]
    while True:
        while p:
            st.append(p)
            p=p.left
        if st:
            p=st.pop()
            print(p.val,end=' ')
            p=p.right
        else:
            break

def iter_bfs_v9(p,q):
    dq=deque([(p,q)])  # use deque, it's FIFO
    while dq:
        p,q=dq.popleft()
        if p and q and p.val==q.val:
            #print(p.val,end=' ')
            dq.append((p.left,q.left))
            dq.append((p.right,q.right))
        elif p or q: #  concise way to say exactly one of p,q is None
            return False
    return True

226. Invert Binary Tree
Easy

5790

82

Add to List

Share
Given the root of a binary tree, invert the tree, and return its root.

 

Example 1:


Input: root = [4,2,7,1,3,6,9]
Output: [4,7,2,9,6,3,1]
Example 2:


Input: root = [2,1,3]
Output: [2,3,1]
Example 3:

Input: root = []
Output: []
 

Constraints:

The number of nodes in the tree is in the range [0, 100].
-100 <= Node.val <= 100

class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return root
        
        queue = []
        queue.append(root)
        
        while queue:
            current_node = queue.pop()
            temp_left = current_node.left
            current_node.left = current_node.right
            current_node.right = temp_left

            # continue traversing with left and right node
            if current_node.right!= None:
                queue.append(current_node.right)
            if current_node.left!= None:
                queue.append(current_node.left)
            
        return root

class Solution:
    def __init__(self):
        self.visited = [] # to store the nodes visited
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return root
        
        self.postOrderTraversal(root)    
        return root
            
            
    def postOrderTraversal(self,node : TreeNode):
        if not node:
            return
        self.postOrderTraversal(node.left)
        self.postOrderTraversal(node.right)
        self.swap(node)
    
    def swap(self,node:TreeNode):
        left = node.left
        right = node.right
        node.right = left
        node.left = right

class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        
        def invert(node):
            if node == None:
                return
            node.left, node.right = node.right,node.left
            invert(node.left)
            invert(node.right)
            
        invert(root)
        return root

124. Binary Tree Maximum Path Sum
Hard

6482

424

Add to List

Share
A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the root of a binary tree, return the maximum path sum of any path.

 

Example 1:


Input: root = [1,2,3]
Output: 6
Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.
Example 2:


Input: root = [-10,9,20,null,null,15,7]
Output: 42
Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.

class Solution(object):
    
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root:
            max_path, _ = self.maxPathRecursive(root)
            return max_path
        return 0

    def maxPathRecursive(self, root):
        if not root:
            return -float('inf'), -float('inf')
        
        max_valid_path = root.val
        curr_max = root.val

        max_right, max_valid_path_right = self.maxPathRecursive(root.right)
        max_left, max_valid_path_left = self.maxPathRecursive(root.left)

        max_valid_path = max(max_valid_path, max_valid_path_right + root.val, max_valid_path_left + root.val)

        curr_max = max(curr_max, max_right, max_left, max_valid_path, max_valid_path_right + max_valid_path_left + root.val)

        return curr_max, max_valid_path

class Solution(object):
    def post_order(self, root, res):
        if not root:
            return 0
        
        left_sum = self.post_order(root.left, res)
        right_sum = self.post_order(root.right, res)
        
        left_only = left_sum + root.val
        right_only = right_sum + root.val
        all_sum = left_only + right_only - root.val
        
        res[0] = max(res[0], max(all_sum, left_only, right_only, root.val))
        return max(left_only, right_only, root.val)
        
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        
        res = [float('-inf')]
        self.post_order(root, res)
        return res[0]

class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        def preorder_traversal(node):
            nonlocal path_max
            v = node.val
            if not node.left and not node.right:
                path_max = max(path_max, v)
                return v
            l_max, r_max = 0, 0
            if node.left:
                l_max = preorder_traversal(node.left)
            if node.right:
                r_max = preorder_traversal(node.right)
            # curr_max represents any max sum resulted from going through
            # current node but does not include sum of left, self, right
            # given a node can't appear twice in a path
            curr_max = max(v+l_max, v+r_max, v)
            path_max = max(path_max, curr_max, v+l_max+r_max)
            return curr_max
        
        path_max = -float('inf')
        preorder_traversal(root)
        return path_max

class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        record = -(math.inf)
        
        def traverse(node):
            # Reached end
            if node is None:
                return 0
            
            left = traverse(node.left)
            right = traverse(node.right)
            
            retval = max(node.val, node.val + left, node.val + right)
            potrecord = max(retval, node.val + left + right)
             
            nonlocal record
            if potrecord > record:
                record = potrecord
            return retval
        
        traverse(root)
        return record

102. Binary Tree Level Order Traversal
Medium

5358

114

Add to List

Share
Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

 

Example 1:


Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]
Example 2:

Input: root = [1]
Output: [[1]]
Example 3:

Input: root = []
Output: []

def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root: return []
        q = deque([root])
        result = []
        level = 0
        while q:
            q_size = len(q)
            result.append([])
            for _ in range(q_size):
                node = q.popleft()
                result[level].append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)

            level += 1
        return result

from collections import deque
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        
        if not root:
            return []
        
        queue = deque()
        
        queue.append(root)
        
        result = []
        
        while queue:
            
            level_result = []
            
            for _ in range(len(queue)):
                node = queue.popleft()
                
                if node.left:
                    queue.append(node.left)
                    
                if node.right:
                    queue.append(node.right)
                    
                level_result.append(node.val)
            
            result.append(level_result)
        
        return result 

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root is None:
            return []
        result, queue, level_array = [], [root], []
        queue_length, counter = len(queue), 0
        while len(queue) > 0 :
            current_node = queue.pop(0)
            counter += 1
            level_array.append(current_node.val)
            if current_node.left:
                queue.append(current_node.left)
            if current_node.right:
                queue.append(current_node.right)
            if counter == queue_length:
                result.append(level_array)
                counter = 0
                level_array = []
                queue_length = len(queue)
        return result

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        result=[]
        def dfs(tree,depth):
            if(tree):
                if(depth==len(result)):
                    result.append([])
                result[depth].append(tree.val)
                dfs(tree.left,depth+1)
                dfs(tree.right,depth+1)
        dfs(root,0)
        return result

572. Subtree of Another Tree
Easy

3824

187

Add to List

Share
Given the roots of two binary trees root and subRoot, return true if there is a subtree of root with the same structure and node values of subRoot and false otherwise.

A subtree of a binary tree tree is a tree that consists of a node in tree and all of this node's descendants. The tree tree could also be considered as a subtree of itself.

 

Example 1:


Input: root = [3,4,5,1,2], subRoot = [4,1,2]
Output: true
Example 2:


Input: root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2]
Output: false

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSubtree(self, root, subRoot):
        """
        :type root: TreeNode
        :type subRoot: TreeNode
        :rtype: bool
        """
        self.res = False
        def isIdentical(tree1,tree2):
            if tree1 == None and tree2 == None:
                return True
            if tree1 == None:
                return False
            if tree2 == None:
                return False
            return isIdentical(tree1.left,tree2.left) and tree1.val == tree2.val and isIdentical(tree1.right,tree2.right)
        
        def preorder(root,subtree):
            if not root:
                return
            preorder(root.left,subtree)
            if isIdentical(root,subtree):
                self.res = True
                return
            
            preorder(root.right,subtree)
        preorder(root,subRoot)
        return self.res

class Solution:
    def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:
        def check_subtree_DFS(t1: TreeNode, t2: TreeNode) -> bool:
            # Base case, if both leaf nodes point to null
            if t1 is None and t2 is None:
                return True
            # Base case, if both leaf nodes do not point to null
            if t1 is None or t2 is None:
                return False

            # Check the current node values match or not, and proceed with the recursion of left and right subtrees
            if t1.val == t2.val and check_subtree_DFS(t1.left, t2.left) and check_subtree_DFS(t1.right, t2.right):
                return True
            else:
                return False

        def traverse_tree(root: TreeNode) -> bool:
            nonlocal subRoot
            if root:
                # Traverse to the extreme bottom and check if any of the node matches
                if traverse_tree(root.left) or traverse_tree(root.right):
                    return True
                return check_subtree_DFS(root, subRoot)

        return traverse_tree(root)

def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:
    	return self.impl(root, subRoot)

def impl(self, root, subRoot, checking=False):
	if root is None and subRoot is None:    return True
	if root is None or subRoot is None: return False

	valid = False
	if root.val == subRoot.val:
		valid = valid or self.impl(root.left, subRoot.left, True) and self.impl(root.right, subRoot.right, True)
	if not checking:
		valid = valid or self.impl(root.left, subRoot) or self.impl(root.right, subRoot)
	return valid

class Solution:
    def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:
        if root: 
            return self.isSameTree(root, subRoot) or self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
        else: return subRoot is None
    
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:  #LC100: https://leetcode.com/problems/same-tree/
        if q is None and p is None: return True
        elif q is None and p is not None: return False
        elif q is not None and p is None: return False
        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right) 

105. Construct Binary Tree from Preorder and Inorder Traversal
Medium

6092

146

Add to List

Share
Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.

 

Example 1:


Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
Example 2:

Input: preorder = [-1], inorder = [-1]
Output: [-1]
 

Constraints:

1 <= preorder.length <= 3000
inorder.length == preorder.length
-3000 <= preorder[i], inorder[i] <= 3000
preorder and inorder consist of unique values.
Each value of inorder also appears in preorder.
preorder is guaranteed to be the preorder traversal of the tree.
inorder is guaranteed to be the inorder traversal of the tree.

def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
    
    if not inorder:
        return
    root=preorder.pop(0)
    temp=inorder.index(root)
    root=TreeNode(root)
  
    root.left=self.buildTree(preorder,inorder[:temp])
    root.right=self.buildTree(preorder,inorder[temp+1:])
    return root

class Solution:
    	def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
		map_inorder = {}
		preorder = preorder[::-1]
		for i,val in enumerate(inorder): map_inorder[val] = i
        
		def helper(low,end):
			if low > end: return None
			root = TreeNode(preorder.pop())
			idx = map_inorder[root.val]
			root.left = helper(low,idx-1)
			root.right = helper(idx+1,end)
        
        
			return root
		return helper(0,len(inorder)-1)

class Solution:
    	def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:

		"""
		Approach

		wkt preOrder = Root,L,R
			InOrder = L,Root,R
		"""
		def helper(preorder,inorder):
			n1 = len(preorder)
			n2 = len(inorder)

			if n1 != n2 or preorder is None or inorder is None or n1 == 0:
				return None

			root = TreeNode(preorder[0])
			idx = inorder.index(root.val)

			root.left = helper(preorder[1:idx+1],inorder[:idx])
			root.right = helper(preorder[idx+1:],inorder[idx+1:]) 

			return root

		return helper(preorder,inorder)

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        inorder_index = {i:count for count, i in enumerate(inorder)}
        head = TreeNode(preorder.pop(0))
        for currVal in preorder:
            myTreeNode = head
            while myTreeNode:
                if inorder_index[currVal] < inorder_index[myTreeNode.val]:
                    parentNode = myTreeNode
                    myTreeNode = myTreeNode.left
                    if not myTreeNode:
                        parentNode.left = TreeNode(currVal)
                        break
                else:
                    parentNode = myTreeNode
                    myTreeNode = myTreeNode.right
                    if not myTreeNode:
                        parentNode.right = TreeNode(currVal)
                        break
        return head

98. Validate Binary Search Tree
Medium

6872

734

Add to List

Share
Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.
 

Example 1:


Input: root = [2,1,3]
Output: true
Example 2:


Input: root = [5,1,4,null,null,3,6]
Output: false
Explanation: The root node's value is 5 but its right child's value is 4.

class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        lst1=list()
        def inorder(root):
            if root == None:
                return 
            else:
                inorder(root.left)
                lst1.append(root.val)
                inorder(root.right)
        
        inorder(root)
     
        for i in range(len(lst1)-1):
            if lst1[i]>=lst1[i+1]:
                return False
        return True

def isValidBST(self, root: TreeNode) -> bool:
        self.prev = None
        def dfs(root):
            if not root:
                return True
            left = dfs(root.left) 
            if self.prev and root.val <= self.prev.val:
                return False
            self.prev = root
            right = dfs(root.right)
            return left and right
        return dfs(root)

def isValidBST(self, root: TreeNode) -> bool:
        stack = []
        prev = None
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            if prev and prev.val >= root.val:
                return False
            prev = root
            root = root.right
        return True

class Solution:
    def __init__(self):
        self.sortedArray = []
        
    def isValidBST(self, root: TreeNode) -> bool:
        self.inorderT(root)
        for i in range(len(self.sortedArray)-1):
            if self.sortedArray[i+1]<=self.sortedArray[i]:
                return False
        return True
    
    def inorderT(self, root):
        if not root:
            return 
        self.inorderT(root.left)
        self.sortedArray.append(root.val)
        self.inorderT(root.right)

230. Kth Smallest Element in a BST
Medium

4433

95

Add to List

Share
Given the root of a binary search tree, and an integer k, return the kth (1-indexed) smallest element in the tree.

 

Example 1:


Input: root = [3,1,4,null,2], k = 1
Output: 1
Example 2:


Input: root = [5,3,6,2,4,null,null,1], k = 3
Output: 3
 

Constraints:

The number of nodes in the tree is n.
1 <= k <= n <= 104
0 <= Node.val <= 104
 

Follow up: If the BST is modified often (i.e., we can do insert and delete operations) and you need to find the kth smallest frequently, how would you optimize?

class Solution:
    def helper(self,tree,result):
        if(not tree):
            return 
        else:
            self.helper(tree.left,result)
            result.append(tree.val)
            self.helper(tree.right,result)
            
        
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        temp=[]
        self.helper(root,temp)
        return temp[k-1]

class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        def make(root,l):
            if not root:
                return
            make(root.left,l)
            l.append(root.val)
            make(root.right,l)
            
        l = []
        make(root,l)
        return l[k-1]

class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        
        val = None
        
        def helper(node):
            nonlocal k, val
            if node and k > -1:
                helper(node.left)
                k -= 1
                if k == 0:
                    val = node.val
                helper(node.right)
            
        helper(root)
        return val

class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        ino=[]
        stack=[]
        cur=root
        while True:
            if cur:
                stack.append(cur)
                cur=cur.left
            elif stack:
                node=stack.pop()
                ino.append(node.val)
                cur=node.right
                if len(ino)==k:
                    return ino[-1]
            else:
                break

class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        # inorder traversal kth smallest element is the answer
        def traversal(root, res):
            if not root:
                return
            # left root right
            if len(res) >= k:
                return
            traversal(root.left, res)
            res.append(root.val)
            if len(res) >= k:
                return
            traversal(root.right, res)
            
        inorder = []
        traversal(root, inorder)
        return inorder[k-1]

235. Lowest Common Ancestor of a Binary Search Tree
Easy

3780

147

Add to List

Share
Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

 

Example 1:


Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
Output: 6
Explanation: The LCA of nodes 2 and 8 is 6.
Example 2:


Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
Output: 2
Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.
Example 3:

Input: root = [2,1], p = 2, q = 1
Output: 2

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
		if root is None: return None
        if root.val > p.val and root.val > q.val: return self.lowestCommonAncestor(root.left, p, q)
        if root.val < p.val and root.val < q.val: return self.lowestCommonAncestor(root.right, p, q)
        return root

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
	while root:
            if root.val > p.val and root.val > q.val:
                root=root.left
            elif root.val < p.val and root.val < q.val:
                root=root.right
            else:
                break
        return root

def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root: return -1
        if p.val < root.val and q.val < root.val: return self.lowestCommonAncestor(root.left, p, q)
        if p.val > root.val and q.val > root.val: return self.lowestCommonAncestor(root.right, p, q)
        return root

def get_low(root, p, q):
    if p.val < root.val and q.val < root.val:
        return get_low(root.left, p, q)
    if p.val > root.val and q.val > root.val:
        return get_low(root.right, p, q)
    else:
        return root


class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        return get_low(root, p, q)

208. Implement Trie (Prefix Tree)
Medium

5033

75

Add to List

Share
A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:

Trie() Initializes the trie object.
void insert(String word) Inserts the string word into the trie.
boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.
 

Example 1:

Input
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
Output
[null, null, true, false, true, null, true]

Explanation
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // return True
trie.search("app");     // return False
trie.startsWith("app"); // return True
trie.insert("app");
trie.search("app");     // return True

class Trie:
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.data = []
        

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        self.data.append(word)
        

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        if(word in self.data):
            return True
        else:
            return False
        

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        for i in range(len(self.data)):
            if(self.data[i][:len(prefix)] == prefix):
                return True
        return False

class Trie:
    children: dict[chr, 'Trie']
    isWord: bool
    
    def __init__(self):
        self.children = {}
        self.isWord = False

    def insert(self, word: str) -> None:
        p: 'Trie' = self
            
        char: chr
        for char in word:
            if char not in p.children:
                p.children[char] = Trie()
            p = p.children[char]
            
        p.isWord = True

    def search(self, word: str) -> bool:
        p: 'Trie' = self
        
        char: chr
        for char in word:
            if char in p.children:
                p = p.children[char]
            else:
                return False
        
        return p.isWord
        

    def startsWith(self, prefix: str) -> bool:
        p: 'Trie' = self
        
        char: chr
        for char in prefix:
            if char in p.children:
                p = p.children[char]
            else:
                return False
        
        return True

class Trie:
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.keys = {}
        self.keyNumberStatic = 1
        

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        
        keyNumber = 0
        for i in word:
            if keyNumber in self.keys and i in self.keys[keyNumber]:
                pass
            elif keyNumber in self.keys and i not in self.keys[keyNumber]:
                current_dict = self.keys[keyNumber]
                current_dict[i] = self.keyNumberStatic
                self.keyNumberStatic+=1
            else:
                self.keys[keyNumber] = {i: self.keyNumberStatic}
                self.keyNumberStatic+=1
            keyNumber = self.keys[keyNumber][i]
        
        if keyNumber in self.keys and "" in self.keys[keyNumber]:
            pass
        elif keyNumber in self.keys and "" not in self.keys[keyNumber]:
            current_dict = self.keys[keyNumber]
            current_dict[""] = self.keyNumberStatic
            self.keyNumberStatic+=1
        else:
            self.keys[keyNumber] = {"": self.keyNumberStatic}
            self.keyNumberStatic+=1

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        start = 0
        for i in word:
            if start not in self.keys:
                return False
            elif i in self.keys[start]:
                start = self.keys[start][i]
            else:
                return False
        
        if "" in self.keys[start]:
            return True
        
        
            
        
    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        start = 0
        for i in prefix:
            if start not in self.keys:
                return False
            elif i in self.keys[start]:
                start = self.keys[start][i]
            else:
                return False

        return True

class Trie:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = dict()
        

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        root = self.root
        
        for char in word:
            if char not in root:
                root[char] = dict()
            root = root[char]
        root["end"] = 1

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        root = self.root
        for char in word:
            if char not in root:
                return False
            root = root[char]
        return "end" in root

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        root = self.root
        for char in prefix:
            if char not in root:
                return False
            root = root[char]
        return True

211. Design Add and Search Words Data Structure
Medium

3294

138

Add to List

Share
Design a data structure that supports adding new words and finding if a string matches any previously added string.

Implement the WordDictionary class:

WordDictionary() Initializes the object.
void addWord(word) Adds word to the data structure, it can be matched later.
bool search(word) Returns true if there is any string in the data structure that matches word or false otherwise. word may contain dots '.' where dots can be matched with any letter.
 

Example:

Input
["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output
[null,null,null,null,false,true,true,true]

Explanation
WordDictionary wordDictionary = new WordDictionary();
wordDictionary.addWord("bad");
wordDictionary.addWord("dad");
wordDictionary.addWord("mad");
wordDictionary.search("pad"); // return False
wordDictionary.search("bad"); // return True
wordDictionary.search(".ad"); // return True
wordDictionary.search("b.."); // return True

import re

class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        # Creating an empty dictionary. The dictionary will have len of words as key
        # and list of words having length = key as value
        # Ex:  { 2 : ['ab', 'bc'], 3 : ['abc', 'acd'] }
        self.words = {}
        

    def addWord(self, word: str) -> None:
        # If len(word) key is not present in the dictionary, we create a new list with
        # that len(word) as key
        if len(word) not in self.words:
            self.words[len(word)] = list()

        # Adding the given word to that len(word) key list
        self.words[len(word)].append(word)

    def search(self, word: str) -> bool:
        # If len(word) key is not present in dictionary, there is no word with that
        # length. So given word also doesn't present. so return False
        if len(word) not in self.words: return False
        
        # If there is no "." in the given word, it is a normal word and so there is 
        # no need to match the words in our dictionary using regular expression 
        # matching. We just need to look whether that word is present in that len(key)
        # list.
        if "." not in word:
            return word in self.words[len(word)]
        
        # If the given word is an regular expression type with "." in it, we loop 
        # through the words in len(word) key and check if that word matches the given
        # regular expression. If yes we return True
        for w in self.words[len(word)]:
            if re.match(word, w): return True
        
        return False

class Node:
    def __init__(self):
        self.children = {}
        self.is_word = False

class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = Node()

    def addWord(self, word: str) -> None:
        n = self.root
        for c in word:
            if c not in n.children:
                n.children[c] = Node()
            n = n.children[c]
        n.is_word = True

    def search(self, word: str) -> bool:
         
        def helper(w, n):
            if not w:
                return n.is_word
            if w[0] != '.':
                if w[0] not in n.children:
                    return False
                return helper(w[1:], n.children[w[0]])
            else:
                return any([helper(w[1:], n.children[x]) for x in n.children])
                
        return helper(word, self.root)

class TrieNode:
    def __init__(self,ch):
        self.children = defaultdict(TrieNode)
        self.isWord = False
        self.char = ch
        
        
        
class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode('*')
       
        

    def addWord(self, word: str) -> None:
        head = self.root
        for ch in word:
            if ch not in head.children:
                head.children[ch]=TrieNode(ch)
            head = head.children[ch]
        head.isWord = True
        

    def search(self, word: str) -> bool:
        head = self.root
        queue = deque([head])
        
        for i,ch in enumerate(word):
            #BFS all the queue node in this level, to find whether ch in the children of the node.
            for _ in range(len(queue)):
                node = queue.popleft()
                if i == len(word)-1: #Find whether find the word in Trie
                    if ch in node.children and node.children[ch].isWord: 
                        return True
                    if ch == '.': #If last ch is '.', explore all the children node to look for isWord
                        for child in node.children:
                            if node.children[child].isWord: return True
                if ch in node.children:
                    queue.append(node.children[ch])
                if ch == '.':
                    queue += node.children.values()
                    
        return False 
        #Two cases that we could exit without return True:
        #1. ch not . and we couldn't find any match for some ch, so queue is empty, then for all the rest of ch, we do nothing
        #2. we complete traversal of word, but didn't find isWord sign.
        #Both above cases we should return False.

class WordDictionary:
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie = {}
        self.count = 0

    def addWord(self, word: str) -> None:
        self.count = max(self.count, len(word))
        m = self.trie
        for w in word:
            m.setdefault(w, {})
            m = m[w]
        m['$'] = word
        
    def search(self, word: str) -> bool:
        if len(word) == 0:
            return False
        if len(word) > self.count:
            return False
        matches = []
        if word[0] == '.':
            matches = list(self.trie.values())
        else:
            d = self.trie.get(word[0], None)
            if d is None:
                return False
            matches = [d]
        for i, w in enumerate(word[1:]):
            m = []
            if w == '.':
                for match in matches:
                    if type(match) == str:
                        continue
                    if '$' in match and len(match) == 1:
                        continue
                    m.extend(list(match.values()))
            else:               
                for match in matches:
                    #print(match)
                    if type(match) == dict and w in match:
                        m.append(match[w])
            if len(m) == 0:
                return False
            matches = m
        for match in matches:
            if '$' in match:
                return True
        return False

212. Word Search II
Hard

4211

152

Add to List

Share
Given an m x n board of characters and a list of strings words, return all words on the board.

Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.

 

Example 1:


Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]
Example 2:


Input: board = [["a","b"],["c","d"]], words = ["abcb"]
Output: []
 

Constraints:

m == board.length
n == board[i].length
1 <= m, n <= 12
board[i][j] is a lowercase English letter.
1 <= words.length <= 3 * 104
1 <= words[i].length <= 10
words[i] consists of lowercase English letters.
All the strings of words are unique.

class Solution:
    def __init__(self, ans=None):
        self.ans = []
        self.dict = Counter()
    def Findwords(self, board, temp, vis, i, j, m, n):
        if i < 0 or j < 0 or i >= m or j >= n or vis[i][j] or len(self.dict) == len(self.ans):
            return None
        vis[i][j] = True
        temp = temp + board[i][j]
        if self.dict[temp]:
            self.ans.add(temp)
        self.Findwords(board, temp, vis, i, j + 1, m, n)
        self.Findwords(board, temp, vis, i + 1, j, m, n)
        self.Findwords(board, temp, vis, i, j - 1, m, n)
        self.Findwords(board, temp, vis, i - 1, j, m, n)
        temp = temp[:-1]
        vis[i][j] = False
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        m, n, temp = len(board), len(board[0]), ''
        self.dict, self.ans = Counter(words), set()
        vis = [[False for i in range(n)] for i in range(m)]
        for i in range(m):
            for j in range(n):
                self.Findwords(board, temp, vis, i, j, m, n)
        return self.ans

    class TrieNode:
        def __init__(self):
            self.mp = Counter()
            self.is_end = False
    class Trie:
         def __init__(self,):
             self.root = TrieNode()
         def insert(self, word):
             curr = self.root
             for char in word:
                 if curr.mp[char] == 0:
                     curr.mp[char] = TrieNode()
                 curr = curr.mp[char]
             curr.is_end = True
    class Solution:
        def __init__(self, ans=None):
            self.ans = []
        def search(self, board, vis, trie, temp, i, j, m, n):
            if i < 0 or j < 0 or i >= m or j >= n or board[i][j] == '#' or trie.mp[board[i][j]] == 0:
                return None
            if trie.mp[board[i][j]]:
                char = board[i][j]
                trie = trie.mp[board[i][j]]
                board[i][j] = '#'
            if trie.is_end:
                self.ans.append(temp + char)
                trie.is_end = False
            self.search(board, vis, trie, temp + char, i, j + 1, m, n)
            self.search(board, vis, trie, temp + char, i + 1, j, m, n)
            self.search(board, vis, trie, temp + char, i - 1, j, m, n)
            self.search(board, vis, trie, temp + char, i, j - 1, m, n)
            board[i][j] = char
        def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
            m, n = len(board), len(board[0])
            self.ans, vis, temp = [], [[False for i in range(n)] for i in range(m)], ''
            trie = Trie()
            for word in words:
                trie.insert(word)
            for i in range(m):
                for j in range(n):
                    self.search(board, vis, trie.root, temp, i, j, m, n)
            return self.ans

class Solution(object):
    def findWords(self, board, words):
        """
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """
        
        def helper(r, c, word, board, prev):
            
                if len(word) == 0:
                    return True
                
                board[r][c] = "."
                flag = 0
                if r-1 >= 0 and board[r-1][c] == word[0] and flag == 0:
                    if helper(r-1, c, word[1:], board, word[0]):
                        flag = 1
                if r+1 < self.m and board[r+1][c] == word[0] and flag == 0:
                    if helper(r+1, c, word[1:], board, word[0]):
                        flag = 1
                if c-1 >= 0 and board[r][c-1] == word[0] and flag == 0:
                    if helper(r, c-1, word[1:], board, word[0]):
                        flag = 1
                if c+1 < self.n and board[r][c+1] == word[0] and flag == 0:
                    if helper(r, c+1, word[1:], board, word[0]):
                        flag = 1
                
                board[r][c] = prev
                if flag:
                    return True
                else:
                    return False
            
        
        retval = []
        self.m = len(board)
        self.n = len(board[0])
        
        hmap = dict()
        for r in range(0, self.m):
            for c in range(0, self.n):
                if board[r][c] not in hmap:
                    hmap[board[r][c]] = 1
                    continue
                hmap[board[r][c]] += 1
        
        cpy_words = []
        for word in words:
            tmp = dict()
            for ch in word:
                if ch not in tmp:
                    tmp[ch] = 1
                    continue
                tmp[ch] += 1
                
            flag = 0
            for k, v in tmp.items():
                if k in hmap and v <= hmap[k]:
                    flag = 1
                else:
                    flag = 0
                    break
                    
            if flag == 1:
                cpy_words.append(word)
        
    
        for r in range(0, self.m):
            for c in range(0, self.n):
                for word in cpy_words:
                    if board[r][c] == word[0] and word not in retval:
                        if helper(r, c, word[1:], board, word[0]):
                            retval.append(word)
        
        return retval

    class Node:
        
    def __init__(self):
        self.children = {}
        self.isWord = False
        self.parentWord = None

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        
        # standard dfs with word check
        # word check is trie with each call being the next level of char in trie 
        # if final word hit then add to res 
        
        trie = Node()
        for word in words:
            cur = trie
            for char in word:
                if char not in cur.children:
                    cur.children[char] = Node()
                cur = cur.children[char]
            cur.isWord = True
            cur.parentWord = word
         
        res = []
        d = ((0,1), (1,0), (0,-1), (-1,0))
        
        for row in range(len(board)):
            for col in range(len(board[0])):
                key = board[row][col]
                if key in trie.children:
                    self.search(board, row, col, trie.children[key], res, d)
        
        return res
    
    def search(self, board, row, col, trie, res, d):

        if trie.isWord:
            trie.isWord = False
            res.append(trie.parentWord)
        
        prev = board[row][col]
        board[row][col] = "#"
        for x, y in d:
            newx = x + row
            newy = y + col
            if newx >= 0 and newy >= 0 and  newx < len(board) and newy < len(board[0]):
                key = board[newx][newy]
                if key in trie.children:
                    self.search(board, newx, newy, trie.children[key], res, d)
                                                      
        board[row][col] = prev

23. Merge k Sorted Lists
Hard

8194

377

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

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        output = []
        for lst in lists:
            while lst:
                output.append(lst.val)
                lst = lst.next
        
        if len(output) == 0: return None
        output = sorted(output)
        head = ListNode(output[0])
        p = head
        for item in output[1:]:
            head.next = ListNode(item)
            head = head.next
        head.next = None
        return p

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        
        starting = tail = None
        stack = []
        for li in lists:
            if li:
                if starting is None:
                    starting = li
                else:
                    tail.next = li

                prev = None
                while li:
                    stack.append(li.val)
                    prev = li
                    li = li.next
                tail = prev
        stack.sort()
        curr = starting
        for x in stack:
            curr.val = x
            curr = curr.next
        return starting

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        
        def merge(head1, head2):
            
            if head1 is None:
                return head2
            elif head2 is None:
                return head1
            
            dummy = ListNode()
            curr = dummy
            while head1 and head2:
                if head1.val <= head2.val:
                    curr.next = head1
                    head1 = head1.next
                else:
                    curr.next = head2
                    head2 = head2.next
                curr = curr.next
            
            curr.next = head1 if head1 else head2
                    
            return dummy.next
        
        if len(lists) == 0:
            return None
        elif len(lists) == 1:
            return lists[0]
        
        curr = merge(lists[0], lists[1])
        for i in range(2, len(lists)):
            curr = merge(curr, lists[i])
        return curr

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        
        def merge(head1, head2):
            
            if head1 is None:
                return head2
            elif head2 is None:
                return head1
            
            dummy = ListNode()
            curr = dummy
            while head1 and head2:
                if head1.val <= head2.val:
                    curr.next = head1
                    head1 = head1.next
                else:
                    curr.next = head2
                    head2 = head2.next
                curr = curr.next
            
            curr.next = head1 if head1 else head2
                    
            return dummy.next
        
        if len(lists) == 0:
            return None
        elif len(lists) == 1:
            return lists[0]
        
        curr = merge(lists[0], lists[1])
        for i in range(2, len(lists)):
            curr = merge(curr, lists[i])
        return curr

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        flattenedList = []
        for list in lists:
            while(list is not None):
                flattenedList.append(list.val)
                list = list.next
        flattenedList.sort()
        #now we build the LL
        head = ListNode()
        if(flattenedList is None or len(flattenedList) == 0):
            return None
        head.val = flattenedList[0]
        length = len(flattenedList) - 1
        i = 1
        prevNode = head
        while(i <= length):
            newNode = ListNode()
            newNode.val = flattenedList[i]
            prevNode.next = newNode
            prevNode = newNode
            i += 1
        return head

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        head = ListNode(0)
        
        l = []
        for i in lists:
            curr = i
            while curr:
                heapq.heappush(l, curr.val)
                curr = curr.next
        p = head
        while len(l):
            i = heapq.heappop(l)
            curr = ListNode(i)
            p.next = curr
            p = p.next
            
        return head.next

import heapq

class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        class Wrapper():
            def __init__(self, node):
                self.node = node
            def __lt__(self, other):
                return self.node.val < other.node.val
        head = point = ListNode(0)
        q = []
        for l in lists:
            if l:
                heapq.heappush(q,Wrapper(l))
        while q:
            node = heapq.heappop(q).node
            point.next = ListNode(node.val)
            point = point.next
            node = node.next
            if node:
                heapq.heappush(q,Wrapper(node))
        return head.next

347. Top K Frequent Elements
Medium

5737

286

Add to List

Share
Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

 

Example 1:

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
Example 2:

Input: nums = [1], k = 1
Output: [1]
 

Constraints:

1 <= nums.length <= 105
k is in the range [1, the number of unique elements in the array].
It is guaranteed that the answer is unique.
 

Follow up: Your algorithm's time complexity must be better than O(n log n), where n is the array's size.

class Solution(object):
    
    def __init__(self):
        self.numMap = defaultdict(lambda : 0)
        self.heap = []
    
    
    def createNumMap(self, nums):
        for elem in nums:
            self.numMap[elem]+=1
            
            
    def applyQuickSelect(self, values, lo, hi):
        
        pi = random.choice(range(lo, hi+1))
        values[pi], values[hi] = values[hi], values[pi]
        pivot = values[hi][1]
        
        i, j = lo-1, lo
        
        while(j < hi):
            if values[j][1] > pivot:
                i+=1
                values[i], values[j] = values[j], values[i]
            j+=1
            
        i+=1
        values[i], values[hi] = values[hi], values[i]
    
        return i
            
            
            
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
       
        self.createNumMap(nums)
        N = len(self.numMap)
        
        keys = list(self.numMap.keys())
        if N <= k:
            return keys
        
        num_li = self.numMap.items()
        lo, hi = 0, len(num_li)-1
        
        
        while(lo < hi):
        
            pivot = self.applyQuickSelect(num_li, lo, hi) 
            if pivot+1 == k:
                return [num_li[i][0] for i in range(pivot+1)]
            elif pivot+1 > k:
                hi = pivot-1
            else:
                lo = pivot+1
        
      
        return [num_li[i][0] for i in range(hi+1)]

class Solution(object):
    def topKFrequent(self, nums, k):
        ans = []
        
        numToCounts = collections.Counter(nums)
        h = [(-numToCounts[num], num) for num in numToCounts]
        
        heapq.heapify(h)
        
        while k>0:
            _, num = heapq.heappop(h)
            ans.append(num)
            k -= 1
        
        return ans

from collections import Counter


class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        d = Counter(nums)
        return dict(d.most_common(k)).keys()

def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        c=Counter(nums)
        heap=[]
        for i in c:
            heappush(heap,(c[i],i))
            
            if len(heap)>k:
                heappop(heap)
      
        return [x[1] for x in heap]

295. Find Median from Data Stream
Hard

5040

88

Add to List

Share
The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value and the median is the mean of the two middle values.

For example, for arr = [2,3,4], the median is 3.
For example, for arr = [2,3], the median is (2 + 3) / 2 = 2.5.
Implement the MedianFinder class:

MedianFinder() initializes the MedianFinder object.
void addNum(int num) adds the integer num from the data stream to the data structure.
double findMedian() returns the median of all elements so far. Answers within 10-5 of the actual answer will be accepted.
 

Example 1:

Input
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
Output
[null, null, null, 1.5, null, 2.0]

Explanation
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0
 

Constraints:

-105 <= num <= 105
There will be at least one element in the data structure before calling findMedian.
At most 5 * 104 calls will be made to addNum and findMedian.
 

Follow up:

If all integer numbers from the stream are in the range [0, 100], how would you optimize your solution?
If 99% of all integer numbers from the stream are in the range [0, 100], how would you optimize your solution?

from heapq import heappush, heappop
class MedianFinder(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.max_heap = []    # stores numbers smaller than the median
        self.min_heap = []    # stores numbers greater than the median

    def addNum(self, num):
        """
        :type num: int
        :rtype: None
        """
        # two invariants that we must keep at all times
        # 1. all elements in the max heap are smaller than those in the min heap
        # 2. number of elements in the two heaps can differ by at most 1

        # handles invariant 1
        if not self.max_heap or num < -self.max_heap[0]:
            heappush(self.max_heap, -num)
        else:
            heappush(self.min_heap, num)
            
        # handles invariant 2
        if len(self.max_heap)-len(self.min_heap) > 1:
            heappush(self.min_heap, -heappop(self.max_heap))
        if len(self.min_heap)-len(self.max_heap) > 1:
            heappush(self.max_heap, -heappop(self.min_heap))
            
    def findMedian(self):
        """
        :rtype: float
        """
        if len(self.max_heap) == len(self.min_heap):
            return float(-self.max_heap[0]+self.min_heap[0])/2
        elif len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        else:
            return self.min_heap[0]

class MedianFinder:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.L = []

    def addNum(self, num: int) -> None:
        if len(self.L) == 0 or num <= self.L[0]:
            i = 0
        elif num >= self.L[-1]:
            i = len(self.L)
        else:
            i = self.binarySearch(num)
        self.L.insert(i, num)

    def findMedian(self) -> float:
        k = len(self.L)-1
        m = k//2
        if k % 2:
            return (self.L[m]+self.L[m+1])/2
        else:
            return self.L[m]
            
    def binarySearch(self, x: int) -> int:
        k = len(self.L)
        upper = k - 1
        mid = upper//2
        lower = 0
        
        while not self.L[mid] <= x <= (self.L[mid+1] if mid+1 < k else 1e5):
            if self.L[mid] > x:
                upper = mid
                mid = (mid-lower)//2 + lower
            else:
                lower = mid
                mid = (upper-mid)//2 + mid
                
        return mid+1

class MedianFinder(object):
    
	def __init__(self):
		"""
		initialize your data structure here.
		"""
		self.max_heap=[]
		self.min_heap=[]
		self.curr_med=float('inf')


	def addNum(self, num):
		"""
		:type num: int
		:rtype: None
		"""
		import heapq
		if self.curr_med==float('inf'):
			self.curr_med=num
			heapq.heappush(self.max_heap, -num)
		else:
			if len(self.max_heap)>len(self.min_heap):
				if num<self.curr_med:
					top=heapq.heappop(self.max_heap)
					heapq.heappush(self.min_heap, -top)
					heapq.heappush(self.max_heap, -num)
				else:
					heapq.heappush(self.min_heap, num)
				self.curr_med=(-self.max_heap[0]+self.min_heap[0])/2.0

			elif len(self.max_heap)<len(self.min_heap):
				if num<self.curr_med:
					heapq.heappush(self.max_heap, -num)
				else:
					top=heapq.heappop(self.min_heap)
					heapq.heappush(self.max_heap, -top)
					heapq.heappush(self.min_heap, num)
				self.curr_med=(-self.max_heap[0]+self.min_heap[0])/2.0      

			else:
				if num<self.curr_med:
					heapq.heappush(self.max_heap, -num)
					self.curr_med=-self.max_heap[0]
				else:
					heapq.heappush(self.min_heap, num)
					self.curr_med=self.min_heap[0]


	def findMedian(self):
		"""
		:rtype: float
		"""
		return self.curr_med

from sortedcontainers import SortedList
class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.mylist = SortedList()
        

    def addNum(self, num: int) -> None:
        self.mylist.add(num)
        

    def findMedian(self) -> float:
        x = len(self.mylist)//2
        if len(self.mylist)%2:
            return self.mylist[x]
        else:
            return (self.mylist[x]+self.mylist[x-1])/2

73. Set Matrix Zeroes
Medium

4291

402

Add to List

Share
Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's, and return the matrix.

You must do it in place.

 

Example 1:


Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]
Example 2:


Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
 

Constraints:

m == matrix.length
n == matrix[0].length
1 <= m, n <= 200
-231 <= matrix[i][j] <= 231 - 1
 

Follow up:

A straightforward solution using O(mn) space is probably a bad idea.
A simple improvement uses O(m + n) space, but still not the best solution.
Could you devise a constant space solution?

class Solution:
    def setZeroes(self, M):
        m, n = len(M[0]), len(M)
        r1 = any(M[0][j] == 0 for j in range(m))
        c1 = any(M[i][0] == 0 for i in range(n))
        for i in range(1, n):
            for j in range(1, m):
                if M[i][j] == 0: M[i][0] = M[0][j] = 0
                
        for i in range(1, n):
            for j in range(1, m):
                if M[i][0] * M[0][j] == 0: M[i][j] = 0
                
        if r1:
            for i in range(m): M[0][i] = 0
                
        if c1:
            for j in range(n): M[j][0] = 0

class Solution:
    	def setZeroes(self, matrix: List[List[int]]) -> None:
		"""
		Do not return anything, modify matrix in-place instead.
		"""
		rows , cols , marker = len(matrix),len(matrix[0]),'M'

		## function for marking the rows and cols when we spot a zero
		## with a marker except the cell which is actually zero

		def markwithmarker(row,col):
			for r in range(rows):
				if matrix[r][col]!=0: 
					matrix[r][col] = 'M'

			for c in range(cols):
				if matrix[row][c]!=0: 
					matrix[row][c] = 'M'

		for row in range(rows):
			for col in range(cols):
				if matrix[row][col] == 0:
					markwithmarker(row,col)

		##Now wherever we mark the cell with M replace those with zeros
		for i in range(rows):
			for j in range(cols):
				if matrix[i][j] == 'M':
					matrix[i][j] = 0

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        i_set, j_set = set(), set()
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == 0:
                    i_set.add(i)
                    j_set.add(j)
        for i in i_set:
            matrix[i] = [0] * len(matrix[i]) 
        for j in j_set:
            for i in range(len(matrix)):
                matrix[i][j] = 0

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        r=len(matrix)
        c=len(matrix[0])
        is_col=False
        for i in range(r):
            if(matrix[i][0]==0):
                is_col=True
            for j in range(1,c):
                if(matrix[i][j]==0):
                    matrix[0][j]=0 
                    matrix[i][0]=0
        for i in range(1,r):
            for j in range(1,c):
                if(not matrix[i][0] or not matrix[0][j]):
                    matrix[i][j]=0 
        if(matrix[0][0]==0):
            for j in range(c):
                matrix[0][j]=0 
        if(is_col):
            for i in range(r):
                matrix[i][0]=0 
        return matrix

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        def transform(m, i, j):
          for k in range(N):
            if m[i][k] != 0:
              m[i][k] = 'Booked' 
          for o in range(M):
            if m[o][j] != 0:
              m[o][j] = 'Booked'
            
        M = len(matrix)
        N = len(matrix[0])
          
        for i in range(M):
          for j in range(len(matrix[i])):
            if matrix[i][j] != 'Booked' and  matrix[i][j] == 0:
              transform(matrix, i, j)
        for i in range(M):
          for j in range(len(matrix[i])):
            if matrix[i][j] == 'Booked':
              matrix[i][j] = 0

54. Spiral Matrix
Medium

4598

698

Add to List

Share
Given an m x n matrix, return all elements of the matrix in spiral order.

 

Example 1:


Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]
Example 2:


Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]
 

Constraints:

m == matrix.length
n == matrix[i].length
1 <= m, n <= 10
-100 <= matrix[i][j] <= 100

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        ans = []
        row_up = 0
        row_down = len(matrix) - 1
        col_left = 0
        col_right = len(matrix[0]) - 1
        
        while True:
            for i in range(col_left, col_right + 1):
                ans.append(matrix[row_up][i])
            
            for i in range(row_up + 1, row_down):
                ans.append(matrix[i][col_right])
            
            if len(ans) < len(matrix) * len(matrix[0]):
                for i in range(col_right, col_left - 1, -1):
                    ans.append(matrix[row_down][i])
            
            if len(ans) < len(matrix) * len(matrix[0]):
                for i in range(row_down - 1, row_up, -1):
                    ans.append(matrix[i][col_left])
            
            if row_up < row_down:
                row_up += 1
                row_down -= 1
            
            if col_left < col_right:
                col_left += 1
                col_right -= 1
                
            if len(ans) == len(matrix) * len(matrix[0]):
                return ans

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m = len(matrix)
        n = len(matrix[0])
        a = []
        i = 0
        j = -1
        while m and n:
            for _ in range(n):
                j += 1
                a.append(matrix[i][j])
            n -= 1
            for _ in range(m - 1):
                i += 1
                a.append(matrix[i][j])
            m -= 1
            if n == 0 or m == 0: break
            for _ in range(n):
                j -= 1
                a.append(matrix[i][j])
            n -= 1
            for _ in range(m - 1):
                i -= 1
                a.append(matrix[i][j])
            m -= 1
        return a

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        l = []
        while len(matrix) > 0:
            # get the first row
            l += matrix.pop(0)
            # give 90 degree angle to the matrix
            matrix =  [row for row in reversed([list(v) for v in list(zip(*matrix))])]
        return l

class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        visited = set()
        queue = [[0,0, "right", "down"]]
        result = []
        maxy = len(matrix[0]) -1
        maxx = len(matrix) -1 
        d2d = {
            "right": "down",
            "down": "left",
            "left": "up",
            "up": "right"
        }
        def addnext(x,y,direction, backupd):
            if direction == "left" and (x, y-1) not in visited and y-1 >= 0:
                queue.append([x,y-1, direction, backupd])
            elif direction == "right" and (x, y+1) not in visited and y+1 <= maxy:
                queue.append([x,y+1, direction, backupd])
            elif direction == "up" and (x-1, y) not in visited and x-1 >= 0:
                queue.append([x-1, y, direction, backupd])
            elif direction == "down" and (x+1, y) not in visited and x+1 <= maxx:
                queue.append([x+1, y, direction, backupd])
            else:
                return False
            

        while queue:
            [x,y, direction, backupd] = queue.pop()

            result.append(matrix[x][y])
            visited.add((x,y))
            valid = addnext(x,y,direction, backupd)
            if valid == False:
                direction = backupd
                backupd = d2d[backupd]
                valid = addnext(x,y,direction, backupd)

                if valid == False:
                    return result

48. Rotate Image
Medium

6012

378

Add to List

Share
You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

 

Example 1:


Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]
Example 2:


Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
Example 3:

Input: matrix = [[1]]
Output: [[1]]
Example 4:

Input: matrix = [[1,2],[3,4]]
Output: [[3,1],[4,2]]
 

Constraints:

matrix.length == n
matrix[i].length == n
1 <= n <= 20
-1000 <= matrix[i][j] <= 1000

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:

        t = l = 0
        b = r = len(matrix)-1
        
        while l < r:
        
            for i in range(r-l):
                tmp = matrix[t][l+i]
                matrix[t][l+i] = matrix[b-i][l]
                matrix[b-i][l] = matrix[b][r-i]
                matrix[b][r-i] = matrix[t+i][r]
                matrix[t+i][r] = tmp
            
            l += 1
            r -= 1
            t += 1
            b -= 1

class Solution:
    
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """

        
        if not matrix or len(matrix) != len(matrix[0]):
            return 
        
        self.transpose(matrix)
        self.reflect(matrix)
        
    def transpose(self,  matrix):
         """ input: 1 2 3
                    4 5 6
                    7 8 9"""
            
        """ output: 1 4 7
                    2 5 8
                    3 6 9"""
   
        for i in range(len(matrix)):
            for j in range(i , len(matrix[0])):
                temp = matrix[i][j]
                matrix[i][j] = matrix[j][i]
                matrix[j][i] = temp
            
    
    def reflect(self, matrix):
        
        # for i in range(len(matrix)):
        #     for j in range(len(matrix[0])//2):
        #         temp = matrix[i][j]
        #         matrix[i][j] = matrix[i][len(matrix[0]) - 1 - j]
        #         matrix[i][len(matrix[0]) - 1 - j] = temp
                
            
        for i in range(len(matrix)):
            matrix[i] = matrix[i][::-1]

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        if matrix is None:
            return 
        if len(matrix)==0 or len(matrix)==1:
            return
        
        n=len(matrix)-1
        
        #matrix row flipping
        for i in range(len(matrix)//2):
            matrix[i],matrix[n]=matrix[n],matrix[i]
            n-=1
        
        #transposing the matrix
        for i in range(len(matrix)):
            for j in range(i+1,len(matrix)):
                matrix[i][j],matrix[j][i]=matrix[j][i],matrix[i][j]
                
        return

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        N = len(matrix)
        for i in range(N): # in-place transpose. temp integer needed. 
            for j in range(i, N):
                temp = matrix[i][j]
                matrix[i][j] = matrix[j][i]
                matrix[j][i] = temp     
        for i in range(N):
            matrix[i] = matrix[i][::-1] # in-place flip 

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        
        for i in range(len(matrix) - 1):
            for j in range(i, len(matrix) - 1 - i):
                matrix[j][-i-1], matrix[-i-1][-j-1], matrix[-j-1][i], matrix[i][j] = matrix[i][j], matrix[j][-i-1], matrix[-i-1][-j-1], matrix[-j-1][i]

46. Permutations
Medium

7160

144

Add to List

Share
Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.

 

Example 1:

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
Example 2:

Input: nums = [0,1]
Output: [[0,1],[1,0]]
Example 3:

Input: nums = [1]
Output: [[1]]
 

Constraints:

1 <= nums.length <= 6
-10 <= nums[i] <= 10
All the integers of nums are unique.

def permute(self, nums):
    return list(itertools.permutations(nums))

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        arr = []
        self.backtracking(nums, 0, arr)
        return arr
        
    def backtracking(self, nums, start, arr):
        if start == len(nums)-1:
            arr.append([*arr])
        else:
            for i in range(start, len(nums)):
                nums[i], nums[start] = nums[start], nums[i]
                self.backtracking(nums, start+1, arr)
                nums[i], nums[start] = nums[start], nums[i]

import itertools 
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        permutations = list(itertools.permutations(nums))
        return permutations

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def insert(nums,element):
            result = [[element] + nums]
            for i in range(len(nums)):
                result.append(nums[0:(i+1)] + [element] + nums[(i+1):len(nums)])
            return result
 
        def permute_n(nums):
            n = len(nums)
            if n == 1:
                return [nums]
            else:
                prev_perm = permute_n(nums[1:n])
                result = []
                element = nums[0]
                for i in prev_perm:
                    result = result + insert(i,element)
                return result
        return permute_n(nums) 

def permute(self, nums: List[int]) -> List[List[int]]:
    res = []
	length = len(nums)
	def perm(nums, inter, res):
		if len(inter) == length: 
			res.append(inter[:])
			return
		for i in range(len(nums)):
			temp = nums[:i]
			temp = temp + nums[i+1:]
			perm(temp, inter+[nums[i]], res)
	perm(nums, [], res)
	return res

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        if len(nums) == 1: 
            res.append(nums)
        else:
            for ele in self.permute(nums[1:]):
                for i in range(len(ele)+1):
                    temp = ele.copy()
                    temp.insert(i,nums[0])
                    res.append(temp)
        return res

78. Subsets
Medium

6778

122

Add to List

Share
Given an integer array nums of unique elements, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.

 

Example 1:

Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
Example 2:

Input: nums = [0]
Output: [[],[0]]
 

Constraints:

1 <= nums.length <= 10
-10 <= nums[i] <= 10
All the numbers of nums are unique.

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        i=0
        arr=[]
        X=[]
        def helper(nums,i,arr,X):
            if i==len(nums):
                X.append(arr[:])
                return
            helper(nums,i+1,arr,X)
            arr.append(nums[i])
            helper(nums,i+1,arr,X)
            arr.pop()
        helper(nums,i,arr,X)
        return X

from itertools import combinations

class Solution:
	def subsets(self, nums: List[int]) -> List[List[int]]:
		lst = []

		for i in range(len(nums) + 1):
			lst += list(combinations(nums, i))

		return lst

class Solution:
    def is_valid_state(self, state, nums): 
        # as long as the nums doesnt duplicates -> done in get_candidates step
        return state <= nums 
    
    def get_candidates(self, state, nums): 
        if not state: # if currently is an empty set 
            return nums 
        # print(curr)
        return set(range(max(state)+1, max(nums)+1)) # limited the occurrence [1,2] -> the no [2,1]
    
    def search(self, state, solutions, nums): 
        if self.is_valid_state(state, nums): 
            solutions.append(state.copy())
        
        for num in self.get_candidates(state, nums): 
            # based on the current set and organize the rest of nums 
            state.add(num)
            self.search(state, solutions, nums)
            state.remove(num)
    
    def subsets(self, nums: List[int]) -> List[List[int]]:
        solutions = [] 
        state = set() 
        self.search(state, solutions, set(nums))
        return solutions 


class Solution:
    def subsets(self, nums: List[int],idx = None) -> List[List[int]]:
        if idx is None:
            idx = len(nums) - 1
            
        if idx < 0:
            return [[]]
        
        ele = nums[idx]
        subset = self.subsets(nums,idx-1)
        for i in range(len(subset)):
            currentSubset = subset[i]
            subset.append(currentSubset+[ele])
            
        return subset

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
            dp = [[] for i in range(len(nums))]
            dp[0].append([])
            dp[0].append([nums[0]])
            for i in range(1, len(nums)):  
                for subset in dp[i-1]: 
                    dp[i].append(subset + [nums[i]])  # with the i'th element
                    dp[i].append(subset)  # without the i'th element
            return dp[len(nums)-1]

class Solution:
    def __init__(self):
        self.all_subset = []
        
    def generate_subsets(self, nums, subset = [], index = 0):
        if index == len(nums):
            self.all_subset.append(subset)
            return
        self.generate_subsets(nums, subset, index+1)
        self.generate_subsets(nums, subset+[nums[index]], index+1)
        
    def subsets(self, nums: List[int]) -> List[List[int]]:
        self.generate_subsets(nums)
        return self.all_subset

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ret = self.subset_helper(nums)
        ret.append([])  # empty list is subset of every list 
        return ret 
    
    def subset_helper(self, nums):
        if len(nums) == 1:
            return [nums]
    
        ret = []
        for i in range(len(nums)):
            ret.append([nums[i]])
            others = nums[(i+1):]
            ret += [[nums[i]] + subset for subset in self.subset_helper(others)]
        return ret 

    def helper(self, nums, path, res):
        res.append(path)
        for i in range(len(nums)):
            self.helper(nums[i+1:], path + [nums[i]], res)
    
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        path = []
        
        self.helper(nums, path, res)
        
        return res

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        
        n = len(nums)
        
        if n == 1:
            return [nums]
        
        res = list(list())
        
        def fact(n):
            if n == 1:
                return 1
            return n * fact(n - 1)
        
        fn = fact(n)
        
        def nextPermutation(nums):
            
            changePos = -1
            changeItem = 0
            
            for i in range(n - 1, 0, -1):
                if nums[i] > nums[i - 1]:
                    changePos = i - 1
                    changeItem = nums[i - 1]
                    break
            
            if changePos == -1:
                return sorted(nums)
            
            first = nums[:changePos]
            second = nums[changePos:]
            
            second.sort()
            
            for i in range(len(second)):
                if second[i] > changeItem:
                    first.append(second[i])
                    second.pop(i)
                    break
            
            return first + second
        
        while (fn):
            nums = nextPermutation(nums)
            res.append(nums)
            fn -= 1
            
        return res

def permute(self, nums):
    	# helper
	def recursive(nums, perm=[], res=[]):
		if not nums: # -- NOTE [1] 
			res.append(perm[::]) #  -- NOTE [2] - append a copy of the perm at the leaf before we start popping/backtracking

		for i in range(len(nums)): # [1,2,3]
			newNums = nums[:i] + nums[i+1:]
			perm.append(nums[i])
			recursive(newNums, perm, res) # - recursive call will make sure I reach the leaf
			perm.pop() # -- NOTE [3] 
		return res

return recursive(nums)

def recursive(nums, perm=[], res=[]):
        
            if not nums: 
                res.append(perm) # --- no need to copy as we are not popping/backtracking. Instead we're passing a new variable each time 

            for i in range(len(nums)): 
                newNums = nums[:i] + nums[i+1:]
                # perm.append(nums[i]) # --- instead of appending to the same variable
                newPerm = perm + [nums[i]] # --- new copy of the data/variable
                recursive(newNums, newPerm, res) 
                # perm.pop()  # --- no need to backtrack
            return res
        
        return recursive(nums)

def recursive(nums):
    	 stack = [(nums, [])]   # -- nums, path (or perms)
	 res = []
	 while stack:
		 nums, path = stack.pop()
		 if not nums:
			 res.append(path)
		 for i in range(len(nums)):   # -- NOTE [4]
			 newNums = nums[:i] + nums[i+1:]
			 stack.append((newNums, path+[nums[i]]))  # --  just like we used to do (path + [node.val]) in tree traversal
	 return res

def recursive(nums):
    	from collections import deque
	q = deque()
	q.append((nums, []))  # -- nums, path (or perms)
	res = []
	while q:
		nums, path = q.popleft()
		if not nums:
			res.append(path)
		for i in range(len(nums)):
			newNums = nums[:i] + nums[i+1:]
			q.append((newNums, path+[nums[i]]))
	return res

def subsets(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    res = []

    def select(c, cur, i):
        if len(cur) == c:
            res.append(list(cur))
            return
        if i >= len(nums):
            return
        cur.append(nums[i])
        select(c, cur, i + 1)
        cur.pop()
        select(c, cur, i + 1)

    for j in range(len(nums) + 1):
        select(j, [], 0)
    return res

class Solution(object):
    def subsets(self, nums):
        ret = []
        self.dfs(nums, [], ret)
        return ret
    
    def dfs(self, nums, path, ret):
        ret.append(path)
        for i in range(len(nums)):
            self.dfs(nums[i+1:], path+[nums[i]], ret)

    def subsets2(self, nums):
        res = []
        nums.sort()
        for i in xrange(1<<len(nums)):
            tmp = []
            for j in xrange(len(nums)):
                if i & 1 << j:  # if i >> j & 1:
                    tmp.append(nums[j])
            res.append(tmp)
        return res

    def subsets(self, nums):
        res = [[]]
        for num in sorted(nums):
            res += [item+[num] for item in res]
        return res

def combinationSum(self, candidates, target):
    res = []
    candidates.sort()
    self.dfs(candidates, target, 0, [], res)
    return res
    
def dfs(self, nums, target, index, path, res):
    if target < 0:
        return  # backtracking
    if target == 0:
        res.append(path)
        return 
    for i in xrange(index, len(nums)):
        self.dfs(nums, target-nums[i], i, path+[nums[i]], res)

def combinationSum2(self, candidates, target):
    res = []
    candidates.sort()
    self.dfs(candidates, target, 0, [], res)
    return res
    
def dfs(self, candidates, target, index, path, res):
    if target < 0:
        return  # backtracking
    if target == 0:
        res.append(path)
        return  # backtracking 
    for i in xrange(index, len(candidates)):
        if i > index and candidates[i] == candidates[i-1]:
            continue
        self.dfs(candidates, target-candidates[i], i+1, path+[candidates[i]], res)

51. N-Queens
Hard

3946

122

Add to List

Share
The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.

Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.

Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space, respectively.

 

Example 1:


Input: n = 4
Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above
Example 2:

Input: n = 1
Output: [["Q"]]

def solveNQueens(self, n):
    res = []
    self.dfs([-1]*n, 0, [], res)
    return res
 
# nums is a one-dimension array, like [1, 3, 0, 2] means
# first queen is placed in column 1, second queen is placed
# in column 3, etc.
def dfs(self, nums, index, path, res):
    if index == len(nums):
        res.append(path)
        return  # backtracking
    for i in xrange(len(nums)):
        nums[index] = i
        if self.valid(nums, index):  # pruning
            tmp = "."*len(nums)
            self.dfs(nums, index+1, path+[tmp[:i]+"Q"+tmp[i+1:]], res)

# check whether nth queen can be placed in that column
def valid(self, nums, n):
    for i in xrange(n):
        if abs(nums[i]-nums[n]) == n -i or nums[i] == nums[n]:
            return False
    return True

class Solution:
    def solveNQueens(self, n: 'int') -> 'List[List[str]]':
        def backtrack(i):
            if i == n:
                res.append(list(board))
            for j in range(n):
                if j not in cols and j-i not in diag and j+i not in off_diag:
                    cols.add(j)
                    diag.add(j-i)
                    off_diag.add(j+i)
                    board.append("."*(j)+"Q"+"."*(n-j-1))
                    backtrack(i+1)
                    board.pop()
                    off_diag.remove(j+i)
                    diag.remove(j-i)
                    cols.remove(j)
        res = []
        board = []
        cols = set()
        diag = set()
        off_diag = set()
        backtrack(0)
        return res

class Solution:
    # @return a list of lists of string
    def solveNQueens(self, n):
        res = []
        stack = []
        for i in range(n):
            stack.append([(0,i)])
        while stack:
            board = stack.pop()
            row = len(board)
            if row == n:
                tmpList = []
                for r,c in board:
                    tmp = []
                    for i in range(n):
                        if i == c:
                            tmp.append('Q')
                        else:
                            tmp.append('.')
                    tmp = ''.join(tmp)
                    tmpList.append(tmp)
                res.append(tmpList)
            for col in range(n):
                tmp = []
                for r, c in board:
                    tmp.append(col != c and abs(row-r) != abs(col-c))
                if all(tmp):
                    stack.append(board+[(row, col)])
        return res

class Solution(object):
    def solveNQueens(self, n):
        """"""
        res, cols, l_diag, r_diag = [], set(), set(), set()
		
        def dfs(r, pos): 
			# r: int, current row to set
			# pos: List[int], previous positioned Queens (at each row)
            if r == n:
                res.append(['.' * i + 'Q' + '.' * (n - i - 1) for i in pos])
                return
            for c in range(n):
                if not c in cols and not r - c in l_diag and not r + c in r_diag:
                    cols.add(c)
                    l_diag.add(r - c)
                    r_diag.add(r + c)
                    dfs(r + 1, pos + [c])
                    cols.remove(c)
                    l_diag.remove(r - c)
                    r_diag.remove(r + c)
            
        dfs(0, [])
        return res

1627. Graph Connectivity With Threshold
Hard

215

20

Add to List

Share
We have n cities labeled from 1 to n. Two different cities with labels x and y are directly connected by a bidirectional road if and only if x and y share a common divisor strictly greater than some threshold. More formally, cities with labels x and y have a road between them if there exists an integer z such that all of the following are true:

x % z == 0,
y % z == 0, and
z > threshold.
Given the two integers, n and threshold, and an array of queries, you must determine for each queries[i] = [ai, bi] if cities ai and bi are connected directly or indirectly. (i.e. there is some path between them).

Return an array answer, where answer.length == queries.length and answer[i] is true if for the ith query, there is a path between ai and bi, or answer[i] is false if there is no path.

 

Example 1:


Input: n = 6, threshold = 2, queries = [[1,4],[2,5],[3,6]]
Output: [false,false,true]
Explanation: The divisors for each number:
1:   1
2:   1, 2
3:   1, 3
4:   1, 2, 4
5:   1, 5
6:   1, 2, 3, 6
Using the underlined divisors above the threshold, only cities 3 and 6 share a common divisor, so they are the
only ones directly connected. The result of each query:
[1,4]   1 is not connected to 4
[2,5]   2 is not connected to 5
[3,6]   3 is connected to 6 through path 3--6
Example 2:


Input: n = 6, threshold = 0, queries = [[4,5],[3,4],[3,2],[2,6],[1,3]]
Output: [true,true,true,true,true]
Explanation: The divisors for each number are the same as the previous example. However, since the threshold is 0,
all divisors can be used. Since all numbers share 1 as a divisor, all cities are connected.
Example 3:


Input: n = 5, threshold = 1, queries = [[4,5],[4,5],[3,2],[2,3],[3,4]]
Output: [false,false,false,false,false]
Explanation: Only cities 2 and 4 share a common divisor 2 which is strictly greater than the threshold 1, so they are the only ones directly connected.
Please notice that there can be multiple queries for the same pair of nodes [x, y], and that the query [x, y] is equivalent to the query [y, x].
 

Constraints:

2 <= n <= 104
0 <= threshold <= n
1 <= queries.length <= 105
queries[i].length == 2
1 <= ai, bi <= cities
ai != bi

class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.size = [1] * n
    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x]) # Path compression
        return self.parent[x]
    def union(self, u, v):
        pu = self.find(u)
        pv = self.find(v)
        if pu == pv: return False
        if self.size[pu] > self.size[pv]: # Union by larger size
            self.size[pu] += self.size[pv]
            self.parent[pv] = pu
        else:
            self.size[pv] += self.size[pu]
            self.parent[pu] = pv
        return True

class Solution(object):
    def areConnected(self, n, threshold, queries):
        uf = UnionFind(n+1)
        for i in range(1, n+1):
            for j in range(i*2, n+1, i): # step by i
                if i > threshold:
                    uf.union(i, j)
        ans = []
        for q in queries:
            pa = uf.find(q[0])
            pb = uf.find(q[1])
            ans.append(pa == pb)
        return ans

def find(par, x):
    if par[x] == x:
        return x
    par[x] = find(par, par[x])
    return par[x]

class Solution:
    def areConnected(self, n: int, threshold: int, queries: List[List[int]]) -> List[bool]:
        if threshold == 0: return [True] * len(queries)
        if threshold >= n / 2: return [False] * len(queries)
		# Initially, each element is in its own single-element group where the element is the parent of that group.
        par = list(range(0, n + 1))
        for d in range(threshold + 1, n//2 + 1):
		    # d already merged to other group (d has smaller divisor)
            if par[d] != d:
                continue
            p1 = par[d]
            for k in range(2*d, n + 1, d): # k = 2*d, 3*d, ...
                p2 = find(par, k)
                if p1 == p2:
                    continue
                if p1 > p2:
                    p1, p2 = p2, p1 # choose the smallest parent as parent for both groups (not necessary)
                par[p2] = p1 # merge groups to single parent

        return [find(par, x) == find(par, y) for x, y in queries]

class Solution:
    def areConnected(self, n: int, threshold: int, queries: List[List[int]]) -> List[bool]:
        def find(x):
            while x in uf:
                while uf[x] in uf:
                    uf[x] = uf[uf[x]]
                x = uf[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px == py: return False
            uf[px] = py
            return True
        
        if not threshold: return [True]*len(queries)
        uf = {}
        for x in range(1,n+1):
            for y in range(2*x,n+1,x):
                if x>threshold: union(x,y)

        return [find(x)==find(y) for x,y in queries]

class Solution:
    def areConnected(self, n: int, threshold: int, queries: List[List[int]]) -> List[bool]:
        disjointSet = [-1] * (n + 1)
        
        for i in range(threshold + 1, n + 1):            
            j = 2
            while i * j <= n:
                u = self.findRoot(disjointSet, i)
                v = self.findRoot(disjointSet, i * j)
                if u > v:
                    disjointSet[u] = v
                
                if u < v:
                    disjointSet[v]= u
                j += 1
        
        return [ self.findRoot(disjointSet, u) == self.findRoot(disjointSet, v) for u, v in queries]

    
    def findRoot(self, disjointSet, i):
        return i if disjointSet[i] == -1 else self.findRoot(disjointSet, disjointSet[i])

121. Best Time to Buy and Sell Stock
Easy

10418

419

Add to List

Share
You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

 

Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
Example 2:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.
 

Constraints:

1 <= prices.length <= 105
0 <= prices[i] <= 104

class Solution(object):
    def maxProfit(self, prices):
        low = float('inf')
        profit = 0
        for i in prices:
            profit = max(profit, i-low)
            low = min(low, i)
        return profit

def maxProfit(self, prices: List[int]) -> int:
    	if not prices:
		return 0

	maxProfit = 0
	minPurchase = prices[0]
	for i in range(1, len(prices)):		
		maxProfit = max(maxProfit, prices[i] - minPurchase)
		minPurchase = min(minPurchase, prices[i])
	return maxProfit

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        
        # It is impossible to have stock to sell on first day, so -infinity is set as initial value
        cur_hold, cur_not_hold = -float('inf'), 0
        
        
        for stock_price in prices:
            
            prev_hold, prev_not_hold = cur_hold, cur_not_hold
            
            # either keep in hold, or just buy today with stock price
            cur_hold = max(prev_hold, -stock_price)
            
            # either keep in not holding, or just sell today with stock price
            cur_not_hold = max(prev_not_hold, prev_hold + stock_price)
            
            
        # max profit must be in not-hold state
        return cur_not_hold if prices else 0

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices: return 0
        sell_prc = max(prices)
        max_prf = 0
        for p in prices:
            if p < sell_prc:
                sell_prc = p
            max_prf = max(p - sell_prc, max_prf)
            
        return max_prf

# Brute force - compare each element, find max diff

def maxProfit(self, p):
    n = len(p)
    max_so_far = 0
    
    for i in range(n):
        for j in range(i+1,n):
            max_so_far = max(max_so_far, p[j] - p[i])
            
    return max_so_far           
# Time: O(n^2)
# Space: O(1)

# Kadane's algorithm
# If the "profit so far" cur becomes negative, start from 0.
# Otherwise add it to current diff p[i] - p[i-1] and update max_so_far (max profit so far).

def maxProfit(self, p):
    n = len(p)
    max_so_far, cur = 0, 0
    
    for i in range(1,n):
        cur = max(cur + p[i] - p[i-1], 0)
        max_so_far = max(cur, max_so_far)
    return max_so_far           
# Time: O(n)
# Space: O(1)

# Comparing with min
# Keep the min_so_far (min element so far) value along with max_so_far (max profit so far).
# Find the diff of p[i] with min_so_far at each step and update max_so_far if it exceeds it.

def maxProfit(self, p):
    n = len(p)
    min_so_far, max_so_far = math.inf, 0
    
    for i in range(n):
        cur = p[i] - min_so_far
        max_so_far = max(cur, max_so_far)
        min_so_far = min(min_so_far, p[i])
    return max_so_far
# Time: O(n)
# Space: O(1)

# Slightly more Pythonic or one-liner solution

def maxProfit(self, p):
        res, min_so_far = 0, math.inf
        
        for p1 in p:
            res, min_so_far = max(res, p1 - min_so_far), min(min_so_far, p1)
        
        return res
# Time: O(n)
# Space: O(1)

1338. Reduce Array Size to The Half
Medium

919

70

Add to List

Share
You are given an integer array arr. You can choose a set of integers and remove all the occurrences of these integers in the array.

Return the minimum size of the set so that at least half of the integers of the array are removed.

 

Example 1:

Input: arr = [3,3,3,3,5,5,5,2,2,7]
Output: 2
Explanation: Choosing {3,7} will make the new array [5,5,5,2,2] which has size 5 (i.e equal to half of the size of the old array).
Possible sets of size 2 are {3,5},{3,2},{5,2}.
Choosing set {2,7} is not possible as it will make the new array [3,3,3,3,5,5,5] which has size greater than half of the size of the old array.
Example 2:

Input: arr = [7,7,7,7,7,7]
Output: 1
Explanation: The only possible set you can choose is {7}. This will make the new array empty.
Example 3:

Input: arr = [1,9]
Output: 1
Example 4:

Input: arr = [1000,1000,3,7]
Output: 1
Example 5:

Input: arr = [1,2,3,4,5,6,7,8,9,10]
Output: 5
 

Constraints:

1 <= arr.length <= 105
arr.length is even.
1 <= arr[i] <= 105

class Solution:
    def minSetSize(self, arr: List[int]) -> int:
        cnt = Counter(arr)
        frequencies = list(cnt.values())
        frequencies.sort()
        
        ans, removed, half = 0, 0, len(arr) // 2
        while removed < half:
            ans += 1
            removed += frequencies.pop()
        return ans

class Solution:
    def minSetSize(self, arr: List[int]) -> int:
        n = len(arr)
        cnt = Counter(arr)

        counting = [0] * (n + 1)
        for freq in cnt.values():
            counting[freq] += 1

        ans, removed, half, freq = 0, 0, n // 2, n
        while removed < half:
            ans += 1
            while counting[freq] == 0: freq -= 1
            removed += freq
            counting[freq] -= 1
        return ans

70. Climbing Stairs
Easy

7959

235

Add to List

Share
You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

 

Example 1:

Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
Example 2:

Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
 

Constraints:

1 <= n <= 45

def climbStairs1(self, n):
    if n == 1:
        return 1
    if n == 2:
        return 2
    return self.climbStairs(n-1)+self.climbStairs(n-2)

def climbStairs2(self, n):
    if n == 1:
        return 1
    res = [0 for i in xrange(n)]
    res[0], res[1] = 1, 2
    for i in xrange(2, n):
        res[i] = res[i-1] + res[i-2]
    return res[-1]

def climbStairs3(self, n):
    if n == 1:
        return 1
    a, b = 1, 2
    for i in xrange(2, n):
        tmp = b
        b = a+b
        a = tmp
    return b

def climbStairs4(self, n):
    if n == 1:
        return 1
    dic = [-1 for i in xrange(n)]
    dic[0], dic[1] = 1, 2
    return self.helper(n-1, dic)

def helper(self, n, dic):
    if dic[n] < 0:
        dic[n] = self.helper(n-1, dic)+self.helper(n-2, dic)
    return dic[n]

def __init__(self):
    self.dic = {1:1, 2:2}
    
def climbStairs(self, n):
    if n not in self.dic:
        self.dic[n] = self.climbStairs(n-1) + self.climbStairs(n-2)
    return self.dic[n]

322. Coin Change
Medium

8252

223

Add to List

Share
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

 

Example 1:

Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
Example 2:

Input: coins = [2], amount = 3
Output: -1
Example 3:

Input: coins = [1], amount = 0
Output: 0
Example 4:

Input: coins = [1], amount = 1
Output: 1
Example 5:

Input: coins = [1], amount = 2
Output: 2
 

Constraints:

1 <= coins.length <= 12
1 <= coins[i] <= 231 - 1
0 <= amount <= 104

class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        if amount == 0:
            return 0
        value1 = [0]
        value2 = []
        nc =  0
        visited = [False]*(amount+1)
        visited[0] = True
        while value1:
            nc += 1
            for v in value1:
                for coin in coins:
                    newval = v + coin
                    if newval == amount:
                        return nc
                    elif newval > amount:
                        continue
                    elif not visited[newval]:
                        visited[newval] = True
                        value2.append(newval)
            value1, value2 = value2, []
        return -1

class Solution:
    """
    @param coins: a list of integer
    @param amount: a total amount of money amount
    @return: the fewest number of coins that you need to make up
    """
    def coinChange(self, coins, amount):
        
        # EDGE CASE
        if amount == 0:
            return 0
        
        # INIT DIMENSIONS
        nrows = len(coins) + 1
        ncols = amount + 1
        
        # BY DEFAULT, 2**64 DENOTES IMPOSSIBLE TO MAKE CHANGE
        dp = [[2**64 for _ in range(ncols)] for _ in range(nrows)]
        
        # BY DEFAULT, IF AMOUNT = 0, WE NEED EXACTLY 0 COINS
        for i in range(nrows):
            dp[i][0] = 0
            
        # OTHER CELLS
        for i in range(1, nrows):
            for j in range(1, ncols):
                
                # CASE 1 - WE MUST LEAVE THE COIN
                if j < coins[i - 1]:
                    dp[i][j] = dp[i - 1][j]
                
                # CASE 2 - WE CAN TAKE OR LEAVE THE COIN
                else:
                    take = 1 + dp[i][j - coins[i - 1]]
                    leave = dp[i - 1][j]
                    dp[i][j] = min(take, leave)
        
        for row in dp:
            print(row)
            
        return -1 if dp[-1][-1] == 2**64 else dp[-1][-1]

class Solution:
      def coinChange(self, coins: List[int], amount: int) -> int:
    coins.sort(reverse = True)
    min_coins = float('inf')
    
    def count_coins(start_coin, coin_count, remaining_amount):
      nonlocal min_coins
      
      if remaining_amount == 0:
        min_coins = min(min_coins, coin_count)
        return
      
      # Iterate from largest coins to smallest coins
      for i in range(start_coin, len(coins)):
        remaining_coin_allowance = min_coins - coin_count
        max_amount_possible = coins[i] * remaining_coin_allowance
        
        if coins[i] <= remaining_amount and remaining_amount < max_amount_possible:
          count_coins(i, coin_count + 1, remaining_amount - coins[i])
      
    count_coins(0, 0, amount)
    return min_coins if min_coins < float('inf') else -1

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        memo = [float("inf")] * (amount+1)
        memo[0] = 0
        for x in range(min(coins), amount+1):
            memo[x] = min([memo[x-c] for c in coins if x-c>=0]) + 1
        v = memo[amount]
        return -1 if v==float("inf") else v

300. Longest Increasing Subsequence
Medium

8798

184

Add to List

Share
Given an integer array nums, return the length of the longest strictly increasing subsequence.

A subsequence is a sequence that can be derived from an array by deleting some or no elements without changing the order of the remaining elements. For example, [3,6,2,7] is a subsequence of the array [0,3,1,6,2,2,7].

 

Example 1:

Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.
Example 2:

Input: nums = [0,1,0,3,2,3]
Output: 4
Example 3:

Input: nums = [7,7,7,7,7,7,7]
Output: 1
 

Constraints:

1 <= nums.length <= 2500
-104 <= nums[i] <= 104
 

Follow up: Can you come up with an algorithm that runs in O(n log(n)) time complexity?

def lengthOfLIS(self, nums):
    tails = [0] * len(nums)
    size = 0
    for x in nums:
        i, j = 0, size
        while i != j:
            m = (i + j) / 2
            if tails[m] < x:
                i = m + 1
            else:
                j = m
        tails[i] = x
        size = max(i + 1, size)
    return size

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n

        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], 1 + dp[j])
                    
        return max(dp)

def lengthOfLIS(self, nums: List[int]) -> int:
    	def max_lis(idx, cur_max):
		if idx == len(nums):
			return 0
		if nums[idx] > cur_max:
			return max(1 + max_lis(idx + 1, nums[idx]), max_lis(idx + 1, cur_max))
		return max_lis(idx + 1, cur_max)
	return max_lis(0, float('-inf'))

from collections import defaultdict
def lengthOfLIS(self, nums: List[int]) -> int:
	cache = defaultdict(dict) # 2D cache of prev_max_idx & cur_idx
	nums.append(float('-inf'))
	def max_lis(idx, prev_max_idx):
		if idx == len(nums) - 1:
			return 0
		if prev_max_idx not in cache or idx not in cache[prev_max_idx]:
			if nums[idx] > nums[prev_max_idx]:
				cache[prev_max_idx][idx] = max(1 + max_lis(idx + 1, idx), max_lis(idx + 1, prev_max_idx))
			else:
				cache[prev_max_idx][idx] = max_lis(idx + 1, prev_max_idx)
		return cache[prev_max_idx][idx]
	return max_lis(0, -1)

def lengthOfLIS(self, nums: List[int]) -> int:
    	if not nums:
		return 0
	dp = [1] * len(nums)
	max_len = 1
	for i in range(1, len(nums)):
		for j in range(0, i):
			if nums[j] < nums[i]:
				dp[i] = max(dp[i], dp[j] + 1)
		max_len = max(max_len, dp[i])
	return max_len

def lengthOfLIS(self, nums: List[int]) -> int:
    	if not nums:
		return 0
	dp = [nums[0]]
	len_dp = 1
	for i in range(1, len(nums)):
		left, right = 0, len(dp) - 1
		while left < right:
			mid = (left + right) // 2
			if dp[mid] < nums[i]:
				left = mid + 1
			else:
				right = mid
		if dp[left] < nums[i]:
			dp.append(nums[i])
			len_dp += 1
		else:
			dp[left] = nums[i]
	return len_dp