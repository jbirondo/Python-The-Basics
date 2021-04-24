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