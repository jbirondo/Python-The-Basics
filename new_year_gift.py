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