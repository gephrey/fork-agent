# 100. 零钱兑换

## 题目

给定硬币面额和金额，求凑成金额的最少硬币数，不能凑成返回 -1。

## 最优思路

完全背包 DP。dp[i] 表示凑成 i 的最少硬币数，枚举金额和硬币更新 dp[i]。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(amount * coins.length)，空间 O(amount)。

## JavaScript 题解

```js
function coinChange(coins, amount) {
  const dp = Array(amount + 1).fill(Infinity);
  dp[0] = 0;
  for (let i = 1;i <= amount;i++) for (const c of coins) if (i >= c) dp[i] = Math.min(dp[i], dp[i-c] + 1);
  return dp[amount] === Infinity ? -1: dp[amount];
}
```
