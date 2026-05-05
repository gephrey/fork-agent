# 065. 路径总和 III

## 题目

统计路径和等于 targetSum 的路径数量，路径可从任意节点向下。

## 最优思路

前缀和。当前前缀为 sum，需要之前出现过 sum-target 的前缀。DFS 进入节点加计数，退出节点恢复。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(h)。

## JavaScript 题解

```js
function pathSum(root, targetSum) {
  const map = new Map([[0, 1]]);
  let ans = 0;
  const dfs = (node, sum) => {
    if (!node) return;
    sum += node.val;
    ans += map.get(sum - targetSum) || 0;
    map.set(sum, (map.get(sum) || 0) + 1);
    dfs(node.left, sum);
    dfs(node.right, sum);
    map.set(sum, map.get(sum) - 1);
  };
  dfs(root, 0);
  return ans;
}
```
