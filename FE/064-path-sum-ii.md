# 064. 路径总和 II

## 题目

返回所有根到叶子节点和为 targetSum 的路径。

## 最优思路

回溯维护当前路径和剩余值。到叶子且满足条件时复制路径加入答案；回溯时弹出当前节点。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(nh)，空间 O(h)，不计答案。

## JavaScript 题解

```js
function pathSum(root, targetSum) {
  const ans = [], path = [];
  const dfs = (node, rest) => {
    if (!node) return;
    path.push(node.val);
    rest -= node.val;
    if (!node.left && !node.right && rest === 0) ans.push([...path]);
    dfs(node.left, rest);
    dfs(node.right, rest);
    path.pop();
  };
  dfs(root, targetSum);
  return ans;
}
```
