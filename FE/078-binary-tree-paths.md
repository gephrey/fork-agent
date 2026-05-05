# 078. 二叉树的所有路径

## 题目

返回所有从根到叶子的路径字符串。

## 最优思路

DFS 维护路径数组。到叶子时 join 成字符串加入答案，回溯时弹出节点。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(nh)，空间 O(h)。

## JavaScript 题解

```js
function binaryTreePaths(root) {
  const ans = [], path = [];
  const dfs = n => {
    if (!n) return;
    path.push(n.val);
    if (!n.left && !n.right) ans.push(path.join('- > '));
    dfs(n.left);
    dfs(n.right);
    path.pop();
  };
  dfs(root);
  return ans;
}
```
