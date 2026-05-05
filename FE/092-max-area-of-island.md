# 092. 岛屿的最大面积

## 题目

返回网格中最大的岛屿面积。

## 最优思路

遇到 1 时 DFS 淹没整座岛，并返回面积。用全局最大值记录答案。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(mn)，空间 O(mn) 最坏递归栈。

## JavaScript 题解

```js
function maxAreaOfIsland(grid) {
  const m = grid.length, n = grid[0].length;
  const dfs = (i, j) => {
    if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] !== 1) return 0;
    grid[i][j] = 0;
    return 1 + dfs(i + 1, j) + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i, j - 1);
  };
  let ans = 0;
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) ans = Math.max(ans, dfs(i, j));
  return ans;
}
```
