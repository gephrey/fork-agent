# 091. 岛屿数量

## 题目

统计网格中由 1 组成的岛屿数量。

## 最优思路

遍历网格，遇到 1 就答案加一，并用 DFS/BFS 把整座岛标记为 0，避免重复统计。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(mn)，空间 O(mn) 最坏递归栈。

## JavaScript 题解

```js
function numIslands(grid) {
  const m = grid.length, n = grid[0].length;
  const dfs = (i, j) => {
    if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] !== '1') return;
    grid[i][j] = '0';
    dfs(i + 1, j);
    dfs(i - 1, j);
    dfs(i, j + 1);
    dfs(i, j - 1);
  };
  let ans = 0;
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
    if (grid[i][j] === '1') {
      ans++;
      dfs(i, j);
    }
  }
  return ans;
}
```
