# 090. 单词搜索

## 题目

判断二维网格中是否存在一条相邻路径组成 word。

## 最优思路

DFS 回溯。从每个匹配首字母的格子出发，向四方向搜索；访问过的位置临时标记，回溯恢复。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(mn*3^L)，空间 O(L)。

## JavaScript 题解

```js
function exist(board, word) {
  const m = board.length, n = board[0].length;
  const dfs = (i, j, k) => {
    if (k === word.length) return true;
    if (i < 0 || i >= m || j < 0 || j >= n || board[i][j] !== word[k]) return false;
    const t = board[i][j];
    board[i][j] = '#';
    const ok = dfs(i + 1, j, k + 1) || dfs(i - 1, j, k + 1) || dfs(i, j + 1, k + 1) || dfs(i, j - 1, k + 1);
    board[i][j] = t;
    return ok;
  };
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) if (dfs(i, j, 0)) return true;
  return false;
}
```
