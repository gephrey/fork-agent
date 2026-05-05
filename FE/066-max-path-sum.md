# 066. 二叉树中的最大路径和

## 题目

求二叉树任意非空路径的最大节点和。

## 最优思路

DFS 返回从当前节点向父节点延伸的最大贡献，只能选一边；全局答案用 left+node+right 更新。负贡献直接舍弃。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(h)。

## JavaScript 题解

```js
function maxPathSum(root) {
  let ans = -Infinity;
  const gain = node => {
    if (!node) return 0;
    const l = Math.max(0, gain(node.left)), r = Math.max(0, gain(node.right));
    ans = Math.max(ans, node.val + l + r);
    return node.val + Math.max(l, r);
  };
  gain(root);
  return ans;
}
```
