# 083. 子集

## 题目

返回数组所有子集。

## 最优思路

回溯枚举每个位置选或不选。更常见写法是每到一个节点先加入当前路径，再继续选择后续元素。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n*2^n)，空间 O(n)。

## JavaScript 题解

```js
function subsets(nums) {
  const ans = [], path = [];
  const dfs = start => {
    ans.push([...path]);
    for (let i = start;i < nums.length;i++) {
      path.push(nums[i]);
      dfs(i + 1);
      path.pop();
    }
  };
  dfs(0);
  return ans;
}
```
