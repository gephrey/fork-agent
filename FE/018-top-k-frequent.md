# 018. 前 K 个高频元素

## 题目

返回数组中出现频率最高的 k 个元素。

## 最优思路

统计频次后做桶排序。频次最大不超过 n，把数字放入对应频次桶，从高频桶往低频桶收集 k 个。

## 关键点

- 先明确状态/指针/边界含义，避免靠试错写分支。
- 面试中要主动说明为什么该做法不会漏解或重复。
- 写完后用空输入、单元素、重复值、边界值各过一遍。

## 复杂度

时间 O(n)，空间 O(n)。

## JavaScript 题解

```js
function topKFrequent(nums, k) {
  const cnt = new Map();
  for (const x of nums) cnt.set(x, (cnt.get(x) || 0) + 1);
  const buckets = Array.from({
    length: nums.length + 1
  }, () => []);
  for (const[num, c] of cnt) buckets[c].push(num);
  const ans = [];
  for (let i = buckets.length - 1; i >= 0 && ans.length < k; i--) ans.push(...buckets[i]);
  return ans.slice(0, k);
}
```
