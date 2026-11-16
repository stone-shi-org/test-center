import java.util.*;
class Solution { public int[] maxSlidingWindow(int[] nums, int k) { int[] r = new int[nums.length - k + 1];
  for (int i = 0; i + k <= nums.length; i++) r[i] = Arrays.stream(nums, i, i + k).max().getAsInt(); return r; } }
