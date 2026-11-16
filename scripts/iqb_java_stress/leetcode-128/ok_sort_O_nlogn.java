import java.util.*;
class Solution { public int longestConsecutive(int[] nums) { if (nums.length == 0) return 0; int[] b = nums.clone(); Arrays.sort(b); int best = 1, run = 1;
  for (int i = 1; i < b.length; i++) { if (b[i] == b[i - 1]) continue; if ((long) b[i] - b[i - 1] == 1) run++; else run = 1; best = Math.max(best, run); } return best; } }
