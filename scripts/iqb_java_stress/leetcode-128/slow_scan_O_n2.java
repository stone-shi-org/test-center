class Solution { public int longestConsecutive(int[] nums) { int best = 0;
  for (int x : nums) { boolean hasPrev = false; for (int y : nums) if (y == x - 1) { hasPrev = true; break; } if (hasPrev) continue;
    int len = 1; while (has(nums, x + len)) len++; best = Math.max(best, len); } return best; }
  private static boolean has(int[] a, int v) { for (int y : a) if (y == v) return true; return false; } }
