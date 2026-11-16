class StressGen {
    static String gen() {
        int n = 100000; int[] a = new int[n];
        java.util.Random r = new java.util.Random(128);
        for (int i = 0; i < n; i++) a[i] = i < 60000 ? 5000000 + i : r.nextInt(2000000001) - 1000000000;
        for (int i = n - 1; i > 0; i--) { int j = r.nextInt(i + 1); int t = a[i]; a[i] = a[j]; a[j] = t; }
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < n; i++) { if (i > 0) sb.append(','); sb.append(a[i]); }
        return sb.append("]\n").toString();
    }
}
