class StressGen {
    static String gen() {
        int n = 100000; int[] a = new int[n]; java.util.Arrays.fill(a, 1);
        java.util.Random r = new java.util.Random(238);
        for (int t = 0; t < 20; t++) a[r.nextInt(n)] = 2;
        for (int t = 0; t < 200; t++) { int i = r.nextInt(n); if (a[i] == 1) a[i] = -1; }
        a[r.nextInt(n)] = 0;
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < n; i++) { if (i > 0) sb.append(','); sb.append(a[i]); }
        return sb.append("]\n").toString();
    }
}
