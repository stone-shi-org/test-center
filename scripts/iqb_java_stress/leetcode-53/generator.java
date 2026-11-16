class StressGen {
    static String gen() {
        int n = 100000;
        java.util.Random r = new java.util.Random(53);
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < n; i++) { if (i > 0) sb.append(','); sb.append(r.nextInt(19) - 9); }
        return sb.append("]\n").toString();
    }
}
