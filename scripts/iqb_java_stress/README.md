# Generated stress tests for iqb-java

Plain stdin cannot carry an input large enough to separate O(n^2) from O(n log n): Piston rejects request
bodies over ~100 KB and caps CPU at ~3 s (JVM startup alone is 1.3-2 s). A stress case therefore generates its
input inside the JVM (`generator.java`, class `StressGen`) and the harness output is reduced to a digest;
see `wrap_stress_harness` in `backend/app/testers/llm_coding/piston.py`.

One directory per problem id:

- `meta.json`        name of the case and the statement constraint it stays within (a stress input must
                     never exceed what the statement promises: that would be an unfair test)
- `generator.java`   `class StressGen { static String gen() { ... } }` returning the exact stdin text
- `slow_*.java`      solutions that MUST fail (time out): proves the case really separates algorithms
- `ok_*.java`        reasonable alternatives that MUST pass (e.g. an O(n log n) version)

`python scripts/export_iqb_java.py --validate-stress [problem ids]` runs these against the Reference on Piston
(reference must pass with CPU headroom; slow_* must time out; ok_* must match the reference's digest) and the
normal export writes `problems/<id>/stress.json` only for definitions that validate.

`meta.json` may set `"enabled": false` (with `disabled_reason`): the definition is kept and still validated by
`--validate-stress`, but the export skips it. leetcode-239 and leetcode-53 are disabled because Piston's 3 s CPU
cap leaves them no reliable headroom (their references take 2.0-2.8 s there, varying with host load).

Rejected while authoring (validation caught them): leetcode-217 (a plain double loop is JIT-vectorised and
finishes in 2.4-2.6 s, so it never times out), leetcode-209 (the naive scan ends at 2.9 s: a coin flip),
leetcode-3 (the input alphabet caps the brute force at O(n*95)), and leetcode-56/-901 (their statements only
allow 10^4 items, where the naive versions are legitimately fast).
