# Performance reports

This directory stores reproducible benchmark methodology and recorded results. The executable workloads live in `tests/benchmarks/` and are built as dedicated CMake targets; they are not part of the pass/fail Google Test suite.

For each implementation version, add a report under its implementation subfolder that records:

- the commit hash, date, CPU, memory configuration, macOS version, and compiler version;
- the CMake configuration and benchmark command;
- at least five runs of every workload, plus the median result;
- a short interpretation of any measured change.

Run the current suites with:

```bash
./build-nested/run_benchmark
./build-nested/run_training_benchmark
```
