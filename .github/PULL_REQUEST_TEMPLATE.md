## Summary

<!-- Briefly describe what changed and why. -->

## Checklist

- [ ] CPU build passes, all tests pass
- [ ] CUDA build passes (if applicable), all three test binaries pass
- [ ] `bytes_allocated() == 0` after relevant test scope
- [ ] (CUDA only) `compute-sanitizer --tool memcheck` passes for `otter_cuda_tests` and `otter_cross_tests`
- [ ] (CUDA only) `compute-sanitizer --tool racecheck` passes for `otter_cuda_tests` and `otter_cross_tests`
- [ ] Commit message follows `<type>(<scope>): <imperative>` format
- [ ] No raw pointer parameters on public functions
- [ ] No graph wiring inside `forward()` or `backward()`
