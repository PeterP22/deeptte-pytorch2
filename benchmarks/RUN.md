These artifacts are from GitHub Actions run [39034019065](https://github.com/PeterP22/deeptte-pytorch2/actions/runs/39034019065) of `.github/workflows/benchmark.yml`.

- Hardware: CPU only, on GitHub-hosted `ubuntu-latest` (4 vCPU). The speed job ran on an AMD EPYC 7763.
- Software: torch 2.14.1+cpu, 2 threads.
- `speed-*.json`: before (commit e6395f5) and after were measured in the same job on the same machine, each as the median of 3.
- `train-*`: each variant was trained in its own job. Settings: seed 42, max 60 epochs, Chengdu sample, evaluated on the 1,400-trip test split.
  - The jobs may have landed on different runner hardware, so treat wall-clock seconds as indicative.
- bf16 evaluation: before and after report "bf16 LSTM unsupported on this CPU", because oneDNN has no bf16 LSTM on this runner. after-masked fell back to fp32.
