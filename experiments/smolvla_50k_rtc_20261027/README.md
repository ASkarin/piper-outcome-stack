# Archived 50k RTC operator recipe

On October 30 the user accepted both closeout checkpoints and this exact RTC recipe as the fixed stage baseline. The controller `freeze.json` records the original assets; `piper smolvla <original|flat> <skill> <layout>` delegates to the unchanged bundle entry. `--verify-only` checks the frozen files without model loading or hardware I/O. The new C development entry shares the unfiltered recipe, but has its own future scene, data and checkpoint configuration; these B weights are not evidence of C capability.

This directory versions the source of the October 27 operator bundle used for the October 27/29 development trials. It is an experiment recipe, not a new default device launcher. The existing controller bundle at `/home/fff/piper-runs/smolvla-50k-rtc-20261027` remains unchanged.

The archived executor uses native six-joint RTC, ten denoising steps, 25-tick requests, structural action-padding constraints and no joint low-pass filter. The generic acceptance entry in `infra/acceptance` retains its separately recorded five-Hz low-pass behavior. Do not interchange these execution recipes when reproducing results.

`spec.json` identifies the two frozen 050000 checkpoints and runtime settings. Shared source comes from this repository; the controller snapshot records exact runtime sources, interpreter and machine configuration. Model weights, raw data, RGB videos and machine environment remain outside Git. Preparation and run commands require the original configured local/training paths; do not execute historical preparation against an existing run root. Hardware startup and motion remain operator-executed.

The original per-run command/outcome records remain in the controller bundle. The planning repository contains the complete 56-attempt closeout report; its small metric/label summaries are versioned with that report.
