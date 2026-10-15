# Acceptance evidence

Snapshot: 2026-10-15T14:14:01.800349+00:00

All 18 cells passed: **false**. Cell counts: `{"missing": 11, "passed": 7}`.

Each seed entry: status; benchmark successes/actual CSV episodes; updates; wall seconds.
Missing metrics are null with reasons in the JSON. Mixed Navigation appears twice.

| Algorithm | Condition | Status | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|---|---|
| ppo | tracking/state | passed | passed; empty:100/100; u=500; 120.6 s | passed; empty:100/100; u=600; 167.93 s | passed; empty:100/100; u=1200; 293.47 s |
| ppo | racing/state | passed | passed; racing:100/100; u=1600; 503.01 s | passed; racing:100/100; u=700; 283.73 s | passed; racing:100/100; u=400; 210.83 s |
| ppo | navigation/static/depth | missing | missing | missing | missing |
| ppo | navigation/dynamic/depth | missing | missing | missing | missing |
| ppo | navigation/static/lidar | missing | missing | missing | missing |
| ppo | navigation/dynamic/lidar | missing | missing | missing | missing |
| apg | tracking/state | passed | passed; empty:100/100; u=300; 99.99 s | passed; empty:100/100; u=300; 88.94 s | passed; empty:100/100; u=300; 65.87 s |
| apg | racing/state | passed | passed; racing:100/100; u=400; 133.49 s | passed; racing:100/100; u=400; 123.33 s | passed; racing:100/100; u=300; 98.54 s |
| apg | navigation/static/depth | passed | passed; S01:25/25, S02:25/25, S03:25/25; u=640; 2989.0 s | passed; S01:25/25, S02:25/25, S03:25/25; u=640; 2575.93 s | passed; S01:25/25, S02:25/25, S03:25/25; u=480; 1785.51 s |
| apg | navigation/dynamic/depth | passed | passed; D01:25/25, D02:25/25, D03:25/25; u=640; 2989.0 s | passed; D01:25/25, D02:25/25, D03:25/25; u=640; 2575.93 s | passed; D01:25/25, D02:25/25, D03:25/25; u=480; 1785.51 s |
| apg | navigation/static/lidar | missing | missing | missing | missing |
| apg | navigation/dynamic/lidar | missing | missing | missing | missing |
| shac | tracking/state | passed | passed; empty:100/100; u=600; 137.44 s | passed; empty:100/100; u=300; 79.68 s | passed; empty:100/100; u=400; 95.66 s |
| shac | racing/state | missing | missing | missing | missing |
| shac | navigation/static/depth | missing | missing | missing | missing |
| shac | navigation/dynamic/depth | missing | missing | missing | missing |
| shac | navigation/static/lidar | missing | missing | missing | missing |
| shac | navigation/dynamic/lidar | missing | missing | missing | missing |

Whole GPU samples with concurrent jobs; not isolated process/runtime measurements.

Shared initialization cost (full transitive ancestry, once per source run):
Formal initialized runs report additional training; their costs exclude pretraining. Source costs cover full recorded runs, not inferred checkpoint prefixes. Unknown historical costs remain unavailable, not zero.

None recorded.

Per-scene outcomes (including S06/D06), every checkpoint evaluation, SHA-256, resolved recipes, provenance, missing reasons and costs are in the companion JSON.
Historical hardcoded physical constants are not independently established by these records. Native S6 is separate and not graded here.
