# P5 scene references: MIGHTY, SANDO, NavRL, P2M and SUPER

This note records the scene facts used to author the fixed P5 candidates. It
separates repository facts from P5 design choices; P5 does not claim that its
hand-authored layouts reproduce another project's complete benchmark.

## MIGHTY

Pinned local source: `/home/tong/tongworkspace/research_dev/mighty`.

The main world directory contains forest (`easy_forest`, `medium_forest`,
`hard_forest`, `dynamic_forest`, `big_forest`, `big_forest_high_res`,
`easy_high_forest`, `forest0/2/3`, `forest3_with_walls`), structured indoor
(`ACL_office`, `office`, `hospital`), tunnel (`tunnel`, `simple_tunnel`) and
several flight/test worlds. `dynamic_forest.world` has a 110 x 50 m ground
plane. `scripts/run_sim.py` uses an approximately 105 m end-to-end goal for the
ordinary forest family and 305 m for `hard_forest`.

The optional dynamic obstacle launcher places obstacles in a long corridor
(`x=5..105 m`, normally `y=-5..5 m`, `z=1..5 m`) and supplies analytic trefoil
trajectories. MIGHTY therefore motivates P5's long route and D05 trefoil scene,
not P5's fixed obstacle coordinates.

## SANDO

Pinned local source: `/home/tong/tongworkspace/research_dev/sando` at
`3a4450dcc5a8ed5ca825c9da7966e2642be091de`.

The paper-aligned simulation families in the local reproduction are:

- IX-C static forest: Easy/Medium/Hard, approximately 41/81/162 trees and
  5/10/20 percent occupied-area density, route `(0,0,3) -> (105,0,3)`.
- IX-D known dynamic: 50/100/200 obstacles in a 100 x 40 m field, about 65%
  moving 0.8 m cubes plus static cylinders, route `(0,0,2) -> (105,0,2)`.
- IX-E STSFC ablation: planner/corridor ablation on the dynamic benchmark
  family rather than a distinct geometry family.
- IX-F D435 unknown dynamic: the same dynamic-scene class under online D435
  perception/tracking rather than privileged current obstacle position.

The dynamic manifest explicitly places obstacle centres in `x=0..100 m`,
`y=-20..20 m`; planner map limits (`x/y=+-200 m`) are broader safety/planning
bounds and must not be confused with the physical benchmark footprint.

## NavRL

Public source: `Zhefan-Xu/NavRL` at
`3725bcc2e7c1be4ecf1455d922299ae85042603a` (MIT).

Isaac training uses `map_range=[20,20,4.5]`, i.e. a 40 x 40 m generated terrain.
The README's large training example uses 350 static and 80 dynamic obstacles.
The ROS/Gazebo simulator additionally contains named world families:

- bridge: `bridge_static`;
- building: 2-floor and 4-floor static worlds;
- corridor: static and `corridor_dynamic_9`;
- floorplan1: static, dynamic 6, dynamic 16;
- floorplan2: static, dynamic 5, dynamic 12;
- floorplan3: static, dynamic 8, box-static, box-dynamic 4;
- floorplan4: static;
- generated environment: initial/static/dynamic/demo plus matching PCDs;
- room, square, tunnel and simple-box/test families, including several tunnel
  shapes and dynamic variants.

Many structured ROS worlds contain explicit wall collision geometry. P5 uses
this as precedent for making its benchmark boundary visible to sensors rather
than relying only on a hidden termination rule.

## P2M

Public source: `arclab-hku/P2M` at
`6aa1f7cf464a158b1464fad314a77ad23ff2bcf5` (MIT).

P2M exposes a parameterized dynamic-clutter family rather than many named
world files. The default Isaac training config uses 40 dynamic obstacles;
static terrain is a 6 x 6 terrain grid with two discrete obstacles per grid
cell by default. Dynamic obstacle origins cover approximately `[-18,18]^2`.
Crucially, the training environment explicitly creates four walls around a
20 m square (`generate_wall_tensor`) and also measures excessive lateral
deviation from the start-goal line (`get_bound_misbehave`, threshold 8 m).

The ROS testing example (`map_generator/launch/sim_test.launch`) is a
10 x 25 x 5 m map with 13 static and 13 moving obstacles by default; obstacle
counts are the intended density control.

This is direct precedent for P5 v3's visible physical boundary walls.

## SUPER

Pinned source: `hku-mars/SUPER` at
`2ad3419c127a617c6d7df6925e81a14175a9c096`.

The standard planner configurations are `click_smooth`, `static_dense` and
`static_high_speed` (plus ROS1/ROS2 variants of click_smooth). Their simulator
counterparts load PCD maps:

| planner mode | simulator config | PCD | measured PCD bounding box | ROG-Map size |
|---|---|---|---|---|
| click_smooth | `click.yaml` | `random_map_150.pcd` | about 50 x 50 x 5.10 m | 50 x 50 x 6 m |
| static_dense | `dense.yaml` | `random_map_2_26609.pcd` | about 15 x 110 x 4.05 m | 15 x 110 x 6 m |
| static_high_speed | `high_speed.yaml` | `random_map_24_6635.pcd` | about 15 x 109.9 x 4.03 m | 15 x 110 x 6 m |

`random_map_50.pcd` (about 50 x 50 x 5.10 m) is also shipped, but is not wired
to one of the three simulator YAMLs above in the pinned tree.

The important interface detail is that SUPER does **not** hand the global PCD
directly to the planner during ordinary simulated perception. `perfect_drone_sim`
loads the PCD as the simulated world, renders/simulates LiDAR, and SUPER receives
the resulting online world-frame point cloud (`/cloud_registered`) to update
ROG-Map. `static_dense.yaml` itself has `load_pcd_en: false` in ROG-Map.

An exact P5 reproduction should therefore import the upstream PCD into a common
static scene/raycast backend and generate MID360 measurements from it. Feeding
the source PCD straight to SUPER would leak the full map and would not reproduce
the upstream sensing boundary.

## P5 fixed-scene policy after user review

The first 20 x 10 m candidate and the later low-density 100 x 40 m candidate
are historical review artifacts. Candidate v3 is 100 x 40 m with `z=0.5..6 m`,
96 m nominal start-goal distance, explicit field obstacle counts of 50/100/150
for Easy/Medium/Hard, and four shared physical boundary walls. The wall geometry
is part of the same scene bank used by depth/LiDAR, collision and RScope.

The catalog still includes an offline `inspection_path` only to prove that a
candidate is traversable. That path is never part of a policy or planner input.
