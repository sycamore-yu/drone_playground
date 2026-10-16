# Scene assets

Simulation loads MJCF/XML geometry and evaluates moving obstacles at simulation
time. Fixed OA maps are reserved for Benchmark; training samples procedural maps.

| Task | Assets | Use |
|---|---|---|
| Tracking | Empty scene | Hover, figure-eight, and configured references |
| Racing | `racing/lsy_level0.xml`, `lsy_native.yaml` | Level0 gate order 1, 2, 3, 4, 2 |
| Static OA | S01, S02, S03, S06 | Fixed static benchmark geometry |
| Dynamic OA | D01, D02, D03, D06 | Fixed moving-obstacle benchmark geometry |

Navigation bounds are `[0,-20,0.5]` to `[100,20,6]` m, with nominal start
`(2,0,3)` and goal `(98,0,3)`. `navigation/catalog.xml` records origin and geometry
metadata; `boundary.xml` defines the physical walls.

## Motion

An obstacle's `body.user` contains a motion type and five parameters:
`0=static`, `1=trefoil`, `2=linear_bounce`. `body.pos` is the rest position or
motion center. Trefoil parameters are `(sx,sy,sz,offset,slower)`:

```text
u = 2*t/slower + offset
dx = sx/6 * (sin(u) + 2*sin(2*u))
dy = sy/5 * (cos(u) - 2*cos(2*u))
dz = -sz/2 * sin(3*u)
```

Linear-bounce parameters are `(ax,ay,az,period,phase)`:

```text
u = t/period + phase
triangle = 2*abs(2*(u-floor(u+0.5))) - 1
position = origin + (ax,ay,az)*triangle
```

D01/D02/D03 use trefoil motion; D06 includes linear bounce. Collision, depth,
LiDAR, and replay use the same geometry pose at each timestamp.

## Attribution

Navigation geometry derives from [SANDO](https://github.com/mit-acl/sando)
construction rules at `3a4450dcc5a8ed5ca825c9da7966e2642be091de`, with this project's
bounds, geometry mixtures, and seeds. Each map has an independent geometry identity.
Racing uses [LSY Drone Racing](https://github.com/learnsyslab/lsy_drone_racing)
Level0 at `b1f5b36adb8e08e8e2adea85de790bd0e0a1d118`; its textures and license remain
in `racing/`. Installed packages load these assets through package resources.
