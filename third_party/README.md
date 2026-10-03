# Third-party source pins

`sources.json` records the upstream repository/revision used by reproducible local
source caches and records provenance for reconstructed or planned methods.

`patches/` contains only repository-wide patches applied to pinned Python/JAX
dependencies by `scripts/tools/setup.py`. Patch filenames describe their purpose,
and their content is verified against the pinned source before a cache is accepted.

Deployment-specific patches stay with the deployment that owns them. In particular,
EGO-Planner and SUPER patches live in `native/ros1/patches/` because they are
applied while building that isolated ROS image, not while installing the Python
package.
