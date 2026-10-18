# Public Drone Playground preview

The GitHub Pages site is a static viewer for authentic, previously recorded
Crazyflow/MuJoCo native-method Episodes. It does not start Python, ROS, JAX,
training or simulation on the visitor's machine.

## Public release contents

- Source: historical native acceptance from 2026-10-17 (172 executed method
  cases and 256 recorded Episodes). Source revisions and actual terminal events
  remain in the exported provenance.
- Public selection: 24 method cases, 30 Episodes, covering Tracking, Racing,
  Static OA, Dynamic OA, both EGO and SUPER, Play and fixed-core Benchmark.
- 550 compatible Train/Play/Benchmark commands remain available in the builder.
  A compatible command is not evidence of a successful trained policy.
- Records and required MJCF/mesh assets are copied into the published tree.
  Raw RScope archives, training checkpoints, native workers and detailed reports
  stay on the research server. These are not browser resources.
- Public artifacts carry no real-time or performance claims beyond their
  recorded results. Failures retain the original outcome and terminal time.

## Build and test before publishing

From the isolated integration checkout:

```bash
npm run build --prefix docs/viewer
python3 tools/build_pages_demo.py \
  --output tmp/pages-preview-root/drone_playground \
  --base-path /drone_playground/
python3 -m http.server 4198 --bind 127.0.0.1 \
  --directory tmp/pages-preview-root
npm run test:pages --prefix docs/viewer
```

Use a new empty output directory for each build. The viewer and asset
loader are tested with the actual `/drone_playground/` project-site prefix.
Open `http://127.0.0.1:4198/drone_playground/` to preview.

## Initial publication

GitHub Pages needs a dedicated branch and an enabled publishing source.
The initial release publishes a separate orphan `gh-pages` branch so it does
not overwrite the existing remote `main`, or publish the ignored `results/`
archive. The website source lives on `feat/github-pages-release-20261018`.

The publishing command requires an authenticated GitHub CLI with the necessary
repository permission. It fails before creating a branch if authentication is
missing:

```bash
gh auth login --web -h github.com --git-protocol https --scopes repo
bash tools/publish_pages.sh
```

It builds a fresh release directory, pushes the static `gh-pages` branch
and the website source branch, and requests GitHub Pages to serve the root of
`gh-pages`. If a site or `gh-pages` branch already exists, inspect the
existing state before attempting another deployment.

Expected site: https://sycamore-yu.github.io/drone_playground/
The `/results/command-builder-native-full-20261017/index.html` link under
the Pages prefix lists all publicly selected method outcomes.
