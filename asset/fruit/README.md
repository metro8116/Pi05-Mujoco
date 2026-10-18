# Rattan basket

`fruit_basket.xml` now uses UV-mapped woven rattan meshes, a wrapped continuous
rim, a woven floor, and a tileable fibre colour texture. These are local assets;
no downloads or additional runtime packages are required.

The 113 original basket contact geoms, their names, dimensions, poses, friction,
and the fixed basket body are preserved. Contact geoms are in hidden group 3;
the four new visual geoms are in visible group 2 with `contype="0"`,
`conaffinity="0"`, and `density="0"`. The original 0.105 m inner contact radius,
0.010 m floor top and 0.096 m rim top (relative to the basket body) still apply.
Do not enable group 3 for normal images: it displays the collision proxies.

Regenerate the checked-in meshes and texture with NumPy and Pillow:

```sh
python tools/generate_basket_visual.py
```

Render two close views using the actual scene lighting, without opening a window:

```sh
python tools/preview_basket.py --output media/basket_after.png
```

MuJoCo's standard renderer uses the colour texture and geometry for this asset;
the fine shadows and interlacing come from real strands rather than a normal map.

## Apple, mug, and table materials

The apple is reconstructed as a closed, smooth, outward-facing mesh from the
original scan, with a nominal 1.08 scale (about 6.3 cm wide). The old grip-band
ridge and non-manifold patch faces are removed. The original scanned colour is
reprojected into continuous spherical UVs; invalid black atlas samples on the
skin are filled locally. Visual and collision geometry share the new surface.
Mass stays 55 g; the inertia and centre of mass scale with the size. The
environment measures the new mesh bottom automatically for spawning.

```sh
python tools/rebuild_apple_visual.py
```

The green mug uses its original geometry and contact shapes, with new UVs and a
muted green glaze colour texture. The rim and foot have subtle glaze pooling;
surface speckles and low-amplitude colour variation are intentionally small.

```sh
python tools/generate_mug_material.py
```

The table retains its photographic wood texture and physics, with a brighter
neutral oak tint and restrained satin highlights. Key/fill lighting and a
separate overhead shadow light in `floor_isaac_style.xml` reveal the materials
without the original harsh oblique self-shadows.

```sh
python tools/preview_materials.py --output media/materials_after.png
python tools/verify_grasp.py --headless --object apple
```

The reconstruction generator requires SciPy in addition to NumPy and Pillow.
Only the checked-in assets are required at runtime.
