# Sprite dataset sources (researched 2026-07-04, URLs verified with curl)

Target: 10k-40k permissively-licensed sprites for 32-64px diffusion training.
Core haul below ≈ 19k sprites from ~350MB of downloads.

## 1. Dungeon Crawl Stone Soup tiles (rltiles) — ~7k usable, public domain/CC0

Individual 32x32 PNGs, transparent background. Tiles are public domain
(see `crawl-ref/source/rltiles/license.txt`); the game code is GPL — keep only
the rltiles dir. Usable dirs: `mon/` (1,581), `item/` (1,118), `player/`
(1,177 equipment layers), `dngn/` (2,300), `effect/` (490), `misc/` (344).
Skip `gui/`, `UNUSED/`.

```sh
git clone --depth 1 --filter=blob:none --sparse https://github.com/crawl/crawl
cd crawl && git sparse-checkout set crawl-ref/source/rltiles
```

## 2. Kenney pixel packs — ~6k, CC0

Static zips at `https://kenney.nl/media/pages/assets/<slug>/<hash>-<ts>/kenney_<slug>.zip`
(hash stable until pack update; re-derive: `curl -s https://kenney.nl/assets/<slug> | grep -oE '...zip'`).

Individual-PNG packs:
- tiny-dungeon (136, 16px): https://kenney.nl/media/pages/assets/tiny-dungeon/f8422efb44-1674742415/kenney_tiny-dungeon.zip
- tiny-town (136, 16px): https://kenney.nl/media/pages/assets/tiny-town/a415fbeb49-1735736916/kenney_tiny-town.zip
- tiny-battle (202, 16px): https://kenney.nl/media/pages/assets/tiny-battle/c1c25ac1f3-1691487575/kenney_tiny-battle.zip
- micro-roguelike (326, 8px): https://kenney.nl/media/pages/assets/micro-roguelike/bbcaf50993-1677578239/kenney_micro-roguelike.zip
- pixel-platformer (240, 18px): https://kenney.nl/media/pages/assets/pixel-platformer/33bb4921eb-1696667883/kenney_pixel-platformer.zip
- pixel-shmup (150, 16px): https://kenney.nl/media/pages/assets/pixel-shmup/640246b9cc-1677495782/kenney_pixel-shmup.zip
- roguelike-modern-city (1,040, 16px): https://kenney.nl/media/pages/assets/roguelike-modern-city/0ff3dfff2b-1677694743/kenney_roguelike-modern-city.zip

Spritesheet packs (need slicing):
- 1-bit-pack (colored_packed.png, 16px grid no spacing, 49x22=1,078): https://kenney.nl/media/pages/assets/1-bit-pack/aa867a1f37-1677578516/kenney_1-bit-pack.zip
- roguelike-rpg-pack (16px + 1px spacing, 57x31=1,767): https://kenney.nl/media/pages/assets/roguelike-rpg-pack/12c03cd78b-1677697420/kenney_roguelike-rpg-pack.zip
- roguelike-characters (16px + 1px spacing, ~450 non-empty): https://kenney.nl/media/pages/assets/roguelike-characters/53ffff4133-1729196490/kenney_roguelike-characters.zip
- roguelike-caves-dungeons (16px + 1px spacing, 522 cells): https://kenney.nl/media/pages/assets/roguelike-caves-dungeons/5195ceb8ca-1677694831/kenney_roguelike-caves-dungeons.zip

GitHub mirror of Kenney Asset Pack 1 (2016, vector-style 64-128px, 5,190 files):
https://codeload.github.com/iwenzhou/kenney/zip/refs/heads/master (~330MB)

## 3. DawnLike tileset — ~6k usable, CC-BY 4.0

Credit DragonDePlatino and DawnBringer. 97 spritesheets, exact 16px grid,
no spacing (~10.5k cells, ~5-7k non-empty after dropping blanks + GUI).
Character sheets come in X0/X1 pairs = 2-frame animation (useful for milestone 4).
https://opengameart.org/sites/default/files/DawnLike_5.zip

## 4. OpenGameArt CC0 mirror on HuggingFace — huge, needs filtering (future)

`nyuuzyou/OpenGameArt-CC0`: 2D Art split = 7.3k assets / 29 zips / 22.9GB, mixed
content (HD art, sheets, sprites). Metadata: `2D_Art.jsonl.zst`. Direct:
`https://huggingface.co/datasets/nyuuzyou/OpenGameArt-CC0/resolve/main/2D_Art_00.zip` (…_28.zip)

## 5. Superpowers asset packs — CC0, optional

8 themed 2D packs, mixed individual/sheets, 16-64px:
https://codeload.github.com/sparklinlabs/superpowers-asset-packs/zip/refs/heads/master

## Rejected

- Liberated Pixel Cup / Wesnoth / Flare: CC-BY-SA / GPL (copyleft) — excluded.
- Urizen 1-bit, 0x72 DungeonTileset II, full Ninja Adventure: itch.io-only, no direct URL.
- Pokémon-style fan sets: unlicensed IP.

## Sizing note

At 64px target, integer nearest-neighbor upscales are exact for 8/16/32px
sources (8x/4x/2x); 18px→3x=54 and 24px→2x=48 get centered padding.
