#!/usr/bin/env bash
# Download the core sprite haul (see docs/sprite-sources.md). Idempotent-ish:
# skips anything whose target dir already exists.
set -e
cd "$(dirname "$0")/../data/raw"

if [ ! -d crawl ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/crawl/crawl
  (cd crawl && git sparse-checkout set crawl-ref/source/rltiles)
fi

mkdir -p kenney
kenney_urls="
tiny-dungeon https://kenney.nl/media/pages/assets/tiny-dungeon/f8422efb44-1674742415/kenney_tiny-dungeon.zip
tiny-town https://kenney.nl/media/pages/assets/tiny-town/a415fbeb49-1735736916/kenney_tiny-town.zip
tiny-battle https://kenney.nl/media/pages/assets/tiny-battle/c1c25ac1f3-1691487575/kenney_tiny-battle.zip
micro-roguelike https://kenney.nl/media/pages/assets/micro-roguelike/bbcaf50993-1677578239/kenney_micro-roguelike.zip
pixel-platformer https://kenney.nl/media/pages/assets/pixel-platformer/33bb4921eb-1696667883/kenney_pixel-platformer.zip
pixel-shmup https://kenney.nl/media/pages/assets/pixel-shmup/640246b9cc-1677495782/kenney_pixel-shmup.zip
roguelike-modern-city https://kenney.nl/media/pages/assets/roguelike-modern-city/0ff3dfff2b-1677694743/kenney_roguelike-modern-city.zip
1-bit-pack https://kenney.nl/media/pages/assets/1-bit-pack/aa867a1f37-1677578516/kenney_1-bit-pack.zip
roguelike-rpg-pack https://kenney.nl/media/pages/assets/roguelike-rpg-pack/12c03cd78b-1677697420/kenney_roguelike-rpg-pack.zip
roguelike-characters https://kenney.nl/media/pages/assets/roguelike-characters/53ffff4133-1729196490/kenney_roguelike-characters.zip
roguelike-caves-dungeons https://kenney.nl/media/pages/assets/roguelike-caves-dungeons/5195ceb8ca-1677694831/kenney_roguelike-caves-dungeons.zip
"
echo "$kenney_urls" | while read -r slug url; do
  [ -z "$slug" ] && continue
  if [ ! -d "kenney/$slug" ]; then
    curl -sL -o "kenney/$slug.zip" "$url"
    unzip -q -o "kenney/$slug.zip" -d "kenney/$slug"
    echo "kenney/$slug: $(find "kenney/$slug" -name '*.png' | wc -l) pngs"
  fi
done

if [ ! -d dawnlike ]; then
  curl -sL -o dawnlike.zip "https://opengameart.org/sites/default/files/DawnLike_5.zip"
  unzip -q -o dawnlike.zip -d dawnlike
  echo "dawnlike: $(find dawnlike -name '*.png' | wc -l) sheets"
fi

echo "ALL DOWNLOADS DONE"
