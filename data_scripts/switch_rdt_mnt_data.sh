#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/personal/xiangpc/0812_Xpolicylab_bench/RDT-1B
ADAPTER="$WORKSPACE/Xpolicylab/policy/RDT_1B"
SETTING=Spark0_bench-cotrain-tianji_marvin_wuji-joint
STAGE="$WORKSPACE/data/mnt20260812_rdt_joint54"
BACKUP="$WORKSPACE/data/backup_20260812_pre_mnt"
LIVE_DATA="$ADAPTER/data/$SETTING"
LIVE_LANG="$ADAPTER/lang_embeds/$SETTING"
LIVE_EMPTY="$ADAPTER/lang_embeds/empty_lang_embed.pt"
STAGED_LANG="$STAGE/lang_embeds/$SETTING"
STAGED_EMPTY="$STAGE/lang_embeds/empty_lang_embed.pt"

test -d /mnt/xspark-data/tjy/spark0_bench
test -d "$STAGED_LANG"
test -f "$STAGED_EMPTY"
test "$(find "$STAGED_LANG" -name lang_embed.pt | wc -l)" -eq 6
mkdir -p "$BACKUP" "$ADAPTER/data" "$ADAPTER/lang_embeds"

if [[ ! -e "$BACKUP/data_link" && ! -L "$BACKUP/data_link" ]]; then
  cp -a "$LIVE_DATA" "$BACKUP/data_link"
fi
tmp_data="$ADAPTER/data/.${SETTING}.mnt-tmp"
ln -sfn /mnt/xspark-data/tjy/spark0_bench "$tmp_data"
mv -Tf "$tmp_data" "$LIVE_DATA"

if [[ ! -e "$BACKUP/lang_embeds" && ! -L "$BACKUP/lang_embeds" ]]; then
  mv "$LIVE_LANG" "$BACKUP/lang_embeds"
fi
tmp_lang="$ADAPTER/lang_embeds/.${SETTING}.mnt-tmp"
ln -sfn "$STAGED_LANG" "$tmp_lang"
mv -Tf "$tmp_lang" "$LIVE_LANG"

if [[ ! -e "$BACKUP/empty_lang_embed.pt" && ! -L "$BACKUP/empty_lang_embed.pt" ]]; then
  cp -a "$LIVE_EMPTY" "$BACKUP/empty_lang_embed.pt"
fi
tmp_empty="$ADAPTER/lang_embeds/.empty_lang_embed.pt.mnt-tmp"
ln -sfn "$STAGED_EMPTY" "$tmp_empty"
mv -Tf "$tmp_empty" "$LIVE_EMPTY"

test "$(readlink -f "$LIVE_DATA")" = /mnt/xspark-data/tjy/spark0_bench
test "$(readlink -f "$LIVE_LANG")" = "$(readlink -f "$STAGED_LANG")"
test "$(readlink -f "$LIVE_EMPTY")" = "$(readlink -f "$STAGED_EMPTY")"
echo "RDT_MNT_DATA_SWITCH_OK data=$(readlink -f "$LIVE_DATA") lang=$(readlink -f "$LIVE_LANG")"
