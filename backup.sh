#!/usr/bin/env bash
# Sao lưu toàn bộ repo để có thể xóa bản local mà không mất gì:
#   - mã + lịch sử git            -> GitHub (origin) + file git bundle trên Google Drive
#   - mọi artifact bị gitignore    -> DVC remote trên Google Drive (trừ secrets)
#   - secrets (.env, mcp.json)     -> veremi_build/secrets/ trên Drive (không bao giờ lên GitHub/DVC)
# Cần ổ G: (Google Drive for Desktop) mount ở /mnt/g.
#
# Khôi phục:
#   git clone git@github.com:odixe06/fl_dagsnet_veremi.git   # hoặc: git clone "$DRIVE/fl_dagsnet_veremi.bundle"
#   cd fl_dagsnet_veremi && dvc pull
#   cp -r "$SECRETS"/. .                                      # .env, */.mcp.json, */.vscode/mcp.json
set -euo pipefail
cd "$(dirname "$0")"

DRIVE="/mnt/g/My Drive/NCKH/veremi_build/fl_dagsnet_veremi"
SECRETS="/mnt/g/My Drive/NCKH/veremi_build/secrets/fl_dagsnet_veremi"
DVC=~/miniforge3/envs/nckh/bin/dvc
SECRET_RE='(^|/)(\.env|\.mcp\.json|\.vscode/?|mcp\.json)$'

[ -d "$DRIVE" ] || { echo "Không thấy $DRIVE — mount ổ G: trước: sudo mount /mnt/g" >&2; exit 1; }

# 1. Mọi đường dẫn bị gitignore (trừ secrets, cache) -> DVC; commit riêng các file .dvc thay đổi.
mapfile -t targets < <(git status --ignored --porcelain | sed -n 's/^!! //p' | sed 's|/$||' \
  | grep -vE "$SECRET_RE" | grep -vE '(^|/)(__pycache__|\.pytest_cache|\.ipynb_checkpoints)$|^\.dvc/')
[ ${#targets[@]} -eq 0 ] || "$DVC" add -q "${targets[@]}"
git add -- '*.dvc' '*.gitignore'
git diff --cached --quiet || git commit -q -m "data: update DVC-tracked artifacts" -- '*.dvc' '*.gitignore'

# 2. Dữ liệu -> Drive.
"$DVC" push

# 3. Secrets -> Drive.
git status --ignored --porcelain | sed -n 's/^!! //p' | grep -E "$SECRET_RE" | while read -r p; do
  mkdir -p "$SECRETS/$(dirname "$p")"; cp -r "$p" "$SECRETS/$(dirname "$p")/"
done

# 4. Git -> GitHub + bundle trên Drive (ghi file tạm rồi đổi tên để Drive không sync bản dở).
git push -q origin --all
git push -q origin --tags
git bundle create -q "$DRIVE/fl_dagsnet_veremi.bundle.tmp" --all
mv -f "$DRIVE/fl_dagsnet_veremi.bundle.tmp" "$DRIVE/fl_dagsnet_veremi.bundle"

if [ -n "$(git status --porcelain)" ]; then
  echo "CẢNH BÁO: còn thay đổi chưa commit — chúng CHƯA được sao lưu:" >&2
  git status --short >&2
fi
echo "Xong. Chờ biểu tượng Google Drive báo đồng bộ xong rồi mới xóa bản local."
