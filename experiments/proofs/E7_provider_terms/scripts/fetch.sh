#!/usr/bin/env bash
# Fetch a URL with curl, save raw body to raw/<name>, append a line to fetch_log.tsv:
# retrieved_utc \t name \t http_code \t final_url \t size_bytes \t content_type \t requested_url
# Usage: scripts/fetch.sh <name> <url>
set -u
cd "$(dirname "$0")/.." || exit 1
name="$1"; url="$2"
ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
meta=$(curl -sL --max-time 60 -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36" \
  -o "raw/${name}" -w '%{http_code}\t%{url_effective}\t%{size_download}\t%{content_type}' "$url")
rc=$?
printf '%s\t%s\t%s\t%s\tcurl_rc=%s\n' "$ts" "$name" "$meta" "$url" "$rc" >> fetch_log.tsv
printf '%s\t%s\t%s\tcurl_rc=%s\n' "$ts" "$name" "$meta" "$rc"
