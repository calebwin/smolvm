#!/bin/sh
# End-to-end tests for remote volumes (-v s3://... and raw rclone remotes).
#
# Fully self-contained: runs a MinIO S3 server inside a smolvm machine and
# verifies every write OUT-OF-BAND from a separate machine via the raw S3 API
# (the vfs write cache survives on the persistent overlay, so reading back
# through the same mount proves nothing).
#
# Usage: ./tests/test_remote_volumes.sh [path-to-smolvm]
set -u
S=${1:-smolvm}
PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); echo "PASS: $1"; }
bad()  { FAIL=$((FAIL+1)); echo "FAIL: $1"; }
check() { if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (want '$3' got '$2')"; fi; }

CREDS="--env AWS_ACCESS_KEY_ID=smoltest --env AWS_SECRET_ACCESS_KEY=smoltest123 --env AWS_ENDPOINT_URL=http://100.96.0.1:9000"
RCLONE_REMOTE=':s3,provider=Minio,access_key_id=smoltest,secret_access_key=smoltest123,endpoint="http://100.96.0.1:9000"'
NAMES="rv-minio rv-ver rv-rw rv-ro rv-notools rv-badremote rv-sf rv-api rv-svc"
cleanup() { for n in $NAMES; do $S machine delete --name "$n" --force >/dev/null 2>&1; done; }
cleanup

# ---- rig: MinIO + out-of-band verifier ------------------------------------
$S machine create --name rv-minio --image minio/minio --net -p 9000:9000 \
  --env MINIO_ROOT_USER=smoltest --env MINIO_ROOT_PASSWORD=smoltest123 \
  -- minio server /data --address :9000 >/dev/null 2>&1
$S machine start --name rv-minio >/dev/null 2>&1
i=0; until curl -s -m 2 http://127.0.0.1:9000/minio/health/live >/dev/null 2>&1; do
  i=$((i+1)); [ $i -gt 30 ] && { echo "ABORT: minio never came up"; cleanup; exit 1; }; sleep 2
done
$S machine create --name rv-ver --image alpine:latest --net --net-backend virtio-net >/dev/null 2>&1
$S machine start --name rv-ver >/dev/null 2>&1
$S machine exec --name rv-ver -- sh -c "apk add -q rclone >/dev/null 2>&1; rclone mkdir '$RCLONE_REMOTE:rv-bucket' 2>/dev/null" >/dev/null 2>&1
# Read an object back OUT-OF-BAND, polling up to ~20s: rclone's vfs cache
# uploads writes asynchronously, so a fixed post-write sleep races the flush
# under load. Poll until the object lands (or give up and return what we have).
oob() {
  _v=""
  i=0; while [ $i -lt 20 ]; do
    _v=$($S machine exec --name rv-ver -- sh -c "rclone cat '$RCLONE_REMOTE:rv-bucket/$1' 2>/dev/null")
    [ -n "$_v" ] && break
    i=$((i+1)); sleep 1
  done
  echo "$_v"
}

# ---- 1. parse rejections happen at create, with hints ---------------------
OUT=$($S machine create --name rv-x --image alpine:latest --net -v "s3://b:relative" -- sleep infinity 2>&1)
echo "$OUT" | grep -q "must be absolute" && ok "relative guest path rejected" || bad "relative guest path rejected"
OUT=$($S machine create --name rv-x --image alpine:latest --net -v ':http,url="https://h":/mnt/x' -- sleep infinity 2>&1)
echo "$OUT" | grep -q "::" && ok "missing remote path colon rejected with :: hint" || bad "missing remote path colon rejected with :: hint"
OUT=$($S machine create --name rv-x --image alpine:latest -v "s3://b:/mnt/x" -- sleep infinity 2>&1)
echo "$OUT" | grep -q "network" && ok "remote volume without network rejected" || bad "remote volume without network rejected"

# ---- 2. missing tools fails the START with the actionable message ---------
$S machine create --name rv-notools --image alpine:latest --net --net-backend virtio-net $CREDS \
  -v "s3://rv-bucket:/mnt/q" -- sleep infinity >/dev/null 2>&1
OUT=$($S machine start --name rv-notools 2>&1)
echo "$OUT" | grep -q "rclone and fuse3" && ok "missing tools fails start with hint" || bad "missing tools fails start with hint"

# ---- 3. a dead mount (garbage backend) fails the START --------------------
$S machine create --name rv-badremote --image alpine:latest --net --net-backend virtio-net \
  --init "apk add -q rclone fuse3" -v ":nosuchbackend,x=1::/mnt/q" -- sleep infinity >/dev/null 2>&1
OUT=$($S machine start --name rv-badremote 2>&1)
echo "$OUT" | grep -q "failed to mount" && ok "dead mount fails start with log pointer" || bad "dead mount fails start with log pointer"

# ---- 4. rw round trip, out-of-band verified -------------------------------
$S machine create --name rv-rw --image alpine:latest --net --net-backend virtio-net $CREDS \
  --init "apk add -q rclone fuse3" -v "s3://rv-bucket:/mnt/rw" -- sleep infinity >/dev/null 2>&1
$S machine start --name rv-rw >/dev/null 2>&1
$S machine exec --name rv-rw -- sh -c 'echo rt-1 > /mnt/rw/rt.txt && sync && sleep 5' >/dev/null 2>&1
check "rw write visible out-of-band" "$(oob rt.txt)" "rt-1"

# ---- 5. restart: mount returns, content intact, appends flush -------------
$S machine stop --name rv-rw >/dev/null 2>&1
$S machine start --name rv-rw >/dev/null 2>&1
GOT=$($S machine exec --name rv-rw -- sh -c 'cat /mnt/rw/rt.txt 2>/dev/null' 2>/dev/null)
check "mount returns after restart with content" "$GOT" "rt-1"

# ---- 6. large-file integrity ----------------------------------------------
H1=$($S machine exec --name rv-rw -- sh -c 'dd if=/dev/urandom of=/tmp/b bs=1M count=8 2>/dev/null; sha256sum /tmp/b | cut -d" " -f1; cp /tmp/b /mnt/rw/b.bin && sync && sleep 8' 2>/dev/null | tail -1)
H2=$($S machine exec --name rv-ver -- sh -c "rclone cat '$RCLONE_REMOTE:rv-bucket/b.bin' 2>/dev/null | sha256sum | cut -d' ' -f1" 2>/dev/null | tail -1)
check "8MB sha256 integrity out-of-band" "$H2" "$H1"

# ---- 7. read-only enforcement ---------------------------------------------
$S machine create --name rv-ro --image alpine:latest --net --net-backend virtio-net $CREDS \
  --init "apk add -q rclone fuse3" -v "s3://rv-bucket:/mnt/q:ro" -- sleep infinity >/dev/null 2>&1
$S machine start --name rv-ro >/dev/null 2>&1
# Assert only once the ro rclone mount is actually live in the container ns —
# before that a write lands in the overlay dir and would falsely "succeed".
OUT=""
n=0; while [ $n -lt 12 ]; do
  if $S machine exec --name rv-ro -- sh -c 'grep -q " /mnt/q " /proc/mounts' 2>/dev/null; then
    OUT=$($S machine exec --name rv-ro -- sh -c 'echo x > /mnt/q/no.txt 2>&1 || true' 2>/dev/null)
    break
  fi
  n=$((n+1)); sleep 1
done
echo "$OUT" | grep -qi "read-only" && ok "ro mount rejects writes" || bad "ro mount rejects writes"

# ---- 8. fork of a remote-volume golden is refused cleanly -----------------
OUT=$($S machine fork --golden rv-ro --name rv-clone 2>&1)
echo "$OUT" | grep -q "cannot be forked yet" && ok "fork refused for remote-volume golden" || bad "fork refused for remote-volume golden"

# ---- 9. Smolfile surface: volumes = ["s3://..."] flows through create -----
TMPD=$(mktemp -d)
cat > "$TMPD/Smolfile" <<'SMOLEOF'
image = "alpine:latest"
net = true
cmd = ["sleep", "infinity"]
env = [
  "AWS_ACCESS_KEY_ID=smoltest",
  "AWS_SECRET_ACCESS_KEY=smoltest123",
  "AWS_ENDPOINT_URL=http://100.96.0.1:9000",
]
volumes = ["s3://rv-bucket:/mnt/sf"]
init = ["apk add -q rclone fuse3"]
SMOLEOF
$S machine create --name rv-sf --smolfile "$TMPD/Smolfile" --net-backend virtio-net >/dev/null 2>&1 \
  && $S machine start --name rv-sf >/dev/null 2>&1 \
  && ok "smolfile machine with remote volume starts" || bad "smolfile machine with remote volume starts"
$S machine exec --name rv-sf -- sh -c 'echo sf-1 > /mnt/sf/sf.txt && sync && sleep 5' >/dev/null 2>&1
check "smolfile write visible out-of-band" "$(oob sf.txt)" "sf-1"
rm -rf "$TMPD"

# ---- 10. serve API surface: structured MountSpec with a remote source ------
# Needs its own serve (the guest-rollout ingress port is exclusive) with the
# egress floor lowered so the guest may dial the host-local MinIO; skipped if
# a serve is already running.
if pgrep -f "smolvm.* serve start" >/dev/null 2>&1; then
  echo "SKIP: serve API cases (a serve is already running)"
else
  RVSOCK=$(mktemp -u /tmp/rv-serve.XXXXXX.sock)
  SMOLVM_EGRESS_FLOOR=metadata "$S" serve start --listen "unix://$RVSOCK" >/dev/null 2>&1 &
  SERVE_PID=$!
  i=0; until [ -S "$RVSOCK" ]; do i=$((i+1)); [ $i -gt 20 ] && break; sleep 1; done
  api() { curl -s -m 300 --unix-socket "$RVSOCK" "$@"; }
  CODE=$(api -o /dev/null -w '%{http_code}' -X POST http://localhost/api/v1/machines \
    -H 'Content-Type: application/json' -d '{
    "name": "rv-api", "image": "rclone/rclone",
    "network": true, "network_backend": "virtio-net",
    "entrypoint": ["sleep"], "cmd": ["infinity"],
    "env": [
      {"name": "AWS_ACCESS_KEY_ID", "value": "smoltest"},
      {"name": "AWS_SECRET_ACCESS_KEY", "value": "smoltest123"},
      {"name": "AWS_ENDPOINT_URL", "value": "http://100.96.0.1:9000"}
    ],
    "mounts": [{"source": "s3://rv-bucket", "target": "/mnt/api"}]}')
  check "api create accepts remote mount" "$CODE" "200"
  CODE=$(api -o /dev/null -w '%{http_code}' -X POST http://localhost/api/v1/machines/rv-api/start)
  check "api start mounts the volume" "$CODE" "200"
  sleep 10
  api -X POST http://localhost/api/v1/machines/rv-api/exec -H 'Content-Type: application/json' \
    -d '{"command":["sh","-c","echo api-1 > /mnt/api/api.txt && sync && sleep 6"]}' >/dev/null 2>&1
  check "api-machine write visible out-of-band" "$(oob api.txt)" "api-1"
  CODE=$(api -o "$TMPD.badjson" -w '%{http_code}' -X POST http://localhost/api/v1/machines \
    -H 'Content-Type: application/json' -d '{
    "name": "rv-api-bad", "image": "alpine:latest", "network": true,
    "mounts": [{"source": ":http,url=\"https://h\"", "target": "/mnt/x"}]}')
  check "api rejects malformed rclone remote (400)" "$CODE" "400"
  grep -q "::" "$TMPD.badjson" 2>/dev/null && ok "api 400 carries the :: hint" || bad "api 400 carries the :: hint"
  rm -f "$TMPD.badjson"
  api -X DELETE "http://localhost/api/v1/machines/rv-api?force=true" >/dev/null 2>&1
  kill "$SERVE_PID" 2>/dev/null
  rm -f "$RVSOCK"
fi

# ---- 11. service image keeps its own entrypoint AND still mounts -----------
# No `--` command: the image's ENTRYPOINT/CMD is resolved inside the guest and
# must still run. The agent wraps the mount AROUND that resolved command
# (`sh -c '<mounts> && exec "$@"' sh <argv>`); a host-side wrap only saw the
# empty request command and would replace nginx with a bare mount stub.
$S machine create --name rv-svc --image nginx:alpine --net --net-backend virtio-net $CREDS \
  --init "apk add -q rclone fuse3" -v "s3://rv-bucket:/mnt/svc" >/dev/null 2>&1
$S machine start --name rv-svc >/dev/null 2>&1
# The wrap execs into the image entrypoint, so nginx becomes PID 1 after a brief
# sh -> docker-entrypoint.sh -> exec nginx transition — poll /proc/1/comm rather
# than probe once (and it sidesteps busybox pgrep quirks).
GOT=
n=0; while [ $n -lt 15 ]; do
  [ "$($S machine exec --name rv-svc -- sh -c 'cat /proc/1/comm' 2>/dev/null)" = nginx ] && { GOT=up; break; }
  n=$((n+1)); sleep 1
done
check "service entrypoint runs under a remote volume" "$GOT" "up"
$S machine exec --name rv-svc -- sh -c 'echo svc-1 > /mnt/svc/svc.txt && sync && sleep 5' >/dev/null 2>&1
check "service-image mount write visible out-of-band" "$(oob svc.txt)" "svc-1"

cleanup
echo ""
echo "remote volumes e2e: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ]
