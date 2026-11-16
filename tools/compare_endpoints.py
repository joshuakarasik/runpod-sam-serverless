#!/usr/bin/env python3
"""Compare a test endpoint with the production endpoint on the same inputs.

For each test image it sends the identical job to:
  - the production endpoint twice (to measure GPU run-to-run variation), and
  - the test endpoint once,
then compares SHA-256 hashes of the returned PNG bytes.

Use only non-sensitive test images. Nothing is written to disk, and no image,
mask or metric value is printed: only hashes (truncated), match flags and
whether `quality` came back.

Usage:
  read -s RUNPOD_API_KEY && export RUNPOD_API_KEY
  python3 tools/compare_endpoints.py --prod 49ciibx2lucfcd --test <TEST_ID> --cases cases.json

cases.json: [{"image": "path/to/test1.jpg", "box": [x1, y1, x2, y2]}, ...]
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.request

API = "https://api.runpod.ai/v2/{endpoint}/{path}"


def call(endpoint, key, payload, timeout_s=600):
    def request(path, body=None):
        req = urllib.request.Request(
            API.format(endpoint=endpoint, path=path),
            data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.load(resp)

    reply = request("runsync?wait=90000", payload)
    deadline = time.time() + timeout_s
    while reply.get("status") in ("IN_QUEUE", "IN_PROGRESS") and time.time() < deadline:
        time.sleep(3)
        reply = request(f"status/{reply['id']}")
    return reply


def digest(reply):
    output = reply.get("output") or {}
    image = output.get("enhancedImageData")
    if reply.get("status") != "COMPLETED" or not image:
        return None, output
    return hashlib.sha256(base64.b64decode(image)).hexdigest(), output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prod", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--cases", required=True)
    args = parser.parse_args()
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        sys.exit("Set RUNPOD_API_KEY in the environment (do not paste it on the command line).")

    cases = json.load(open(args.cases))
    rows, all_ok = [], True
    for n, case in enumerate(cases, 1):
        data = base64.b64encode(open(case["image"], "rb").read()).decode()
        payload = {"input": {"imageData": data, "boundingBox": case["box"]}}
        prod_a, _ = digest(call(args.prod, key, payload))
        prod_b, _ = digest(call(args.prod, key, payload))
        test, out = digest(call(args.test, key, payload))
        quality = out.get("quality")
        q = "null" if quality is None else f"{len(quality)} keys, {quality.get('worker_version')}"
        prod_stable = prod_a is not None and prod_a == prod_b
        match = test is not None and test in (prod_a, prod_b)
        all_ok &= match
        short = lambda h: h[:12] if h else "FAILED"
        rows.append(f"{n:>2}  prod {short(prod_a)} {short(prod_b)}  test {short(test)}  "
                    f"prod_stable={prod_stable}  test_matches_prod={match}  quality={q}")
    print("\n".join(rows))
    print("ALL MATCH" if all_ok else "MISMATCH: see rows above (if prod_stable is False, GPU variation, not the change)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
