#!/usr/bin/env python3
# TON-2873: restore run_raft_r1.sh train block to iter0's certified HP
# (run_raft_8b.sh): io-cap 0, max-length 2048, batch 1, grad-accum 4, lr 1e-4,
# eval disabled, output-dir. run_raft_r1.sh had dropped all of these, causing a
# defaults-based train (io-cap 2000, lr 2e-4, ...) = science-lever divergence.
import re, sys
p = "run_raft_r1.sh"
s = open(p).read()
# match the sft_train.py invocation block (from the call through --epochs $EPOCHS)
pat = re.compile(r'python sft_train\.py \\\n(?:  .*\\\n)*  --epochs \$EPOCHS\n')
new = (
    'python sft_train.py \\\n'
    '  --model "$BASE" --init-adapter "$POLICY_ADAPTER" \\\n'
    '  --data "$RAFT" --io-cap 0 --eval-data /tmp/_no_eval_$$ \\\n'
    '  --epochs "$EPOCHS" --max-length 2048 --batch-size 1 --grad-accum 4 \\\n'
    '  --lr 1e-4 --output-dir "$OUT"\n'
    'RC=$?; [ $RC -ne 0 ] && { echo "[$(date +%T)] TRAIN FAILED rc=$RC"; exit $RC; }\n'
)
m = pat.search(s)
if not m:
    print("NO_MATCH"); sys.exit(1)
s2 = s[:m.start()] + new + s[m.end():]
open(p, "w").write(s2)
print("REPLACED")
print("--- new block ---")
print(new)
