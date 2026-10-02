# Item 01 — model-copy independence (PASS)

New independent_wssl.py removes method closures, preserves old state keys, and provides explicit SSL forward input plus an instance-local compatibility bridge. Fresh-instance copy restores ordinary training RNG and clones ephemeral feature storage. No architecture, metric definition or formal artifact changed.

All 45 frozen best checkpoints: real 8-subject development batches have bit-identical old/new logits. Maximum difference from archived probabilities 1.11e-16. Separate parameter storage and cache mutation isolation, safe deepcopy of new wrapper, fresh checkpoint reload, RNG preservation and finite gradient update passed. Classifier has 143172 trainable parameters and no buffers. HarNet remains frozen cached representation.

Old wrapper deepcopy demonstrably captures the original wrapper in its method closure. This identifies a hazardous copy operation; historical experiments constructed separate instances and are not invalidated merely by its existence. No full training or new performance candidate has run at this step. See analysis/independence_test.json and independence_45checkpoints.csv.
