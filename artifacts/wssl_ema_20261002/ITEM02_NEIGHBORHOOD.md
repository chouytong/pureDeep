# Best epoch neighborhood — actual archived logs

|Offset from ordinary BA best|Runs|Validation BA|Validation AUROC|Online train main CE|Validation main CE|
|---:|---:|---:|---:|---:|---:|
|-2|44|0.662961|0.749907|0.374382|0.693586|
|-1|45|0.666127|0.754599|0.328680|0.704509|
|0|45|0.719644|0.759553|0.291046|0.679387|
|1|45|0.670580|0.757319|0.247215|0.755399|
|2|45|0.671875|0.755120|0.211281|0.759714|

These are observed dropout-active online train metrics and ordinary eval-mode validation metrics, not intermediate checkpoint reevaluations. Early best epochs can lack offset -2 observations. Mean BA peaks at offset 0 by its selection rule; this is descriptive, not an independent test.
