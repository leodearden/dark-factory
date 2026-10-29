# Cross-family re-rate of the write-triage eval set

- Raters: gpt-6.1-sol:high; calls 320; spend 4.05 USD at list price.
- Discordant pairs: 220 selected, 220 rated, 0 failed.
- Control pairs: 100 rated, 0 failed.

| selection | binary | versus | n | agree | rate | wilson95 | kappa |
|---|---|---|---|---|---|---|---|
| discordant | home | resolved | 220 | 173 | 0.7864 | [0.7275, 0.8354] | 0.5666 |
| discordant | home | primary | 220 | 166 | 0.7545 | [0.6936, 0.8067] | — |
| discordant | home | second | 220 | 177 | 0.8045 | [0.7471, 0.8515] | — |
| discordant | conflict | resolved | 219 | 150 | 0.6849 | [0.6207, 0.7428] | 0.383 |
| discordant | conflict | primary | 219 | 143 | 0.653 | [0.5878, 0.7129] | — |
| discordant | conflict | second | 219 | 146 | 0.6667 | [0.6018, 0.7258] | — |
| discordant | scope | resolved | 84 | 68 | 0.8095 | [0.7128, 0.8792] | 0.6285 |
| discordant | scope | primary | 52 | 44 | 0.8462 | [0.7248, 0.9199] | — |
| discordant | scope | second | 65 | 51 | 0.7846 | [0.6703, 0.8671] | — |
| discordant | action_changing | resolved | 82 | 73 | 0.8902 | [0.8044, 0.9412] | 0.6276 |
| discordant | action_changing | primary | 52 | 45 | 0.8654 | [0.7473, 0.9332] | — |
| discordant | action_changing | second | 65 | 60 | 0.9231 | [0.8322, 0.9667] | — |
| discordant | loss | resolved | 46 | 30 | 0.6522 | [0.5077, 0.7732] | 0.3622 |
| discordant | loss | primary | 34 | 25 | 0.7353 | [0.5688, 0.854] | — |
| discordant | loss | second | 32 | 11 | 0.3438 | [0.2041, 0.5169] | — |
| control | home | resolved | 100 | 98 | 0.98 | [0.93, 0.9945] | 0.8643 |
| control | home | primary | 100 | 98 | 0.98 | [0.93, 0.9945] | — |
| control | home | second | 100 | 98 | 0.98 | [0.93, 0.9945] | — |
| control | conflict | resolved | 100 | 95 | 0.95 | [0.8882, 0.9785] | 0.7346 |
| control | conflict | primary | 100 | 95 | 0.95 | [0.8882, 0.9785] | — |
| control | conflict | second | 100 | 95 | 0.95 | [0.8882, 0.9785] | — |
| control | scope | resolved | 8 | 8 | 1.0 | [0.6756, 1.0] | 1.0 |
| control | scope | primary | 8 | 8 | 1.0 | [0.6756, 1.0] | — |
| control | scope | second | 8 | 8 | 1.0 | [0.6756, 1.0] | — |
| control | action_changing | resolved | 6 | 5 | 0.8333 | [0.4365, 0.9699] | 0.0 |
| control | action_changing | primary | 8 | 6 | 0.75 | [0.4093, 0.9285] | — |
| control | action_changing | second | 8 | 6 | 0.75 | [0.4093, 0.9285] | — |
| control | loss | resolved | 6 | 5 | 0.8333 | [0.4365, 0.9699] | 0.0 |
| control | loss | primary | 6 | 5 | 0.8333 | [0.4365, 0.9699] | — |
| control | loss | second | 6 | 5 | 0.8333 | [0.4365, 0.9699] | — |

HOME on pairs whose Opus raters split on HOME: n 69, cross-family YES 58, agrees with the resolved label 36.

| v1 word | YES | NO | UNCLEAR |
|---|---|---|---|
| RELATED/UNRELATED | 1 | 18 | 0 |
| SAME/SUBSUMED | 6 | 0 | 0 |
| EXTENDS | 33 | 13 | 0 |
| CORRECTS | 13 | 2 | 0 |
