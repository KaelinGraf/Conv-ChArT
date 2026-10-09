# Minimal fine-tuning ladder (headline target 99.0% ID on the 1k in-loop val; trained-board reference 99.53%)

| arm | trainable params | steps to 99% | wall to 99% | peak VRAM B16 (MB) | ms/step B16 | final ID 10k | p95 px | correct pose | wrong accepted | train wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| head | 882k | 37968 | n/a | n/a | 2068 | 22.4 | 57.41% | 0.6753 | 0.4% | 0.0% | 44 min |
| head | 3-board | 37968 | n/a | n/a | 2068 | 22.4 | 50.98% | 0.7000 | 0.1% | 0.0% | 43 min |
| bneck 0.1x | 882k | 437105 | n/a | n/a | 3543 | 47.9 | 90.52% | 0.6748 | 69.6% | 0.4% | 44 min |
| bneck 0.1x | 3-board | 437105 | n/a | n/a | 3543 | 47.9 | 72.48% | 0.6994 | 34.3% | 0.3% | 43 min |
| bneck 1x | 882k | 437105 | n/a | n/a | 3543 | 47.9 | 97.10% | 0.6736 | 84.5% | 0.4% | 41 min |
| bneck 1x | 3-board | 437105 | n/a | n/a | 3543 | 47.9 | 86.73% | 0.6993 | 64.1% | 0.5% | 43 min |
| post-bneck | 882k | 191890 | n/a | n/a | 2695 | 40.7 | 72.27% | 0.6747 | 15.0% | 0.0% | 43 min |
| post-bneck | 3-board | 191890 | n/a | n/a | 2695 | 40.7 | 67.28% | 0.6993 | 9.4% | 0.1% | 43 min |
| post-bneck | 15-board | 191890 | n/a | n/a | 2695 | 40.7 | 29.97% | 0.6895 | 0.0% | 0.0% | 43 min |

Steps to each threshold (1k in-loop val): head | 882k: 80%->never, 90%->never, 95%->never, 99%->never; head | 3-board: 80%->never, 90%->never, 95%->never, 99%->never; bneck 0.1x | 882k: 80%->1500, 90%->never, 95%->never, 99%->never; bneck 0.1x | 3-board: 80%->never, 90%->never, 95%->never, 99%->never; bneck 1x | 882k: 80%->500, 90%->750, 95%->1750, 99%->never; bneck 1x | 3-board: 80%->1500, 90%->never, 95%->never, 99%->never; post-bneck | 882k: 80%->never, 90%->never, 95%->never, 99%->never; post-bneck | 3-board: 80%->never, 90%->never, 95%->never, 99%->never; post-bneck | 15-board: 80%->never, 90%->never, 95%->never, 99%->never
