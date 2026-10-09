# Minimal fine-tuning ladder (headline target 99.0% ID on the 1k in-loop val; trained-board reference 99.53%)

| arm | trainable params | steps to 99% | wall to 99% | peak VRAM B16 (MB) | ms/step B16 | final ID 10k | p95 px | correct pose | wrong accepted | train wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 head | 37968 | n/a | n/a | 2068 | 22.4 | 57.41% | 0.6753 | 0.4% | 0.0% | 44 min |
| 2 head+bneck+hm 0.1x | 437105 | n/a | n/a | 3543 | 47.9 | 90.52% | 0.6748 | 69.6% | 0.4% | 44 min |
| 3 head+bneck 0.01x+hm 0.1x | 437105 | n/a | n/a | 3543 | 47.7 | 69.33% | 0.6753 | 8.6% | 0.2% | 43 min |
| 4 full, LLRD 0.3 | 882402 | n/a | n/a | 4851 | 66.8 | 71.92% | 0.6745 | 14.7% | 0.2% | 43 min |
| 5 head+gate3 | 44177 | n/a | n/a | 2702 | 32.8 | 57.38% | 0.6751 | 0.3% | 0.0% | 44 min |
| 6 head+d3 0.1x | 148688 | n/a | n/a | 2679 | 32.2 | 61.25% | 0.6748 | 1.3% | 0.0% | 43 min |
| 7 head re-init | 37968 | n/a | n/a | 2075 | 22.4 | 58.93% | 0.6753 | 0.7% | 0.1% | 45 min |
| 8 BitFit | 42194 | n/a | n/a | 4571 | 55.0 | 73.27% | 0.6759 | 15.1% | 0.3% | 44 min |
| 9 full 0.1x | 882402 | n/a | n/a | 4851 | 66.0 | 94.34% | 0.6752 | 79.2% | 0.3% | 44 min |
| 10 full 1x | 882402 | n/a | n/a | 4851 | 65.9 | 98.36% | 0.6771 | 86.9% | 0.5% | 43 min |

Steps to each threshold (1k in-loop val): 1 head: 80%->never, 90%->never, 95%->never, 99%->never; 2 head+bneck+hm 0.1x: 80%->1500, 90%->never, 95%->never, 99%->never; 3 head+bneck 0.01x+hm 0.1x: 80%->never, 90%->never, 95%->never, 99%->never; 4 full, LLRD 0.3: 80%->never, 90%->never, 95%->never, 99%->never; 5 head+gate3: 80%->never, 90%->never, 95%->never, 99%->never; 6 head+d3 0.1x: 80%->never, 90%->never, 95%->never, 99%->never; 7 head re-init: 80%->never, 90%->never, 95%->never, 99%->never; 8 BitFit: 80%->never, 90%->never, 95%->never, 99%->never; 9 full 0.1x: 80%->1000, 90%->1750, 95%->never, 99%->never; 10 full 1x: 80%->500, 90%->500, 95%->1000, 99%->never
