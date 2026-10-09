# Minimal fine-tuning ladder (headline target 99.0% ID on the 1k in-loop val; trained-board reference 99.53%)

| arm | trainable params | steps to 99% | wall to 99% | peak VRAM B16 (MB) | ms/step B16 | final ID 10k | p95 px | correct pose | wrong accepted | train wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bneck 0.01x | 437105 | n/a | n/a | 3543 | 47.7 | 69.33% | 0.6753 | 8.6% | 0.2% | 43 min |
| bneck 0.1x | 437105 | n/a | n/a | 3543 | 47.9 | 90.52% | 0.6748 | 69.6% | 0.4% | 44 min |
| bneck 0.3x | 437105 | n/a | n/a | 3543 | 47.9 | 93.46% | 0.6748 | 78.5% | 0.3% | 43 min |
| bneck 1x | 437105 | n/a | n/a | 3543 | 47.9 | 97.09% | 0.6739 | 84.8% | 0.4% | 43 min |
| bneck 3x | 437105 | n/a | n/a | 3543 | 47.9 | 98.22% | 0.6739 | 86.2% | 0.3% | 41 min |
| bneck+e4 1x | 658801 | n/a | n/a | 3630 | 49.1 | 97.84% | 0.6745 | 85.4% | 0.5% | 43 min |
| full 1x | 882402 | n/a | n/a | 4851 | 65.9 | 98.36% | 0.6771 | 86.9% | 0.5% | 43 min |
| full 3x | 882402 | 3500 | 19.6 min | 4851 | 65.9 | 98.88% | 0.6818 | 88.3% | 0.3% | 30 min |
| bneck 0.1x, 20k | 437105 | n/a | n/a | 3543 | 47.9 | 95.97% | 0.6733 | 82.0% | 0.2% | 156 min |
| bneck 3x, 20k | 437105 | 17500 | 92.7 min | 3543 | 47.9 | 99.06% | 0.6719 | 87.6% | 0.2% | 108 min |

Steps to each threshold (1k in-loop val): bneck 0.01x: 80%->never, 90%->never, 95%->never, 98%->never, 99%->never; bneck 0.1x: 80%->1500, 90%->never, 95%->never, 98%->never, 99%->never; bneck 0.3x: 80%->1000, 90%->2000, 95%->never, 98%->never, 99%->never; bneck 1x: 80%->500, 90%->750, 95%->1750, 98%->never, 99%->never; bneck 3x: 80%->500, 90%->500, 95%->1000, 98%->3250, 99%->never; bneck+e4 1x: 80%->500, 90%->500, 95%->1250, 98%->never, 99%->never; full 1x: 80%->500, 90%->500, 95%->1000, 98%->3500, 99%->never; full 3x: 80%->250, 90%->500, 95%->750, 98%->1250, 99%->3500; bneck 0.1x, 20k: 80%->1500, 90%->3000, 95%->10500, 98%->never, 99%->never; bneck 3x, 20k: 80%->500, 90%->500, 95%->1000, 98%->3000, 99%->17500
