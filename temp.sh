
Latest training run:
python hnefatafl_train.py --game-name Brandubh --batch --train-attacker --train-defender --enhanced-encoding --probabilistic-symmetry -v44 --load-latest --cache-model-every 1000 > brandubh_v44.log

Verify status:
Get-Content .\brandubh_v44.log -Tail 20

Run latest games:
python hnefatafl_train.py --game-name Brandubh --interactive --ai-attacker --ai-defender --enhanced-encoding -v44 --load-latest

