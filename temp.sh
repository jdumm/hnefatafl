
Latest training run:
python hnefatafl_train.py --game-name Brandubh --batch --train-model --probabilistic-symmetry --model-preset brandubh -v 46 --load-latest --cache-model-every 1000 --log-level 1 --log-every 200 *> brandubh_v46.log

Verify status:
Get-Content .\brandubh_v46.log -Tail 20

Run latest games:
python hnefatafl_train.py --game-name Brandubh --interactive --ai-attacker --ai-defender -v46 --load-latest

