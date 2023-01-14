# The Simulated Emergence of Chord Function
This repository is the implementation of [The Simulated Emergence of Chord Function (Uehara and Tojo; EvoMUSART2021)](https://link.springer.com/chapter/10.1007/978-3-030-72914-1_18) and the extended paper [Chord Function Recognition as Latent State Transition (SN Computer Science 3:508 2022)](https://link.springer.com/article/10.1007/s42979-022-01395-4).

## Requirements
- Python 3.8.2
- Required packages are listed in: [requirements.txt](requirements.txt)

## Training
- Set the flag ``--do_train`` for training.

### Hidden Markov Models (HMM)
- Set the model type to ``--model_type hmm`` for HMM.

#### Baseline HMM
- Set the special flag for the baseline model: ``--base_model``
```buildoutcfg
python run_nhmm.py --model_type hmm --base_model --scale any --num_state 8 --seed 0 --device cpu --num_epochs 1000 --do_train
```

#### Neural HMM with all additional contexts
- Additional contexts can be set by flags: ``--use_lstm``, ``--use_histogram``, ``--use_pitch``
```buildoutcfg
python run_nhmm.py --model_type hmm --use_lstm --use_histogram --use_pitch --scale any --num_state 8 --seed 0 --device cpu --num_epochs 1000 --do_train
```

### Hidden Semi-Markov Models (HSMM)
- Set the model type to ``--model_type hsmm`` for HSMM.

#### Baseline HSMM
- Set the special flag for the baseline model: ``--base_model``
```buildoutcfg
python run_nhmm.py --model_type hsmm --base_model --scale any --num_state 8 --seed 0 --device cpu --num_epochs 500 --do_train
```

#### Neural HSMM with all additional contexts
- Additional contexts can be set by flags: ``--use_lstm``, ``--use_histogram``, ``--use_pitch``, ``--use_beat``
- Note that ``--use_beat`` can be set only for the Neural HSMM.
```buildoutcfg
python run_nhmm.py --model_type hsmm --use_lstm --use_histogram --use_pitch --use_beat --scale any --num_state 8 --seed 0 --device cpu --num_epochs 500 --do_train
```
## Evaluation and Visualizing
- To evaluate a trained model, set these options and flags same to it: ``--model_type``, ``--num_state``, ``--base_model``, ``--use_lstm``, ``--use_histogram``, ``--use_pitch``, ``--use_beat`` 
- Specify a trained model by ``--model_to_eval <path to a model for evaluation>`` and set the flag ``--do_eval``.
- A scale for evaluating can be selected by ``--eval_scale`` from four options {any, major, minor, dorian}.
#### Sample of Evaluating Neural HSMM with the Major Scale
```buildoutcfg
python run_nhmm.py --model_type hsmm --use_lstm --use_histogram --use_pitch --use_beat --eval_scale major --num_state 8 --do_eval --model_to_eval <path to a model for evaluation>
```

## Samples of Our Trained Models
Samples of our trained models can be found at: [trained_models](trained_models)

## Note
Although the resulting values are similar, the latest code changes the way perplexity is averaged in the evaluation.
For more information on this, see Appendix B of my [thesis](http://hdl.handle.net/10119/18139).

## License and Reference
MIT License: [LICENSE](LICENSE)


