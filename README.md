# Speech and Visual Dialogue Agents

This repository contains four dialogue-system projects that go beyond ordinary typed two-person chat. They cover speech recognition, clarification when a request is unclear, predicting who will speak next in a meeting, and a visual guessing game played by two agents.

The main models are Whisper for speech recognition, a hierarchical BiLSTM for next-speaker prediction, and Qwen3.5-0.8B for visual dialogue. The clarification system is deliberately rule-based so uncertain slot values can be handled before a language model or external tool is called.

## Spoken questions with Whisper

The spoken-dialogue pipeline is:

```text
speech
  ↓
Whisper
  ↓
text question
  ↓
retrieval + language model
  ↓
text answer
  ↓
gTTS
  ↓
spoken answer
```

I first compared four Whisper sizes on English FLEURS audio. References and predictions were normalized with Whisper's `BasicTextNormalizer` before calculating word error rate.

| Whisper model | Word error rate | Recorded runtime |
| --- | ---: | ---: |
| tiny | 15.38% | 188.53 s |
| base | 10.71% | 196.05 s |
| small | 6.96% | 291.42 s |
| medium | **5.13%** | 471.30 s |

The larger models made fewer word errors on this clean benchmark, but they also took longer to run.

I then recorded six additional clips to see what happens when the audio is less clean. The conditions included whispering, music in the background, singing, overlapping speech, and clean versions of two questions.

A few examples show why the transcription itself matters to the rest of the dialogue system:

| Audio condition | tiny | base | small | medium |
| --- | --- | --- | --- | --- |
| Music + “What is Barbican?” | “What is barbecue?” | “What is barbeque?” | “What is Barbie?” | **“What is Barbican?”** |
| Clean “What is Barbican?” | “What is barbeque?” | **correct** | **correct** | **correct** |
| Sung question | “Do you say…” | **correct** | **correct** | **correct** |
| Cross-talk + distance question | heavily corrupted | “I mean the kilometers is one and more.” | “Our main kilometers is one mile.” | **correct** |
| Clean distance question | **correct** | **correct** | **correct** | **correct** |

In these recordings, Whisper-medium was the only model that recovered the intended wording in every condition. The smaller models sometimes produced fluent substitutions such as `Barbican → Barbie`. Those are especially problematic for retrieval because the next stage can confidently answer the wrong question.

The personal audio recordings themselves are not included in the repository.

## Asking for clarification before calling an LLM

The clarification system sits in front of a restaurant-booking agent. Its job is to decide whether a user request is clear enough to continue.

For each restaurant constraint, it distinguishes five cases:

- `mapped` — an exact known value or synonym
- `substring` — a known value found inside a longer utterance
- `fuzzy` — a likely typo or ASR error such as `chainese → chinese`
- `unknown` — the user appears to mention the slot, but the value cannot be mapped
- `missing` — the user did not provide the slot at all

Fuzzy matches are split into confidence bands. Below `0.84`, the system asks the user to confirm the value directly. Between `0.84` and `0.94`, it carries the likely interpretation into the next question. Exact and high-confidence values can continue without another clarification turn.

For example:

```text
User: I want chainese food.
System: OK, in what area are you looking for a chinese restaurant?
```

but:

```text
User: I want asian food.
System: I didn't catch the food clearly. Did you mean one of these: chinese, italian, indian, japanese, vegetarian, or british?
```

The policy was evaluated on 12 cases covering missing values, spelling errors, out-of-vocabulary terms, and noisy ASR-style text.

| Input group | Correct | Accuracy |
| --- | ---: | ---: |
| Missing values | 3 / 3 | 100% |
| Ambiguous spellings | 1 / 2 | 50% |
| Out-of-vocabulary values | 3 / 3 | 100% |
| Noisy ASR text | 4 / 4 | 100% |
| **Overall** | **11 / 12** | **91.7%** |

The one failed case was `senter`. Its fuzzy similarity to `center` falls just below the threshold for implicit clarification, so the implementation asks for explicit confirmation instead. The cleaned code reproduces the same **11/12** result.

A useful property of this setup is that unclear requests can be handled before the generative model runs. The system does not need an LLM to decide whether `asian` is a supported cuisine or whether `chainese` probably means `chinese`.

## Predicting who speaks next in a four-person meeting

The next-speaker model uses meetings with speakers `A`, `B`, `C`, and `D` from the AMI Meeting Corpus.

Each training example contains the previous eight turns, the current turn, the identities of the speakers in that history, and the identity of the current speaker. The label is the person who speaks next.

The model works in two stages:

```text
text of each turn
      ↓
utterance BiLSTM
      ↓
utterance vectors + speaker IDs
      ↓
context BiLSTM over the turn sequence
      ↓
current speaker embedding
      ↓
A / B / C / D
```

The main recorded test result was:

| Metric | Test result |
| --- | ---: |
| Accuracy | **71.58%** |
| Macro precision | 71.41% |
| Macro recall | 71.41% |
| Macro F1 | **71.39%** |

For comparison, random guessing over four speakers is about 25% accuracy, while always predicting the most frequent test speaker is about 28%.

I also reran the model with different speaker information removed. These ablations used the same hyperparameters and a fixed random seed, so their numbers are reported separately from the main run above.

| Inputs | Test macro F1 | Test accuracy |
| --- | ---: | ---: |
| Text + current speaker + history speakers | **70.70%** | **71.15%** |
| Remove current speaker | 67.62% | 68.01% |
| Remove history speaker IDs | 37.68% | 38.79% |
| Text only | 27.88% | 29.48% |

The largest drop comes from removing the identities of the previous speakers. In this dataset, the sequence of who has been speaking carries much more information about the next turn than the words alone.

## Visual dialogue with a Questioner and Oracle

The visual-dialogue system plays GuessWhat?! with Qwen3.5-0.8B.

Each image contains several candidate objects. Their bounding boxes are converted to a 0–999 coordinate scale and listed in the prompt with an index:

```text
[0] person {"bbox_2d": [590, 452, 803, 803]}
[1] surfboard {"bbox_2d": [665, 489, 860, 708]}
...
```

Two copies of the model play different roles:

- the **Questioner** sees the image and candidate list and asks one yes/no question at a time
- the **Oracle** knows which candidate is the hidden target and answers `Yes`, `No`, or `N/A`

After five turns, the Questioner receives the dialogue history again and returns one candidate index such as `[3]`.

### Zero-shot result

The prompt-only system was evaluated on 20 games from GuessWhat?!:

| Metric | Result |
| --- | ---: |
| Target selected correctly | **20.00%** |
| Oracle outputs exactly `Yes`, `No`, or `N/A` | **100.00%** |
| Oracle outputs recoverable by the normalizer | **100.00%** |

The main problem was not answer formatting. The Questioner often repeated the same candidate instead of moving through the remaining options, which wastes a large part of the five-turn limit.

### LoRA fine-tuning of the Oracle

I then fine-tuned only the Oracle with LoRA while leaving the Questioner unchanged.

The data construction sampled 120 GuessWhat?! games. The first 20 were kept for the recorded comparison, and the remaining games produced **446 Oracle question-answer examples**. Those examples were split 90/10 for training and evaluation.

The LoRA configuration was:

```text
rank: 16
alpha: 32
dropout: 0.05
targets: q_proj, k_proj, v_proj, o_proj
epochs: 2
learning rate: 2e-4
effective batch size: 4
```

Training ran for 202 optimization steps. Validation loss fell quickly and was around `1.66` near the end of training.

The same 20 games were then run again:

| Metric | Zero-shot | Fine-tuned Oracle |
| --- | ---: | ---: |
| Target-selection accuracy | 20.00% | **25.00%** |
| Oracle strict answer format | **100.00%** | 99.00% |
| Oracle answer recoverable | 100.00% | 100.00% |

Because the test contains only 20 games, the five-point increase is **one additional correctly identified target**, so I do not treat it as a strong general performance claim.

The dialogue traces also reveal a downside. The fine-tuned Oracle sometimes gives different answers when the Questioner repeats exactly the same question. The human GuessWhat?! answers used for training are based on visual and category judgements, while the zero-shot Oracle prompt uses a strict candidate-index rule. Fine-tuning makes the Oracle more like the dataset, but less consistent with that hand-written rule.

## Repository structure

```text
speech-and-visual-dialogue-agents/
├── src/
│   └── speech_visual_dialogue/
│       ├── speech.py
│       ├── clarification.py
│       ├── multiparty.py
│       ├── visual_dialogue.py
│       └── visual_finetuning.py
├── experiments/
│   ├── asr_results.py
│   ├── clarification_eval.py
│   ├── multiparty_results.py
│   └── visual_dialogue_results.py
├── tests/
├── data/
│   └── README.md
├── pyproject.toml
└── README.md
```

## Running the code

Install the core package:

```bash
pip install -e .
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

The cleaned project has **22 tests** covering slot normalization, clarification decisions, state updates, the next-speaker model, candidate bounding boxes, Questioner and Oracle prompts, answer normalization, selector parsing, and the LoRA configuration.

Run the clarification evaluation:

```bash
python experiments/clarification_eval.py
```

Install the optional model dependencies before running Whisper or Qwen experiments:

```bash
pip install -e ".[models]"
```

The full ASR, AMI, and GuessWhat?! results above are recorded from the original experiments. They are not rerun by the unit-test suite because the datasets and pretrained model weights are not bundled with this repository.