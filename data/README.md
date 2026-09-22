# Data

The datasets and personal audio recordings used for the recorded experiments are not included in this repository.

The spoken-language evaluation used the English subset of **FLEURS**. Six additional recordings were used to test whispering, background music, singing, cross-talk, and clean speech. The recordings are omitted because they contain the author's voice.

The next-speaker model uses four-speaker meetings from the **AMI Meeting Corpus**. Consecutive words from the same speaker are merged into turns, and each example uses the previous eight turns plus the current turn to predict the next speaker.

The visual dialogue experiments use the **GuessWhat?!** dataset loaded from `jxu124/guesswhat`. The first 20 streamed test samples were used for the recorded zero-shot and post-fine-tuning comparison. A larger sample was used to construct Oracle training examples.
