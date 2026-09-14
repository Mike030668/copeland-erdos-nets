| Experiment | Scientific question | Factor / intervention | Seed block | Primary endpoint |
|---|---|---|---|---|
| R010 | Does the historical gain survive an attention-only same-path intervention? | Q/K/V/O attention initialization; non-target tensors parity-matched | 42–46 | Held-out test PPL at the validation-selected checkpoint |
| R011 | Which historical parameter group carries the reconstructed benefit? | Token-embedding condition × attention condition factorial | 42–46 | Held-out test PPL at the validation-selected checkpoint |
| R012 | Is the embedding effect attributable to initial RMS scale or redraw/direction? | Embedding RMS scale × redraw/direction factorial | 47–51 | Held-out test PPL at the validation-selected checkpoint |
| R013 | How does held-out performance vary over a fresh-seed embedding-scale ladder? | Five target RMS doses | 52–56 | Held-out test PPL at the validation-selected checkpoint |
| R014 | Does the tested scale effect persist at a larger model capacity? | Three target RMS doses in a 4-layer, d_model=256 model | 57–61 | Held-out test PPL at the validation-selected checkpoint |
