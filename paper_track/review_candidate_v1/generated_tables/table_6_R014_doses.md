| Dose | Target RMS | Factor vs constructor | Mean held-out test PPL | 95% CI |
|---|---|---|---|---|
| D_xavier | 0.00629236 | 0.00629287 | 200.3252 | [198.5897, 202.0608] |
| D_mid1 | 0.03408242 | 0.03408521 | 205.4083 | [204.3242, 206.4924] |
| D_ctor | 0.99991812 | 1.00000000 | 235.0694 | [232.3218, 237.8171] |

| Paired contrast | Mean Δ test PPL | 95% CI | CI excludes zero |
|---|---|---|---|
| D_xavier_minus_D_ctor | -34.7442 | [-37.5676, -31.9208] | true |
| D_xavier_minus_D_mid1 | -5.0831 | [-7.2794, -2.8868] | true |
| D_mid1_minus_D_ctor | -29.6611 | [-33.3032, -26.0191] | true |
