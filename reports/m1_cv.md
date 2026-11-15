# M1 cross-validation

Best C: 1; high-recall threshold: 0.4308.

| Model | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| B0 | 0.000 | 0.000 | 0.000 | 0.319 |
| B1 | 0.319 | 1.000 | 0.483 | 0.319 |
| M1 | 0.470 | 0.902 | 0.618 | 0.653 |

## Recall by subgroup

- bay_local: 0.901
- housing_money: 0.906
- work: 0.895
- bay: 0.904
- national: 0.900
- short: 0.667
- long: 0.906

## Type models

- housing: {'status': 'ok', 'positives': 31, 'f1': 0.6086956521739131, 'pr_auc': 0.8220862359582308}
- homelessness: {'status': 'ok', 'positives': 25, 'f1': 0.21428571428571427, 'pr_auc': 0.6483878360595463}
- food: {'status': 'ok', 'positives': 12, 'f1': 0.15384615384615385, 'pr_auc': 0.7712539711232516}
- medical: {'status': 'ok', 'positives': 19, 'f1': 0.48, 'pr_auc': 0.8714751763017448}
- job_income: {'status': 'ok', 'positives': 69, 'f1': 0.7941176470588235, 'pr_auc': 0.8942009956185153}
- bills_debt: {'status': 'ok', 'positives': 22, 'f1': 0.2962962962962963, 'pr_auc': 0.5772796229138197}
- animal: {'status': 'ok', 'positives': 9, 'f1': 0.6153846153846154, 'pr_auc': 0.9658119658119659}
