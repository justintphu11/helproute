# Silver-label pilot (claude-haiku-5-5, effort low)

100 of 100 posts labeled (0 errors).

| Label | Human + | Claude + | Agreement | Cohen's kappa |
|---|---:|---:|---:|---:|
| is_need | 44 | 56 | 0.700 | 0.409 |
| housing | 1 | 18 | 0.830 | 0.088 |
| homelessness | 10 | 13 | 0.950 | 0.755 |
| food | 2 | 12 | 0.900 | 0.260 |
| medical | 3 | 9 | 0.920 | 0.302 |
| job_income | 24 | 42 | 0.680 | 0.302 |
| bills_debt | 7 | 19 | 0.840 | 0.314 |
| animal | 0 | 0 | 1.000 | nan |
| crisis | 1 | 5 | 0.940 | -0.017 |

is_need kappa, person-labeled posts (n=100): 0.409

## is_need disagreements (30)

- human=0 claude=1 — Poster's partner faces a large unaffordable hospital bill sent to a collection agency after an insurer's claim stalled despite a payment guarantee.
- human=0 claude=1 — Currently employed but weighing a startup offer after interviewing, which counts as job hunting under tie-breaker 1; no financial hardship is described, so it is borderline.
- human=0 claude=1 — Laid off months ago, 400+ applications with no job, unemployment benefits ran out, and stressing about paying bills: current job/income and money need.
- human=0 claude=1 — Recently laid off with official exit in a month and job hunting ahead; job search counts as a job/income need.
- human=1 claude=0 — Venting about a past resolved career gap now filled by a consulting gig; the unemployment was months ago, so it is not a current need, though job-related venting makes it borderline.
- human=0 claude=1 — Poster is camping in woods with no stable shelter, is short on water and waiting for food tonight, so homelessness and food needs are current.
- human=1 claude=0 — Poster mocks a 62-year-old shelter resident's TV tantrum; no need is described for the poster or anyone they directly help, and the shelter expulsion is a past event with no current need.
- human=0 claude=1 — Poster was just fired and is actively job hunting with interviews coming up, and asks how the no-rehire status will affect those job searches.
- human=1 claude=0 — General career advice about layoffs and upskilling; poster describes no current job loss or hardship.
- human=0 claude=1 — Poster was fired for reporting fraud, is pregnant, and is seeking legal help; job loss is the core need, with a whistleblower and legal dispute as context, and the post is cut off mid-sentence.
- human=1 claude=0 — Opinion and policy discussion about 401k fund fees; poster is helping a brother, not describing a current need HelpRoute could route, so a non-need opinion post.
- human=0 claude=1 — Poster was laid off and has been job hunting for 13 months, a current job/income need under the guide.
- human=0 claude=1 — Poster expects a layoff within weeks and asks how to prepare, which is a current job/income need.
- human=0 claude=1 — Unemployed poster needs about $100 within a day to cover rent and avoid eviction, a current housing and income problem.
- human=1 claude=0 — Parent asking for a list of high-paying sales careers for their daughter; general career advice, not a current need of the poster.
- human=0 claude=1 — Poster loses their car by Friday, can't afford transport to a job paying $18/hr, faces a loan decision with $1.5k savings, $894 medical debt, and possible return to parents who are physically and verbally abusive, which is a safety concern.
- human=1 claude=0 — Shares general advice on cash-pay medical pricing and itemized bills, describing past experience helping a family member, with no current problem the poster or someone they directly help is facing.
- human=0 claude=1 — Recently laid off and now job hunting again, with a stated dread of applying to postings; per the guide, job hunting counts as a need, though the post is mainly venting about emotional distress rather than asking for resources.
- human=0 claude=1 — Poster has possible debt in collections under a false name, is getting daily collection calls, and asks how to be absolved of the debt, which is a current bills/debt problem.
- human=0 claude=1 — Poster has no income until the June 1 start date, has rent due, a pending internet shutoff, loans in arrears, and no food now; distress alone is not crisis.
- human=0 claude=1 — Poster is being laid off with last day next Friday, applying for unemployment and job hunting, which counts as job/income need.
- human=1 claude=0 — Currently employed with a job offer in process; asking whether a job-change process is worth it, a career decision rather than a current unmet need, though it touches on wanting more income.
- human=0 claude=1 — Poster seeks advice on handling a deceased father's medical debt and unpaid taxes through a small estate, including whether collectors could pursue them personally; it is a current debt problem but the poster is not clearly personally liable.
- human=0 claude=1 — Poster describes a friend laid off and job hunting without interviews; the post is mostly general discussion, but the friend's job search counts as a need under the job/income rule, so labeled 1 and unsure.
- human=1 claude=0 — Poster was laid off but already has an accepted better job starting in 3 weeks plus 5 months severance and benefits, so no current unmet need; the post is a success story and advice to others, not a request for help.
- human=0 claude=1 — Poster is behind on affordability with rising electric bills and fixed income, seeking ways to cut the bill; the edit saying they are overwhelmed is mild stress, not crisis, so it is a bills need that could read as a simple efficiency question.
- human=0 claude=1 — Poster is job hunting after a recent layoff and deciding between two accounting job offers, which the guide counts as a job/income need.
- human=1 claude=0 — Poster was laid off but is venting about discrimination and the company's stated reasons; no current need for job, income, or housing help is requested, so it reads as opinion/rant about a past layoff.
- human=0 claude=1 — Probationary employee worried about possible layoffs amid a hiring freeze and financial crisis, which counts as an about-to-be-laid-off job/income need within weeks; unsure because the poster is currently employed and well-regarded.
- human=0 claude=1 — Recently laid off network engineer actively job hunting and asking whether a paid placement service is worth it.
