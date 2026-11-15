# HelpRoute labeling guide

One row per post. Read the title + body, then fill the columns.

## `is_need` (1 or 0): the main label
**1 = real need:** the poster, or someone they are directly helping, has a **current** problem that HelpRoute could route to help:
- Housing / landlord dispute (eviction, can't pay rent, deposit withheld, unfixed mold or repairs)
- Homelessness (no stable place to sleep now, living in a car, in a shelter)
- Food (can't afford enough food, food bank / CalFresh questions for themselves)
- Medical cost or access (uninsured, can't afford care, Medi-Cal doctor or dentist access, medical debt)
- Job / income (laid off, hours cut, unemployment running out, job hunting, 401k/retirement or mortgage planning)
- Bills & debt (utility shutoff, behind on bills, payday loans, collections)

Counts as 1:
- Asking for resources or advice about their own situation
- Venting while still in the situation ("we live in our van and they towed it")
- Helper posts ("my mom is getting evicted, where can she go?")

**0 = not a need:**
- News articles or headlines, even about evictions or homelessness
- Opinion, policy or politics posts, rants about the city or the market
- Scam warnings / PSAs
- Volunteer, donation or nonprofit promotion posts
- Past, resolved situations ("I was homeless 10 years ago")
- Cost-of-living comparisons, general apartment hunting (job hunting and 401k/mortgage planning DO count, see rules 1-2)

## `unsure` (1 or blank)
Still pick 0 or 1 for `is_need`, but tick `unsure` if it could reasonably go either way (e.g., venting about a big rent increase, a medical bill they already paid).

## `crisis` (1 or blank)
A safety situation (see rule 5): suicide/self-harm, abuse or violence, or anything that needs police or emergency social services. Mark it and still fill `is_need` and types. These go to the safety rule (988 / 911 / Adult or Child Protective Services / DV hotline + moderator), not the regular resource reply. Take a break if you need one.

## Need types (only if `is_need` = 1; tick every one that applies)
`housing`, `homelessness`, `food`, `medical`, `job_income`, `bills_debt`, `animal`
- `animal`: an animal needs urgent help (foster/rehoming, euthanasia list, can't keep a pet because of a move or eviction). Animal posts CAN be `is_need` = 1. If the person also has their own need (e.g., an eviction), tick that type too.
A post can have several (e.g., laid off and about to lose housing = `job_income` + `housing`).

## `why_not` (optional, only if `is_need` = 0)
news / opinion_rant / pet / scam_warning / volunteer_donation / past_situation / other

## Tie-breaker rules (agreed after comparing the two test labelers)
1. **Job / income (`job_income`):** need = unemployed, laid off, about to be laid off, hours cut, OR job hunting (including while still employed).
2. **Money:** need = can't cover basics, behind on bills, unmanageable debt, can't afford medical care. **401k / retirement planning and mortgage planning also count as needs, type `job_income`.** Still 0 unless the team decides otherwise: savings-account questions, buying a car, credit-score optimization, budgeting while doing fine.
3. **Resource lists, PSAs, food bank hauls, recipe sharing:** 0, unless the poster says they themselves are struggling.
4. **Housing search:** 0, unless they're losing housing or can't afford any place.
5. **Crisis:** suicidal or self-harm language ("want to die", "end it all"); abuse or violence (domestic violence, elder abuse, child abuse); or a vulnerable person who needs police or emergency social services (e.g., a confused elderly person alone in a car). Distress, stress or crying alone is not crisis. Still fill `is_need` and types on crisis posts.

## Batch A only: `changed_from_draft`
`draft_verdict` is Claude's AI guess. Put 1 if your label differs from it, 0 if it matches. Don't trust the draft; read the post.

## Rules
- Test sheets are **blind**: don't discuss posts with the other labeler until you've both finished.
- Never copy usernames anywhere.
- When in doubt, re-read this guide; if still stuck, label your best guess + `unsure`.
