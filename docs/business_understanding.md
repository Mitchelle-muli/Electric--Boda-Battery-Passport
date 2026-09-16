Business Understanding
Kore2 Motorcycle Battery Performance Analysis — Team TITANS
CRISP-DM Phase 1 — prepared as a standalone branch contribution, structured to be read on its own or lifted directly into the team presentation.

Contents
1.	The Story
2.	Determine Business Objectives
3.	Assess Situation
4.	Determine Data Mining Goals
5.	Produce Project Plan
6.	Stakeholders

1. The Story
Kenya's boda boda economy runs on roughly three million motorcycles and well over a million riders, and for most of them a shift is a simple, self-contained unit of survival: fuel or charge in the morning, a day of fares, cash in hand by evening. For almost all of that history, the machine underneath a rider was a petrol engine — simple, forgiving, and nobody's daily worry. That is changing fast. Kenya's electric motorcycle fleet grew roughly thirty-fold between 2022 and early 2026, pushed along by zero-rated VAT on e-bikes and lithium batteries, dedicated green number plates, and a government ambition north of 200,000 electric units on the road. Every one of those new electric bodas depends on a part a petrol boda never had to think about: a lithium-ion battery pack that has to survive fast-charging, tropical heat, rough roads, and a 10-to-12-hour shift, day after day, swap station after swap station.
KOFA is one of the companies betting on that shift. Having proven its swap network and its Kore2 battery in West Africa, it is now expanding into Kenya — arguably the most competitive e-mobility market on the continent, going up against Roam, Spiro, Ampersand, ARC Ride, and M-KOPA-financed fleets, all chasing the same riders. Entering a market like that while carrying a battery-reliability question mark is not a small risk. Boda riders talk to each other constantly, on the road and off it; a bad reputation for a battery spreads faster than any marketing campaign can outrun it.
That is roughly where this project starts. As KOFA scaled its Kenyan pilot, Customer Service began hearing the same complaint arrive from different riders, in different words, often enough that it stopped looking like noise: the battery drains too fast, and sometimes the bike just switches off — no warning, sometimes mid-trip. For a boda rider, that is not a minor inconvenience. A shutdown with a passenger on board, or halfway through a delivery run, is a lost fare at best and a roadside safety incident at worst — and every rider who has that experience is a rider who tells three others about it before the day is out.
That leaves the business with a genuinely hard question, and an expensive one to get wrong in either direction: is this a handful of defective units, a design flaw across the whole Kore2 line, or complaints that don't hold up once you look closely? Each answer points to a completely different response — a quiet, targeted parts swap; an expensive fleet-wide recall that KOFA can ill afford while still establishing itself in a new market; or no action at all, betting that the complaints fade on their own. Answering that question properly, before committing to any of those three paths, is what CRISP-DM's first phase — Business Understanding — exists to do.

2. Determine Business Objectives
2.1 Background
KOFA is a battery-swap operator entering Kenya's e-boda market with its Kore2 battery, competing directly with established players (Roam, Spiro, Ampersand, ARC Ride) and financing-led fleets (M-KOPA) for the same pool of riders. Its Kenyan fleet is small today and, if the market follows the sector-wide trend, will grow quickly — which means whatever is discovered now, at a few dozen or a few hundred units, needs to still hold true at a few thousand.
Customer Service has logged a recurring complaint pattern — rapid battery drain and unprompted mid-ride shutdowns — that has not yet been formally investigated. No one currently knows whether this reflects a real, isolated hardware problem, a broader design or manufacturing issue, or inconsistent/inaccurate rider reporting.
2.2 Business Objectives
#	Objective
B1 (primary)	Determine, with defensible confidence, whether the rapid-drain / shutdown complaints trace back to a specific subset of defective batteries, a fleet-wide design or manufacturing issue, or unreliable reporting — and act on whichever turns out to be true.
B2	Protect and rebuild rider trust during a competitive market entry where reputation moves rider-to-rider, not through advertising.
B3	Avoid the two expensive wrong answers: dismissing a genuine defect (safety and reputational exposure) and launching an unnecessary fleet-wide recall (direct cost, operational disruption, and a signal of instability to a market KOFA is still trying to win).
B4	Produce a reusable diagnostic capability, not a one-off investigation — this complaint pattern will not be the last, and the fleet this method needs to work on is about to get much bigger.

2.3 Business Success Criteria
Success is measured in business terms, independent of which specific technique ends up being used:
·	A concrete, serial-number-level answer — not a probability, not "likely fine" — that Engineering and Customer Service can act on the same week the analysis is delivered.
·	A response that is proportionate: if the problem is 4 batteries, the fix is 4 batteries, not a fleet-wide recall (and vice versa, if it turns out to be systemic).
·	A method that KOFA's Product/Operations team can run again on the next batch of telemetry without commissioning a fresh investigation each time a complaint pattern emerges.
·	A finding that Customer Service can put in front of a rider in plain language — "your unit is one of the four affected, here's your replacement" — rather than a generic "within spec" response that erodes trust further.

3. Assess Situation
3.1 Inventory of Resources
·	Data: a 12-battery Kore2 telemetry sample — 720 readings per battery at 30-second intervals over a ~6-hour window — covering operating state (riding / charging / idle), speed, voltage, current, state of charge, self-reported state of health, and ambient temperature.
·	Documentation: this project's accompanying industry brief, covering Kenya's e-mobility sector context and a full CRISP-DM walkthrough of the SOH-tracking problem.
·	Systems: KOFA's existing Fleet Ops telemetry dashboard, which this analysis is designed to eventually feed rather than duplicate.
·	People: Engineering, Customer Service, and Product/Operations, plus KOFA management and financing/insurance partners as downstream stakeholders (Section 6).
·	Tooling: a Python data-science stack (pandas, numpy, matplotlib, scikit-learn, scipy) in a Jupyter environment, already in active use for the modelling work in this project.
3.2 Requirements, Assumptions, and Constraints
Requirements
·	Findings must be explainable to non-technical stakeholders — Customer Service, riders, and potentially regulators — not just statistically correct.
·	The method must be visibly trustworthy under scrutiny, since it is the basis for a real recall / no-recall decision with cost attached either way.
Assumptions
·	The 12-battery sample's behaviour during this telemetry window is a reasonable early signal for how the wider Kenyan fleet is behaving, even though it is not yet a full population.
·	The batteries' self-reported SOC and SOH values are usable analytical signals, even though they come from each unit's own battery management system rather than an independent bench test.
Constraints
·	A single ~6-hour telemetry snapshot, not a longitudinal history — this is a snapshot of one day, not a lifecycle view of these batteries.
·	A small sample (12 units) relative to the fleet size this method eventually needs to serve, which limits how confidently any single finding generalises without re-validation.
3.3 Risks and Contingencies
Risk	Contingency
The sample is too small or unrepresentative to generalise to the full fleet.	Treat findings as a validated hypothesis to re-test as the fleet scales, not a closed population-level claim — carried forward explicitly as a limitation rather than glossed over.
A naive "compare each battery to the fleet average" approach can fail silently, exactly when the true fault rate is high enough to distort the average itself.	Identified directly in this project's Modeling phase, and mitigated by using a clustering approach that doesn't depend on a possibly-contaminated baseline.
Reputational risk either way — publicly over- or under-reacting to a pattern before it's confirmed.	Keep findings internal to Engineering and Customer Service until the diagnosis is confirmed; act on confirmed serial numbers only, not on the complaint pattern alone.
Telemetry alone can identify which batteries are behaving abnormally, but not why at a component level.	Hand flagged units to Engineering for physical teardown; this analysis narrows the search, it doesn't replace it.

3.4 Terminology
Term	Meaning
SOC (State of Charge)	How full the battery is right now, as a percentage — the "fuel gauge."
SOH (State of Health)	How much usable capacity and performance remains compared to when the battery was new — the "condition of the tank itself," not just what's in it.
Drain rate	Percentage of SOC lost per hour while the motorcycle is being ridden.
Voltage stability	How steady a battery's voltage stays under load; a healthy pack holds a tight band, a failing one sags and spikes.
Control-limit / threshold test	A statistical rule that flags anything more than a set distance (e.g. 2 standard deviations) from the fleet's own average.
Clustering	A method that groups batteries by how similar their behaviour is, without being told in advance which group is "healthy" and which is "faulty."
Flagged vs. normal battery	The two groups this analysis ultimately sorts every battery into — flagged meaning statistically and behaviourally abnormal on multiple signals at once.

3.5 Costs and Benefits
The cost of doing this analysis properly is small: analyst time against an existing, already-collected telemetry sample. The cost of not doing it is not small: every unresolved complaint is a rider closer to switching to a competitor, every generic "within spec" response is a small erosion of trust that compounds across a fleet, and a genuine defect left unaddressed is a safety liability. The benefit of getting the diagnosis right is proportional and durable — a fix sized to the actual problem, a screening method KOFA keeps rather than discards, and a concrete, specific answer to give both engineering and riders instead of reassurance.

4. Determine Data Mining Goals
4.1 Data Mining Goals
Translating the business objective (B1) into an analytical one: given four behavioural signals per battery — drain rate, ride duration before shutdown, voltage stability, and self-reported SOH — identify which batteries, if any, are statistical outliers relative to the rest of the fleet.
Critically, the method has to stay reliable even if a large share of the fleet turns out to be affected — a real possibility that a naive "compare to average" approach handles badly, since a high fault rate distorts the very average it's being compared to. This turns out to be a genuine finding of this project's Modeling phase, not a hypothetical concern.
The output needs to be a specific, serial-number-level list of flagged batteries that Engineering and Customer Service can act on directly — not a risk score or a probability that still requires human interpretation before anyone can act on it.
4.2 Data Mining Success Criteria
·	A statistically clean, non-overlapping separation between "flagged" and "normal" groups — a bimodal split, not a vague continuum with ambiguous borderline cases.
·	That separation confirmed two independent ways (an unsupervised grouping method and a significance test), rather than relying on the judgement of a single technique.
·	Every flagged battery anomalous on multiple independent metrics at once, guarding against a single noisy sensor reading producing a false positive.
·	Plausible confounding explanations (starting with ambient temperature) explicitly tested and ruled out before attributing the pattern to the batteries themselves.
·	An output that is operationally reusable — a new battery's telemetry should be scoreable against the fitted model directly, without re-running the full analysis from scratch each time.

5. Produce Project Plan
5.1 Project Plan
Phase	Key Task	Output
Data Understanding	Load and structurally validate the 12-battery telemetry sample; check completeness; look at raw distributions and correlations before any modelling	Confirmed data quality, an early (unlabelled) visual hint of the pattern
Data Preparation	Collapse ~720 raw readings per battery into one row of behavioural summary features per battery	An analysis-ready, one-row-per-battery feature table
Modeling	Attempt a transparent statistical threshold first; diagnose why it fails on this fleet; apply clustering as the corrected approach	A validated 2-group split of the fleet, with the method's own limits documented
Evaluation	Test the split for statistical significance; rule out ambient temperature as a confound; assemble supporting visual evidence; document limitations	Confidence that the split reflects a real pattern, not sampling noise
Deployment	Turn the diagnosis into stakeholder-specific recommendations; define a monitoring cadence for the live fleet	Actioned recommendations, and a path to a repeatable screen

5.2 Initial Assessment of Tools and Techniques
Tools: Python (pandas, numpy) for data handling; matplotlib for visualisation; scikit-learn (StandardScaler, KMeans, clustering-quality metrics) for modelling; scipy.stats for significance testing — all already in use and proven on this dataset.
Techniques considered:
·	Statistical control limits (mean ± 2σ, cross-checked with the 1.5×IQR rule) — the natural first choice for its transparency: any stakeholder can see exactly why a unit was flagged. Planned as the first attempt, with the explicit understanding that it needs to be checked against reality rather than assumed to work.
·	Unsupervised clustering (K-Means) — planned as the fallback if the threshold approach proves unreliable, precisely because it doesn't require deciding in advance what "normal" looks like; it asks whether the fleet's own data splits into distinct groups instead.
Both are established, explainable techniques rather than a black box — a deliberate choice given that the ultimate audience for this analysis includes non-technical stakeholders (Customer Service, and by extension, riders) who need to trust the result, not just the data scientist who produced it.

6. Stakeholders
Behind every row of this table is someone whose day changes depending on what this analysis finds — not an abstract audience.
Stakeholder	What "success" looks like for them	What they do with this analysis
Engineering	A short, defensible list of which physical units are faulty, and what their behaviour actually looks like — not a vague "some batteries might be bad."	Pull the flagged units for teardown; trace manufacturing batch and date codes for a common root cause.
Customer Service	A serial-number-level list to check every incoming complaint against, so they can stop guessing.	Replace confirmed-flagged units without hesitation; stop issuing generic "within spec" responses that damage trust.
Product / Operations	Confirmation of whether this is isolated or systemic, and a method they can run again next quarter without a fresh investigation.	Decide whether fleet growth continues as planned; fold the screening logic into the live telemetry pipeline.
KOFA management & investors	A cost-of-inaction estimate and a right-sized response, at a moment when the company is trying to prove itself in a new, competitive market.	Approve (and, if needed, publicly explain) the response.
Riders	Either a fast, confirmed replacement, or confidence that their unit was actually checked rather than dismissed.	The direct, human reason this analysis exists — everything above ultimately serves this row.
Financing & insurance partners (e.g. M-KOPA-financed fleets)	Evidence that failures are tracked and resolved systematically, not anecdotally.	Factor into loan and insurance risk pricing for financed e-bodas.

This document — Business Understanding — is the phase everything else in the project has to answer to. Data Understanding, Modeling, and Evaluation can all be technically correct and still miss the point if they don't ultimately resolve B1: is this a few batteries, or is it Kore2 itself? Every later phase in this project's notebook is written to close that loop back t
