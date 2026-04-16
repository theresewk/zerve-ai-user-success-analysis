**Executive Summary**

I analysed Zerve’s event data to define what “successful usage” means in a workflow product and to identify the early behaviours most strongly associated with it. I looked at operationalised success as a ladder: `Activate → Habit → Ship`, where activation is first execution, habit is repeated execution over time, and ship is completed output. This framing is more meaningful than traffic or sign-ups because Zerve creates value when users actually do and complete work.

On the modelled base of `4,771` users, `16.77%` activated, `2.47%` formed habit, and `0.29%` shipped. This means most users entered the product, but only a minority progressed into meaningful workflow usage, and very few reached repeated or completed outcomes. Among users who did progress, the early journey happened quickly: first canvas, build, and run typically occurred within minutes or the same hour.

The strongest descriptive divider in the dataset is not subtle feature usage but user mode: `81.2%` of users were `observer-only`, while the active cohorts (`agent-led` and `manual-led`) were dramatically more likely to activate and progress. Workflow-native entry points such as `canvas` and `notebook` were also associated with much stronger activation than broad surfaces such as `home`.

Credits friction appeared mainly among highly active users, but high activity under friction did not translate proportionally into shipped outcomes. Overall, the data suggests that Zerve’s core challenge is not speeding up successful users, but increasing the share of users who enter real workflow execution at all.

---

## Zerve Success Ladder + Drivers Analysis

**Objective**

Define what “successful usage” means in Zerve and identify the early behaviours, workflows, and friction patterns most strongly associated with it, using only observed event data.

**Success Framework**

Successful usage was operationalised as:

- `Activate`: user runs within 7 days of `t0`
- `Habit`: user runs on 2+ distinct days within 14 days
- `Ship`: user reaches a shipped state
- `Maturity signal`: ship-then-use

This ladder fits Zerve’s product reality as a workflow tool: value is not just arrival or exploration; but execution, repeated use, and completed output.

**What Happened Overall**

On the modelled base of `4,771` users:

- `16.77%` activated (`800`)
- `2.47%` formed habit (`118`)
- `0.29%` shipped (`14`)
- `0.06%` shipped and used again (`3`)

This means most users entered the product, but only a minority reached real workflow execution, and only a very small minority progressed to repeat usage or shipped output.

For users who did progress, the early journey happened fast:

- median time to first canvas: `0.0h`
- median time to first build: `0.1h`
- median time to first run: `0.1h`
- median time to first ship: `94.3h`

*Implication*:

The biggest challenge is not speed to first action among successful users. It is that most users never enter meaningful workflow execution at all.

**Who Succeeds**

Persona analysis shows the clearest structural split in the user base:

- `observer-only`: `81.2%`
- `agent-led`: `12.0%`
- `manual-led`: `6.8%`

Success rates differ sharply by persona:

- `agent-led`: `91.6%` run within 7d
- `manual-led`: `84.3%` run within 7d
- `observer-only`: `0.0%` run within 7d

*Implication:*

The user base is not one funnel. It is a large passive cohort plus two much smaller active cohorts. The biggest divide is between users who begin doing real work early and users who remain observers.

**What Successful Workflows Look Like**

Session analysis found `11,949` sessions across `4,771` users, averaging `2.5` sessions per user, with median collapsed session length of `1.0`. Most sessions were short and simple.

The strongest workflow-like transitions were around:

- `CANVAS → AGENT_TOOL`
- `BUILD → RUN`
- actions following `RUN` and `AGENT_TOOL`

The Markov flow also showed many transitions back into generic activity (`OTHER`), meaning real work patterns exist, but they sit inside a broader layer of exploratory or uncategorised interaction.

*Implication:*

The product does show identifiable workflow paths, but the dominant behavioural pattern is still not a tight linear progression. Real work exists, but it is concentrated in a smaller active subset.

**Where Success Starts**

Entry context matters materially.

By surface, users entering workflow-native areas perform much better than users entering broad or informational surfaces:

- `canvas`: `57.6%` run within 7d, `1.5%` shipped
- `notebook`: `52.3%` run within 7d
- `home`: `5.8%` run within 7d

By device:

- `desktop`: `19.1%` run within 7d
- `mobile`: `5.1%` run within 7d

*Implication:*

How users enter the product strongly shapes whether they become active workflow users.

**What Friction Shows**

Early friction was relatively rare in the full base, but concentrated in active users:

- credits warning in 7d: `5.11%`
- credits exceeded in 7d: `1.57%`
- agent-error friction in 7d: `1.38%`
- stop events in 7d: `0.78%`

Credits cohorts showed:

- `no_friction`: `13.0%` activate, `2.1%` habit, `0.3%` ship
- `warn_only`: `78.9%` activate, `5.1%` habit, `0.6%` ship
- `exceeded`: `98.7%` activate, `20.0%` habit, `0.0%` ship

*Implication:*

Friction is not evidence of healthy outcomes. It appears mostly among highly active users, but that activity does not convert proportionally into shipped output. In particular, exceeded-credit users are highly active but show no observed shipping in this sample.

**Conclusion**

The clearest overall conclusion is that Zerve’s observed user base is dominated by entrants who never become active workflow users. Success is strongly associated with early entry into real execution, active working mode, and workflow-native entry surfaces. Friction appears among engaged users, but engaged usage does not automatically translate into completed output.