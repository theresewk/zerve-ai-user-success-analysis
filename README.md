# zerve-ai-user-success-analysis
Behavioral analytics project analyzing 4,700+ users to identify what drives success in a data platform. Combines persona segmentation, activation metrics, funnel analysis, and Markov chains to uncover friction points, usage patterns, and predictors of long-term user success.

> **Hackathon submission for Zerve.AI x HackerEarth March 2026 Data Challenge**

# 4 Lenses on Success: Execution, Persona, Entry, Patterns

## Overview

This project explores a central question: **what drives successful usage of a data platform?**

Using real-world event data from 4,771+ users, we analyze behavioral patterns, workflows, and friction points to identify what differentiates successful users from the rest.

The project is structured around four analytical lenses:

* **Execution** – how users interact with core features
* **Persona** – behavioral segmentation of users
* **Entry** – onboarding and early activation signals
* **Patterns** – session flows and long-term usage behavior

---

## Objectives

* Define and quantify "user success"
* Identify behaviors that predict long-term engagement
* Detect friction points that block activation
* Analyze how workflows evolve over time

---

## Dataset

The dataset consists of platform event logs, including:

* Canvas creation
* Code execution events
* Deployments
* Agent interactions
* Session-level activity

__Dataset provided by Zerve.AI for the hackathon.__

Due to size and privacy constraints, the dataset is not included in this repository.

---

## Methodology

### 1. Persona Segmentation

Users are grouped into behavioral personas:

* **Agent-led** – rely heavily on automation/agents
* **Manual-led** – primarily execute workflows manually
* **Observer-only** – minimal interaction, mostly passive

---

### 2. Success Metrics (Activation Ladder)

We define success using a progression of 8 activation flags, representing increasing engagement and platform adoption.

Examples include:

* First execution
* Multi-step workflows
* Deployment activity
* Repeated usage patterns

---

### 3. Funnel Analysis

We track user progression through key milestones:

* Entry → First action → Activation → Advanced usage

This helps identify:

* Drop-off points
* Conversion rates
* Time-to-milestone distributions

---

### 4. Friction Analysis

We identify blockers that prevent user success:

* Credit limitations
* Execution failures
* Agent errors
* Seat constraints

We measure how each friction point impacts:

* Conversion rates
* Retention
* Depth of usage

---

### 5. Session Pattern Analysis (Markov Chains)

User sessions are modeled as state transitions to uncover:

* Common workflows
* High-performing paths
* Dead-end or inefficient flows

This reveals how successful users navigate the platform differently.

---

## Key Insights

* Clear behavioral differences exist between high- and low-success users
* Early activation signals strongly predict long-term engagement
* Certain friction points disproportionately impact conversion
* Successful users follow more structured and repeatable workflows

---

## Outputs & Visualizations

* Conversion funnels
* Persona distributions
* Time-to-activation charts
* Friction impact analysis
* Markov transition diagrams

All outputs are reproducible within the Zerve environment.

---

## Reproducibility

This project is fully reproducible:

* All analysis is contained within the Zerve project
* Data transformations and models are documented
* Visualizations are generated programmatically

---

## Tech & Approach

* Event-driven behavioral analysis
* Statistical aggregation
* Markov chain modeling
* Funnel and cohort analysis

---

## Conclusion

Success in a data platform is not defined by a single action, but by a combination of:

* Early activation
* Workflow depth
* Consistency of usage
* Ability to overcome friction

This project provides a structured framework to measure and predict that success.

---
