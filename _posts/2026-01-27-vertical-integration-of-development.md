---
layout: single
title: "The Vertical Integration of Development: How LLMs Are Revolutionizing the Software Development Cycle"
date: 2026-01-27
categories: [software-development, ai, llm]
tags: [claude, mcp, ai-assisted-development, developer-tools, productivity]
excerpt: "We're witnessing a fundamental shift in how software gets built—not just in what tools we use, but in how the entire development process connects together."
---

In a recent [post](https://x.com/karpathy/status/2015883857489522876?s=20) on X, the inimitable [Andrej Karpathy](https://karpathy.ai/) shared his impressions on the current state of coding using LLMs (and in particular Claude Code). In a nutshell, he believes we've crossed a critical threshold where going forward we are likely to have most of the code be written by LLMs.  I think many of us who are writing software using these tools feel this phase transition happening as well, especially going from where things were a year ago.

I believe _more is true_. We're witnessing a fundamental shift in how software gets built. Not just in *what* tools we use, but in *how* the entire development process connects together. I call this shift the **Vertical Integration of Development**—and I believe if you are a technical leader you should be paying attention to this, or you risk your organization being left in the dust. 

_ As always, any opinions expressed are solely my own and do not express the views or opinions of my employer. I welcome feedback and discussion on this post - find me via social links on this site!_


## What Do I Mean by "Vertical Integration"?

Traditionally, software development has been a fairly "horizontal" affair. You write code in your IDE. You have an observability stack of some sort to comb through logs and metrics. You search documentation in a browser. You coordinate with teammates in Slack. You manage deployments through yet another interface. Each tool is a silo, and ***you* are the integration layer—constantly context-switching, copy-pasting, and mentally stitching together information from disparate sources**.

The vertical integration of development flips this model. With modern LLM-powered tools like Claude Code, **the AI becomes the integration hub**. It doesn't just write code—it can simultaneously query your logs in Grafana, inspect your Kubernetes cluster state, read across multiple codebases, hit live API endpoints, traverse git history, and synthesize all of this into actionable insights. The development cycle collapses from a sprawling horizontal landscape into a unified vertical stack, with the LLM serving as the orchestration layer. 

This is pretty revolutionary in itself, but **the real a-ha moment comes when you experience how tools such as Claude Code use these integrations in a feedback loop**; Inspecting logs feeds into looking at the code, checking git commit history, inspecting Kubernetes cluster state, checking API response, etc etc.

This isn't incremental improvement. It's a phase change in how software engineering happens.

## The Enabling Technologies

In hindsight, I believe the culmination of these three recent developments has made this possible. Interestingly, these have all been initiatives started by Anthropic:

**Model Context Protocol (MCP)** provides a [standardized](https://modelcontextprotocol.io/) way for LLMs to connect to external tools and data sources. Instead of copy-pasting Grafana logs into a chat window, the LLM can query Grafana directly, in real-time, with full context about what it's looking for. A vast catalog of MCP servers already exists, see for example [here](https://www.pulsemcp.com/servers).

**Skills and Plugins** allow teams to package domain expertise into reusable components. These aren't just prompts—they're structured knowledge about *how* to accomplish specific tasks within your organization's context. The growing ecosystem (see [skillsmp.com](https://skillsmp.com)) means you're not starting from scratch. The [Agent Skills open standard](https://agentskills.io/home) has been recently also adopted by Cursor, so this seems to be something we'll see industry consolidation around.

**Rules and Memory (CLAUDE.md)** let teams codify conventions, patterns, and guardrails that keep the LLM aligned with your codebase's idioms. This is crucial: without steering, LLMs have biases that can introduce subtle inconsistencies—dropping inline imports, using magic literals, creating unnecessary backward-compatibility shims. Rules transform the LLM from a generic assistant into one that understands *your* team's way of working. A recent shift is happening toward a more general notion of [Memory](https://code.claude.com/docs/en/memory#determine-memory-type) as a sort of queryable store that can have multiple files referencing each other.

## Case Studies: Vertical Integration in Practice

The above all sounds nice and well in theory, but what does it look like in practice? I'll share below a few use cases, all based on my actual usage from just the last few weeks. These are real-world scenarios where **this workflow saved us hours and days**, and in some cases helped find extremely subtle issues that might as well have taken weeks to resolve, if at all.

### Debugging Across the Stack

Here's a scenario that previously would have taken hours of context-switching: we had a tricky QA issue that spanned multiple services. Using Claude Code with MCP integrations, we were able to:

1. Query Grafana logs directly to identify the failure pattern
2. Cross-reference three separate codebases (backend, frontend, gateway) all checked out locally
3. Inspect responses from live API endpoints via HTTP calls
4. Traverse git commit history to understand when the behavior changed

The LLM synthesized all of this information simultaneously, pinpointing the root cause in minutes rather than the hours of "Grafana spam" that would have been required otherwise. This is what vertical integration looks like in practice—the AI isn't just a coding assistant, it's operating across the entire observability and debugging stack.

### Rapid Prototyping at Unprecedented Speed

We've been building proof-of-concept implementations in 2-3 days that would have previously taken weeks. More importantly, when requirements change—say, migrating to a different framework—iteration happens in *hours*, not days. The key enabler is that we can point Claude Code at existing implementations and say "align with this pattern," and it understands the broader architectural context.

Having Claude Code build an End-to-end POC is quite impressive, and I've found that prompting here can make for a big difference in quality. I would typically mention desired architecture (e.g. RESTful API service, Event processing daemon, etc.), frameworks (e.g. fastapi, FastStream, React) and some design patterns, high-level architecture I'd like to follow. This is usually enough for it to go off and build a working system which I can then iterate on. 

An interesting side-effect here is that its worthwhile keeping PoCs as a monolithic repository / codebase for as long as possible - this makes it very easy to update the system as a whole (e.g. API, persistent storage layer and FE can all change in tandem) using something like Claude Code.

### Tool Building Becomes Viable

Here's an underappreciated impact: tools that were never worth building before suddenly have positive ROI.

At a startup, you rarely have bandwidth to build internal tooling. The maintenance burden isn't worth it for a 10-person team. But when the LLM can scaffold these tools in hours and help maintain them, the calculus changes completely. This relates to the "building is fun again" sentiment sweeping through X apropos of using these tools to write code. 

These past few months I found myself building tools that I now use daily, with some gaining traction across teams:

- **[Mockstack](https://github.com/promptromp/mockstack)**: A smart proxy and microservice mocking layer that enables running complex event-based flows locally without replicating the entire platform.
- **[pytest-impacted](https://github.com/promptromp/pytest-impacted)**: A pytest plugin that selectively runs unit-tests based on introspection of git commit history, performing AST and code dependency graph analysis to figure out impacted tests.
- **Custom Agent Skills**: Including a [remote Python process debugging skill](https://github.com/promptromp/python-remote-debug-skill/tree/main) leveraging the recent Python 3.14 remote debugging capabilities.
- **Custom MCP servers**: Including a Grafana integration that makes log analysis conversational
- Internal tooling for pulling up all relevant links and metadata for various platform resources, saving precious minutes of manually using Postman or otherwise clicking through links to find UUIDs, etc etc.

None of these would have likely seen the light of day if I had no LLM agents to take care of a lot of the time consuming drudgery involved in creating such tooling. All of them now pay dividends daily.

### Kubernetes Debugging Without the Pain

Anyone who has debugged production Kubernetes issues knows the pain: you're constantly jumping between `kubectl` commands, config files, logs, and documentation. In a recent case, we had Claude Code introspect our dev cluster state, cross-reference it with our platform configuration and Grafana logs, and identify that a service was down due to a misconfigured port. This kind of issue is notoriously difficult to debug because it requires holding multiple contexts simultaneously—exactly what vertical integration excels at.

## What's Changed in the Last Six Months

If you tried LLM-assisted development a year ago and were underwhelmed, it's time to revisit. The improvements have been substantial:

**Model quality**: Recent models (Claude 4.5, etc.) are dramatically better at maintaining context, following complex instructions, and producing idiomatic code.

**Application layer maturity**: Tools like Claude Code have moved from interesting experiments to genuinely productive workflows. The CLI-based interaction model, combined with MCP support, creates a much tighter feedback loop than chat interfaces.

**Ecosystem growth**: The marketplace for skills, plugins, and MCP servers means you can leverage community work rather than building everything yourself.

## A Vision for the Next 12-24 Months

Where is this heading? I see several trends converging:

**The LLM Agent as command hub**: Tools like [Claude Code](https://code.claude.com/docs/en/overview), [Codex](https://developers.openai.com/codex/cli/), [Cursor CLI](https://cursor.com/cli) etc. become the primary interface for development work, with MCP providing connectivity to everything else—observability, deployment, documentation, communication. Right now the trend is for these to be TUI / CLI style tools although IDE integrations (E.g. Cursor IDE, VS Code) is fairly popular too. exact form factor TBD.

**Organizational knowledge codified**: Teams will invest heavily in CLAUDE.md style rules and custom skills that capture institutional knowledge. The competitive advantage shifts from "who has the best developers" to "who has best captured their development practices in a form LLMs can leverage.". These can be distributed e.g. via a GitHub repository (public or private) serving as a "plugin / skill marketplace" and rules files can be distributed either per repository or in a centralized way (I suspect [Git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) might rear their ugly head here for some quick wins :-))

**Cloud-hybrid execution**: The ability to report issues and open candidate PRs directly from Slack (via cloud-based execution) while maintaining the power of local CLI tools creates flexibility in how and where work happens. Claude Code in particular recently introduced [Slack Integration](https://code.claude.com/docs/en/slack) and Google has been experimenting with [Jules](https://jules.google/) for a while now, letting you access it via the web and create PRs on the fly. This holds a lot of promise, although for the time being I find the power of having a full development environment with all the needed permissions on my local dev workstation a clear winner.

**Cross-repository development**: This is still an edge case today, but rapidly improving. As context windows grow and tools better support multi-project workspaces, the artificial boundaries between repositories will matter less. Having BE microservices, GraphQL API gateways, IaaC repositories, all available for the LLM to switch back and forth between to triage issues will become a common workflow.

## The Remaining Challenges

This isn't all roses. Some open questions:

**Context management**: Even with larger context windows, understanding complex systems requires loading a lot of information. Curated rules and skills help, but there's still work to do here. I currently find myself using `/clear` and `/compact` a lot at key points, or asking the LLM to summarize state into an .md file I will pull back later (essentially relying on the notion of queryable 'Memory' mentioned above). The idea of [Progressive disclosure](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices#progressive-disclosure-patterns) will probably play a big role here. 

**Cross-repository coherence**: Working across multiple repos is possible but spotty. The LLM needs significant context to understand how services interact, and there's no great solution yet for keeping that understanding current. Per-repository CLAUDE.md files that are periodically refreshed do the trick for me for now, in the future there might be some automation around keeping these rules files up to date automatically (Note to self, another fun project idea: perhaps a [pre-commit](https://pre-commit.com/) hook that triggers claude code CLI to refresh CLAUDE.md based on latest git commits?)

**Verification overhead**: As LLMs take on more complex tasks, verifying their work becomes its own challenge. The time saved in generation can be eaten by review if you're not careful. Part of my workflow is emphasizing Test-driven development at every corner and every turn, and also relying on integration testing by running processes locally and asking Claude Code to interact with them. I typically have a process running in background, tailing to a log file, using some hot-reload functionality (e.g. via [Watchdog](https://github.com/gorakhargosh/watchdog)). This lets the LLM look at the log, make changes, see those get updated automatically, look at the log again etc. This often finds issues that aren't caught in unit-tests alone.

## The Bottom Line

The vertical integration of development isn't a productivity hack—it's a fundamental restructuring of how software gets built. The developers who thrive in this new world won't necessarily be the ones who write the most code; they'll be the ones who are most effective at orchestrating LLM capabilities across the entire development lifecycle.

If you're still thinking of LLMs as "fancy autocomplete," you're missing the bigger picture. The integration layer is where the leverage is, and the tools to build that layer are here now.

---

*What's your experience with vertically integrated development workflows? I'd love to hear what's working (or not) for your team.*
