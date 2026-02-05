---
layout: single
title: "Projects"
permalink: /projects/
author_profile: true
header:
  image: /assets/images/ai-pixelart.png
  image_description: "Pixel art AI and technology landscape"
classes: wide
---

<style>
  .page__hero--overlay {
    padding: 3em 0 !important;
    min-height: 480px !important;
  }
  .page__hero-image {
    object-fit: cover !important;
    object-position: center 55% !important;
    max-height: 480px !important;
    width: 100% !important;
  }
</style>

## Projects

Here you'll find a collection of my technical projects, experiments, and open-source contributions.

### PromptRomp - Dev Tools for the LLM Generation

I'm a core member of [PromptRomp](https://github.com/promptromp/), an open-source organization building developer productivity tools for the age of AI-assisted development. Our mission is to create practical tooling that enhances developer workflows, particularly for teams leveraging LLM-enabled development practices.

**GitHub**: [github.com/promptromp](https://github.com/promptromp/)

**Key Projects**:

- **[pytest-impacted](https://github.com/promptromp/pytest-impacted)** - A smart testing utility that selectively runs only the tests affected by your code changes. Using git analysis and dependency mapping, it dramatically reduces test execution time while maintaining confidence in your test coverage.

- **[mockstack](https://github.com/promptromp/mockstack)** - An API mocking workhorse designed to streamline API development and testing workflows. Built to handle complex API simulation scenarios with ease.

- **[pdfalive](https://github.com/promptromp/pdfalive)** - A Python library and CLI toolkit that enhances PDF files using LLM capabilities. Features include automatic table of contents generation with clickable bookmarks, intelligent OCR detection for scanned documents, and smart batch file renaming using natural language instructions. Supports multiple LLM providers via LangChain (OpenAI, Anthropic, Ollama).

- **[diastra](https://github.com/promptromp/diastra)** - DIaSTRA (Distributed Architecture Specification Transpiler) is a specification and framework ecosystem for writing large-scale, distributed, stateful cloud-first software. It leverages LLMs to transpile high-level system specifications (using concepts like Entities, State Machines, Events, Workflows and Sagas) into microservices, code, and durable storage definitions.

- **[python-remote-debug-skill](https://github.com/promptromp/python-remote-debug-skill)** - A Claude Code skill for debugging running Python processes using Python 3.14+ remote debugging via `sys.remote_exec()`. Inject debugging scripts into live processes to get stack traces without stopping them, with special handling for gevent-based workers (like Celery with `-P gevent`).

- **[aws-bootstrap-g4dn](https://github.com/promptromp/aws-bootstrap-g4dn)** - One command to go from zero to a fully configured GPU dev box on AWS. Handles spot instance launching, automatic SSH config, CUDA-aware PyTorch installation, Jupyter Lab setup, and GPU benchmarking. Supports Jupyter server-client, VSCode Remote SSH, and NVIDIA Nsight remote debugging workflows.

All PromptRomp projects are open-source and welcome contributions from the community.

---

Feel free to also explore more of my work on [GitHub](https://github.com/adamhadani).
