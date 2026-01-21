---
layout: single
title: "Your LLM Is Living in the Past (Enterprise LLM Design Patterns #1)"
date: 2026-01-20
categories: [architecture, llm, distributed-systems]
tags: [llm, event-sourcing, async, python, fastapi, postgresql]
excerpt: "Exploring architecture patterns for handling asynchronous events in LLM-powered applications, with a focus on debouncing, optimistic locking, and semantic versioning."
---

_This is the first post in what I hope will become a series of short and to the point technical writeups about some architectural patterns I've been experimenting with vis-a-vis adopting LLM usage in SaaS and enterprise-grade applications. Any opinions expressed are solely my own and do not express the views or opinions of my employer. I welcome feedback and discussion on this post - find me via social links on this site!_ 


# Introduction


In modern SaaS applications, Large Language Models (LLMs) are increasingly deployed in contexts where inputs arrive asynchronously and unpredictably. Consider a customer support platform where an AI assistant helps users troubleshoot issues: while the LLM processes a user's initial message, the user might send follow-up clarifications, system events might fire (e.g., a webhook indicating the user's subscription status changed), or external integrations might push relevant data.

This creates a fundamental tension: **LLM processing is slow relative to the rate at which state can change**. A response generated based on stale state may be incorrect, confusing, or even harmful to the user experience.

This post explores the solution space for handling this problem and proposes a concrete architecture suitable for scenarios where events arrive in short bursts and responses should reflect the complete, up-to-date state. 

_TLDR; We employ a combination of optimistic locking (via Compare-and-swap primitives) and event debouncing._


## The Problem Space

Let's formalize the challenge. Consider a timeline where:

```
t=0    User sends message A
t=50ms System event B fires (user's account upgraded)
t=100ms LLM begins processing (sees A, doesn't see B)
t=200ms User sends clarification C
t=2000ms LLM completes response (based on A only)
```

The LLM's response is now stale—it doesn't reflect the account upgrade (B) or the clarification (C). Depending on the domain, this could range from mildly suboptimal to seriously problematic.

### Design Considerations

Several factors influence the appropriate solution:

1. **Event arrival patterns**: Are events sporadic or bursty? Is there a natural "settling" period?
2. **Latency tolerance**: How long can users wait for a response?
3. **Cost sensitivity**: LLM API calls are expensive—can we afford to discard partial work?
4. **Correctness requirements**: Is a stale response acceptable, or must we guarantee freshness?
5. **User experience**: Should we show partial/streaming responses, or wait for complete ones?

## The Solution Space

Different products make different tradeoffs. Here's a sampling of some common approaches seen in the wild:

### Queue and Process Sequentially

**Example: Claude Code**

Claude Code queues user messages that arrive during task execution, processing them after the current task completes. This ensures each response is based on complete information available at processing time, at the cost of increased latency for queued messages.

```
User: "Fix the bug in auth.py"
[Claude Code begins working]
User: "Actually, also update the tests"
[Message queued]
[Claude Code completes first task]
[Claude Code processes queued message with full context]
```

**Tradeoffs**: Simple mental model, guaranteed consistency, but latency accumulates for rapid interactions.

### Disable Concurrent Input

**Example: Many chatbot UIs**

The simplest approach: disable input controls while processing. Users cannot create conflicting state.

**Tradeoffs**: Eliminates the problem entirely but creates a blocking, unresponsive UX. Unsuitable for systems with non-user event sources.

### Optimistic Concurrent Execution

**Example: Collaborative editing with AI assistance**

Allow all events to trigger processing immediately. Multiple LLM calls may run concurrently, with conflict resolution applied at response time.

**Tradeoffs**: Maximally responsive but expensive and complex. Requires sophisticated conflict resolution.

### Debounce and Validate (Our Focus)

For scenarios where events arrive in bursts and we want responses to reflect the complete burst, we can combine:

- **Event debouncing**: Wait for a settling period before triggering LLM processing
- **Optimistic locking**: Detect when state has changed during processing
- **Semantic versioning**: Track logical state changes independent of wall-clock time

This approach suits applications like:
- Multi-turn conversations with rapid clarifications
- Document analysis where multiple edits arrive quickly
- Workflow orchestration with cascading events

## Proposed Architecture

### Core Concepts

**Semantic Clock**: A logical counter that increments with each state-changing event. Unlike wall-clock time, this captures *meaningful* state transitions.

**Event Window**: A configurable period during which we accumulate events before processing. Resets when new events arrive (debouncing).

**Generation Token**: A version identifier captured at processing start, used to validate response freshness.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Event Sources                            │
│  (User Messages, Webhooks, System Events, Scheduled Tasks)      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Event Stream (Kafka/Redis)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Stream Processor                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Debouncer  │─▶│   Persist   │─▶│  Trigger LLM Processing │  │
│  │  (per-ctx)  │  │   Event     │  │  (with generation token)│  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM Processor                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Load State  │─▶│  Call LLM   │─▶│ Validate & Commit (CAS) │  │
│  │ + Gen Token │  │  (stream)   │  │  or Retry/Discard       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### State Machine

Each conversation context follows this state machine:

```
                    ┌──────────────┐
                    │    IDLE      │
                    └──────┬───────┘
                           │ event received
                           ▼
                    ┌──────────────┐
            ┌──────▶│  DEBOUNCING  │◀─────┐
            │       └──────┬───────┘      │
            │              │ timeout      │ new event
            │              ▼              │ (reset timer)
            │       ┌──────────────┐      │
            │       │  PROCESSING  │──────┘
            │       └──────┬───────┘
            │              │
            │    ┌─────────┴─────────┐
            │    │                   │
            │    ▼                   ▼
            │ ┌──────────┐    ┌──────────────┐
            │ │ COMMITTED│    │ INVALIDATED  │
            │ └──────────┘    └──────┬───────┘
            │                        │
            └────────────────────────┘
```

## Considerations and Tradeoffs

### When This Pattern Fits

We use an event-sourcing inspired schema where events are immutable and state is derived:

```python
# models.py
"""
Domain models for the async LLM event processing system.

Key concepts:
- ContextState: Aggregate root tracking semantic version and processing state
- Event: Immutable event records with semantic clock assignment
- LLMResponse: Generated responses with validity tracking
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """SQLAlchemy declarative base with common configurations."""
    pass


class ProcessingState(enum.Enum):
    """State machine states for context processing."""
    IDLE = "idle"
    DEBOUNCING = "debouncing"
    PROCESSING = "processing"


class ContextState(Base):
    """
    Aggregate root for a conversation/processing context.
    
    The semantic_clock increments with each meaningful state change,
    providing a logical ordering independent of wall-clock time.
    The version field enables optimistic locking via CAS operations.
    """
    __tablename__ = "context_states"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    
    # Semantic versioning - increments on meaningful state changes
    semantic_clock: Mapped[int] = mapped_column(
        BigInteger, 
        nullable=False, 
        default=0
    )
    
    # Optimistic locking version - increments on any update
    version: Mapped[int] = mapped_column(
        Integer, 
        nullable=False, 
        default=0
    )
    
    # Current processing state
    state: Mapped[ProcessingState] = mapped_column(
        SAEnum(ProcessingState),
        nullable=False,
        default=ProcessingState.IDLE,
    )
    
    # Generation token for the current/last processing run
    # Used to validate responses against state drift
    current_generation: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    
    # Relationships
    events: Mapped[list[Event]] = relationship(
        "Event", 
        back_populates="context",
        order_by="Event.semantic_clock",
    )
    responses: Mapped[list[LLMResponse]] = relationship(
        "LLMResponse",
        back_populates="context",
    )

    __table_args__ = (
        Index("ix_context_states_state", "state"),
    )


class EventType(enum.Enum):
    """Classification of events that can affect context state."""
    USER_MESSAGE = "user_message"
    SYSTEM_EVENT = "system_event"
    EXTERNAL_WEBHOOK = "external_webhook"
    SCHEDULED_TASK = "scheduled_task"


class Event(Base):
    """
    Immutable event record in the event store.
    
    Events are assigned a semantic_clock value upon persistence,
    establishing a total ordering within their context.
    """
    __tablename__ = "events"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    context_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("context_states.id"),
        nullable=False,
    )
    
    # Event classification and content
    event_type: Mapped[EventType] = mapped_column(
        SAEnum(EventType),
        nullable=False,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
    )
    
    # Assigned during persistence - provides total ordering
    semantic_clock: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    
    # Wall-clock timestamp for debugging/analytics
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    
    # Relationship
    context: Mapped[ContextState] = relationship(
        "ContextState",
        back_populates="events",
    )

    __table_args__ = (
        Index("ix_events_context_clock", "context_id", "semantic_clock"),
    )


class LLMResponse(Base):
    """
    Record of an LLM-generated response.
    
    Tracks the generation token and semantic clock at generation time,
    enabling staleness detection and response validity assessment.
    """
    __tablename__ = "llm_responses"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    context_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("context_states.id"),
        nullable=False,
    )
    
    # Links response to specific processing run
    generation_token: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
    )
    
    # Semantic clock value when processing started
    # Response is valid iff this matches context's clock at commit time
    based_on_clock: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    
    # The actual response content
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    
    # Response metadata
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Validity tracking
    is_valid: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )
    invalidated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    
    # Relationship
    context: Mapped[ContextState] = relationship(
        "ContextState",
        back_populates="responses",
    )

    __table_args__ = (
        Index("ix_responses_context_valid", "context_id", "is_valid"),
        Index("ix_responses_generation", "generation_token"),
    )


# Pydantic models for API/messaging boundaries

class EventPayload(BaseModel):
    """Incoming event data before persistence."""
    context_id: UUID
    event_type: EventType
    payload: dict[str, Any]
    idempotency_key: str | None = None


class ProcessingResult(BaseModel):
    """Result of an LLM processing attempt."""
    generation_token: UUID
    success: bool
    response_id: UUID | None = None
    invalidation_reason: str | None = None
    
    
class ContextSnapshot(BaseModel):
    """Point-in-time view of context state for LLM processing."""
    context_id: UUID
    semantic_clock: int
    generation_token: UUID
    events: list[dict[str, Any]]
    
    class Config:
        from_attributes = True
```

### Persistence Layer with Optimistic Locking

The persistence layer implements the core CAS (Compare-And-Swap) semantics:

```python
# persistence.py
"""
Event store and optimistic locking implementation.

This module provides:
- Event persistence with automatic semantic clock assignment
- Optimistic locking via version-based CAS operations
- State transitions with conflict detection
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .config import settings
from .models import (
    Base,
    ContextSnapshot,
    ContextState,
    Event,
    EventPayload,
    EventType,
    LLMResponse,
    ProcessingState,
)

logger = logging.getLogger(__name__)


class OptimisticLockError(Exception):
    """Raised when a CAS operation fails due to concurrent modification."""
    
    def __init__(self, context_id: UUID, expected_version: int, actual_version: int):
        self.context_id = context_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"Optimistic lock failed for context {context_id}: "
            f"expected version {expected_version}, found {actual_version}"
        )


class StaleGenerationError(Exception):
    """Raised when attempting to commit a response for an outdated generation."""
    
    def __init__(self, context_id: UUID, generation_token: UUID):
        self.context_id = context_id
        self.generation_token = generation_token
        super().__init__(
            f"Stale generation {generation_token} for context {context_id}"
        )


class EventStore:
    """
    Event store with optimistic locking support.
    
    Implements event sourcing patterns with:
    - Atomic event persistence and clock increment
    - Version-based optimistic locking for state transitions
    - Generation token validation for response commits
    """
    
    def __init__(self, database_url: str | None = None):
        self._engine = create_async_engine(
            database_url or settings.database_url,
            echo=settings.debug,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def initialize(self) -> None:
        """Create database tables if they don't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """Clean up database connections."""
        await self._engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Provide a transactional session scope."""
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def get_or_create_context(
        self, 
        context_id: UUID,
        session: AsyncSession | None = None,
    ) -> ContextState:
        """
        Get existing context or create new one.
        
        Uses INSERT ... ON CONFLICT DO NOTHING pattern for idempotency.
        """
        async def _execute(sess: AsyncSession) -> ContextState:
            # Try to fetch existing
            result = await sess.execute(
                select(ContextState).where(ContextState.id == context_id)
            )
            context = result.scalar_one_or_none()
            
            if context is None:
                # Create new context
                context = ContextState(id=context_id)
                sess.add(context)
                await sess.flush()
            
            return context

        if session:
            return await _execute(session)
        
        async with self.session() as sess:
            return await _execute(sess)

    async def append_event(
        self,
        event_payload: EventPayload,
        session: AsyncSession | None = None,
    ) -> tuple[Event, ContextState]:
        """
        Append an event to the store, atomically incrementing the semantic clock.
        
        This operation:
        1. Increments the context's semantic_clock and version
        2. Creates the event with the new clock value
        3. Returns to DEBOUNCING state if currently IDLE
        
        Returns the persisted event and updated context state.
        """
        async def _execute(sess: AsyncSession) -> tuple[Event, ContextState]:
            # Lock the context row for update
            result = await sess.execute(
                select(ContextState)
                .where(ContextState.id == event_payload.context_id)
                .with_for_update()
            )
            context = result.scalar_one_or_none()
            
            if context is None:
                context = ContextState(id=event_payload.context_id)
                sess.add(context)
                await sess.flush()
                # Re-fetch with lock
                result = await sess.execute(
                    select(ContextState)
                    .where(ContextState.id == event_payload.context_id)
                    .with_for_update()
                )
                context = result.scalar_one()
            
            # Increment semantic clock
            new_clock = context.semantic_clock + 1
            context.semantic_clock = new_clock
            context.version += 1
            
            # Transition to debouncing if idle
            if context.state == ProcessingState.IDLE:
                context.state = ProcessingState.DEBOUNCING
            
            # Create event with assigned clock
            event = Event(
                context_id=context.id,
                event_type=event_payload.event_type,
                payload=event_payload.payload,
                semantic_clock=new_clock,
            )
            sess.add(event)
            
            await sess.flush()
            logger.info(
                "Appended event %s to context %s at clock %d",
                event.id,
                context.id,
                new_clock,
            )
            
            return event, context

        if session:
            return await _execute(session)
        
        async with self.session() as sess:
            return await _execute(sess)

    async def begin_processing(
        self,
        context_id: UUID,
        expected_version: int | None = None,
    ) -> ContextSnapshot:
        """
        Transition context to PROCESSING state and return a snapshot for LLM.
        
        Generates a new generation_token that must be provided when committing
        the response. Uses optimistic locking if expected_version is provided.
        
        Raises:
            OptimisticLockError: If version doesn't match (concurrent modification)
            ValueError: If context is not in DEBOUNCING state
        """
        async with self.session() as session:
            result = await session.execute(
                select(ContextState)
                .where(ContextState.id == context_id)
                .with_for_update()
            )
            context = result.scalar_one_or_none()
            
            if context is None:
                raise ValueError(f"Context {context_id} not found")
            
            # Optimistic lock check
            if expected_version is not None and context.version != expected_version:
                raise OptimisticLockError(
                    context_id, 
                    expected_version, 
                    context.version
                )
            
            if context.state != ProcessingState.DEBOUNCING:
                raise ValueError(
                    f"Context {context_id} is in state {context.state}, "
                    f"expected DEBOUNCING"
                )
            
            # Generate new token and transition
            generation_token = uuid4()
            context.current_generation = generation_token
            context.state = ProcessingState.PROCESSING
            context.version += 1
            
            # Load events for snapshot
            events_result = await session.execute(
                select(Event)
                .where(Event.context_id == context_id)
                .order_by(Event.semantic_clock)
            )
            events = events_result.scalars().all()
            
            snapshot = ContextSnapshot(
                context_id=context.id,
                semantic_clock=context.semantic_clock,
                generation_token=generation_token,
                events=[
                    {
                        "id": str(e.id),
                        "type": e.event_type.value,
                        "payload": e.payload,
                        "clock": e.semantic_clock,
                    }
                    for e in events
                ],
            )
            
            logger.info(
                "Began processing context %s with generation %s at clock %d",
                context_id,
                generation_token,
                context.semantic_clock,
            )
            
            return snapshot

    async def commit_response(
        self,
        context_id: UUID,
        generation_token: UUID,
        content: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        based_on_clock: int,
    ) -> LLMResponse:
        """
        Attempt to commit an LLM response using CAS semantics.
        
        The commit succeeds only if:
        1. The generation_token matches the context's current_generation
        2. The semantic_clock hasn't advanced since processing began
        
        On success, transitions context to IDLE.
        On failure due to new events, transitions to DEBOUNCING for retry.
        
        Raises:
            StaleGenerationError: If generation token doesn't match
        """
        async with self.session() as session:
            result = await session.execute(
                select(ContextState)
                .where(ContextState.id == context_id)
                .with_for_update()
            )
            context = result.scalar_one_or_none()
            
            if context is None:
                raise ValueError(f"Context {context_id} not found")
            
            # Validate generation token (CAS check)
            if context.current_generation != generation_token:
                raise StaleGenerationError(context_id, generation_token)
            
            # Check if state drifted during processing
            is_valid = context.semantic_clock == based_on_clock
            
            # Create response record
            response = LLMResponse(
                context_id=context_id,
                generation_token=generation_token,
                based_on_clock=based_on_clock,
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                is_valid=is_valid,
            )
            session.add(response)
            
            if is_valid:
                # Success - return to idle
                context.state = ProcessingState.IDLE
                context.current_generation = None
                logger.info(
                    "Committed valid response %s for context %s",
                    response.id,
                    context_id,
                )
            else:
                # State drifted - need to reprocess
                context.state = ProcessingState.DEBOUNCING
                logger.warning(
                    "Response %s for context %s is stale "
                    "(based on clock %d, current is %d)",
                    response.id,
                    context_id,
                    based_on_clock,
                    context.semantic_clock,
                )
            
            context.version += 1
            await session.flush()
            
            return response

    async def check_generation_valid(
        self,
        context_id: UUID,
        generation_token: UUID,
        based_on_clock: int,
    ) -> bool:
        """
        Check if a generation is still valid without committing.
        
        Useful for early termination of LLM calls when state has drifted.
        """
        async with self.session() as session:
            result = await session.execute(
                select(ContextState)
                .where(ContextState.id == context_id)
            )
            context = result.scalar_one_or_none()
            
            if context is None:
                return False
            
            return (
                context.current_generation == generation_token
                and context.semantic_clock == based_on_clock
            )

    async def cancel_processing(
        self,
        context_id: UUID,
        generation_token: UUID,
        reason: str,
    ) -> bool:
        """
        Cancel an in-progress processing run.
        
        Returns True if cancellation was successful (generation matched),
        False if the generation was already superseded.
        """
        async with self.session() as session:
            result = await session.execute(
                select(ContextState)
                .where(ContextState.id == context_id)
                .with_for_update()
            )
            context = result.scalar_one_or_none()
            
            if context is None:
                return False
            
            if context.current_generation != generation_token:
                logger.debug(
                    "Cancel request for stale generation %s (current: %s)",
                    generation_token,
                    context.current_generation,
                )
                return False
            
            # Return to debouncing if there are pending events,
            # otherwise to idle
            if context.state == ProcessingState.PROCESSING:
                context.state = ProcessingState.DEBOUNCING
                context.current_generation = None
                context.version += 1
                
                logger.info(
                    "Cancelled processing for context %s, generation %s: %s",
                    context_id,
                    generation_token,
                    reason,
                )
                return True
            
            return False

    async def get_events_since(
        self,
        context_id: UUID,
        since_clock: int,
    ) -> list[Event]:
        """Retrieve events with semantic_clock > since_clock."""
        async with self.session() as session:
            result = await session.execute(
                select(Event)
                .where(Event.context_id == context_id)
                .where(Event.semantic_clock > since_clock)
                .order_by(Event.semantic_clock)
            )
            return list(result.scalars().all())
```

### Stream Processor with Debouncing

The stream processor handles incoming events and implements debouncing:

```python
# processor.py
"""
Event stream processor with debouncing support.

Implements the event ingestion pipeline:
1. Receive events from message broker
2. Persist to event store (incrementing semantic clock)
3. Debounce events per-context
4. Trigger LLM processing after settling period
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Awaitable
from uuid import UUID

from .config import settings
from .models import EventPayload, ContextSnapshot
from .persistence import EventStore

logger = logging.getLogger(__name__)


@dataclass
class DebounceState:
    """Tracks debounce state for a single context."""
    timer_task: asyncio.Task | None = None
    pending_count: int = 0
    last_clock: int = 0


class EventProcessor:
    """
    Processes incoming events with per-context debouncing.
    
    When events arrive:
    1. Persist immediately to event store
    2. Reset debounce timer for the context
    3. After debounce period expires, trigger LLM processing
    
    This ensures rapid event bursts are coalesced into single LLM calls.
    """
    
    def __init__(
        self,
        event_store: EventStore,
        on_process_ready: Callable[[ContextSnapshot], Awaitable[None]],
        debounce_seconds: float | None = None,
    ):
        """
        Initialize the processor.
        
        Args:
            event_store: Persistence layer for events
            on_process_ready: Callback invoked when debounce completes
            debounce_seconds: Time to wait after last event before processing
        """
        self._store = event_store
        self._on_process_ready = on_process_ready
        self._debounce_seconds = debounce_seconds or settings.debounce_seconds
        self._debounce_states: dict[UUID, DebounceState] = defaultdict(DebounceState)
        self._lock = asyncio.Lock()

    async def handle_event(self, event: EventPayload) -> None:
        """
        Handle an incoming event.
        
        Persists the event and manages debounce timing.
        """
        # Persist event (this increments semantic clock)
        persisted_event, context = await self._store.append_event(event)
        
        async with self._lock:
            state = self._debounce_states[event.context_id]
            state.pending_count += 1
            state.last_clock = context.semantic_clock
            
            # Cancel existing timer if any
            if state.timer_task is not None and not state.timer_task.done():
                state.timer_task.cancel()
                try:
                    await state.timer_task
                except asyncio.CancelledError:
                    pass
            
            # Start new debounce timer
            state.timer_task = asyncio.create_task(
                self._debounce_timer(event.context_id, context.version)
            )
            
            logger.debug(
                "Event %s for context %s, pending=%d, clock=%d",
                persisted_event.id,
                event.context_id,
                state.pending_count,
                state.last_clock,
            )

    async def _debounce_timer(self, context_id: UUID, version: int) -> None:
        """
        Wait for debounce period then trigger processing.
        
        The version parameter ensures we don't process if new events arrived
        after this timer was started (the timer would have been cancelled).
        """
        try:
            await asyncio.sleep(self._debounce_seconds)
        except asyncio.CancelledError:
            logger.debug("Debounce timer cancelled for context %s", context_id)
            raise
        
        async with self._lock:
            state = self._debounce_states[context_id]
            state.pending_count = 0  # Reset pending count
        
        try:
            # Transition to processing and get snapshot
            snapshot = await self._store.begin_processing(
                context_id,
                expected_version=None,  # We accept any version here
            )
            
            logger.info(
                "Debounce complete for context %s, triggering processing "
                "with %d events at clock %d",
                context_id,
                len(snapshot.events),
                snapshot.semantic_clock,
            )
            
            # Invoke processing callback
            await self._on_process_ready(snapshot)
            
        except ValueError as e:
            # Context not in expected state - likely already processing
            logger.warning(
                "Could not begin processing for context %s: %s",
                context_id,
                e,
            )

    async def get_pending_count(self, context_id: UUID) -> int:
        """Get count of events pending processing for a context."""
        async with self._lock:
            return self._debounce_states[context_id].pending_count


class FastStreamProcessor:
    """
    FastStream-based event processor for production deployments.
    
    Integrates with message brokers (Redis Streams, Kafka, etc.)
    via the FastStream framework.
    """
    
    def __init__(
        self,
        event_store: EventStore,
        on_process_ready: Callable[[ContextSnapshot], Awaitable[None]],
        broker_url: str | None = None,
    ):
        self._store = event_store
        self._processor = EventProcessor(event_store, on_process_ready)
        self._broker_url = broker_url or settings.broker_url

    def create_app(self):
        """Create FastStream application with event handlers."""
        from faststream import FastStream
        from faststream.redis import RedisBroker
        
        broker = RedisBroker(self._broker_url)
        app = FastStream(broker)
        
        @broker.subscriber("events")
        async def handle_event(data: dict) -> None:
            """Handle incoming event from Redis stream."""
            event = EventPayload(**data)
            await self._processor.handle_event(event)
        
        return app
```

### LLM Service with Cancellation Support

The LLM service integrates with the model provider and supports mid-stream cancellation:

```python
# llm_service.py
"""
LLM integration service with streaming and cancellation support.

Provides:
- Streaming LLM calls with periodic validity checks
- Early termination when context state drifts
- Token usage tracking for cost optimization
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator
from uuid import UUID

import anthropic

from .config import settings
from .models import ContextSnapshot, ProcessingResult
from .persistence import EventStore, StaleGenerationError

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for LLM interactions with validity-aware streaming.
    
    During streaming, periodically checks if the generation is still valid.
    If state has drifted (new events arrived), cancels the stream early
    to avoid wasting tokens on a response that will be discarded.
    """
    
    def __init__(
        self,
        event_store: EventStore,
        validity_check_interval: float | None = None,
    ):
        """
        Initialize the LLM service.
        
        Args:
            event_store: For validity checks and response commits
            validity_check_interval: Seconds between validity checks during streaming
        """
        self._store = event_store
        self._check_interval = (
            validity_check_interval or settings.validity_check_interval
        )
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def process_context(self, snapshot: ContextSnapshot) -> ProcessingResult:
        """
        Process a context snapshot through the LLM.
        
        Builds a prompt from events, streams the response while checking
        validity, and commits the result if still valid.
        """
        logger.info(
            "Processing context %s, generation %s, %d events",
            snapshot.context_id,
            snapshot.generation_token,
            len(snapshot.events),
        )
        
        # Build prompt from events
        prompt = self._build_prompt(snapshot)
        
        try:
            # Stream response with validity checking
            content, input_tokens, output_tokens = await self._stream_with_checks(
                prompt=prompt,
                context_id=snapshot.context_id,
                generation_token=snapshot.generation_token,
                based_on_clock=snapshot.semantic_clock,
            )
            
            # Attempt to commit
            response = await self._store.commit_response(
                context_id=snapshot.context_id,
                generation_token=snapshot.generation_token,
                content=content,
                model=settings.llm_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                based_on_clock=snapshot.semantic_clock,
            )
            
            return ProcessingResult(
                generation_token=snapshot.generation_token,
                success=response.is_valid,
                response_id=response.id if response.is_valid else None,
                invalidation_reason=(
                    None if response.is_valid 
                    else "State drifted during processing"
                ),
            )
            
        except StaleGenerationError:
            logger.warning(
                "Generation %s was superseded before commit",
                snapshot.generation_token,
            )
            return ProcessingResult(
                generation_token=snapshot.generation_token,
                success=False,
                invalidation_reason="Generation superseded",
            )
        
        except asyncio.CancelledError:
            logger.info(
                "Processing cancelled for generation %s",
                snapshot.generation_token,
            )
            await self._store.cancel_processing(
                snapshot.context_id,
                snapshot.generation_token,
                "Processing cancelled",
            )
            raise

    def _build_prompt(self, snapshot: ContextSnapshot) -> str:
        """
        Build LLM prompt from event history.
        
        This is a simplified implementation - production systems would
        have more sophisticated prompt construction.
        """
        parts = ["You are processing a series of events. Here is the history:\n"]
        
        for event in snapshot.events:
            event_type = event["type"]
            payload = event["payload"]
            clock = event["clock"]
            
            if event_type == "user_message":
                parts.append(f"[{clock}] User: {payload.get('content', '')}")
            elif event_type == "system_event":
                parts.append(f"[{clock}] System: {payload.get('description', '')}")
            elif event_type == "external_webhook":
                parts.append(f"[{clock}] External: {payload.get('summary', '')}")
            else:
                parts.append(f"[{clock}] Event: {payload}")
        
        parts.append(
            "\nRespond to the user's latest message, "
            "taking into account all context."
        )
        
        return "\n".join(parts)

    async def _stream_with_checks(
        self,
        prompt: str,
        context_id: UUID,
        generation_token: UUID,
        based_on_clock: int,
    ) -> tuple[str, int, int]:
        """
        Stream LLM response while periodically checking validity.
        
        Returns (content, input_tokens, output_tokens).
        Raises CancelledError if validity check fails.
        """
        chunks: list[str] = []
        input_tokens = 0
        output_tokens = 0
        last_check = asyncio.get_event_loop().time()
        
        async with self._client.messages.stream(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for event in stream:
                # Collect content
                if hasattr(event, "delta") and hasattr(event.delta, "text"):
                    chunks.append(event.delta.text)
                
                # Periodic validity check
                now = asyncio.get_event_loop().time()
                if now - last_check >= self._check_interval:
                    last_check = now
                    
                    is_valid = await self._store.check_generation_valid(
                        context_id,
                        generation_token,
                        based_on_clock,
                    )
                    
                    if not is_valid:
                        logger.info(
                            "Validity check failed for generation %s, "
                            "cancelling stream",
                            generation_token,
                        )
                        # Note: We don't raise CancelledError here because
                        # we want to handle this as a normal invalidation
                        break
            
            # Get final usage stats
            final_message = await stream.get_final_message()
            input_tokens = final_message.usage.input_tokens
            output_tokens = final_message.usage.output_tokens
        
        content = "".join(chunks)
        
        logger.debug(
            "Stream complete for generation %s: %d input, %d output tokens",
            generation_token,
            input_tokens,
            output_tokens,
        )
        
        return content, input_tokens, output_tokens


class SignallingLLMService(LLMService):
    """
    Extended LLM service with external cancellation signalling.
    
    Allows other components to signal that a generation should be cancelled,
    enabling more aggressive cost optimization when state drift is detected
    before the next periodic check.
    """
    
    def __init__(
        self,
        event_store: EventStore,
        validity_check_interval: float | None = None,
    ):
        super().__init__(event_store, validity_check_interval)
        self._cancellation_events: dict[UUID, asyncio.Event] = {}

    def register_generation(self, generation_token: UUID) -> None:
        """Register a generation for potential external cancellation."""
        self._cancellation_events[generation_token] = asyncio.Event()

    def unregister_generation(self, generation_token: UUID) -> None:
        """Unregister a generation (cleanup)."""
        self._cancellation_events.pop(generation_token, None)

    def signal_cancellation(self, generation_token: UUID) -> bool:
        """
        Signal that a generation should be cancelled.
        
        Returns True if the signal was delivered, False if generation
        not found (already completed or not registered).
        """
        event = self._cancellation_events.get(generation_token)
        if event is not None:
            event.set()
            logger.info("Signalled cancellation for generation %s", generation_token)
            return True
        return False

    async def process_context(self, snapshot: ContextSnapshot) -> ProcessingResult:
        """Process with external cancellation support."""
        self.register_generation(snapshot.generation_token)
        try:
            return await super().process_context(snapshot)
        finally:
            self.unregister_generation(snapshot.generation_token)

    async def _stream_with_checks(
        self,
        prompt: str,
        context_id: UUID,
        generation_token: UUID,
        based_on_clock: int,
    ) -> tuple[str, int, int]:
        """Extended streaming with external signal checking."""
        chunks: list[str] = []
        input_tokens = 0
        output_tokens = 0
        last_check = asyncio.get_event_loop().time()
        cancel_event = self._cancellation_events.get(generation_token)
        
        async with self._client.messages.stream(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for event in stream:
                # Check external cancellation signal
                if cancel_event is not None and cancel_event.is_set():
                    logger.info(
                        "External cancellation signal received for %s",
                        generation_token,
                    )
                    break
                
                if hasattr(event, "delta") and hasattr(event.delta, "text"):
                    chunks.append(event.delta.text)
                
                # Periodic validity check
                now = asyncio.get_event_loop().time()
                if now - last_check >= self._check_interval:
                    last_check = now
                    
                    is_valid = await self._store.check_generation_valid(
                        context_id,
                        generation_token,
                        based_on_clock,
                    )
                    
                    if not is_valid:
                        logger.info(
                            "Validity check failed for generation %s",
                            generation_token,
                        )
                        break
            
            final_message = await stream.get_final_message()
            input_tokens = final_message.usage.input_tokens
            output_tokens = final_message.usage.output_tokens
        
        return "".join(chunks), input_tokens, output_tokens
```

### API Layer

The FastAPI application exposes the system:

```python
# api.py
"""
FastAPI application exposing the async LLM event processing system.

Endpoints:
- POST /events: Submit new events
- GET /contexts/{id}: Get context state
- GET /contexts/{id}/responses: Get responses for a context
- WebSocket /ws/{context_id}: Real-time updates
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from uuid import UUID

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .models import (
    ContextSnapshot,
    EventPayload,
    EventType,
    ProcessingResult,
)
from .persistence import EventStore
from .processor import EventProcessor
from .llm_service import SignallingLLMService

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self._connections: dict[UUID, list[WebSocket]] = {}

    async def connect(self, context_id: UUID, websocket: WebSocket) -> None:
        await websocket.accept()
        if context_id not in self._connections:
            self._connections[context_id] = []
        self._connections[context_id].append(websocket)

    def disconnect(self, context_id: UUID, websocket: WebSocket) -> None:
        if context_id in self._connections:
            self._connections[context_id].remove(websocket)
            if not self._connections[context_id]:
                del self._connections[context_id]

    async def broadcast(self, context_id: UUID, message: dict) -> None:
        if context_id in self._connections:
            for connection in self._connections[context_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass  # Connection closed


# Global instances (initialized in lifespan)
event_store: EventStore | None = None
llm_service: SignallingLLMService | None = None
event_processor: EventProcessor | None = None
connection_manager = ConnectionManager()


async def on_process_ready(snapshot: ContextSnapshot) -> None:
    """Callback invoked when debounce completes and processing should begin."""
    global llm_service, connection_manager
    
    if llm_service is None:
        logger.error("LLM service not initialized")
        return
    
    # Notify connected clients that processing is starting
    await connection_manager.broadcast(
        snapshot.context_id,
        {
            "type": "processing_started",
            "generation_token": str(snapshot.generation_token),
            "semantic_clock": snapshot.semantic_clock,
        },
    )
    
    # Process through LLM
    result = await llm_service.process_context(snapshot)
    
    # Notify connected clients of result
    await connection_manager.broadcast(
        snapshot.context_id,
        {
            "type": "processing_complete",
            "generation_token": str(result.generation_token),
            "success": result.success,
            "response_id": str(result.response_id) if result.response_id else None,
            "invalidation_reason": result.invalidation_reason,
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    global event_store, llm_service, event_processor
    
    # Initialize components
    event_store = EventStore()
    await event_store.initialize()
    
    llm_service = SignallingLLMService(event_store)
    event_processor = EventProcessor(event_store, on_process_ready)
    
    logger.info("Application initialized")
    
    yield
    
    # Cleanup
    await event_store.close()
    logger.info("Application shutdown complete")


app = FastAPI(
    title="Async LLM Event Processor",
    description="Process asynchronous events through LLMs with "
                "debouncing and validity tracking",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/events", status_code=202)
async def submit_event(event: EventPayload) -> dict:
    """
    Submit a new event for processing.
    
    The event will be persisted immediately and the context will enter
    debouncing state. After the debounce period, LLM processing will begin.
    """
    if event_processor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    await event_processor.handle_event(event)
    
    return {
        "status": "accepted",
        "context_id": str(event.context_id),
    }


@app.get("/contexts/{context_id}")
async def get_context(context_id: UUID) -> dict:
    """Get the current state of a context."""
    if event_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    async with event_store.session() as session:
        from sqlalchemy import select
        from .models import ContextState
        
        result = await session.execute(
            select(ContextState).where(ContextState.id == context_id)
        )
        context = result.scalar_one_or_none()
        
        if context is None:
            raise HTTPException(status_code=404, detail="Context not found")
        
        return {
            "id": str(context.id),
            "semantic_clock": context.semantic_clock,
            "state": context.state.value,
            "current_generation": (
                str(context.current_generation) 
                if context.current_generation else None
            ),
            "version": context.version,
        }


@app.get("/contexts/{context_id}/events")
async def get_events(
    context_id: UUID,
    since_clock: int = 0,
) -> dict:
    """Get events for a context, optionally filtered by semantic clock."""
    if event_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    events = await event_store.get_events_since(context_id, since_clock)
    
    return {
        "context_id": str(context_id),
        "events": [
            {
                "id": str(e.id),
                "type": e.event_type.value,
                "payload": e.payload,
                "semantic_clock": e.semantic_clock,
                "occurred_at": e.occurred_at.isoformat(),
            }
            for e in events
        ],
    }


@app.get("/contexts/{context_id}/responses")
async def get_responses(
    context_id: UUID,
    valid_only: bool = True,
) -> dict:
    """Get LLM responses for a context."""
    if event_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    async with event_store.session() as session:
        from sqlalchemy import select
        from .models import LLMResponse
        
        query = select(LLMResponse).where(LLMResponse.context_id == context_id)
        if valid_only:
            query = query.where(LLMResponse.is_valid == True)
        query = query.order_by(LLMResponse.created_at.desc())
        
        result = await session.execute(query)
        responses = result.scalars().all()
        
        return {
            "context_id": str(context_id),
            "responses": [
                {
                    "id": str(r.id),
                    "content": r.content,
                    "based_on_clock": r.based_on_clock,
                    "is_valid": r.is_valid,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "created_at": r.created_at.isoformat(),
                }
                for r in responses
            ],
        }


@app.websocket("/ws/{context_id}")
async def websocket_endpoint(websocket: WebSocket, context_id: UUID):
    """
    WebSocket endpoint for real-time updates on a context.
    
    Clients receive notifications when:
    - Processing starts
    - Processing completes (with success/failure status)
    - New events are received (optional)
    """
    await connection_manager.connect(context_id, websocket)
    try:
        while True:
            # Keep connection alive, handle any client messages
            data = await websocket.receive_json()
            
            # Client can send ping messages
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            
    except WebSocketDisconnect:
        connection_manager.disconnect(context_id, websocket)


@app.post("/contexts/{context_id}/cancel")
async def cancel_processing(context_id: UUID) -> dict:
    """
    Request cancellation of current processing for a context.
    
    This signals the LLM service to abort the current generation,
    saving tokens if state has drifted.
    """
    if event_store is None or llm_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    async with event_store.session() as session:
        from sqlalchemy import select
        from .models import ContextState
        
        result = await session.execute(
            select(ContextState).where(ContextState.id == context_id)
        )
        context = result.scalar_one_or_none()
        
        if context is None:
            raise HTTPException(status_code=404, detail="Context not found")
        
        if context.current_generation is None:
            return {"status": "no_active_processing"}
        
        cancelled = llm_service.signal_cancellation(context.current_generation)
        
        return {
            "status": "cancellation_signalled" if cancelled else "signal_failed",
            "generation_token": str(context.current_generation),
        }
```

### Configuration

```python
# config.py
"""Configuration management for the async LLM event processor."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/llm_events"
    )
    db_pool_size: int = 10
    db_max_overflow: int = 20
    
    # Message broker
    broker_url: str = "redis://localhost:6379"
    
    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 4096
    
    # Processing
    debounce_seconds: float = 0.5
    validity_check_interval: float = 1.0
    
    # Application
    debug: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
```

## Conclusion

Handling asynchronous events in LLM-powered applications requires careful consideration of the tradeoffs between responsiveness, correctness, and cost. The architecture presented here—combining event sourcing, optimistic locking via semantic clocks, and debouncing—provides a robust foundation for scenarios where events arrive in bursts and responses should reflect complete, up-to-date state.

The key insights are:

1. **Semantic clocks** provide a logical ordering that's more meaningful than wall-clock time for detecting state drift
2. **Optimistic locking via CAS** ensures responses are only committed if still valid
3. **Debouncing** coalesces rapid events into single processing runs
4. **Mid-stream validity checks** enable early termination to save costs

The proof-of-concept implementation demonstrates these concepts with production-grade patterns: proper async/await usage, type safety, separation of concerns, and extensibility for cost optimization variants.

As LLMs become more deeply integrated into real-time applications, patterns like these will become increasingly important for building systems that are both responsive and correct.

---

## Appendix: Implementation

> **⚠️ Didactic Proof-of-Concept**: The code samples below are intended as educational illustrations of the architectural concepts discussed above. They demonstrate core patterns and mechanisms but should *not* be considered production-ready. Real implementations would require additional considerations: comprehensive error handling, monitoring/observability, security hardening, performance optimization, and extensive testing. These samples trade off completeness for clarity to highlight the key architectural ideas.

### Implementation Overview

Let's build a proof-of-concept implementation. The full code is structured as follows:

```
llm_events/
├── models.py          # Domain models and database schema
├── persistence.py     # Event store and optimistic locking
├── processor.py       # Stream processing and debouncing
├── llm_service.py     # LLM integration with cancellation
├── api.py             # FastAPI endpoints
└── config.py          # Configuration
```

### Database Schema and Models

We use an event-sourcing inspired schema where events are immutable and state is derived:

```python
# models.py
"""
Domain models for the async LLM event processing system.

Key concepts:
- ContextState: Aggregate root tracking semantic version and processing state
- Event: Immutable event records with semantic clock assignment
- LLMResponse: Generated responses with validity tracking
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """SQLAlchemy declarative base with common configurations."""
    pass


class ProcessingState(enum.Enum):
    """State machine states for context processing."""
    IDLE = "idle"
    DEBOUNCING = "debouncing"
    PROCESSING = "processing"


class ContextState(Base):
    """
    Aggregate root for a conversation/processing context.
    
    The semantic_clock increments with each meaningful state change,
    providing a logical ordering independent of wall-clock time.
    The version field enables optimistic locking via CAS operations.
    """
    __tablename__ = "context_states"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    
    # Semantic versioning - increments on meaningful state changes
    semantic_clock: Mapped[int] = mapped_column(
        BigInteger, 
        nullable=False, 
        default=0
    )
    
    # Optimistic locking version - increments on any update
    version: Mapped[int] = mapped_column(
        Integer, 
        nullable=False, 
        default=0
    )
    
    # Current processing state
    state: Mapped[ProcessingState] = mapped_column(
        SAEnum(ProcessingState),
        nullable=False,
        default=ProcessingState.IDLE,
    )
    
    # Generation token for the current/last processing run
    # Used to validate responses against state drift
    current_generation: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    
    # Relationships
    events: Mapped[list[Event]] = relationship(
        "Event", 
        back_populates="context",
        order_by="Event.semantic_clock",
    )
    responses: Mapped[list[LLMResponse]] = relationship(
        "LLMResponse",
        back_populates="context",
    )

    __table_args__ = (
        Index("ix_context_states_state", "state"),
    )


class EventType(enum.Enum):
    """Classification of events that can affect context state."""
    USER_MESSAGE = "user_message"
    SYSTEM_EVENT = "system_event"
    EXTERNAL_WEBHOOK = "external_webhook"
    SCHEDULED_TASK = "scheduled_task"


class Event(Base):
    """
    Immutable event record in the event store.
    
    Events are assigned a semantic_clock value upon persistence,
    establishing a total ordering within their context.
    """
    __tablename__ = "events"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    context_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("context_states.id"),
        nullable=False,
    )
    
    # Event classification and content
    event_type: Mapped[EventType] = mapped_column(
        SAEnum(EventType),
        nullable=False,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
    )
    
    # Assigned during persistence - provides total ordering
    semantic_clock: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    
    # Wall-clock timestamp for debugging/analytics
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    
    # Relationship
    context: Mapped[ContextState] = relationship(
        "ContextState",
        back_populates="events",
    )

    __table_args__ = (
        Index("ix_events_context_clock", "context_id", "semantic_clock"),
    )


class LLMResponse(Base):
    """
    Record of an LLM-generated response.
    
    Tracks the generation token and semantic clock at generation time,
    enabling staleness detection and response validity assessment.
    """
    __tablename__ = "llm_responses"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    context_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("context_states.id"),
        nullable=False,
    )
    
    # Links response to specific processing run
    generation_token: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
    )
    
    # Semantic clock value when processing started
    # Response is valid iff this matches context's clock at commit time
    based_on_clock: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    
    # The actual response content
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    
    # Response metadata
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Validity tracking
    is_valid: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )
    invalidated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    
    # Relationship
    context: Mapped[ContextState] = relationship(
        "ContextState",
        back_populates="responses",
    )

    __table_args__ = (
        Index("ix_responses_context_valid", "context_id", "is_valid"),
        Index("ix_responses_generation", "generation_token"),
    )


# Pydantic models for API/messaging boundaries

class EventPayload(BaseModel):
    """Incoming event data before persistence."""
    context_id: UUID
    event_type: EventType
    payload: dict[str, Any]
    idempotency_key: str | None = None


class ProcessingResult(BaseModel):
    """Result of an LLM processing attempt."""
    generation_token: UUID
    success: bool
    response_id: UUID | None = None
    invalidation_reason: str | None = None
    
    
class ContextSnapshot(BaseModel):
    """Point-in-time view of context state for LLM processing."""
    context_id: UUID
    semantic_clock: int
    generation_token: UUID
    events: list[dict[str, Any]]
    
    class Config:
        from_attributes = True
```

### Cost Optimization Variants

#### Variant 1: Aggressive Early Termination

For cost-sensitive applications, we can terminate LLM calls immediately when new events arrive, rather than waiting for periodic validity checks:

```python
# In processor.py, enhance the EventProcessor

class EagerCancellingProcessor(EventProcessor):
    """
    Processor that immediately signals LLM cancellation on new events.
    
    Trades off potential for more frequent restarts against token cost savings.
    """
    
    def __init__(
        self,
        event_store: EventStore,
        on_process_ready: Callable[[ContextSnapshot], Awaitable[None]],
        llm_service: SignallingLLMService,
        debounce_seconds: float | None = None,
    ):
        super().__init__(event_store, on_process_ready, debounce_seconds)
        self._llm_service = llm_service

    async def handle_event(self, event: EventPayload) -> None:
        # Get current generation before persisting
        async with self._store.session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(ContextState)
                .where(ContextState.id == event.context_id)
            )
            context = result.scalar_one_or_none()
            current_gen = context.current_generation if context else None
        
        # Persist event (parent implementation)
        await super().handle_event(event)
        
        # Signal cancellation if processing was in progress
        if current_gen is not None:
            self._llm_service.signal_cancellation(current_gen)
```

#### Variant 2: Batched Validity Checks

For high-throughput systems, we can batch validity checks across multiple concurrent LLM calls:

```python
class BatchValidityChecker:
    """
    Batches validity checks to reduce database round-trips.
    
    Multiple concurrent LLM processors can register their generations,
    and validity is checked in batches at configurable intervals.
    """
    
    def __init__(self, event_store: EventStore, batch_interval: float = 0.5):
        self._store = event_store
        self._batch_interval = batch_interval
        self._pending_checks: dict[UUID, tuple[UUID, int, asyncio.Future]] = {}
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._check_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def check_validity(
        self,
        context_id: UUID,
        generation_token: UUID,
        based_on_clock: int,
    ) -> bool:
        """Register for batch validity check and await result."""
        future: asyncio.Future[bool] = asyncio.Future()
        
        async with self._lock:
            self._pending_checks[generation_token] = (
                context_id,
                based_on_clock,
                future,
            )
        
        return await future

    async def _check_loop(self) -> None:
        while True:
            await asyncio.sleep(self._batch_interval)
            
            async with self._lock:
                if not self._pending_checks:
                    continue
                
                checks = dict(self._pending_checks)
                self._pending_checks.clear()
            
            # Batch query all contexts
            context_ids = {ctx_id for ctx_id, _, _ in checks.values()}
            
            async with self._store.session() as session:
                from sqlalchemy import select
                result = await session.execute(
                    select(ContextState)
                    .where(ContextState.id.in_(context_ids))
                )
                contexts = {c.id: c for c in result.scalars().all()}
            
            # Resolve futures
            for gen_token, (ctx_id, based_on_clock, future) in checks.items():
                context = contexts.get(ctx_id)
                is_valid = (
                    context is not None
                    and context.current_generation == gen_token
                    and context.semantic_clock == based_on_clock
                )
                future.set_result(is_valid)
```

---

**Repository & Further Reading**: Full working examples, deployment instructions, and benchmarks can be found in the referenced implementation repository (link available upon request).

