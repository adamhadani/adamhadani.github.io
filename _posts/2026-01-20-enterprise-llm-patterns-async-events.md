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

**Generation Token**: A unique identifier captured at processing start, used to validate response freshness. We use UUIDs for global uniqueness without coordination overhead.

These three concepts work in concert to handle asynchronous events gracefully. The semantic clock provides a logical ordering independent of wall-clock time, allowing us to detect when state has drifted. Generation tokens serve as unique identifiers for each processing run, enabling Compare-and-Swap validation—if the token stored at commit time differs from the one captured at start, we know another processing run has superseded ours. The event window ensures we coalesce rapid bursts into single processing runs, optimizing for cost while maintaining responsiveness.

> **Implementation Note**: Generation tokens can use UUID v4 (random), UUIDv7 (time-ordered), or sequential integers. UUID v4 suffices for identity checks, but **UUIDv7 is often preferable** in production: its time-ordering makes debugging and audit trails significantly easier, with no additional coordination overhead. Sequential integers are compact but require centralized generation. Since the semantic clock already provides ordering for state transitions, the generation token's primary job is identity, not ordering—but time-correlation is valuable for operations.

Here's how they flow through the system:

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

The state machine ensures clear transitions and prevents race conditions:

- **IDLE**: Initial state. No processing in progress. The system is ready to accept new events.
  
- **DEBOUNCING**: Events have arrived and we're waiting for the settling period before processing. If new events arrive while debouncing, the timer resets, extending the window. This coalesces bursts into single processing runs, reducing the number of LLM calls needed.

- **PROCESSING**: Actively calling the LLM with the accumulated event history. All events arriving in this state are persisted and the semantic clock incremented, but they don't trigger a new processing run yet. At processing start, we record both a generation token (UUID) and the current semantic clock value—this pair is critical for later validation.

- **COMMITTED**: Processing completed successfully and the semantic clock hasn't advanced (no new events arrived). The response is authoritative and the context returns to IDLE, ready for the next cycle.

- **INVALIDATED**: Processing completed but the semantic clock advanced (new events arrived during LLM processing). The response is marked invalid (unsuitable for user consumption) but preserved in the audit log for debugging. The context transitions back to DEBOUNCING, triggering reprocessing with fresh state. Consider adding exponential backoff here to prevent tight retry loops under sustained event bursts.

This design ensures both correctness and efficiency: responses always reflect current state, but we avoid unnecessary reprocessing through debouncing and don't waste tokens on responses we know are stale.

## Considerations and Tradeoffs

### When This Pattern Fits

This architecture works well when:

- Events arrive in bursts with natural settling periods
- Response freshness is more important than minimal latency
- Cost optimization through avoiding wasted tokens matters
- You need an audit trail of state changes (event sourcing)

### When to Consider Alternatives

Consider different approaches when:

- **Latency is critical**: Debouncing adds inherent delay
- **Events are truly continuous**: No natural settling period to exploit
- **Simple request/response suffices**: Don't over-engineer
- **Partial responses are acceptable**: Streaming to user might be better than invalidation

### Operational Considerations

- **Monitor debounce effectiveness**: Track how often responses are invalidated
- **Tune debounce period**: Too short = frequent invalidations; too long = poor UX
- **Set up alerting**: Detect runaway retry loops or stuck contexts
- **Consider circuit breakers**: Prevent cascading failures from LLM timeouts

## Conclusion

Handling asynchronous events in LLM-powered applications requires careful consideration of the tradeoffs between responsiveness, correctness, and cost. The architecture presented here—combining event sourcing, optimistic locking via semantic clocks, and debouncing—provides a robust foundation for scenarios where events arrive in bursts and responses should reflect complete, up-to-date state.

The key insights are:

1. **Semantic clocks** provide a logical ordering that's more meaningful than wall-clock time for detecting state drift
2. **Optimistic locking via CAS** ensures responses are only committed if still valid
3. **Debouncing** coalesces rapid events into single processing runs
4. **Mid-stream validity checks** enable early termination to save costs

The proof-of-concept implementation illustrates these concepts with idiomatic patterns: proper async/await usage, type safety, separation of concerns, and PostgreSQL-native constructs (CTEs, `UPDATE...RETURNING`) that minimize lock contention compared to explicit `SELECT FOR UPDATE`. Production implementations would require additional hardening around error handling, observability, and edge cases.

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

The persistence layer implements CAS (Compare-And-Swap) semantics using PostgreSQL-idiomatic patterns that minimize lock contention:

- **CTE-based atomic operations** for `append_event`: Combines UPDATE and INSERT in a single statement. The row lock is implicit and held only for statement duration—no explicit `FOR UPDATE` needed.
- **`UPDATE ... RETURNING`** for state transitions like `begin_processing`: Atomic update with immediate value return, minimal lock duration.
- **`UPDATE ... WHERE`** (pure optimistic) for validation operations like `commit_response`: The WHERE clause encodes the CAS predicate; we check `rowcount` to detect conflicts.

These patterns avoid the contention and deadlock risks of explicit `SELECT FOR UPDATE` while maintaining atomicity.

> **Alternative: Sequences for Clock Assignment**
> For use cases where gaps in the semantic clock are acceptable, PostgreSQL sequences offer an even simpler approach: `SELECT nextval('context_' || context_id || '_seq')` or a global sequence. Sequences are lock-free and highly concurrent—no CTE needed, just a simple `INSERT`. The tradeoff: gaps occur on transaction rollback, and you'd still need *some* state tracking for the processing state machine and generation tokens (though this could live in Redis or be derived from event patterns). For maximum simplicity with relaxed ordering guarantees, sequences are worth considering.

```python
# persistence.py
"""
Event store and optimistic locking implementation.
"""

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
    pass


class EventStore:
    """
    Event store with optimistic locking support.

    Implements event sourcing patterns with:
    - Atomic event persistence and clock increment
    - Version-based optimistic locking for state transitions
    - Generation token validation for response commits
    """

    async def append_event(
        self,
        event_payload: EventPayload,
        session: AsyncSession | None = None,
    ) -> tuple[Event, ContextState]:
        """
        Append an event to the store, atomically incrementing the semantic clock.

        Uses a CTE (Common Table Expression) to combine UPDATE and INSERT in a
        single atomic statement. This is more idiomatic than SELECT FOR UPDATE:
        - Single round-trip to the database
        - Row lock is implicit and minimal duration
        - No explicit locking, avoiding contention and deadlock risks
        """
        async def _execute(sess: AsyncSession) -> tuple[Event, ContextState]:
            # Ensure context exists (upsert pattern)
            await sess.execute(
                insert(ContextState)
                .values(id=event_payload.context_id)
                .on_conflict_do_nothing(index_elements=["id"])
            )

            # CTE: atomically increment clock and insert event in one statement
            # The UPDATE implicitly locks the row for minimal duration
            result = await sess.execute(
                text("""
                    WITH updated_context AS (
                        UPDATE context_states
                        SET semantic_clock = semantic_clock + 1,
                            version = version + 1,
                            state = CASE
                                WHEN state = 'idle' THEN 'debouncing'
                                ELSE state
                            END,
                            updated_at = now()
                        WHERE id = :context_id
                        RETURNING id, semantic_clock, version, state
                    )
                    INSERT INTO events (id, context_id, event_type, payload, semantic_clock)
                    SELECT :event_id, id, :event_type, :payload, semantic_clock
                    FROM updated_context
                    RETURNING context_id, semantic_clock
                """),
                {
                    "context_id": event_payload.context_id,
                    "event_id": uuid4(),
                    "event_type": event_payload.event_type.value,
                    "payload": json.dumps(event_payload.payload),
                },
            )
            row = result.fetchone()

            # Fetch the updated entities for return
            context = await sess.get(ContextState, event_payload.context_id)
            event = await sess.execute(
                select(Event)
                .where(Event.context_id == event_payload.context_id)
                .where(Event.semantic_clock == row.semantic_clock)
            )
            return event.scalar_one(), context
        # ... session handling omitted

    async def begin_processing(
        self,
        context_id: UUID,
        expected_version: int | None = None,
    ) -> ContextSnapshot:
        """
        Transition context to PROCESSING state and return a snapshot for LLM.

        Uses UPDATE...RETURNING for atomic state transition with minimal lock
        duration. The WHERE clause encodes preconditions (state must be DEBOUNCING,
        optionally version must match), making this a CAS operation.
        """
        async with self.session() as session:
            generation_token = uuid4()

            # Atomic state transition via UPDATE...RETURNING
            # WHERE clause enforces preconditions; rowcount tells us if it succeeded
            query = (
                update(ContextState)
                .where(ContextState.id == context_id)
                .where(ContextState.state == ProcessingState.DEBOUNCING)
            )
            if expected_version is not None:
                query = query.where(ContextState.version == expected_version)

            result = await session.execute(
                query.values(
                    state=ProcessingState.PROCESSING,
                    current_generation=generation_token,
                    version=ContextState.version + 1,
                )
                .returning(ContextState.id, ContextState.semantic_clock, ContextState.version)
            )
            row = result.fetchone()

            if row is None:
                # Transition failed - either wrong state or version mismatch
                ctx = await session.get(ContextState, context_id)
                if ctx is None:
                    raise ValueError(f"Context {context_id} not found")
                if expected_version is not None and ctx.version != expected_version:
                    raise OptimisticLockError(context_id, expected_version, ctx.version)
                raise ValueError(f"Context {context_id} in state {ctx.state}, expected DEBOUNCING")

            # Load events (immutable, no lock needed)
            events_result = await session.execute(
                select(Event)
                .where(Event.context_id == context_id)
                .order_by(Event.semantic_clock)
            )
            events = events_result.scalars().all()

            return ContextSnapshot(
                context_id=context_id,
                semantic_clock=row.semantic_clock,
                generation_token=generation_token,
                events=[{"id": str(e.id), "type": e.event_type.value,
                        "payload": e.payload, "clock": e.semantic_clock} for e in events],
            )

    async def commit_response(
        self,
        context_id: UUID,
        generation_token: UUID,
        content: str,
        based_on_clock: int,
        # ... other params
    ) -> LLMResponse:
        """
        Attempt to commit an LLM response using CAS semantics.

        Uses pure optimistic locking: an atomic UPDATE with WHERE conditions
        encoding the CAS predicate. This is more idiomatic for PostgreSQL than
        SELECT FOR UPDATE when we're validating preconditions rather than
        doing read-modify-write.

        The commit succeeds only if:
        1. The generation_token matches the context's current_generation
        2. The semantic_clock hasn't advanced since processing began

        On success, transitions context to IDLE.
        On failure due to new events, transitions to DEBOUNCING for retry.
        """
        async with self.session() as session:
            # Atomic CAS: UPDATE only if generation matches and clock unchanged
            result = await session.execute(
                update(ContextState)
                .where(ContextState.id == context_id)
                .where(ContextState.current_generation == generation_token)
                .where(ContextState.semantic_clock == based_on_clock)
                .values(
                    state=ProcessingState.IDLE,
                    current_generation=None,
                    version=ContextState.version + 1,
                )
                .returning(ContextState.id)
            )
            committed = result.scalar_one_or_none() is not None

            if committed:
                # CAS succeeded - response is valid
                response = LLMResponse(
                    context_id=context_id,
                    generation_token=generation_token,
                    based_on_clock=based_on_clock,
                    content=content,
                    is_valid=True,
                    # model, input_tokens, output_tokens passed from caller
                )
                session.add(response)
                return response

            # CAS failed - determine why
            ctx = await session.execute(
                select(ContextState).where(ContextState.id == context_id)
            )
            context = ctx.scalar_one_or_none()

            if context is None or context.current_generation != generation_token:
                # Our processing run was superseded entirely
                raise StaleGenerationError(context_id, generation_token)

            # Generation matches but clock advanced - state drifted during processing
            # Record the invalid response for audit, transition to DEBOUNCING
            response = LLMResponse(
                context_id=context_id,
                generation_token=generation_token,
                based_on_clock=based_on_clock,
                content=content,
                is_valid=False,
                # model, input_tokens, output_tokens passed from caller
            )
            session.add(response)

            await session.execute(
                update(ContextState)
                .where(ContextState.id == context_id)
                .values(state=ProcessingState.DEBOUNCING, version=ContextState.version + 1)
            )

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
                select(ContextState).where(ContextState.id == context_id)
            )
            context = result.scalar_one_or_none()

            return (
                context is not None
                and context.current_generation == generation_token
                and context.semantic_clock == based_on_clock
            )
```

### Stream Processor with Debouncing

The stream processor handles incoming events and implements debouncing:

```python
# processor.py
"""
Event stream processor with debouncing support.
"""

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
    """

    def __init__(
        self,
        event_store: EventStore,
        on_process_ready: Callable[[ContextSnapshot], Awaitable[None]],
        debounce_seconds: float | None = None,
    ):
        self._store = event_store
        self._on_process_ready = on_process_ready
        self._debounce_seconds = debounce_seconds or settings.debounce_seconds
        self._debounce_states: dict[UUID, DebounceState] = defaultdict(DebounceState)
        self._lock = asyncio.Lock()

    async def handle_event(self, event: EventPayload) -> None:
        """Handle an incoming event. Persists the event and manages debounce timing."""
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

    async def _debounce_timer(self, context_id: UUID, version: int) -> None:
        """Wait for debounce period then trigger processing."""
        try:
            await asyncio.sleep(self._debounce_seconds)
        except asyncio.CancelledError:
            raise

        async with self._lock:
            state = self._debounce_states[context_id]
            state.pending_count = 0

        # Transition to processing and get snapshot
        snapshot = await self._store.begin_processing(context_id)

        # Invoke processing callback
        await self._on_process_ready(snapshot)
```

### LLM Service with Cancellation Support

The LLM service integrates with the model provider and supports mid-stream cancellation:

```python
# llm_service.py
"""
LLM integration service with streaming and cancellation support.
"""

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
        self._store = event_store
        self._check_interval = validity_check_interval or settings.validity_check_interval
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def process_context(self, snapshot: ContextSnapshot) -> ProcessingResult:
        """
        Process a context snapshot through the LLM.

        Builds a prompt from events, streams the response while checking
        validity, and commits the result if still valid.
        """
        prompt = self._build_prompt(snapshot)

        try:
            content, input_tokens, output_tokens = await self._stream_with_checks(
                prompt=prompt,
                context_id=snapshot.context_id,
                generation_token=snapshot.generation_token,
                based_on_clock=snapshot.semantic_clock,
            )

            response = await self._store.commit_response(
                context_id=snapshot.context_id,
                generation_token=snapshot.generation_token,
                content=content,
                based_on_clock=snapshot.semantic_clock,
                # ...
            )

            return ProcessingResult(
                generation_token=snapshot.generation_token,
                success=response.is_valid,
                response_id=response.id if response.is_valid else None,
                invalidation_reason=None if response.is_valid else "State drifted during processing",
            )

        except StaleGenerationError:
            return ProcessingResult(
                generation_token=snapshot.generation_token,
                success=False,
                invalidation_reason="Generation superseded",
            )

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
        """
        chunks: list[str] = []
        last_check = asyncio.get_event_loop().time()

        async with self._client.messages.stream(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for event in stream:
                if hasattr(event, "delta") and hasattr(event.delta, "text"):
                    chunks.append(event.delta.text)

                # Periodic validity check
                now = asyncio.get_event_loop().time()
                if now - last_check >= self._check_interval:
                    last_check = now

                    is_valid = await self._store.check_generation_valid(
                        context_id, generation_token, based_on_clock,
                    )

                    if not is_valid:
                        break  # Early termination - state drifted

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
"""

app = FastAPI(
    title="Async LLM Event Processor",
    description="Process asynchronous events through LLMs with debouncing and validity tracking",
)

@app.post("/events", status_code=202)
async def submit_event(event: EventPayload) -> dict:
    """
    Submit a new event for processing.

    The event will be persisted immediately and the context will enter
    debouncing state. After the debounce period, LLM processing will begin.
    """
    await event_processor.handle_event(event)
    return {"status": "accepted", "context_id": str(event.context_id)}


@app.get("/contexts/{context_id}")
async def get_context(context_id: UUID) -> dict:
    """Get the current state of a context."""
    # ... returns id, semantic_clock, state, current_generation, version


@app.get("/contexts/{context_id}/responses")
async def get_responses(context_id: UUID, valid_only: bool = True) -> dict:
    """Get LLM responses for a context."""
    # ... returns list of responses with validity status


@app.websocket("/ws/{context_id}")
async def websocket_endpoint(websocket: WebSocket, context_id: UUID):
    """
    WebSocket endpoint for real-time updates on a context.

    Clients receive notifications when:
    - Processing starts
    - Processing completes (with success/failure status)
    """
    await connection_manager.connect(context_id, websocket)
    # ... handle real-time updates


@app.post("/contexts/{context_id}/cancel")
async def cancel_processing(context_id: UUID) -> dict:
    """
    Request cancellation of current processing for a context.
    Signals the LLM service to abort the current generation.
    """
    # ... signal cancellation via llm_service
```

### Configuration

```python
# config.py
"""Configuration management for the async LLM event processor."""

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/llm_events"
    db_pool_size: int = 10

    # Message broker
    broker_url: str = "redis://localhost:6379"

    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 4096

    # Processing
    debounce_seconds: float = 0.5
    validity_check_interval: float = 1.0

    class Config:
        env_file = ".env"


settings = Settings()
```

### Cost Optimization Variants

The base implementation uses periodic validity checks during streaming. For more aggressive cost optimization, consider these variants:

#### Variant 1: Aggressive Early Termination

For cost-sensitive applications, we can terminate LLM calls immediately when new events arrive, rather than waiting for periodic validity checks. This requires extending `LLMService` with a cancellation signalling mechanism:

```python
# Extend LLMService with external cancellation signals
class SignallingLLMService(LLMService):
    """LLMService with external cancellation support via asyncio.Event per generation."""

    def __init__(self, event_store: EventStore, ...):
        super().__init__(event_store, ...)
        self._cancellation_events: dict[UUID, asyncio.Event] = {}

    def signal_cancellation(self, generation_token: UUID) -> bool:
        """Signal that a generation should be cancelled."""
        if event := self._cancellation_events.get(generation_token):
            event.set()
            return True
        return False


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

