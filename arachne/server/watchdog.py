"""A request's deadline, kept by a thread of its own.

The scene's code a request runs can't be interrupted from inside the
process: a scene that sleeps, loops or waits holds the thread serving the
request. So the deadline is kept beside it. Each request is one
:class:`Ticket`, which ends exactly one way: answered (:meth:`Watchdog.end`,
the request's answer written) or expired, its deadline passed first. An
expiry calls what the server gave for the request, ``expired(respond)``,
which ends the process: with ``respond`` true when the answer is still the
expiry's to give, false when the request had already begun writing its own
(:meth:`Watchdog.answering`), so that a request is never answered twice.
A deadline that has passed is expired by whichever thread sees it first:
the watchdog's, or the request's own as it moves the deadline or claims
its answer, so a request already past its deadline neither runs nor
answers.
"""

from __future__ import annotations

__all__ = ["Ticket", "Watchdog"]

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(eq=False)
class Ticket:
    """One request's deadline: ``at``, a :func:`time.monotonic` instant, and
    how far it is: ``"running"``, ``"answering"`` (the request writes its
    answer), ``"ended"`` or ``"expired"``.
    """

    at: float
    expired: Callable[[bool], object]
    state: str = "running"


class Watchdog:
    """The deadline of the request running now, one at a time: :meth:`arm`
    a ticket as a request starts, :meth:`answering` as it begins writing its
    answer, :meth:`end` once written. Past the ticket's instant, its
    ``expired`` is called, once, on the watchdog's thread.
    """

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._ticket: Ticket | None = None
        threading.Thread(target=self._keep, name="deadline", daemon=True).start()

    def arm(self, at: float, expired: Callable[[bool], object]) -> Ticket:
        """A request's ticket, expiring at ``at``: the request kept now."""
        ticket = Ticket(at, expired)
        with self._changed:
            self._ticket = ticket
            self._changed.notify()
        return ticket

    def move(self, ticket: Ticket, at: float) -> None:
        """Move ``ticket``'s deadline to ``at``: a deadline already past
        expires it now, on the calling thread, before the request runs.
        """
        with self._changed:
            ticket.at = at
            self._changed.notify()
        if at <= time.monotonic():
            self.expire(ticket)

    def answering(self, ticket: Ticket) -> bool:
        """Claim ``ticket``'s answer for the request: false when it expired
        first, and the request must answer nothing. A request whose deadline
        passed before it claimed its answer expired, whether or not the
        watchdog's thread has seen it yet: it is expired now, on the calling
        thread. Its deadline still holds while the answer is written. A
        request that claimed its answer already, to publish what it made
        (:meth:`claim`), has it.
        """
        with self._changed:
            if ticket.state == "answering":
                return True
            if ticket.state != "running":
                return False
            if ticket.at > time.monotonic():
                ticket.state = "answering"
                return True
        self.expire(ticket)
        return False

    def claim(self, ticket: Ticket, action: Callable[[], object]) -> bool:
        """Claim ``ticket``'s answer and run ``action`` under the claim, which
        publishes what the request made (a render's file renamed onto the one
        its caller named): false, running nothing, when the request expired
        first, so that what a request publishes and its deadline's error are
        one or the other, never both. ``action`` runs while the deadline can't
        pass, so it must be brief. From then on the request is answering, as
        :meth:`answering` makes it.
        """
        with self._changed:
            if ticket.state == "running" and ticket.at > time.monotonic():
                action()
                ticket.state = "answering"
                return True
            late = ticket.state == "running"
        if late:
            self.expire(ticket)
        return False

    def end(self, ticket: Ticket) -> None:
        """``ticket``'s request is over: once this returns, it can't expire."""
        with self._changed:
            if ticket.state in ("running", "answering"):
                ticket.state = "ended"
            if self._ticket is ticket:
                self._ticket = None
            self._changed.notify()

    def expire(self, ticket: Ticket) -> None:
        """Expire ``ticket`` now, as its deadline does: its ``expired`` is
        called unless the request ended first.
        """
        with self._changed:
            state = ticket.state
            if state not in ("running", "answering"):
                return
            ticket.state = "expired"
        ticket.expired(state == "running")

    def _keep(self) -> None:
        while True:
            with self._changed:
                ticket = self._ticket
                if ticket is None:
                    self._changed.wait()
                    continue
                left = ticket.at - time.monotonic()
                if left > 0:
                    # A wait is at most the longest a thread can make.
                    self._changed.wait(min(left, threading.TIMEOUT_MAX))
                    continue
                self._ticket = None
            self.expire(ticket)
