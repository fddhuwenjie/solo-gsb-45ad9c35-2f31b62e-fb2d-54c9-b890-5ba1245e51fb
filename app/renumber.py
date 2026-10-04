"""Whole-pool renumbering: move a pool to a new same-size CIDR.

``POST /pools/{id}/renumber`` relocates an existing pool to a new CIDR of
the *same prefix length*.  The pool keeps its id and ``created_at``; every
exclusion and every allocation is translated by the same constant address
offset (new pool start - old pool start), so each range keeps its relative
position inside the pool.  Prefix lengths, exclusion/allocation row ids,
``request_id`` values and ``created_at`` values are all preserved verbatim.

The response gives an old -> new CIDR comparison for the pool itself, every
exclusion and every allocation.  After a successful renumber every existing
endpoint (status query, snapshot, ``request_id`` idempotent retry, release,
reverse-DNS plan, ...) sees the new addresses, because only the stored CIDRs
change -- the allocation rows are never replaced.

Everything runs inside one ``BEGIN IMMEDIATE`` write transaction:

* the new pool CIDR must not overlap any other existing pool;
* every translated exclusion/allocation must still lie completely inside
  the new pool and the translated ranges must stay mutually non-overlapping
  (redundant given a constant offset, but enforced explicitly);
* the pool is refused while it has a ``planned`` or ``active`` BGP
  announcement plan (the plan pins a digest of the old pool state), or while
  it appears in a completed batch reservation or cross-pool allocation
  migration idempotency record (replaying such a historical operation must
  keep returning the original, old addresses).

Any failure rolls the transaction back: the pool and every record keep
their previous CIDRs.
"""

from __future__ import annotations

import ipaddress
import sqlite3

from .cidrmath import parse_network
from .db import write_transaction
from .service import (
    ApiError,
    _pool_allocations,
    _pool_exclusions,
    _row_pool,
)


def _renumber_error(detail: str, code: str, status_code: int = 422) -> ApiError:
    return ApiError(status_code, detail, code=code)


def _translate(
    net: ipaddress.IPv4Network,
    offset: int,
) -> ipaddress.IPv4Network:
    """Translate ``net`` by ``offset`` addresses, keeping its prefix length.

    The new start is constructed from the old *network* address plus the
    pool offset, so it is always aligned to ``net``'s own prefix; callers
    additionally re-verify containment inside the new pool.
    """

    new_address = ipaddress.IPv4Address(int(net.network_address) + offset)
    return ipaddress.IPv4Network((new_address, net.prefixlen), strict=False)


def _ensure_no_blocking_history(conn: sqlite3.Connection, pool_id: int) -> None:
    """Refuse renumbering while pinned/historical records reference the pool.

    * a ``planned``/``active`` BGP plan pins a digest of the old state;
    * a completed reservation batch replays its stored old CIDR mapping;
    * a completed cross-pool allocation migration replays old CIDRs and the
      pool may appear on either side of it (source or target).
    """

    plan = conn.execute(
        "SELECT plan_id, status FROM bgp_plans WHERE pool_id = ? LIMIT 1",
        (pool_id,),
    ).fetchone()
    if plan is not None:
        raise _renumber_error(
            f"pool {pool_id} has a {plan['status']} BGP announcement plan "
            f"({plan['plan_id']!r}) pinned to its current CIDR; cancel or "
            "finish it before renumbering",
            "renumber_blocked_by_bgp_plan",
            status_code=409,
        )

    batch = conn.execute(
        "SELECT batch_id FROM reservation_batches WHERE pool_id = ? LIMIT 1",
        (pool_id,),
    ).fetchone()
    if batch is not None:
        raise _renumber_error(
            f"pool {pool_id} appears in completed batch reservation "
            f"{batch['batch_id']!r}; its idempotent replay must keep returning "
            "the original addresses",
            "renumber_blocked_by_reservation_batch",
            status_code=409,
        )

    migration = conn.execute(
        "SELECT operation_id, source_pool_id FROM allocation_migrations "
        "WHERE source_pool_id = ? OR target_pool_id = ? LIMIT 1",
        (pool_id, pool_id),
    ).fetchone()
    if migration is not None:
        side = "source" if migration["source_pool_id"] == pool_id else "target"
        raise _renumber_error(
            f"pool {pool_id} is the {side} pool of completed cross-pool "
            f"migration {migration['operation_id']!r}; its idempotent replay "
            "must keep returning the original addresses",
            "renumber_blocked_by_allocation_migration",
            status_code=409,
        )


def renumber_pool(pool_id: int, new_cidr: str) -> dict:
    """Move pool ``pool_id`` to ``new_cidr`` (same prefix length) atomically."""

    try:
        new_pool_net = parse_network(new_cidr)
    except ValueError as exc:
        raise _renumber_error(f"invalid new pool CIDR: {exc}", "invalid_cidr")

    # The whole check-and-rewrite runs inside one immediate write
    # transaction: SQLite serialises writers, so a concurrent allocation,
    # reservation or another renumber cannot interleave with the translate.
    with write_transaction() as conn:
        pool = _row_pool(conn, pool_id)
        old_pool_net = parse_network(pool["cidr"])

        if new_pool_net.prefixlen != old_pool_net.prefixlen:
            raise _renumber_error(
                f"new CIDR {new_pool_net} has prefix length /{new_pool_net.prefixlen} "
                f"but pool {old_pool_net} has /{old_pool_net.prefixlen}; a "
                "renumber must keep the same prefix length",
                "renumber_prefix_length_changed",
            )

        # Pinned plans / replayable historical records make renumbering
        # unsafe, regardless of whether the CIDR actually changes.
        _ensure_no_blocking_history(conn, pool_id)

        # The new range must not overlap any other pool.  Pool CIDRs are not
        # otherwise required to be disjoint (cross-pool migrations allow
        # overlap), so this is specific to renumbering.
        other_pools = conn.execute(
            "SELECT id, cidr FROM pools WHERE id != ?",
            (pool_id,),
        ).fetchall()
        for other in other_pools:
            other_net = parse_network(other["cidr"])
            if new_pool_net.overlaps(other_net):
                raise _renumber_error(
                    f"new pool CIDR {new_pool_net} overlaps existing pool "
                    f"{other['id']} ({other_net})",
                    "renumber_pool_conflict",
                    status_code=409,
                )

        # Constant offset between the old and new pool starts.
        offset = int(new_pool_net.network_address) - int(old_pool_net.network_address)

        exclusion_rows = _pool_exclusions(conn, pool_id)
        allocation_rows = _pool_allocations(conn, pool_id)

        translated_exclusions: list[tuple[sqlite3.Row, ipaddress.IPv4Network]] = []
        for row in exclusion_rows:
            old_net = parse_network(row["cidr"])
            new_net = _translate(old_net, offset)
            if not new_net.subnet_of(new_pool_net):
                raise _renumber_error(
                    f"translated exclusion {old_net} -> {new_net} is not "
                    f"completely inside the new pool {new_pool_net}",
                    "renumber_exclusion_outside_pool",
                )
            translated_exclusions.append((row, new_net))

        translated_allocations: list[tuple[sqlite3.Row, ipaddress.IPv4Network]] = []
        for row in allocation_rows:
            old_net = parse_network(row["cidr"])
            new_net = _translate(old_net, offset)
            if not new_net.subnet_of(new_pool_net):
                raise _renumber_error(
                    f"translated allocation {old_net} -> {new_net} is not "
                    f"completely inside the new pool {new_pool_net}",
                    "renumber_allocation_outside_pool",
                )
            translated_allocations.append((row, new_net))

        # All translated ranges must stay mutually non-conflicting.  A pure
        # translation by one constant offset preserves every overlap
        # relation, but the requirement is enforced explicitly so a corrupt
        # input can never pass.
        all_ranges = [net for _, net in translated_exclusions] + [
            net for _, net in translated_allocations
        ]
        all_ranges.sort(key=lambda n: int(n.network_address))
        for earlier, later in zip(all_ranges, all_ranges[1:]):
            if earlier.overlaps(later):
                raise _renumber_error(
                    f"translated ranges {earlier} and {later} overlap after "
                    f"renumbering the pool to {new_pool_net}",
                    "renumber_translated_range_conflict",
                    status_code=409,
                )

        # All checks passed: rewrite CIDRs in place.  Row ids, request_ids,
        # prefix lengths and timestamps never change; the pool row itself is
        # updated rather than deleted/recreated, so the pool id and its
        # foreign-key relationships survive untouched.
        try:
            conn.execute(
                "UPDATE pools SET cidr = ? WHERE id = ?",
                (str(new_pool_net), pool_id),
            )
            conn.executemany(
                "UPDATE exclusions SET cidr = ? WHERE id = ? AND pool_id = ?",
                [
                    (str(new_net), row["id"], pool_id)
                    for row, new_net in translated_exclusions
                ],
            )
            conn.executemany(
                "UPDATE allocations SET cidr = ? WHERE id = ? AND pool_id = ?",
                [
                    (str(new_net), row["id"], pool_id)
                    for row, new_net in translated_allocations
                ],
            )
        except sqlite3.IntegrityError:
            raise _renumber_error(
                f"renumber of pool {pool_id} to {new_pool_net} conflicts with "
                "existing data",
                "renumber_conflict",
                status_code=409,
            )

        # Old -> new comparison, deterministically ordered by the old
        # numeric address so the response is independent of row-id order.
        exclusion_pairs = sorted(
            translated_exclusions,
            key=lambda p: int(parse_network(p[0]["cidr"]).network_address),
        )
        allocation_pairs = sorted(
            translated_allocations,
            key=lambda p: int(parse_network(p[0]["cidr"]).network_address),
        )

    return {
        "pool_id": pool_id,
        "old_cidr": str(old_pool_net),
        "new_cidr": str(new_pool_net),
        "exclusions": [
            {"old_cidr": row["cidr"], "new_cidr": str(net)}
            for row, net in exclusion_pairs
        ],
        "allocations": [
            {
                "request_id": row["request_id"],
                "old_cidr": row["cidr"],
                "new_cidr": str(net),
                "prefix_length": row["prefix_length"],
                "allocation_id": row["id"],
                "created_at": row["created_at"],
            }
            for row, net in allocation_pairs
        ],
    }
