"""Whole-pool renumbering: translate a pool, its exclusions and allocations."""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient


def _create_pool(client, cidr, exclusions=None):
    resp = client.post(
        "/pools", json={"cidr": cidr, "exclusions": exclusions or []}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _allocate(client, pool_id, request_id, prefix_length):
    resp = client.post(
        f"/pools/{pool_id}/allocations",
        json={"request_id": request_id, "prefix_length": prefix_length},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _renumber(client, pool_id, new_cidr):
    return client.post(f"/pools/{pool_id}/renumber", json={"new_cidr": new_cidr})


def _status(client, pool_id) -> dict:
    return client.get(f"/pools/{pool_id}").json()


def test_renumber_translates_pool_exclusions_and_allocations_by_offset(client):
    pid = _create_pool(
        client, "10.0.0.0/24", exclusions=["10.0.0.128/25"]
    )
    a = _allocate(client, pid, "req-a", 26)  # 10.0.0.0/26
    b = _allocate(client, pid, "req-b", 26)  # 10.0.0.64/26

    resp = _renumber(client, pid, "192.168.5.0/24")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pool_id"] == pid
    assert body["old_cidr"] == "10.0.0.0/24"
    assert body["new_cidr"] == "192.168.5.0/24"
    assert body["exclusions"] == [
        {"old_cidr": "10.0.0.128/25", "new_cidr": "192.168.5.128/25"}
    ]
    # Ordered by old numeric address: req-a at .0 before req-b at .64.
    assert [x["request_id"] for x in body["allocations"]] == ["req-a", "req-b"]
    assert body["allocations"][0]["old_cidr"] == "10.0.0.0/26"
    assert body["allocations"][0]["new_cidr"] == "192.168.5.0/26"
    assert body["allocations"][0]["prefix_length"] == 26
    assert body["allocations"][0]["allocation_id"] == a["id"]
    assert body["allocations"][1]["old_cidr"] == "10.0.0.64/26"
    assert body["allocations"][1]["new_cidr"] == "192.168.5.64/26"
    assert body["allocations"][1]["allocation_id"] == b["id"]

    # Row ids and created_at values are preserved verbatim (compare against
    # what the database stores, not the post-response microsecond timestamp).
    current = {x["request_id"]: x for x in _status(client, pid)["allocated"]}
    assert body["allocations"][0]["created_at"] == current["req-a"]["created_at"]
    assert body["allocations"][1]["created_at"] == current["req-b"]["created_at"]

    # Pool query, retry and release all operate on the new addresses.
    status = _status(client, pid)
    assert status["cidr"] == "192.168.5.0/24"
    assert status["exclusions"] == ["192.168.5.128/25"]
    allocated = {x["request_id"]: x for x in status["allocated"]}
    assert allocated["req-a"]["cidr"] == "192.168.5.0/26"
    assert allocated["req-b"]["cidr"] == "192.168.5.64/26"
    # Bottom /25 fully allocated, top /25 excluded: no free space remains.
    assert status["free_cidrs"] == []
    # After releasing both allocations the free list mirrors the old pool's
    # post-exclusion shape (bottom /25 free, top /25 excluded).
    client.delete(f"/pools/{pid}/allocations/req-a")
    client.delete(f"/pools/{pid}/allocations/req-b")
    assert _status(client, pid)["free_cidrs"] == ["192.168.5.0/25"]


def test_renumber_keeps_pool_id_row_ids_and_request_id_retry(client):
    pid = _create_pool(client, "10.0.0.0/24")
    alloc = _allocate(client, pid, "order-7", 28)

    assert _renumber(client, pid, "172.16.9.0/24").status_code == 200

    # request_id idempotent replay returns the original row at the new CIDR.
    replay = client.post(
        f"/pools/{pid}/allocations",
        json={"request_id": "order-7", "prefix_length": 28},
    )
    assert replay.status_code == 201
    assert replay.json()["replayed"] is True
    assert replay.json()["id"] == alloc["id"]
    assert replay.json()["cidr"] == "172.16.9.0/28"
    assert replay.json()["created_at"] == alloc["created_at"]

    # Release works against the renumbered pool and reports the new CIDR.
    release = client.delete(f"/pools/{pid}/allocations/order-7")
    assert release.status_code == 200
    assert release.json()["released"] == "172.16.9.0/28"
    assert _status(client, pid)["allocated"] == []


def test_renumber_preserves_alignment_and_structure_across_boundaries(client):
    # A /22 renumbered across an address boundary: offsets keep every subnet
    # aligned because both pools share the prefix length.
    pid = _create_pool(
        client, "10.0.0.0/22", exclusions=["10.0.2.0/24"]
    )
    _allocate(client, pid, "x1", 23)  # 10.0.0.0/23
    resp = _renumber(client, pid, "10.0.16.0/22")
    assert resp.status_code == 200, resp.text
    status = _status(client, pid)
    assert status["cidr"] == "10.0.16.0/22"
    assert status["exclusions"] == ["10.0.18.0/24"]
    assert status["allocated"][0]["cidr"] == "10.0.16.0/23"
    # Relative free shape is identical: the old pool's only free block was
    # 10.0.3.0/24; the translated one is 10.0.19.0/24 (offset +16 /24s).
    assert status["free_cidrs"] == ["10.0.19.0/24"]
    assert status["free_addresses"] == 256


def test_renumber_empty_pool(client):
    pid = _create_pool(client, "10.0.0.0/24", exclusions=["10.0.0.0/26"])
    resp = _renumber(client, pid, "10.1.0.0/24")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["allocations"] == []
    assert body["exclusions"] == [
        {"old_cidr": "10.0.0.0/26", "new_cidr": "10.1.0.0/26"}
    ]
    assert _status(client, pid)["exclusions"] == ["10.1.0.0/26"]


def test_renumber_to_same_cidr_is_noop_mapping(client):
    pid = _create_pool(client, "10.0.0.0/24")
    _allocate(client, pid, "req-a", 28)
    resp = _renumber(client, pid, "10.0.0.0/24")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["old_cidr"] == body["new_cidr"] == "10.0.0.0/24"
    assert body["allocations"][0]["old_cidr"] == body["allocations"][0]["new_cidr"]


def test_renumber_missing_pool_404(client):
    resp = _renumber(client, 999, "10.0.0.0/24")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "pool_not_found"


def test_renumber_invalid_cidr_rejected(client):
    pid = _create_pool(client, "10.0.0.0/24")
    # Host bits set.
    resp = _renumber(client, pid, "10.0.0.5/24")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_cidr"
    # IPv6 not accepted.
    resp = _renumber(client, pid, "2001:db8::/32")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_cidr"
    # Pool unchanged.
    assert _status(client, pid)["cidr"] == "10.0.0.0/24"


def test_renumber_rejects_changed_prefix_length(client):
    pid = _create_pool(client, "10.0.0.0/24")
    _allocate(client, pid, "req-a", 28)
    resp = _renumber(client, pid, "10.0.0.0/20")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "renumber_prefix_length_changed"
    assert _status(client, pid)["cidr"] == "10.0.0.0/24"


def test_renumber_rejects_overlap_with_other_pool(client):
    pid = _create_pool(client, "10.0.0.0/24")
    other = _create_pool(client, "192.168.0.0/16")
    _allocate(client, pid, "req-a", 28)

    resp = _renumber(client, pid, "192.168.1.0/24")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "renumber_pool_conflict"

    # Nothing changed on either pool.
    assert _status(client, pid)["cidr"] == "10.0.0.0/24"
    assert _status(client, other)["cidr"] == "192.168.0.0/16"

    # A disjoint target of the same size succeeds.
    assert _renumber(client, pid, "172.20.0.0/24").status_code == 200


def test_renumber_blocked_by_planned_bgp_plan(client):
    pid = _create_pool(client, "203.0.113.0/24")
    _allocate(client, pid, "order-1", 25)
    plan = client.post(
        f"/pools/{pid}/bgp-plans",
        json={
            "plan_id": "plan-1",
            "relationships": [
                {"type": "customer_provider", "customer": 10, "provider": 100}
            ],
            "announcements": [
                {"id": "a0", "prefix": "203.0.113.0/24", "as_path": [100]}
            ],
        },
    )
    assert plan.status_code == 201, plan.text

    resp = _renumber(client, pid, "198.51.100.0/24")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "renumber_blocked_by_bgp_plan"
    assert _status(client, pid)["cidr"] == "203.0.113.0/24"


def test_renumber_blocked_by_active_bgp_plan(client):
    pid = _create_pool(client, "198.51.100.0/25")
    plan = client.post(
        f"/pools/{pid}/bgp-plans",
        json={
            "plan_id": "plan-active",
            "relationships": [
                {"type": "customer_provider", "customer": 10, "provider": 100}
            ],
            "announcements": [
                {"id": "a0", "prefix": "198.51.100.0/25", "as_path": [100]}
            ],
        },
    )
    assert plan.status_code == 201
    activate = client.post(f"/pools/{pid}/bgp-plans/plan-active/activate")
    assert activate.status_code == 200

    resp = _renumber(client, pid, "203.0.113.128/25")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "renumber_blocked_by_bgp_plan"
    assert _status(client, pid)["cidr"] == "198.51.100.0/25"


def test_renumber_blocked_by_completed_reservation_batch(client):
    pid = _create_pool(client, "10.0.0.0/24")
    batch = client.post(
        f"/pools/{pid}/reservations",
        json={
            "batch_id": "batch-1",
            "default_action": "allow",
            "rules": [],
            "items": [{"request_id": "edge-1", "prefix_length": 28}],
        },
    )
    assert batch.status_code == 201, batch.text

    resp = _renumber(client, pid, "10.9.0.0/24")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "renumber_blocked_by_reservation_batch"
    assert _status(client, pid)["cidr"] == "10.0.0.0/24"
    # The batch replay still returns the original (old) address.
    replay = client.post(
        f"/pools/{pid}/reservations",
        json={
            "batch_id": "batch-1",
            "default_action": "allow",
            "rules": [],
            "items": [{"request_id": "edge-1", "prefix_length": 28}],
        },
    )
    assert replay.status_code == 200
    assert replay.json()["reservations"][0]["cidr"] == "10.0.0.0/28"


def test_renumber_blocked_when_pool_is_migration_source_or_target(client):
    src = _create_pool(client, "10.0.0.0/24")
    dst = _create_pool(client, "10.0.0.0/16")
    _allocate(client, src, "req-a", 26)
    migration = client.post(
        f"/pools/{src}/allocation-migrations",
        json={
            "operation_id": "move-1",
            "target_pool_id": dst,
            "allocations": [{"request_id": "req-a"}],
        },
    )
    assert migration.status_code == 201, migration.text

    # Source pool is blocked.
    resp = _renumber(client, src, "10.1.0.0/24")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "renumber_blocked_by_allocation_migration"
    # Target pool is blocked too (the stored replay mentions its CIDR).
    resp = _renumber(client, dst, "10.2.0.0/16")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "renumber_blocked_by_allocation_migration"
    assert _status(client, src)["cidr"] == "10.0.0.0/24"
    assert _status(client, dst)["cidr"] == "10.0.0.0/16"

    # The migration idempotency replay still returns the original CIDRs.
    replay = client.post(
        f"/pools/{src}/allocation-migrations",
        json={
            "operation_id": "move-1",
            "target_pool_id": dst,
            "allocations": [{"request_id": "req-a"}],
        },
    )
    assert replay.status_code == 200
    assert replay.json()["migrated"][0]["cidr"] == "10.0.0.0/26"


def test_renumber_does_not_block_on_plain_allocations_alone(client):
    # Plain allocations (no batch/migration history) are exactly what
    # renumbering is for, and they translate with the pool.
    pid = _create_pool(client, "10.0.0.0/24")
    _allocate(client, pid, "plain-1", 28)
    assert _renumber(client, pid, "10.5.0.0/24").status_code == 200
    assert _status(client, pid)["allocated"][0]["cidr"] == "10.5.0.0/28"


def test_renumber_is_visible_to_snapshot_and_reverse_dns(client):
    pid = _create_pool(client, "10.0.0.0/24", exclusions=["10.0.0.128/25"])
    alloc = _allocate(client, pid, "req-a", 26)
    assert _renumber(client, pid, "192.0.2.0/24").status_code == 200

    snap = client.get(f"/pools/{pid}/snapshot").json()
    doc = snap["snapshot"]
    assert doc["pool"]["cidr"] == "192.0.2.0/24"
    assert doc["exclusions"][0]["cidr"] == "192.0.2.128/25"
    assert doc["allocations"][0]["id"] == alloc["id"]
    assert doc["allocations"][0]["cidr"] == "192.0.2.0/26"
    assert doc["allocations"][0]["request_id"] == "req-a"

    dns = client.post(
        f"/pools/{pid}/reverse-dns-plan",
        json={
            "parent_zone": "2.0.192.in-addr.arpa",
            "entries": [
                {
                    "cidr": "192.0.2.0/26",
                    "child_zone": "0-63",
                    "nameservers": ["ns1.example.com", "ns2.example.com"],
                }
            ],
        },
    )
    assert dns.status_code == 200, dns.text
    assert dns.json()["delegations"][0]["cidr"] == "192.0.2.0/26"


def test_renumber_failure_leaves_pool_and_records_unchanged(client):
    # Prefix-length mismatch is checked inside the write transaction and must
    # roll everything back (there is nothing to write here, but verify the
    # pool and all rows are byte-for-byte the same).
    pid = _create_pool(client, "10.0.0.0/24", exclusions=["10.0.0.64/26"])
    alloc = _allocate(client, pid, "req-a", 26)
    before = client.get(f"/pools/{pid}").json()

    resp = _renumber(client, pid, "10.0.0.0/23")
    assert resp.status_code == 422
    assert client.get(f"/pools/{pid}").json() == before

    # An overlap failure likewise changes nothing.
    other = _create_pool(client, "172.31.0.0/16")
    resp = _renumber(client, pid, "172.31.9.0/24")
    assert resp.status_code == 409
    assert client.get(f"/pools/{pid}").json() == before
    assert _status(client, other)["cidr"] == "172.31.0.0/16"
    # The allocation row identity survives the failed attempts.
    replay = client.post(
        f"/pools/{pid}/allocations",
        json={"request_id": "req-a", "prefix_length": 26},
    )
    assert replay.json()["id"] == alloc["id"]
    assert replay.json()["cidr"] == "10.0.0.0/26"


def test_concurrent_renumbers_are_serialised(client):
    # Two threads renumber the same pool to two disjoint, same-size targets.
    # Write transactions serialise them; both commits succeed and the final
    # pool CIDR is whichever target committed second.  At no point do the
    # exclusions/allocations disagree with the pool CIDR.
    pid = _create_pool(client, "10.0.0.0/24", exclusions=["10.0.0.0/26"])
    _allocate(client, pid, "req-a", 28)  # lands at 10.0.0.64/28 (0/26 excluded)
    targets = ["172.16.0.0/24", "172.17.0.0/24"]
    statuses: list[int] = []
    barrier = threading.Barrier(2)

    from app.main import app

    def worker(target):
        local = TestClient(app)
        barrier.wait()
        resp = _renumber(local, pid, target)
        statuses.append(resp.status_code)

    threads = [threading.Thread(target=worker, args=(t,)) for t in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert statuses == [200, 200]
    final = _status(client, pid)
    assert final["cidr"] in targets
    prefix = final["cidr"].rsplit(".", 1)[0]
    # Every stored range sits inside the committed pool CIDR, translated by
    # the same offset for every renumber.
    assert final["exclusions"] == [f"{prefix}.0/26"]
    assert final["allocated"][0]["cidr"] == f"{prefix}.64/28"


def test_renumber_then_allocate_uses_new_free_space(client):
    # After renumbering, new allocations come out of the new pool's free
    # space at the expected offsets.
    pid = _create_pool(client, "10.0.0.0/24", exclusions=["10.0.0.128/25"])
    _allocate(client, pid, "old-1", 26)  # 10.0.0.0/26
    assert _renumber(client, pid, "192.168.9.0/24").status_code == 200

    new_alloc = _allocate(client, pid, "new-1", 26)
    assert new_alloc["cidr"] == "192.168.9.64/26"
