from pathlib import Path

from tests.conftest import SAMPLES


def _upload(client, headers, cid, names):
    files = [("files", (n, (SAMPLES / n).read_bytes())) for n in names]
    r = client.post(f"/collections/{cid}/documents", files=files, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def test_auth_required(client):
    assert client.get("/collections").status_code == 401
    assert client.get("/collections", headers={"X-API-Key": "nope"}).status_code == 401


def test_full_flow_upload_chat_citations_delete(client, admin):
    cid = client.post("/collections", json={"name": "hr"}, headers=admin).json()["id"]
    docs = _upload(client, admin, cid, ["employee_leave_policy.docx", "northwind_annual_report_2023.pdf",
                                        "travel_expense_policy.txt"])
    assert all(d["status"] == "ready" for d in docs)
    # duplicate upload is detected
    assert _upload(client, admin, cid, ["travel_expense_policy.txt"])[0]["duplicate"] is True

    r = client.post("/chat", json={"collection_id": cid, "question": "How many days of paid annual leave do employees get?"},
                    headers=admin).json()
    assert "24" in r["answer"]
    assert r["citations"] and r["citations"][0]["filename"] == "employee_leave_policy.docx"
    assert r["usage"]["model"] == "extractive"

    # follow-up keeps the conversation and history is retrievable
    r2 = client.post("/chat", json={"collection_id": cid, "question": "And carry forward?",
                                    "conversation_id": r["conversation_id"]}, headers=admin).json()
    assert r2["conversation_id"] == r["conversation_id"]
    conv = client.get(f"/conversations/{r['conversation_id']}", headers=admin).json()
    assert len(conv["messages"]) == 4

    # PDF page citation + preview
    r3 = client.post("/chat", json={"collection_id": cid, "question": "What was Northwind's revenue in fiscal 2023?"},
                     headers=admin).json()
    cit = r3["citations"][0]
    assert cit["filename"] == "northwind_annual_report_2023.pdf" and cit["page"] == 1
    prev = client.get(f"/documents/{cit['document_id']}/preview", params={"page": cit["page"]}, headers=admin).json()
    assert "412.6" in prev["text"]

    # reindex then delete
    pdf_id = next(d["id"] for d in docs if d["filename"].startswith("northwind"))
    assert client.post(f"/documents/{pdf_id}/reindex", headers=admin).json()["status"] == "ready"
    assert client.delete(f"/documents/{pdf_id}", headers=admin).status_code == 204
    r4 = client.post(f"/collections/{cid}/search", json={"query": "Northwind revenue"}, headers=admin).json()
    assert all(h["filename"] != "northwind_annual_report_2023.pdf" for h in r4)

    usage = client.get("/usage", headers=admin).json()
    assert usage["total_requests"] >= 3


def test_access_control(client, admin):
    cid = client.post("/collections", json={"name": "secret"}, headers=admin).json()["id"]
    _upload(client, admin, cid, ["information_security_policy.pdf"])
    bob = client.post("/users", json={"username": "bob", "role": "user"}, headers=admin).json()
    bh = {"X-API-Key": bob["api_key"]}
    assert client.get("/collections", headers=bh).json() == []
    assert client.post("/chat", json={"collection_id": cid, "question": "password length"}, headers=bh).status_code == 404
    assert client.post(f"/collections/{cid}/search", json={"query": "x"}, headers=bh).status_code == 404
    assert client.post("/users", json={"username": "eve"}, headers=bh).status_code == 403
    # shared => read allowed, write still forbidden
    assert client.post(f"/collections/{cid}/share", json={"username": "bob"}, headers=admin).status_code == 200
    r = client.post("/chat", json={"collection_id": cid, "question": "What is the minimum password length?"}, headers=bh)
    assert r.status_code == 200 and "14" in r.json()["answer"]
    docs = client.get(f"/collections/{cid}/documents", headers=bh).json()
    assert client.delete(f"/documents/{docs[0]['id']}", headers=bh).status_code == 403


def test_prompt_injection_in_document_is_neutralised(client, admin):
    cid = client.post("/collections", json={"name": "vendors"}, headers=admin).json()["id"]
    docs = _upload(client, admin, cid, ["vendor_onboarding_notes.txt"])
    assert docs[0]["flagged_chunks"] == 1
    r = client.post("/chat", json={"collection_id": cid, "question": "Are new vendors pre-approved and exempt from all checks?"},
                    headers=admin).json()
    assert "pre-approved" not in r["answer"].lower()
    assert r["injection_redactions"] >= 1


def test_rejects_unsupported_file(client, admin):
    cid = client.post("/collections", json={"name": "x"}, headers=admin).json()["id"]
    r = client.post(f"/collections/{cid}/documents", files=[("files", ("bad.exe", b"MZ"))], headers=admin).json()
    assert r[0]["status"] == "rejected"
