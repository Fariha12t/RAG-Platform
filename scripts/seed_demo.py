"""Seed demo users, collections and sample documents (no API server needed).

    python -m scripts.seed_demo

Creates: admin (key from ADMIN_API_KEY), alice (admin) and bob (user, access to HR & Policies only).
"""
from pathlib import Path

from sqlalchemy import select

from app.db import Collection, CollectionAccess, User, init_db, session_scope
from app.ingestion.pipeline import ingest_file
from app.services import auth

DOCS = Path(__file__).resolve().parent.parent / "data" / "sample_docs"
LAYOUT = {
    "Finance": ["northwind_annual_report_2023.pdf", "contoso_retail_annual_report_2023.pdf",
                "rag_customer_support_study.pdf"],
    "HR & Policies": ["employee_leave_policy.docx", "travel_expense_policy.txt",
                      "information_security_policy.pdf", "data_privacy_compliance_guide.docx"],
    "Product & Vendors": ["aurorasense_x200_manual.pdf", "vendor_onboarding_notes.txt"],
}


def main():
    init_db()
    with session_scope() as db:
        auth.bootstrap_admin(db)
        keys = {}
        for name, role in (("alice", "admin"), ("bob", "user")):
            user = db.scalar(select(User).where(User.username == name))
            if not user:
                user, keys[name] = auth.create_user(db, name, role)
        admin = db.scalar(select(User).where(User.role == "admin"))
        bob = db.scalar(select(User).where(User.username == "bob"))
        for cname, files in LAYOUT.items():
            col = db.scalar(select(Collection).where(Collection.name == cname))
            if not col:
                col = Collection(name=cname, owner_id=admin.id)
                db.add(col)
                db.flush()
            for fn in files:
                doc, created = ingest_file(db, col, DOCS / fn, fn, admin.id)
                print(f"[{cname}] {fn}: {doc.status} ({doc.n_chunks} chunks){'' if created else ' - already indexed'}")
            if cname == "HR & Policies" and not db.get(CollectionAccess, (col.id, bob.id)):
                db.add(CollectionAccess(collection_id=col.id, user_id=bob.id))
    print("\nNew API keys (shown once):", keys or "(users already existed)")


if __name__ == "__main__":
    main()
