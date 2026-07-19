"""
check_connection.py — run this locally (not in a restricted sandbox) to:
  1. confirm Atlas connectivity with your credentials
  2. list actual collection names in both databases (so you can confirm
     the real 2025 collection name — only the DB name 'DK' was given)
  3. print one sample document from each so you can eyeball field names
     against schema_mapping.py

Run:
    python check_connection.py
"""

from pymongo import MongoClient
from config import SIRConfig

cfg = SIRConfig()
print(f"Connecting to Atlas...")
client = MongoClient(cfg.mongo_uri, serverSelectionTimeoutMS=8000)
client.admin.command("ping")
print("Connected OK.\n")

for db_name in [cfg.mongo_db_2002, cfg.mongo_db_2025]:
    db = client[db_name]
    print(f"Database '{db_name}' collections: {db.list_collection_names()}")

print()
print(f"--- Sample doc from {cfg.mongo_db_2002}.{cfg.mongo_collection_2002} ---")
doc = client[cfg.mongo_db_2002][cfg.mongo_collection_2002].find_one()
print(doc)

print()
print(f"--- Sample doc from {cfg.mongo_db_2025}.{cfg.mongo_collection_2025} ---")
doc = client[cfg.mongo_db_2025][cfg.mongo_collection_2025].find_one()
print(doc)

print()
print(f"Doc counts: 2002={client[cfg.mongo_db_2002][cfg.mongo_collection_2002].count_documents({})}, "
      f"2025={client[cfg.mongo_db_2025][cfg.mongo_collection_2025].count_documents({})}")
