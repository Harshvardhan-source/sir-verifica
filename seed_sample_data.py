"""
seed_sample_data.py — Inserts small synthetic sample datasets into the two
MongoDB collections so you can test the pipeline end-to-end before pointing
it at the real Dakshina Kannada 2002/2025 voter rolls.

Deliberately includes planted anomalies (duplicate EPIC, overcrowded house,
lineage break, deceased-still-active, relation drift) so you can verify the
anomaly_detector.py output looks right.

Run:
    python seed_sample_data.py
"""

from pymongo import MongoClient
from config import SIRConfig

cfg = SIRConfig()
client = MongoClient(cfg.mongo_uri)
db = client[cfg.mongo_db]

voters_2002 = [
    {"epic_no": "KA1000001", "door_no": "12-45", "voter_name": "Suresh Rao", "relation_name": "Ganapathi Rao",
     "relation_type": "Father", "age": 45, "gender": "M", "part_no": "101", "serial_no": "1",
     "constituency": "Dakshina Kannada", "status": "ACTIVE", "address": "Kadri, Mangalore"},
    {"epic_no": "KA1000002", "door_no": "12-45", "voter_name": "Lakshmi Rao", "relation_name": "Suresh Rao",
     "relation_type": "Husband", "age": 40, "gender": "F", "part_no": "101", "serial_no": "2",
     "constituency": "Dakshina Kannada", "status": "ACTIVE", "address": "Kadri, Mangalore"},
    {"epic_no": "KA1000003", "door_no": "7-9", "voter_name": "Abdul Rahman", "relation_name": "Ismail Sait",
     "relation_type": "Father", "age": 60, "gender": "M", "part_no": "102", "serial_no": "3",
     "constituency": "Dakshina Kannada", "status": "DECEASED", "address": "Bunder, Mangalore"},
    {"epic_no": "KA1000004", "door_no": "3-1", "voter_name": "Shreya Shetty", "relation_name": "Ravi Shetty",
     "relation_type": "Father", "age": 30, "gender": "F", "part_no": "103", "serial_no": "4",
     "constituency": "Dakshina Kannada", "status": "ACTIVE", "address": "Surathkal"},
    # will "break lineage" - not present in 2025, no fuzzy match
    {"epic_no": "KA1000005", "door_no": "8-8", "voter_name": "Ganesh Poojary", "relation_name": "Krishna Poojary",
     "relation_type": "Father", "age": 55, "gender": "M", "part_no": "104", "serial_no": "5",
     "constituency": "Dakshina Kannada", "status": "ACTIVE", "address": "Ullal"},
]

voters_2025 = [
    {"epic_no": "KA1000001", "door_no": "12-45", "voter_name": "Suresh Rao", "relation_name": "Ganapathi Rao",
     "relation_type": "Father", "age": 68, "gender": "M", "part_no": "101", "serial_no": "1",
     "constituency": "Dakshina Kannada", "status": "ACTIVE", "address": "Kadri, Mangalore"},
    {"epic_no": "KA1000002", "door_no": "12-45", "voter_name": "Lakshmi Rao", "relation_name": "Suresh Rao",
     "relation_type": "Husband", "age": 63, "gender": "F", "part_no": "101", "serial_no": "2",
     "constituency": "Dakshina Kannada", "status": "ACTIVE", "address": "Kadri, Mangalore"},
    # deceased in 2002 but still active in 2025 -> anomaly
    {"epic_no": "KA1000003", "door_no": "7-9", "voter_name": "Abdul Rahman", "relation_name": "Ismail Sait",
     "relation_type": "Father", "age": 83, "gender": "M", "part_no": "102", "serial_no": "3",
     "constituency": "Dakshina Kannada", "status": "ACTIVE", "address": "Bunder, Mangalore"},
    # relation name drastically changed under same EPIC -> anomaly
    {"epic_no": "KA1000004", "door_no": "3-1", "voter_name": "Shreya Shetty", "relation_name": "Vijay Kumar",
     "relation_type": "Husband", "age": 53, "gender": "F", "part_no": "103", "serial_no": "4",
     "constituency": "Dakshina Kannada", "status": "ACTIVE", "address": "Surathkal"},
    # duplicate EPIC within 2025 roll -> anomaly
    {"epic_no": "KA1000099", "door_no": "5-2", "voter_name": "Manjunath Kamath", "relation_name": "Ramesh Kamath",
     "relation_type": "Father", "age": 34, "gender": "M", "part_no": "105", "serial_no": "6",
     "constituency": "Dakshina Kannada", "status": "ACTIVE", "address": "Bejai"},
    {"epic_no": "KA1000099", "door_no": "5-2", "voter_name": "Manjunath Kamath D.", "relation_name": "Ramesh Kamath",
     "relation_type": "Father", "age": 34, "gender": "M", "part_no": "105", "serial_no": "7",
     "constituency": "Dakshina Kannada", "status": "ACTIVE", "address": "Bejai"},
    # duplicate person, different EPIC, same name/relation/age -> anomaly
    {"epic_no": "KA1000100", "door_no": "9-3", "voter_name": "Sunitha Bhandary", "relation_name": "Prakash Bhandary",
     "relation_type": "Father", "age": 29, "gender": "F", "part_no": "106", "serial_no": "8",
     "constituency": "Dakshina Kannada", "status": "ACTIVE", "address": "Kankanady"},
    {"epic_no": "KA1000101", "door_no": "10-1", "voter_name": "Sunitha Bhandary", "relation_name": "Prakash Bhandary",
     "relation_type": "Father", "age": 29, "gender": "F", "part_no": "106", "serial_no": "9",
     "constituency": "Dakshina Kannada", "status": "ACTIVE", "address": "Attavar"},
    # new genuine registration, no 2002 lineage -> GHOST_ENTRY flag (expected/benign)
    {"epic_no": "KA1000102", "door_no": "2-7", "voter_name": "Akash Pai", "relation_name": "Ganesh Pai",
     "relation_type": "Father", "age": 21, "gender": "M", "part_no": "107", "serial_no": "10",
     "constituency": "Dakshina Kannada", "status": "ACTIVE", "address": "Kottara"},
]

# overcrowded house test: add 16 voters at the same door number
for i in range(16):
    voters_2025.append({
        "epic_no": f"KA20{i:05d}", "door_no": "OVERCROWD-1", "voter_name": f"Test Voter {i}",
        "relation_name": "Test Relation", "relation_type": "Father", "age": 25 + i, "gender": "M",
        "part_no": "108", "serial_no": str(100 + i), "constituency": "Dakshina Kannada",
        "status": "ACTIVE", "address": "Test Colony",
    })

db[cfg.mongo_collection_2002].delete_many({})
db[cfg.mongo_collection_2025].delete_many({})
db[cfg.mongo_collection_2002].insert_many(voters_2002)
db[cfg.mongo_collection_2025].insert_many(voters_2025)

print(f"Seeded {len(voters_2002)} docs into '{cfg.mongo_collection_2002}'")
print(f"Seeded {len(voters_2025)} docs into '{cfg.mongo_collection_2025}'")
