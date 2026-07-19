#!/usr/bin/env python3
"""
MongoDB Voter Community + Religion Classifier  (ML-powered)
───────────────────────────────────────────────────────────
Reads voter documents directly from MongoDB, classifies Community and
Religion, and writes results back to the same documents — no Excel,
no row limits. Built for collections of any size via batched bulk writes.

Religion classification:
    Pass 1 — Community-derived (rule-based, from curated surname dict)
    Pass 2 — Jain dictionary match (the ML model has no Jain class)
    Pass 3 — trained ML model (hmc_classifier_final.joblib) predicts
             Hindu / Muslim / Christian for everything left over,
             batched for speed. Falls back to dictionary matching
             automatically if the model file can't be loaded.

Requirements:
    pip install pymongo scikit-learn joblib pandas

    The model file (hmc_classifier_final.joblib) must sit in the same
    folder as this script, or pass --model-path /full/path/to/model.joblib

CONFIGURE MONGO_URI BELOW, then run:

    # Step 1 — ALWAYS run a dry run first (no writes to DB):
    python mongo_voter_classifier.py --dry-run --target 2025 --sample-size 50
    python mongo_voter_classifier.py --dry-run --target 2002 --sample-size 50

    # Step 2 — once the dry run looks correct, run the full classification:
    python mongo_voter_classifier.py --full --target 2025
    python mongo_voter_classifier.py --full --target 2002

    # Resume an interrupted full run (skips already-classified docs):
    python mongo_voter_classifier.py --full --target 2025 --resume

    # Force re-classify everything, including already-classified docs:
    python mongo_voter_classifier.py --full --target 2025 --no-resume

Named targets (both in database 'SurveyDataBase'):
    2002  ->  collection 'DK_2002_new2'
    2025  ->  collection 'DK'           (default)

    # Use a model file from a different location:
    python mongo_voter_classifier.py --full --target 2025 --model-path /path/to/model.joblib

    # Skip the ML model entirely, dictionary matching only:
    python mongo_voter_classifier.py --full --target 2025 --no-ml
"""

import re
import sys
import argparse
from datetime import datetime
import pandas as pd
import joblib
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError


# ═══════════════════════════════════════════════════════════════════════
#  CONNECTION CONFIG
# ═══════════════════════════════════════════════════════════════════════

# ── EDIT THIS: your MongoDB connection string ────────────────────────
MONGO_URI = "mongodb+srv://ravindraacharya0512:2Dlb9csFBkM9n9Bs@cluster0.ynaiaut.mongodb.net/"

# ── Named targets — pick one with --target 2002 or --target 2025 ────
# Both collections live in the same database, 'SurveyDataBase'.
TARGETS = {
    '2002': {'db': 'SurveyDataBase', 'collection': 'DK_2002_new2'},
    '2025': {'db': 'SurveyDataBase', 'collection': 'DK'},
}
DEFAULT_TARGET = '2025'   # used if --target is not specified

# Field names — leave as None to auto-detect from a sample document,
# or set explicitly if you already know them, e.g. NAME_FIELD = "Name"
NAME_FIELD     = None
RELATION_FIELD = None

BATCH_SIZE = 2000   # documents per bulk_write batch during full run

# ── Trained ML model for Hindu/Muslim/Christian prediction ───────────
# Used as the fallback classifier whenever Community is Unclassified
# (and the name isn't caught by the Jain dictionary). Replaces the old
# manual name-token dictionary matching for H/M/C.
ML_MODEL_PATH = "hmc_classifier_final.joblib"   # <-- must sit next to this script

# Populated by load_ml_model() at startup. Stays None if loading fails,
# in which case the script automatically falls back to dictionary
# matching so a missing model file never stops a production run.
ML_MODEL  = None
ML_INFO   = {}
ML_LABELS = {'H': 'Hindu', 'M': 'Muslim', 'C': 'Christian'}


def load_ml_model(path=ML_MODEL_PATH):
    """Load the trained HMC pipeline once at startup. Never raises —
    on failure, ML_MODEL stays None and the script uses the dictionary
    fallback instead, so a missing/corrupt model file never kills a
    1.7M-record production run."""
    global ML_MODEL, ML_INFO
    try:
        bundle = joblib.load(path)
        ML_MODEL = bundle['pipeline']
        ML_INFO  = bundle
        print(f"ML model loaded : {bundle.get('model_type', '?')}")
        print(f"  trained on    : {bundle.get('trained_on_rows', '?'):,} rows")
        print(f"  held-out acc  : {bundle.get('held_out_accuracy', 0):.4f}   "
              f"macro F1: {bundle.get('held_out_macro_f1', 0):.4f}")
    except Exception as e:
        print(f"WARNING: could not load ML model from '{path}'  ({e})")
        print("  Falling back to dictionary-based Hindu/Muslim/Christian matching.")
        ML_MODEL = None


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 1 — COMMUNITY SURNAME DICTIONARY
# ═══════════════════════════════════════════════════════════════════════

SURNAME_DICT = {

    # ── BUNT  (GC) ───────────────────────────────────────────────────
    'SHETTY'        : ('Bunt', 'Hindu - GC'),
    'SHETTI'        : ('Bunt', 'Hindu - GC'),
    'HEGDE'         : ('Bunt', 'Hindu - GC'),
    'RAI'           : ('Bunt', 'Hindu - GC'),
    'ALVA'          : ('Bunt', 'Hindu - GC'),
    'BALLAL'        : ('Bunt', 'Hindu - GC'),
    'BHANDARY'      : ('Bunt', 'Hindu - GC'),
    'BHANDARI'      : ('Bunt', 'Hindu - GC'),
    'CHOWTA'        : ('Bunt', 'Hindu - GC'),
    'AJILA'         : ('Bunt', 'Hindu - GC'),
    'GANIGA'        : ('Saphalya','Hindu - OBC'),
    # ── OBC (SHET / SHETTIGAR) ───────────────────────────────────────
    'SHET'          : ('OBC (Shet)',       'Hindu - OBC'),
    'SHETTIGAR'     : ('OBC (Shettigar)', 'Hindu - OBC'),

    # ── GSB  (GC) ────────────────────────────────────────────────────
    'KAMATH'        : ('GSB (Goud Saraswat Brahmin)', 'Hindu - GC'),
    'KAMAT'         : ('GSB (Goud Saraswat Brahmin)', 'Hindu - GC'),
    'SHENOY'        : ('GSB (Goud Saraswat Brahmin)', 'Hindu - GC'),
    'PAI'           : ('GSB (Goud Saraswat Brahmin)', 'Hindu - GC'),
    'PRABHU'        : ('GSB (Goud Saraswat Brahmin)', 'Hindu - GC'),
    'BALIGA'        : ('GSB (Goud Saraswat Brahmin)', 'Hindu - GC'),
    'KINI'          : ('GSB (Goud Saraswat Brahmin)', 'Hindu - GC'),
    'MALLYA'        : ('GSB (Goud Saraswat Brahmin)', 'Hindu - GC'),
    'KUDVA'         : ('GSB (Goud Saraswat Brahmin)', 'Hindu - GC'),
    
    'PURANIK'       : ('GSB (Goud Saraswat Brahmin)', 'Hindu - GC'),
    'SHANBHAG'      : ('GSB (Goud Saraswat Brahmin)', 'Hindu - GC'),
    
    'SHENAI'        : ('GSB (Goud Saraswat Brahmin)', 'Hindu - GC'),
 
    'NAYAK'         : ('GSB (Goud Saraswat Brahmin)', 'Hindu - GC'),

    # ── SHIVALLI / BRAHMIN / GSB  (GC) ───────────────────────────────
    'BHAT'          : ('Brahmin/GSB', 'Hindu - GC'),
    'BHATT'         : ('Brahmin/GSB', 'Hindu - GC'),
    'AITHAL'        : ('Shivalli Brahmin', 'Hindu - GC'),
    'UDUPA'         : ('Shivalli Brahmin', 'Hindu - GC'),
    'ADIGA'         : ('Shivalli Brahmin', 'Hindu - GC'),
    'THINGALAYA'    : ('GSB (Goud Saraswat Brahmin)', 'Hindu - GC'),
    'TANTRI'        : ('GSB (Goud Saraswat Brahmin)', 'Hindu - GC'),
    'KARANTH'       : ('GSB (Goud Saraswat Brahmin)', 'Hindu - GC'),
    'UPADHYAYA'     : ('GSB (Goud Saraswat Brahmin)', 'Hindu - GC'),


    # ── COASTAL / HAVYAKA BRAHMIN  (GC) ──────────────────────────────
    'HOLLA'         : ('Brahmin (Coastal)', 'Hindu - GC'),
    'MAYYA'         : ('Brahmin (Coastal)', 'Hindu - GC'),
    'SHARMA'        : ('Brahmin (Coastal)', 'Hindu - GC'),
    'JOSHI'         : ('Brahmin (Coastal)', 'Hindu - GC'),

    # ── BILLAVA  (OBC) ───────────────────────────────────────────────
    'SUVARNA'       : ('Billava', 'Hindu - OBC'),
    'BELCHADA'      : ('Billava', 'Hindu - OBC'),
    'SUVARN'        : ('Billava', 'Hindu - OBC'),
    'SUVARANA'      : ('Billava', 'Hindu - OBC'),
    'SUVARMA'       : ('Billava', 'Hindu - OBC'),
    'SUWARNA'       : ('Billava', 'Hindu - OBC'),
    'POOJARY'       : ('Billava', 'Hindu - OBC'),
    'POOJARI'       : ('Billava', 'Hindu - OBC'),
    'PUJARI'        : ('Billava', 'Hindu - OBC'),
    'SALIAN'        : ('Billava', 'Hindu - OBC'),
    'SALIYAN'       : ('Billava', 'Hindu - OBC'),
    'SALYAN'        : ('Billava', 'Hindu - OBC'),
    'KOTIAN'        : ('Billava', 'Hindu - OBC'),
    'KOTYAN'        : ('Billava', 'Hindu - OBC'),
    'ANCHAN'        : ('Billava', 'Hindu - OBC'),
    'KANCHAN'       : ('Billava', 'Hindu - OBC'),
    'PUTRAN'        : ('Billava', 'Hindu - OBC'),
    'PUTHRAN'       : ('Billava', 'Hindu - OBC'),
    'MENDON'        : ('Billava', 'Hindu - OBC'),
    'KUNDAR'        :  ('Billava,bestha', 'Hindu - OBC'),
     'KUNDER'        :  ('Billava,bestha', 'Hindu - OBC'),
    # ── BILLAVA / MOGAVEERA  (OBC) ───────────────────────────────────
    'BANGERA'       : ('Billava/Mogaveera', 'Hindu - OBC'),
    'BANGER'        : ('Billava/Mogaveera', 'Hindu - OBC'),
    'KARKERA'       : ('Billava/Mogaveera', 'Hindu - OBC'),
    'KARKER'        : ('Billava/Mogaveera', 'Hindu - OBC'),
    'SANIL'         : ('Billava/Mogaveera', 'Hindu - OBC'),
    'SALIN'         : ('Billava/Mogaveera', 'Hindu - OBC'),

    # ── MOGAVEERA  (OBC) ─────────────────────────────────────────────
    'KHARVI'        : ('Mogaveera', 'Hindu - OBC'),
    'KHARWI'        : ('Mogaveera', 'Hindu - OBC'),
    'KHARYV'        : ('Mogaveera', 'Hindu - OBC'),
    'KHARV'         : ('Mogaveera', 'Hindu - OBC'),
    'MENDAN'        : ('Mogaveera', 'Hindu - OBC'),
    'MENDON'        : ('Mogaveera', 'Hindu - OBC'),


    # ── DEVADIGA  (OBC) ──────────────────────────────────────────────
    'DEVADIGA'      : ('Devadiga', 'Hindu - OBC'),

    # ── VISHWAKARMA  (OBC) ───────────────────────────────────────────
    'ACHARYA'       : ('Vishwakarma', 'Hindu - OBC'),
    'AACHARYA'      : ('Vishwakarma', 'Hindu - OBC'),
    'AACHARA'       : ('Vishwakarma', 'Hindu - OBC'),
    'ACHARI'        :('Vishwakarma', 'Hindu - OBC'),
    'ACHAR'         : ('Vishwakarma', 'Hindu - OBC'),

    # ── NAIK  (Multiple communities OBC/SC/ST) ───────────────────────
    'NAIK'          : ('Multiple communities (Naik)', 'Hindu - OBC/SC/ST'),
    'NAIKA'          : ('Multiple communities (Naik)', 'Hindu - OBC/SC/ST'),
    'NAIK'          : ('PARIWARA BUNT (Naik)', 'Hindu - OBC'),

    # ── VOKKALIGA  (GC) ──────────────────────────────────────────────
    'GOWDA'         : ('Vokkaliga', 'Hindu - GC'),

    # ── KOTTARI / MOOLYA / GANIGA  (OBC) ─────────────────────────────
    'SAPALYA'       : ('Ganiga (OBC)',   'Hindu - OBC'),
    'KOTTARY'       : ('Kottari (OBC)', 'Hindu - OBC'),
    'KOTTARI'       : ('Kottari (OBC)', 'Hindu - OBC'),
    'MOOLYA'        : ('Moolya (OBC)',  'Hindu - OBC'),

    # ── NAIR / NAMBIAR  (Kerala GC) ──────────────────────────────────
    'NAIR'          : ('Nair (Kerala GC)',    'Hindu - GC'),
    'NAYAR'         : ('Nair (Kerala GC)',    'Hindu - GC'),
    'NAMBIAR'       : ('Nambiar (Kerala GC)', 'Hindu - GC'),

    # ── VAIDYA  (Kerala OBC) ─────────────────────────────────────────
    'PANIKAR'       : ('Vaidya (Kerala OBC)', 'Hindu - OBC'),
    'PANIKER'       : ('Vaidya (Kerala OBC)', 'Hindu - OBC'),
    'PANIKERA'      : ('Vaidya (Kerala OBC)', 'Hindu - OBC'),

    # ── TRADING / GC ─────────────────────────────────────────────────
    'PAREKH'        : ('Trading/agricultural groups', 'Hindu - GC'),

    # ── MANGALOREAN CATHOLIC ─────────────────────────────────────────
    'DSOUZA'        : ('Mangalorean Catholic', 'Christian - OC'),
    'SOUZA'         : ('Mangalorean Catholic', 'Christian - OC'),
    'DSILVA'        : ('Mangalorean Catholic', 'Christian - OC'),
    'SILVA'         : ('Mangalorean Catholic', 'Christian - OC'),
    'DCOSTA'        : ('Mangalorean Catholic', 'Christian - OC'),
    'COSTA'         : ('Mangalorean Catholic', 'Christian - OC'),
    'DCUNHA'        : ('Mangalorean Catholic', 'Christian - OC'),
    'CUNHA'         : ('Mangalorean Catholic', 'Christian - OC'),
    'DMELLO'        : ('Mangalorean Catholic', 'Christian - OC'),
    'MELLO'         : ('Mangalorean Catholic', 'Christian - OC'),
    'DCRUZ'         : ('Mangalorean Catholic', 'Christian - OC'),
    'CRUZ'          : ('Mangalorean Catholic', 'Christian - OC'),
    'DALMEIDA'      : ('Mangalorean Catholic', 'Christian - OC'),
    'PINTO'         : ('Mangalorean Catholic', 'Christian - OC'),
    'LOBO'          : ('Mangalorean Catholic', 'Christian - OC'),
    'FERNANDES'     : ('Mangalorean Catholic', 'Christian - OC'),
    'RODRIGUES'     : ('Mangalorean Catholic', 'Christian - OC'),
    'RODRIQUES'     : ('Mangalorean Catholic', 'Christian - OC'),
    'SEQUEIRA'      : ('Mangalorean Catholic', 'Christian - OC'),
    'PEREIRA'       : ('Mangalorean Catholic', 'Christian - OC'),
    'SALDANHA'      : ('Mangalorean Catholic', 'Christian - OC'),
    'NORONHA'       : ('Mangalorean Catholic', 'Christian - OC'),
    'MONTEIRO'      : ('Mangalorean Catholic', 'Christian - OC'),
    'MENEZES'       : ('Mangalorean Catholic', 'Christian - OC'),
    'MIRANDA'       : ('Mangalorean Catholic', 'Christian - OC'),
    'FURTADO'       : ('Mangalorean Catholic', 'Christian - OC'),
    'CRASTA'        : ('Mangalorean Catholic', 'Christian - OC'),
    'MASCARENHAS'   : ('Mangalorean Catholic', 'Christian - OC'),
    'CASTELINO'     : ('Mangalorean Catholic', 'Christian - OC'),
    'GONSALVES'     : ('Mangalorean Catholic', 'Christian - OC'),
    'ALMEIDA'       : ('Mangalorean Catholic', 'Christian - OC'),
    'COLACO'        : ('Mangalorean Catholic', 'Christian - OC'),
    'COELHO'        : ('Mangalorean Catholic', 'Christian - OC'),
    'DIAS'          : ('Mangalorean Catholic', 'Christian - OC'),
    'REGO'          : ('Mangalorean Catholic', 'Christian - OC'),
    'QUADROS'       : ('Mangalorean Catholic', 'Christian - OC'),
    'NAZARETH'      : ('Mangalorean Catholic', 'Christian - OC'),
    'REBELLO'       : ('Mangalorean Catholic', 'Christian - OC'),
    'CORREA'        : ('Mangalorean Catholic', 'Christian - OC'),
    'MONIS'         : ('Mangalorean Catholic', 'Christian - OC'),
    'ROSARIO'       : ('Mangalorean Catholic', 'Christian - OC'),
    'ALVARES'       : ('Mangalorean Catholic', 'Christian - OC'),
    'FERRAO'        : ('Mangalorean Catholic', 'Christian - OC'),
    'GOVEAS'        : ('Mangalorean Catholic', 'Christian - OC'),
    'LASRADO'       : ('Mangalorean Catholic', 'Christian - OC'),
    'TAURO'         : ('Mangalorean Catholic', 'Christian - OC'),
    'CUTINHA'       : ('Mangalorean Catholic', 'Christian - OC'),
    'BARBOZA'       : ('Mangalorean Catholic', 'Christian - OC'),
    'BARRETTO'      : ('Mangalorean Catholic', 'Christian - OC'),
    'ANDRADE'       : ('Mangalorean Catholic', 'Christian - OC'),
    'GOMES'         : ('Mangalorean Catholic', 'Christian - OC'),
    'SERRAO'        : ('Mangalorean Catholic', 'Christian - OC'),
    'MARTIS'        : ('Mangalorean Catholic', 'Christian - OC'),
    'KARKADA'       : ('Mangalorean Catholic', 'Christian - OC'),
    'MENDONCA'      : ('Mangalorean Catholic', 'Christian - OC'),
    'VAZ'           : ('Mangalorean Catholic', 'Christian - OC'),
    'VAS'           : ('Mangalorean Catholic', 'Christian - OC'),
    'MABEN'         : ('Mangalorean Catholic', 'Christian - OC'),
    'MACHADO'       : ('Mangalorean Catholic', 'Christian - OC'),
    'ARANHA'        : ('Mangalorean Catholic', 'Christian - OC'),
    'ALBUQUERQUE'   : ('Mangalorean Catholic', 'Christian - OC'),
    'VIEGAS'        : ('Mangalorean Catholic', 'Christian - OC'),
    'MISQUITH'      : ('Mangalorean Catholic', 'Christian - OC'),
    'RASQUINHA'     : ('Mangalorean Catholic', 'Christian - OC'),
    'RASQUINA'      : ('Mangalorean Catholic', 'Christian - OC'),
    'MORAS'         : ('Mangalorean Catholic', 'Christian - OC'),
    'MORASA'        : ('Mangalorean Catholic', 'Christian - OC'),
    'MENDONSA'      : ('Mangalorean Catholic', 'Christian - OC'),
    'MENDONZA'      : ('Mangalorean Catholic', 'Christian - OC'),
    'MEDONZA'       : ('Mangalorean Catholic', 'Christian - OC'),
    'FERNANDIS'     : ('Mangalorean Catholic', 'Christian - OC'),
    'PHERNANDIS'    : ('Mangalorean Catholic', 'Christian - OC'),
    'PHERNANDISA'   : ('Mangalorean Catholic', 'Christian - OC'),
    'MONTHERO'      : ('Mangalorean Catholic', 'Christian - OC'),

    # ── CHRISTIAN  (Protestant / other) ─────────────────────────────
    'DISOJA'        : ('Christian', 'Christian - OC'),
    'MABINA'        : ('Christian', 'Christian - OC'),
    'MOBENA'        : ('Christian', 'Christian - OC'),
    'JATHANNA'      : ('Christian', 'Christian - OC'),
    'CARLO'         : ('Christian', 'Christian - OC'),
    'SONS'          : ('Christian', 'Christian - OC'),
    'SONSA'         : ('Christian', 'Christian - OC'),
    'RENJAL'        : ('Christian', 'Christian - OC'),
    'VARTHIK'       : ('Christian', 'Christian - OC'),
    'LEWIS'         : ('Christian', 'Christian - OC'),
    'PATRAO'        : ('Christian', 'Christian - OC'),
    'SALINS'        : ('Christian', 'Christian - OC'),
    'AMANNA'        : ('Christian', 'Christian - OC'),

    # ── MUSLIM family surnames ────────────────────────────────────────
    'BYARI'         : ('Muslim', 'Muslim'),
    'BYARY'         : ('Muslim', 'Muslim'),
    'BAVA'          : ('Muslim', 'Muslim'),
    'THANGAL'       : ('Muslim', 'Muslim'),
    'KALANDAR'      : ('Muslim', 'Muslim'),
    'MOIDEEN'       : ('Muslim', 'Muslim'),
    'MOIDIN'        : ('Muslim', 'Muslim'),
    'ROWTHER'       : ('Muslim', 'Muslim'),
    'HAJI'          : ('Muslim', 'Muslim'),
    'LEBBE'         : ('Muslim', 'Muslim'),
    'MARIKAR'       : ('Muslim', 'Muslim'),
    'SAHEB'         : ('Muslim', 'Muslim'),
}

# Add your own entries here
EXTRA_RULES = {
    # 'SURNAME': ('Community', 'Category'),
}
SURNAME_DICT.update(EXTRA_RULES)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 2 — RELIGION DICTIONARIES
#  Global first-name and surname token sets for religion detection.
#  Used when Community is Unclassified.
# ═══════════════════════════════════════════════════════════════════════

# ── Community → Religion mapping ─────────────────────────────────────
COMMUNITY_RELIGION = {
    # Hindu communities
    'Bunt'                           : 'Hindu',
    'GSB (Goud Saraswat Brahmin)'    : 'Hindu',
    'Shivalli Brahmin'               : 'Hindu',
    'Brahmin/GSB'                    : 'Hindu',
    'Brahmin (Coastal)'              : 'Hindu',
    'Billava'                        : 'Hindu',
    'Billava/Mogaveera'              : 'Hindu',
    'Ganiga'                         : 'Hindu',
    'Mogaveera'                      : 'Hindu',
    'Devadiga'                       : 'Hindu',
    'Vishwakarma'                    : 'Hindu',
    'Vokkaliga'                      : 'Hindu',
    'Nair (Kerala GC)'               : 'Hindu',
    'Nambiar (Kerala GC)'            : 'Hindu',
    'Vaidya (Kerala OBC)'            : 'Hindu',
    'OBC (Shet)'                     : 'Hindu',
    'OBC (Shettigar)'                : 'Hindu',
    'Multiple communities (Naik)'    : 'Hindu',
    'Ganiga (OBC)'                   : 'Hindu',
    'Kottari (OBC)'                  : 'Hindu',
    'Moolya (OBC)'                   : 'Hindu',
    'Trading/agricultural groups'    : 'Hindu',
    # Christian communities
    'Mangalorean Catholic'           : 'Christian',
    'Christian'                      : 'Christian',
    'Possibly Christian'             : 'Christian',
    # Muslim
    'Muslim'                         : 'Muslim',
    'Muslim (Unverified)'            : 'Muslim',
    # Jain (if added)
    'Jain'                           : 'Jain',
}

# ── Muslim name tokens ────────────────────────────────────────────────
MUSLIM_NAMES = {
    # Classical Islamic names
    'ABDUL HAKEEM','HAMMABBA',
    'ABDUL',
    'MOHAMMED','MOHAMMAD','MUHAMMED','MUHAMMAD','MOHD','MOHMMED','MOHAMAD',
    'ABUBAKAR','ABUBAKKAR','ABOOBAKKAR','ABOOBAKKER','ABOOBAKAR',
    'ABOOBACKER','ABUBABAKAR',
    'IBRAHIM','IBRAHEEM','ISMAIL','ISMAYIL','ISMAYEEL',
    'HAMEED','HAMID','HAMZA','HANZA','HAMSA',
    'HUSSAIN','HYDER','HUSEN','HUSAIN',
    'RASHEED','RASHID','FAROOK','FAROOQ','FARUQ',
    'THASLEEM','TASLEEM','ALTHAF','HASAINAR',
    'MUNEER','MUNIRA','KALANDAR','KULSU','AADAM','ADAM',
    'SAMRAN','SHAFHI','SHAPHI','SHAFI',
    'MAMTHAJ','MAMTHAZ','MAMTAZ','RIJWANA','JOHRA','RAJAK',
    'NAPHISA','NAFISA','IRPHANA','IRFANA',
    'MUSA','MOOSA','JUBEDA','ZUBEIDA','MUZAIDA','MUZAIFA',
    'SHAHUL','NOUSEENA','NOWSHEENA','SHAMEENA','SHAMEEMA',
    'NAUSHAD','NOUSHAD','JAFFER','ASHFAK','ASIF','IQBAL','IMRAN',
    'MUSTHAPHA','MUSTHAF','MUKTHAR','MAMMIKUNHI','MAHMOOD','MAHMUD',
    'NAWAZ','HANIF','WAHAB','WAHIDA','AZWEENA','ABIDA',
    'MUSRATH','RAZAQ','RAZAK','BASIR','BASHEER','BASHIR',
    'RAFIK','RAPHEEQ','REHAMAN','REHMAN','RAHMAN',
    'BYARI','BYARY','YAKUB','YAQUB','YUSUF','YUSUPH','YUSOOF',
    'THANGAL','THAHIRA','THAYIRA','ZAREENA','ZAHIDA',
    'NEBISA','NABEESA','NAZIYA','NAZEEMA','NAZREENA',
    'NASREENA','NASIMA','KHALID','KABEER','UMMER','UMAR',
    'MOIDIN','MOHIDEEN','SUHAIL','SUHANA','ANWAR',
    'SADIK','SADHIK','SIDDIQ','NIZAR','NISAR','NAVAZ',
    'FAIZAL','FAYAZ','SHABEER','THOUSEEF','MAINAZ','MEHARAZ',
    'SALIM','SALEEM','NASIR','SIRAJ','JABBAR','ASLAM',
    'SHAHEENA','SHAHEEN','PARVEEN','PARVEENA',
    'SABEENA','SABEERA','HALEEMA','HAFSA','NOORJAHAN',
    'BEGUM','SULTANA','KHAN','KASIM','GHOUSE','KADRI',
    'SHEIKH','SHAIKH','SHEIK','SHAIK','SYED','SAYYED','PASHA','BANU',
    'KHATHIJA','KHATIJA','KATHIJA','KATHEEJA','KATEEJA',
    'FATIMA','PHATHIMA','FATHIMA','FHATHIMATH',
    'AYISHA','AYSHA','AYESHA','ZAINABA','JAINABA','JAINABU',
    'JAMEELA','JAMILA','SUMAYYA','NASEEMA','NAFEESA','NAFISA',
    'MUMTAZ','MUMTHAZ','YASMEEN','YASMIN','REHANA','RAMEEZA','RAZIYA',
    'FARZANA','BUSHRA','KAUSAR','SAFIYA','SAJIDA',
    'SHAHINA','SHAHIDA','SHAHANA','RUKHIYA','RUKIYA','RUKSANA',
    'BIPATHUMA','BIPATHUMMA','MISRIYA','MISHRIYA','RAMLATH','RAMIATH',
    'LATHEEF','LATHIF','HAFEEZ','SHAMEER','SINAN',
    'IRSHAD','AZEEZ','AZIZ','RIYAZ','RIZWAN','SAMEER',
    'USMAN','SULAIMAN','MANSOOR','KHADER','KHADAR','MEHASINA','ASHRAF',
    # Urdu / North Indian Muslim names
    'FAIZ','ATHAR','ZAFAR','TARIQ','ARIF','AAMIR','AMIR','ADIL',
    'AKBAR','AKRAM','ANSAR','AQIL','ARSHAD','ARFAN',
    'DILSHAD','FARHAN','FIRDAUS','FURQAN','GHULAM','HASNAIN',
    'HILAL','HUZAIF','IFTIKHAR','INAYAT','INTESAR',
    'JAMEEL','JAVED','JUNAID','KAMAL','KAMRAN','KASHIF',
    'KHALEEL','KHURSHID','LUQMAN','MAJID','MANZOOR','MASOOD',
    'MEHBOOB','MOHSIN','MUNAWAR','MUSHTAQ','MUZAMMIL',
    'NADEEM','NAEEM','NASEEM','NAZIM','NOMAN','OBAID',
    'PERVAIZ','RAEES','RAIHAN','RAUF','RIZWAN','ROSHAN',
    'SAEED','SAJID','SALMAN','SHAHZAD','SHAKEEL','SHAUKAT',
    'SHOAIB','SIKANDER','SUBHAN','TAHA','TANVEER','TARIQ',
    'WASEEM','WAQAR','YASIR','YOUNUS','YOUSUF','ZAID','ZEESHAN',
    'ZUBAIR','ZUBER','ZUBEIR',
    'AISHA','ALIYA','AMARA','AMNA','ARFA','ASMA','AZRA',
    'FARIDA','FOZIA','HINA','HUMA','IQRA','LUBNA','MADIHA',
    'MARYAM','MEHWISH','NAILA','NAZIA','NEHA','NIDA','NOOR',
    'RUBINA','SAIMA','SANA','SHAZIA','SIDRA','TABASSUM','UZMA',
    'ZARA','ZOBIA','ZUNAIRA',
    # Byari / Coastal Karnataka Muslim names
    'BYARI','BAVA','MOIDEEN','ROWTHER','LEBBE','MARIKAR',
    'HAJI','SAHEB','KUNHI','MARAKKAR','KUTTY',
}

# ── Christian name tokens (first names + surnames) ────────────────────
CHRISTIAN_NAMES = {
    # Mangalorean Catholic surnames
    'DSOUZA','SOUZA','DSILVA','SILVA','DCOSTA','COSTA','DCUNHA','CUNHA',
    'DMELLO','MELLO','DCRUZ','CRUZ','DALMEIDA','PINTO','LOBO','FERNANDES',
    'RODRIGUES','RODRIQUES','SEQUEIRA','PEREIRA','SALDANHA','NORONHA',
    'MONTEIRO','MENEZES','MIRANDA','FURTADO','CRASTA','MASCARENHAS',
    'CASTELINO','GONSALVES','ALMEIDA','COLACO','COELHO','DIAS','REGO',
    'QUADROS','NAZARETH','REBELLO','CORREA','MONIS','ROSARIO','ALVARES',
    'FERRAO','GOVEAS','LASRADO','TAURO','CUTINHA','BARBOZA','BARRETTO',
    'ANDRADE','GOMES','SERRAO','MARTIS','KARKADA','MENDONCA','VAZ','VAS',
    'MABEN','MACHADO','ARANHA','ALBUQUERQUE','VIEGAS','MISQUITH',
    'RASQUINHA','RASQUINA','MORAS','MORASA','MENDONSA','MENDONZA',
    'MEDONZA','FERNANDIS','PHERNANDIS','PHERNANDISA','MONTHERO',
    'DISOJA','MABINA','MOBENA','JATHANNA','CARLO','SONS','SONSA',
    'RENJAL','VARTHIK','LEWIS','PATRAO','SALINS','AMANNA',
    # Syrian Malabar / Kerala Christian surnames
    'VARGHESE','MATHEW','KURIAN','CHACKO','THAMPI','GEORGE','THOMAS',
    'PHILIP','POTHEN','OOMMEN','LUKOSE','KOSHY','KOSHI','KURUVILA',
    'IYPE','ITTY','ITTOOP','LONAPPAN','MAMMEN','MATHAI','NETTIKKADAN',
    'PAILY','PAULOSE','PUNNOOSE','RAJAN','THARAKAN','THARIAN',
    'VERGHESE','ZACHARIAH','ZACHARIAS','ABRAHAM','ACHAN','ANNAMMA',
    'AYYAPPAN','CHERIAN','DEVASIA','EAPEN','ELIAS','ELAMMA',
    # Protestant / general Christian first names
    'JOHN','JAMES','PETER','PAUL','JOSEPH','GEORGE','THOMAS','STEPHEN',
    'MICHAEL','MICHAEL','DAVID','DANIEL','SAMUEL','ANDREW','MARK',
    'LUKE','MATTHEW','BENJAMIN','JOSHUA','ELIJAH','AARON','ADAM',
    'NATHAN','SIMON','TIMOTHY','JACOB','LEVI','PHILIP','BARNABAS',
    # Female Christian names
    'MARY','MARIA','ANNA','ANNE','SARAH','ELIZABETH','CATHERINE',
    'THERESA','TERESA','CLARA','ROSA','GRACE','ANGELA','AGNES',
    'CLAIRE','ESTHER','EVE','HELEN','JESSICA','JOANNA','JUDITH',
    'LEAH','LYDIA','MARTHA','MIRIAM','NAOMI','PRISCILLA','RACHEL',
    'REBECCA','RUTH','VERONICA','VIRGINIA',
    # Popular Christian first names (South Indian)
    'SUNNY','SHINY','BINU','BIJI','BOBY','BOBBY','LIJO','LINO',
    'LENNY','LENI','GLEN','GLENSON','GLENDON','MELVIN','ALVIN',
    'CALVIN','KEVIN','BRIAN','RYAN','IVAN','IRENE','JOYAL','JESSY',
    'JUSTIN','ANTONY','VINCENT','LAWRENCE','BERNAD','OLIVER','MAXIM',
    'JACINTHA','AIDA','ASUNTHA','JEBIN','JINS','DINTO','DIBIN',
    'DINU','DINO','ELDO','ELVIN','ERVIN','FELIX','FREDY','GEORGY',
    'GINSON','GINCY','GIGY','GILSON','GIBIN','GIGO','GILBY',
    'JAINCY','JAIMY','JAISE','JAISON','JIJO','JIJESH','JIKKU',
    'JIBIN','JIJOY','JILL','JIMIN','JIMCY','JIMMY','JINCE','JINS',
    'JINCY','JINO','JINSON','JISMY','JISNA','JITHIN','JITHU',
    'JIYA','JIYASH','JOBBY','JOBY','JOEMON','JOFIN','JOGI',
    'JOHNY','JOJI','JOJO','JOMON','JOMY','JONEY','JONCY',
    'JOSHY','JOSNA','JOSY','JOTHISH','JOYSON','JUDITH','JULFY',
    'JULIAN','JULIUS','JUMIN','JUMY','JUNAS','JUNE','JUNESH',
    'JUSTIN','DENNIS','DENIS','GLADS','GLADSON','GIBSON',
    'SEBASTIAN','SEBASTIN','CHRISTOPHER','ALEXANDER', 'Igneshiyas Sikwera', 'Sikwera'
}

# ── Hindu name tokens ─────────────────────────────────────────────────
HINDU_NAMES = {
    # Male first names
    'Vijayan',
    'RAVI','SURESH','RAMESH','MAHESH','DINESH','GANESH','GIRISH','HARISH',
    'NARESH','YOGESH','RAJESH','LOKESH','RAKESH','UMESH','MUKESH','NITESH',
    'RITESH','SATISH','MANISH','VIRESH','MOHAN','VIJAY','AJAY','SANJAY',
    'MANOJ','PRAMOD','PRAVEEN','NAVEEN','DEEPAK','SHEKHAR','ABHISHEK',
    'ANAND','ANIL','ASHOK','SUNIL','SUHAS','RAHUL','PRADEEP','SANDEEP',
    'SUNDEEP','SANDEEP','ARUN','TARUN','VARUN','KARUN',
    'SHANKAR','SHANKARA','SHIVAPRASAD','SHIVARAMAIAH','SHIVARAMU',
    'SHIVAKUMAR','SHIVAPPA','SHIVAJI','SHIVARAM','SHIVANAND',
    'KRISHNA','KRISHNAPPA','KRISHNAMURTHY','KRISHNADAS','KRISHNASWAMY',
    'VENKATESH','VENKATESHA','VENKATARAMANA','VENKATASUBRAMANIAN',
    'NARAYANA','NARAYANASWAMY','NARAYANDAS',
    'BALAKRISHNA','BALKRISHNA','BALKISHAN',
    'RAMAKRISHNA','RAMASWAMY','RAMADAS','RAMANATHAN','RAMANUJAM',
    'GOPALAKRISHNA','GOPALKRISHNA','GOPAL','GOPALA',
    'LAKSHMIKANTHA','LAKSHMINARAYAN','LAKSHMINARAYANA',
    'SUBRAMANYA','SUBRAMANIAN','SUBRAMANIAM','SUBRAMANYAM',
    'SIDDHARTH','SIDDHARTHA','SIDDESH',
    'PARAMESWARA','PARAMESHWARA','PARASHURAM','PARASHURAMA',
    'DAYANANDA','DAYANAND','SADANANDA','SADASHIVA','SADASHIV',
    'JAGANNATH','JAGADISH','JAGADEESHA','JAGADEESH',
    'MANJUNATH','MANJUNATHA','MANJESH','MANJU',
    'BASAVRAJ','BASAVANNA','BASAVARAJ','BASAVA',
    'CHANDRASHEKHAR','CHANDRASHEKAR','CHANDRA','CHANDRAKANT',
    'HEMANT','HEMANTHA','HEMANTH','HEMANTHU',
    'PRASAD','PRASHANTH','PRASHANT','PRASHAN',
    'VIVEK','VIVEKA','VIVEKANANDA','VIVEKANAND',
    'KAPIL','KARTHIK','KARTHIKEYA','KARTIK',
    'ARJUN','ARJUNA','ARJUNAPPA',
    'BHARAT','BHARATH','BHARATESH',
    'DILEEP','DILIP','DILEEPKUMAR',
    'PRAKASH','PRAKASHA','PAVANKUMAR','PAVAN','MAHENDRA','MAHENDRAN',
    'AJITH','AJITHKUMAR','AJITKUMAR',
    'AKSHAY','AKSHAYA','AKSHAI',
    'AMITH','AMITH','AMIT',
    'DHANANJAY','DHANARAJ','DHANRAJ','DHANESH',
    'GOKUL','GOKULKRISHNA','GOKULANAND',
    'GOVIND','GOVINDA','GOVINDAPPA','GOVINDARAJU','GOVINDRAJ',
    'HARI','HARIHAR','HARIKRISHNA','HARINATH','HARIPRASAD',
    'INDRA','INDRESH','INDRANEEL',
    'JAGAN','JAGANNATH','JAGANNADH',
    'KIRAN','KIRANRAJ','KIRANAPPA',
    'MADHAV','MADHAVA','MADHUSUDAN',
    'NAGESH','NAGENDRA','NAGRAJ',
    'NARENDRA','NARENDAR','NARENDRANATH',
    'NIKHIL','NIKHILESH',
    'OMKAR','OMKARESHWARA',
    'PRABHAKARA','PRABHAKAR','PRABHAKARAN',
    'RAGHAVENDRA','RAGHUNATH','RAGHUNANDAN','RAGHU',
    'RAJENDRA','RAJENDAR','RAJENDRAN',
    'RAJIV','RAJEEV','RAJIVKUMAR',
    'RAMANAND','RAMAMURTHY','RAMARAO',
    'RATAN','RATNA','RATHNA',
    'SACHIDANANDA','SACHINDRA',
    'SHREEDHARA','SHREEDHAR','SREEDHARA','SRIDHAR','SRIDHARA',
    'SREEKUMAR','SRIKUMAR','SRIKANT','SRIKANTH',
    'SRIRAM','SREERAMA','SRIRAMAIAH',
    'SUDHARSHAN','SUDARSHAN','SUDARSHANAIAH',
    'SURYANARAYANA','SURYANARAYAN','SURYANAND',
    'THIRUMALA','TIRUMALAIAH','TIRUMALA',
    'UMAPATI','UMAPATHY',
    'VENKATARAMAN','VENKATRAMAIAH','VENKATESH',
    'VIJAYKUMAR','VIJAYAKUMAR','VIJAYENDRA',
    'VISWANATH','VISHWANATH','VISHWANATHA',
    'YALLAPPA','YELLAPPA','YAMANAPPA',
    # Female first names
    'PRIYA','DEEPA','REKHA','KAVITA','SUNITA','ANITA','LALITA','SAVITA',
    'ASHA','USHA','MEENA','SEEMA','GEETA','SITA','RADHA','PARVATI',
    'LAKSHMI','SARASWATI','DURGA','DEVI','KAMALA','KUMARI','PUSHPA',
    'PADMA','SUDHA','SHOBHA','SAVITHA','KAVITHA','BHAVITHA','BHAVNA',
    'BHAVANA','GEETHA','SHANTHA','SHANTA','SHEELA','SHEILA','SHILPA',
    'SHRUTI','SHRUTHI','SHWETA','SHWETHA','SOWMYA','SOWNDARYA',
    'SOUNDARYA','SOWJANYA','SOUMYA','SNEHA','SINDHU','SINDHU',
    'SANGEETHA','SANGEETA','SANDHYA','SARITHA','SAROJA','SARALA',
    'SAROJINI','SRIDEVI','SUSHMA','SUSHILA','SUSMITHA','SUPRIYA',
    'SUPRABHA','SUMA','SUMITHRA','SUMITRA','SUKANYA','SUHASINI',
    'SUJATHA','SUJATA','SUJAYA','SUKANYA','RADHIKA','RAMYA',
    'RASHMI','RATNA','RATHNA','ROJA','ROOPA','ROSHAN','ROHINI',
    'PAVITHRA','PAVITRA','POOJA','PUJA','POORNIMA','POURNAMI',
    'PREETI','PREETHI','PREETHMA','PREMA','PREMILA','PRIYADARSHINI',
    'NIRUPAMA','NIRMALA','NIRANJANA','NIRANJANI','NISHA','NISHITHA',
    'NAGALAKSHMI','NAGAMMA','NALINI','NAMITHA','NAMITA','NAMRATHA',
    'MYTHILI','MYTHRI','MYTHREYI','MRINALINI','MRIDULA',
    'MEENAKSHI','MEERA','MADHURI','MADHURA','MADHUVANTHI',
    'LATHA','LATA','LATHADEVI','LAVANYA','LEELAVATHI','LEELA',
    'KALPANA','KAMAKSHI','KAMALAKSHI','KANCHANA','KARTHYAYINI',
    'JAYALAKSHMI','JAYASHREE','JAYANTHI','JAYA','JANAKI','JALAJA',
    'INDIRA','INDUMATHI','INDU','INDUMATI',
    'HEMA','HEMAVATHI','HEMAVATI','HEMASHREE',
    'GOURI','GAURI','GIRIJA','GEETHA',
    'DIVYA','DEVYANI','DEVAKI','DEVATHA',
    'CHAMPAVATHI','CHAITRA','CHANDRA','CHANDRIKA','CHANDRAMATHI',
    'BHUVANA','BHAGYALAKSHMI','BHAGYA','BHARATHI','BHARATI',
    'ARCHANA','ARUNA','ARUNDHATHI','ARUNDHATI',
    'ANURADHA','ANUPAMA','ANUJA','ANUSHREE','ANUSHKA','ANUSHA',
    'AMBIKA','AMRUTHA','AMRITA','AMALA','AMALA',
    # South Indian Telugu/Kannada specific names
    'NAGESWARA','NAGABHUSHANA','NAGABHUSHANAIAH',
    'PUTTASWAMY','PUTTARAJU','PUTTARAJA',
    'SHIVEGOWDA','SHIVAMURTHY','SHIVAMURTHI',
    'THIMMAIAH','THIMMARAYAPPA','THIMMARAJU',
    'LAKSHMAIAH','LAKSHMANA','LAKSHMINARAYANA',
    # Surnames (Hindu-specific, not community-specific)
    'SHARMA','VERMA','MISHRA','TRIVEDI','PANDEY','TIWARI','DUBEY',
    'SINGH','KUMAR','YADAV','GUPTA','AGARWAL','MEHTA','PATEL',
    'REDDY','PILLAI','MENON','NAMBOODIRI','NAMBOOTHIRI','IYER',
    'IYENGAR','AIYER','MUDALIAR','CHETTIAR','NAIDU','GOUNDAR',
    'DEVAR','VELLALAR','MUDALI','PILLAI','PANIKKAR',
    # Hindu naming suffixes (South Indian)
    'SWAMY','SWAMI','APPA','AIAH','ANNA','AMMA',

     "Aadhavan", "Aadhithyan", "Aakash", "Abhijith", "Abhinav", "Adarsh",
    "Adhavan", "Adinarayanan", "Adithya", "Aditya", "Ajay", "Ajith",
    "Ajithan", "Akash", "Akhil", "Akhilan", "Akhilesh", "Amarnath",
    "Ambujan", "Anand", "Anandan", "Ananth", "Ananthan", "Anbarasan",
    "Anbu", "Anil", "Anirudh", "Anish", "Anoop", "Aravind",
    "Aravindan", "Arjun", "Arjunan", "Arul", "Arulanantham", "Arulanandan",
    "Arun", "Arunan", "Arunkumar", "Ashok", "Ashokan", "Ashwin",
    "Balachandran", "Balaji", "Balakrishnan", "Balan", "Balaraman",
    "Baskaran", "Bhanuprasad", "Bharath", "Bharathan", "Bhaskar",
    "Bhaskaran", "Bhuvan", "Chaitanya", "Chandran", "Chandrasekhar",
    "Chandrasekaran", "Charan", "Chethan", "Damodaran", "Darshan",
    "Dayanand", "Deepak", "Dhananjayan", "Dhanush", "Dharan",
    "Dilip", "Dinesh", "Divakaran", "Easwaran", "Ganapathy",
    "Ganesh", "Ganeshan", "Gireesh", "Girish", "Gokul",
    "Gopal", "Gopalan", "Gopinath", "Goutham", "Govindan",
    "Hari", "Haridas", "Harikrishnan", "Harish", "Harshan",
    "Harshavardhan", "Hemanth", "Hemachandran", "Jagadeesh",
    "Jagadish", "Jagan", "Jaganathan", "Janardhan", "Jayakumar",
    "Jayan", "Jayaraj", "Jayaram", "Jayaraman", "Jeevan",
    "Kailasan", "Kannan", "Karthick", "Karthikeyan", "Karthik",
    "Kasi", "Kasinathan", "Kesavan", "Keshavan", "Kiran",
    "Krishna", "Krishnan", "Krishnakumar", "Krishnamoorthy",
    "Kumaran", "Kumar", "Kumaran", "Kumaravel", "Lokesh",
    "Madhavan", "Mahadevan", "Mahesh", "Manikandan", "Mani",
    "Manivannan", "Manjunath", "Mohan", "Mohanan", "Murali",
    "Muralidharan", "Murugan", "Murugesan", "Muthu", "Muthuraman",
    "Nagarajan", "Nandakumar", "Nandan", "Narayanan", "Narendran",
    "Natarajan", "Navaneethan", "Naveen", "Nikhil", "Niranjan",
    "Parameswaran", "Parthiban", "Pavithran", "Perumal", "Prabhakaran",
    "Pradeep", "Prakash", "Pranav", "Prasanth", "Prashanth",
    "Prem", "Purushothaman", "Radhakrishnan", "Raghavan",
    "Raghunandan", "Rajan", "Rajanikanth", "Rajasekar",
    "Rajasekaran", "Rajendran", "Rajesh", "Rajiv", "Rajkumar",
    "Ramakrishnan", "Raman", "Ramachandran", "Ramesh",
    "Ranjith", "Ravichandran", "Ravi", "Ravikumar", "Riyas",
    "Sabarinathan", "Sachin", "Sadanandan", "Sakthivel",
    "Sampath", "Sanjay", "Santhosh", "Saravanan", "Sasidharan",
    "Sathish", "Satyan", "Selvakumar", "Senthil", "Senthilkumar",
    "Shankar", "Shankaran", "Shanmugan", "Shivakumar",
    "Shivananthan", "Sivakumar", "Sivaraman", "Sivaranjan",
    "Soman", "Sreenath", "Sreekanth", "Sreekumar", "Sreenivasan",
    "Sridhar", "Srinath", "Srinivasan", "Subash", "Subramanian",
    "Sudhakaran", "Sukumaran", "Sundar", "Sundaran", "Suresh",
    "Sureshkumar", "Thangavel", "Udayakumar", "Unnikrishnan",
    "Vaidyanathan", "Vaishakan", "Varadarajan", "Varun",
    "Vasanthan", "Vasudevan", "Velan", "Velmurugan",
    "Venkatesan", "Venkatesh", "Vetrivel", "Vignesh",
    "Vijay", "Vijayan", "Vijayakumar", "Vikram", "Vinay",
    "Vinayan", "Vinod", "Vinodan", "Vishakan", "Vishnu",
    "Vivek", "Yogesh",

    # Female
    "Aarthi", "Abhirami", "Adhira", "Akhila", "Akshaya",
    "Amritha", "Anagha", "Anitha", "Anjana", "Anjali",
    "Anupama", "Anusha", "Archana", "Arundhathi", "Asha",
    "Ashwini", "Bhagyalakshmi", "Bhanu", "Bhargavi",
    "Bhuvana", "Chaitra", "Chandrika", "Deepa", "Deepika",
    "Devika", "Dhanya", "Divya", "Gayathri", "Geetha",
    "Gomathi", "Harini", "Hemalatha", "Indu", "Janaki",
    "Jayanthi", "Jyothi", "Kala", "Kalpana", "Kalyani",
    "Kanchana", "Karthika", "Kavitha", "Kavya", "Keerthana",
    "Krithika", "Lakshmi", "Lalitha", "Lekha", "Madhavi",
    "Malathi", "Manasa", "Meena", "Meenakshi", "Meera",
    "Meghana", "Nandhini", "Nandita", "Nayana", "Neethu",
    "Nirmala", "Padma", "Padmini", "Parvathi", "Pavithra",
    "Poornima", "Prabha", "Preetha", "Priya", "Radha",
    "Rajalakshmi", "Rajeshwari", "Rakshitha", "Ramya",
    "Ranjani", "Revathi", "Rohini", "Sandhya", "Saranya",
    "Savitha", "Shanthi", "Sharada", "Shilpa", "Shobana",
    "Shravya", "Shruthi", "Sindhu", "Sowmya", "Sreeja",
    "Sreelakshmi", "Subhashini", "Sudha", "Sujatha",
    "Sunitha", "Supriya", "Sushmitha", "Swathi", "Uma",
    "Usha", "Vaidehi", "Vaishnavi", "Vani", "Varalakshmi",
    "Varsha", "Vasundhara", "Veena", "Vidhya", "Vidya",
    "Vijayalakshmi", "Vimala", "Vinaya", "Vinodha",
    "Vishalakshi"

}

# ── Jain name tokens ──────────────────────────────────────────────────
JAIN_NAMES = {
    # Jain surnames
    'JAIN','SHAH','MEHTA','OSWAL','PORWAL','KOTHARI','DOSHI','SANGHVI',
    'LODHA','LOHIA','SINGHAL','GARG','SANCHETI','SURANA','PATNI',
    'SARAOGI','RANKA','BAID','DUGAR','KHATOR','LAKHOTIA','MURARKA',
    'TOSHNIWAL','KANKARIA','CHHAJER','BOTHRA','BENGANI',
    'CHOPRA','NAHATA','MANDHANA','JHUNJHUNWALA','HIRAWAT',
    'GOLCHA','SONI','NAGORI','AGARWAL',
    # Jain first names (male)
    'MAHAVIR','MAHAVIRA','VARDHAMANA','PARSHWA','PARSHVANATH',
    'RISHABH','RISHABHDEV','ADINATH','AJITNATH','NEMINATH',
    'JINENDRA','JINESH','JIGAR','VIRAL','DHAVAL','DHRUVAL',
    'CHIRAG','CHINTAN','DARSHAN','DEVANG','DHARMESH','DILIP',
    'HARDIK','HARSH','HIREN','JATIN','KETAN','KEYUR',
    'MEHUL','MITEN','MITESH','NIMISH','NIRAV','NISHANT',
    'PARTH','PARTHIV','PRATIK','PREET','PRIYEN','PUJAN',
    'RAJAN','RAJESH','RITESH','RONAK','RUCHIT','RUPESH',
    'SAGAR','SANJAY','SHAILESH','SHALIN','SHYAM','SUNIL',
    'TEJAS','TIRTH','TUSHAR','UDAY','UMANG','UTKARSH',
    'VIMAL','VIRAM','VISHAL','VIVEK','YASH',
    # Jain first names (female)
    'JINAL','DRASHTI','KHUSHBU','KHUSHBOO','DHARTI','DHARA',
    'DIVYA','HETAL','HINA','HIRAL','JALPA','JALSA',
    'KAVYA','KINJAL','MANSI','MEERA','MISHA','MONA',
    'NIDHI','NIKITA','NISHA','PAYAL','POOJA','PRACHI',
    'RADHA','RIDDHI','RIYA','SEJAL','SHEFALI','SHREYA',
    'SONAL','SWATI','TEJAL','URVASHI','VARSHA','VIDHI',
}

# ── Hindu name-pattern suffixes (end of token) ────────────────────────
HINDU_SUFFIXES = (
    'APPA', 'AIAH', 'ANNA', 'AMMA', 'AKKA', 'AYYA',
    'SWAMY', 'SWAMI', 'MURTHY', 'MURTHI', 'KRISHNA',
    'RAJU', 'RAJAN', 'BABU', 'REDDY', 'GOWDA',
    'DEVI', 'BHAI', 'RAO',
)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 3 — COMMUNITY CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════

CONCAT_SUFFIXES = sorted([
    'POOJARY','POOJARI','KARKERA','AACHARYA','ACHARYA','SUVARNA','SUVARN',
    'SHETTY','SHETTI','HEGDE','GOWDA','KAMATH','PRABHU','NAIK','NAYAK',
    'BANGERA','SALIAN','KOTIAN','DEVADIGA','BHANDARY','ACHAR','SHENOY',
    'BHATT','BHAT',
], key=len, reverse=True)

MUSLIM_TOKENS  = MUSLIM_NAMES          # reuse the same set
CHRISTIAN_TOKENS = CHRISTIAN_NAMES     # reuse


def tokenize(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    cleaned = str(text).upper().replace("'", "").replace("`", "")
    return [t for t in re.split(r'[\s\.\,\-\/\(\)]+', cleaned) if len(t) >= 2]


def fused_suffix(token):
    for suf in CONCAT_SUFFIXES:
        if token.endswith(suf) and len(token) > len(suf) + 2:
            return suf
    return None


def classify_community(voter_name, relation_name=''):
    v = tokenize(voter_name)
    r = tokenize(relation_name)
    if not v:
        return None
    for tok in v:
        res = SURNAME_DICT.get(tok)
        if res:
            return res[0], res[1], 'HIGH', 'voter_name'
    for tok in v:
        suf = fused_suffix(tok)
        if suf:
            res = SURNAME_DICT.get(suf)
            if res:
                return res[0], res[1], 'HIGH', 'voter_fused'
    for tok in v:
        if tok in MUSLIM_TOKENS:
            return 'Muslim', 'Muslim', 'HIGH', 'voter_muslim'
    if r:
        res = SURNAME_DICT.get(r[-1])
        if res:
            return res[0], res[1], 'MEDIUM', 'relation_surname'
    for tok in r[:-1]:
        res = SURNAME_DICT.get(tok)
        if res:
            return res[0], res[1], 'MEDIUM', 'relation_other'
    for tok in r:
        suf = fused_suffix(tok)
        if suf:
            res = SURNAME_DICT.get(suf)
            if res:
                return res[0], res[1], 'MEDIUM', 'relation_fused'
    for tok in r:
        if tok in MUSLIM_TOKENS:
            return 'Muslim', 'Muslim', 'MEDIUM', 'relation_muslim'
    for tok in v:
        if tok in CHRISTIAN_TOKENS:
            return 'Christian', 'Christian - OC', 'LOW', 'voter_christian'
    return None


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 4 — RELIGION CLASSIFIER  (community rules + Jain dict + ML model)
#
#  Pass 1 — derive from Community label (highest precision; unchanged,
#           still rule-based since it comes from confirmed surname maps)
#  Pass 2 — Jain dictionary check (the ML model was trained on a 3-class
#           H/M/C problem and has no Jain label, so Jain still needs a
#           dictionary lookup regardless of whether the model is loaded)
#  Pass 3 — trained ML model predicts Hindu / Muslim / Christian for
#           everything community + Jain didn't resolve. This REPLACES
#           the old manual MUSLIM_NAMES / CHRISTIAN_NAMES / HINDU_NAMES
#           token matching, which now only runs as an automatic
#           fallback if the model file failed to load.
# ═══════════════════════════════════════════════════════════════════════

def classify_religion_dict_fallback(voter_name, relation_name=''):
    """
    Old manual token-dictionary classifier for Hindu / Muslim / Christian.
    Only used automatically if ML_MODEL failed to load — keeps a 1.7M-row
    production run from dying just because the model file is missing.
    """
    v_toks = tokenize(voter_name)
    r_toks = tokenize(relation_name or '')
    all_toks = v_toks + r_toks

    if not all_toks:
        return 'Unknown', 'unknown'

    for tok in all_toks:
        if tok in MUSLIM_NAMES:
            return 'Muslim', 'dict_fallback'
    for tok in all_toks:
        if tok in CHRISTIAN_NAMES:
            return 'Christian', 'dict_fallback'
    for tok in all_toks:
        if tok in HINDU_NAMES:
            return 'Hindu', 'dict_fallback'
    for tok in all_toks:
        for suf in HINDU_SUFFIXES:
            if tok.endswith(suf) and len(tok) > len(suf) + 1:
                return 'Hindu', 'dict_fallback_pattern'

    return 'Unknown', 'unknown'


def classify_religion_batch(records):
    """
    Classify Religion for a BATCH of documents at once. This is the core
    of the ML integration — instead of calling the model once per name
    (slow), every record that needs the model is collected first, then
    predicted in a single vectorised call. This is what makes the model
    practical at 1.7M rows: TF-IDF + LinearSVC on a batch of 2,000 names
    takes a fraction of a second; calling it 2,000 times separately does not.

    records : list of dicts, each with keys 'voter_name', 'relation_name',
              'community'  (community may be 'Unclassified')

    Returns : list of (Religion, Religion_Source, Confidence) tuples,
              same order and length as `records`. Confidence is None for
              non-ML-derived rows.
    """
    results = [None] * len(records)
    ml_indices = []
    ml_rows = []

    for i, rec in enumerate(records):
        v_name = rec['voter_name']
        r_name = rec['relation_name']
        community = rec['community']

        # ── Pass 1: community → religion (unchanged, rule-based) ────────
        if community and community != 'Unclassified':
            rel = COMMUNITY_RELIGION.get(community)
            if rel:
                results[i] = (rel, 'community_derived', None)
                continue

        v_toks = tokenize(v_name)
        r_toks = tokenize(r_name or '')
        all_toks = v_toks + r_toks

        if not all_toks:
            results[i] = ('Unknown', 'unknown', None)
            continue

        # ── Pass 2: Jain dictionary (model has no Jain class) ───────────
        jain_hit = False
        for tok in all_toks:
            if tok in JAIN_NAMES:
                results[i] = ('Jain', 'name_token_jain', None)
                jain_hit = True
                break
        if jain_hit:
            continue

        # ── Defer everything else to the ML model, batched ──────────────
        ml_indices.append(i)
        ml_rows.append({'Name': v_name or '', 'Relation_Name': r_name or ''})

    # ── Pass 3: one batched model call for every deferred record ────────
    if ml_indices:
        if ML_MODEL is not None:
            X = pd.DataFrame(ml_rows)
            preds  = ML_MODEL.predict(X)
            probas = ML_MODEL.predict_proba(X)
            confs  = probas.max(axis=1)
            for idx, pred, conf in zip(ml_indices, preds, confs):
                religion = ML_LABELS.get(pred, pred)
                results[idx] = (religion, 'ml_model', round(float(conf), 4))
        else:
            # Model not loaded — fall back to dictionary matching so the
            # run still completes instead of crashing.
            for idx in ml_indices:
                rec = records[idx]
                religion, src = classify_religion_dict_fallback(
                    rec['voter_name'], rec['relation_name'])
                results[idx] = (religion, src, None)

    return results


def classify_religion(voter_name, relation_name='', community='Unclassified'):
    """
    Single-record convenience wrapper around classify_religion_batch,
    for the dry-run preview and any one-off lookups. Internally this is
    just a batch of size 1 — full runs use classify_religion_batch()
    directly for speed.
    """
    rel, src, conf = classify_religion_batch(
        [{'voter_name': voter_name, 'relation_name': relation_name, 'community': community}]
    )[0]
    return rel, src, conf


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 5 — FIELD AUTO-DETECTION  (operates on a MongoDB document)
# ═══════════════════════════════════════════════════════════════════════

def detect_fields(sample_doc):
    """
    Auto-detect the voter name field and relation name field from a
    sample MongoDB document's keys. Same logic as the Excel column
    detector, adapted for dict keys instead of DataFrame columns.
    """
    keys = list(sample_doc.keys())
    non_voter = ['assembly', 'constituency', 'section', 'ward', 'polling',
                 'booth', 'part', 'address', 'station', 'relative', 'relation']

    name_field = None
    for pattern in ['voter name', 'voter full name', 'first name',
                    'full name', 'name of voter', 'applicant name']:
        for k in keys:
            if k.strip().lower() == pattern:
                name_field = k
                break
        if name_field:
            break
    if not name_field:
        for k in keys:
            if k.strip().lower() == 'name':
                name_field = k
                break
    if not name_field:
        for k in keys:
            kl = k.strip().lower()
            if kl.endswith('name') and not any(x in kl for x in non_voter):
                name_field = k
                break

    rel_field = None
    for pattern in ['relative name', 'relation name', 'father name',
                    'husband name', 'mother name', 'guardian name',
                    'relatives name', 'relations name']:
        for k in keys:
            if k.strip().lower() == pattern:
                rel_field = k
                break
        if rel_field:
            break
    if not rel_field:
        for k in keys:
            if 'relative' in k.strip().lower():
                rel_field = k
                break
    if not rel_field:
        for k in keys:
            kl = k.strip().lower()
            if 'relation' in kl and 'type' not in kl and 'no' not in kl:
                rel_field = k
                break
    if not rel_field:
        for k in keys:
            kl = k.strip().lower()
            if 'father' in kl or 'husband' in kl or 'guardian' in kl:
                rel_field = k
                break

    return name_field, rel_field


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 6 — DRY RUN  (no writes — verification only)
# ═══════════════════════════════════════════════════════════════════════

def dry_run(collection, name_field=None, relation_field=None, sample_size=25):
    """
    Pulls a random sample of documents via $sample, runs the classifier
    on each, and prints a before/after table. Makes NO changes to the
    database. Always run this before --full on a new collection.
    """
    print("=" * 78)
    print("  DRY RUN  —  no documents will be modified")
    print("=" * 78)

    total_docs = collection.estimated_document_count()
    print(f"  Collection            : {collection.full_name}")
    print(f"  Estimated total docs   : {total_docs:,}")

    sample = list(collection.aggregate([{"$sample": {"size": sample_size}}]))
    if not sample:
        print("  ERROR: collection appears to be empty. Nothing to sample.")
        return False

    # Auto-detect fields from the first sampled document if not provided
    if not name_field or not relation_field:
        auto_name, auto_rel = detect_fields(sample[0])
        name_field     = name_field or auto_name
        relation_field = relation_field or auto_rel

    print(f"  Sample size            : {len(sample)}")
    print(f"  Detected name field    : '{name_field}'")
    print(f"  Detected relation field: '{relation_field}'")
    print()

    if not name_field:
        print("  ERROR: could not detect a voter name field.")
        print(f"  Document keys found: {list(sample[0].keys())}")
        print("  Set NAME_FIELD explicitly at the top of this script and retry.")
        return False

    # ── Community classification: still per-row (fast regex/dict work) ──
    community_results = []
    for doc in sample:
        v_name = doc.get(name_field, '')
        r_name = doc.get(relation_field, '') if relation_field else ''
        c_res = classify_community(v_name, r_name)
        if c_res:
            comm, cat, conf, csrc = c_res
        else:
            comm, cat, conf, csrc = 'Unclassified', 'Unknown', '', 'unclassified'
        community_results.append((v_name, r_name, comm, cat, conf, csrc))

    # ── Religion classification: ONE batched call (community + Jain +
    #    a single vectorised ML prediction for everything left over) ───
    religion_records = [
        {'voter_name': v, 'relation_name': r, 'community': c}
        for (v, r, c, _, _, _) in community_results
    ]
    religion_results = classify_religion_batch(religion_records)

    rows = []
    for doc, (v_name, r_name, comm, cat, conf, csrc), (religion, rsrc, rconf) in zip(
            sample, community_results, religion_results):
        rows.append({
            '_id'              : str(doc.get('_id', '')),
            'Voter Name'       : v_name,
            'Relation Name'    : r_name,
            'Community'        : comm,
            'Category'         : cat,
            'Religion'         : religion,
            'Religion_Conf'    : rconf if rconf is not None else '',
            'Comm_Source'      : csrc,
            'Rel_Source'       : rsrc,
        })

    df_preview = pd.DataFrame(rows)
    with pd.option_context('display.max_colwidth', 26, 'display.width', 170):
        print(df_preview.to_string(index=False))

    print()
    print("-" * 78)
    print("  Dry run summary (sample only):")
    print(f"    Community classified : {(df_preview['Community'] != 'Unclassified').sum()} / {len(df_preview)}")
    print(f"    Religion identified  : {(df_preview['Religion']  != 'Unknown').sum()} / {len(df_preview)}")
    ml_used = (df_preview['Rel_Source'] == 'ml_model').sum()
    print(f"    Religion via ML model: {ml_used} / {len(df_preview)}")
    print("-" * 78)
    print()
    print("  No documents were modified. Review the table above.")
    print("  If this looks correct, run with --full to classify all documents.")
    print("=" * 78)
    return True


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 7 — FULL CLASSIFICATION  (batched bulk writes)
# ═══════════════════════════════════════════════════════════════════════

def classify_all(collection, name_field=None, relation_field=None,
                  batch_size=2000, resume=True):
    """
    Classifies every document in the collection and writes Community,
    Category, Religion, Confidence, Classification_Source, and
    Religion_Source back to each document via batched bulk_write.

    resume=True  -> only processes documents where 'Community' field
                    does not yet exist (safe to re-run after interruption)
    resume=False -> re-classifies every document, overwriting existing
                    Community/Religion fields
    """
    # Detect fields from one sample doc if not provided
    if not name_field or not relation_field:
        probe = collection.find_one({})
        if not probe:
            print("ERROR: collection is empty.")
            return
        auto_name, auto_rel = detect_fields(probe)
        name_field     = name_field or auto_name
        relation_field = relation_field or auto_rel

    if not name_field:
        print("ERROR: could not detect voter name field. Set NAME_FIELD explicitly.")
        return

    query_filter = {"Community": {"$exists": False}} if resume else {}

    total = collection.count_documents(query_filter)
    if total == 0:
        print("Nothing to classify — all documents already have a Community field.")
        print("Use --no-resume to force re-classification of all documents.")
        return

    print("=" * 78)
    print("  FULL CLASSIFICATION RUN")
    print("=" * 78)
    print(f"  Collection      : {collection.full_name}")
    print(f"  Name field      : '{name_field}'")
    print(f"  Relation field  : '{relation_field}'")
    print(f"  Resume mode     : {resume}  (skip already-classified: {resume})")
    print(f"  Documents to do : {total:,}")
    print(f"  Batch size      : {batch_size:,}")
    print("=" * 78)

    projection = {name_field: 1}
    if relation_field:
        projection[relation_field] = 1

    cursor = collection.find(query_filter, projection).batch_size(batch_size)

    processed   = 0
    by_comm_src = {}
    by_rel_src  = {}
    comm_counts = {}
    rel_counts  = {}
    start_time  = datetime.now()
    doc_batch   = []   # raw docs, accumulated until we hit batch_size

    def flush_batch(docs):
        """
        Process one full batch:
          1. Community classification per-doc (cheap regex/dict work)
          2. ONE batched call to classify_religion_batch() for the whole
             batch — this is what makes the ML model fast at 1.7M rows
          3. Build bulk UpdateOne ops and write them in a single round trip
        """
        nonlocal processed
        if not docs:
            return

        community_results = []
        for doc in docs:
            v_name = doc.get(name_field, '')
            r_name = doc.get(relation_field, '') if relation_field else ''
            c_res = classify_community(v_name, r_name)
            if c_res:
                comm, cat, conf, csrc = c_res
            else:
                comm, cat, conf, csrc = 'Unclassified', 'Unknown', '', 'unclassified'
            community_results.append((v_name, r_name, comm, cat, conf, csrc))

        religion_records = [
            {'voter_name': v, 'relation_name': r, 'community': c}
            for (v, r, c, _, _, _) in community_results
        ]
        religion_results = classify_religion_batch(religion_records)

        ops = []
        for doc, (v_name, r_name, comm, cat, conf, csrc), (religion, rsrc, rconf) in zip(
                docs, community_results, religion_results):
            ops.append(UpdateOne(
                {"_id": doc["_id"]},
                {"$set": {
                    "Community"            : comm,
                    "Category"             : cat,
                    "Religion"             : religion,
                    "Religion_Source"      : rsrc,
                    "Religion_Confidence"  : rconf,
                    "Confidence"           : conf,
                    "Classification_Source": csrc,
                    "Classified_At"        : datetime.utcnow(),
                }}
            ))
            comm_counts[comm] = comm_counts.get(comm, 0) + 1
            rel_counts[religion] = rel_counts.get(religion, 0) + 1
            by_comm_src[csrc] = by_comm_src.get(csrc, 0) + 1
            by_rel_src[rsrc]  = by_rel_src.get(rsrc, 0) + 1
            processed += 1

        try:
            collection.bulk_write(ops, ordered=False)
        except PyMongoError as e:
            print(f"\n  Bulk write error: {e}")
            raise

        elapsed = (datetime.now() - start_time).total_seconds()
        rate    = processed / elapsed if elapsed > 0 else 0
        pct     = processed / total * 100
        eta_sec = (total - processed) / rate if rate > 0 else 0
        print(f"  {processed:>9,} / {total:,}  ({pct:5.1f}%)  "
              f"{rate:,.0f} docs/sec  ETA {eta_sec/60:.1f} min", end='\r')

    for doc in cursor:
        doc_batch.append(doc)
        if len(doc_batch) >= batch_size:
            flush_batch(doc_batch)
            doc_batch = []

    flush_batch(doc_batch)   # flush any remaining partial batch

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n\n{'='*78}")
    print(f"  COMPLETE")
    print(f"  Processed       : {processed:,} documents")
    print(f"  Time taken      : {elapsed/60:.1f} minutes  ({processed/elapsed:,.0f} docs/sec)")
    print(f"\n  Community breakdown:")
    for comm, cnt in sorted(comm_counts.items(), key=lambda x: -x[1]):
        bar = chr(9608) * int(cnt / processed * 30)
        print(f"    {comm:<46}  {cnt:>9,}  {bar}")
    print(f"\n  Religion breakdown:")
    for rel, cnt in sorted(rel_counts.items(), key=lambda x: -x[1]):
        bar = chr(9608) * int(cnt / processed * 35)
        print(f"    {rel:<14}  {cnt:>9,}  {bar}")
    print(f"\n  Community classification source:")
    for src, cnt in sorted(by_comm_src.items(), key=lambda x: -x[1]):
        print(f"    {src:<24}  {cnt:>9,}")
    print(f"\n  Religion classification source:")
    for src, cnt in sorted(by_rel_src.items(), key=lambda x: -x[1]):
        print(f"    {src:<24}  {cnt:>9,}")
    print("=" * 78)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 8 — ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Classify voter Community + Religion directly in MongoDB."
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Sample documents and preview results WITHOUT writing to DB')
    parser.add_argument('--sample-size', type=int, default=25,
                        help='Number of documents to sample for dry run (default: 25)')
    parser.add_argument('--full', action='store_true',
                        help='Run full classification on the entire collection')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Documents per bulk_write batch (default: {BATCH_SIZE})')
    parser.add_argument('--resume', dest='resume', action='store_true', default=True,
                        help='Skip documents that already have a Community field (default)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='Re-classify ALL documents, overwriting existing results')
    parser.add_argument('--target', type=str, default=DEFAULT_TARGET,
                        choices=list(TARGETS.keys()),
                        help=f"Which voter list to process: {list(TARGETS.keys())} "
                             f"(default: {DEFAULT_TARGET})")
    parser.add_argument('--uri', type=str, default=None,
                        help='Override MONGO_URI for this run')
    parser.add_argument('--db', type=str, default=None,
                        help='Override the database name for this run')
    parser.add_argument('--collection', type=str, default=None,
                        help='Override the collection name for this run')
    parser.add_argument('--model-path', type=str, default=ML_MODEL_PATH,
                        help=f'Path to the trained .joblib model (default: {ML_MODEL_PATH})')
    parser.add_argument('--no-ml', action='store_true',
                        help='Skip the ML model entirely and use dictionary matching only')
    args = parser.parse_args()

    if not args.dry_run and not args.full:
        print("Specify either --dry-run (recommended first) or --full")
        print("Example:")
        print("  python mongo_voter_classifier.py --dry-run --sample-size 50")
        print("  python mongo_voter_classifier.py --full")
        sys.exit(1)

    target_cfg = TARGETS.get(args.target, {})
    uri        = args.uri or MONGO_URI
    db_name    = args.db or target_cfg.get('db')
    coll_name  = args.collection or target_cfg.get('collection')

    if not db_name or not coll_name:
        print(f"ERROR: no database/collection resolved for target '{args.target}'.")
        print(f"Available targets: {TARGETS}")
        sys.exit(1)

    print(f"Connecting to MongoDB: {uri}")
    print(f"Target    : {args.target}")
    print(f"Database  : {db_name}")
    print(f"Collection: {coll_name}")
    print()

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=8000)
        client.admin.command('ping')   # fail fast if unreachable
    except PyMongoError as e:
        print(f"ERROR: could not connect to MongoDB.\n  {e}")
        sys.exit(1)

    db         = client[db_name]
    collection = db[coll_name]

    print(f"Surname dictionary : {len(SURNAME_DICT):,} entries")
    print(f"Jain names         : {len(JAIN_NAMES):,}  (Jain has no ML class, dictionary only)")
    print(f"Dict fallback sets : Muslim={len(MUSLIM_NAMES):,}  "
          f"Christian={len(CHRISTIAN_NAMES):,}  Hindu={len(HINDU_NAMES):,}  "
          f"(used only if ML model fails to load)")
    print()

    if args.no_ml:
        print("--no-ml set: skipping model load, using dictionary matching for H/M/C.")
    else:
        load_ml_model(args.model_path)
    print()

    if args.dry_run:
        dry_run(collection, NAME_FIELD, RELATION_FIELD, sample_size=args.sample_size)
    elif args.full:
        classify_all(collection, NAME_FIELD, RELATION_FIELD,
                    batch_size=args.batch_size, resume=args.resume)

    client.close()


if __name__ == '__main__':
    main()