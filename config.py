# config.py — KOFA Battery SOH System configuration
#
# Twilio credentials are read from environment variables, NOT hardcoded.
# Set them before running anything that sends SMS alerts:
#
#   export TWILIO_SID="your_account_sid"
#   export TWILIO_TOKEN="your_auth_token"
#   export TWILIO_NUMBER="+1XXXXXXXXXX"
#
# Or place them in a local .env file (see .env.example) and load it with
# python-dotenv. Never commit real credentials to this file or to .env —
# both .env and any *_secrets.py are listed in .gitignore for that reason.

import os

TWILIO_SID = os.environ.get("TWILIO_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN", "")
TWILIO_NUMBER = os.environ.get("TWILIO_NUMBER", "")

# Per-battery rider phone numbers for SMS alerts.
# Placeholder numbers below — replace with your fleet's real mapping,
# ideally loaded from a config file or database rather than hardcoded here.
RIDER_PHONES = {
    "KF-A100": os.environ.get("RIDER_PHONE_KF_A100", ""),
    "KF-B101": os.environ.get("RIDER_PHONE_KF_B101", ""),
    "KF-C102": os.environ.get("RIDER_PHONE_KF_C102", ""),
    "KF-D103": os.environ.get("RIDER_PHONE_KF_D103", ""),
    "KF-E104": os.environ.get("RIDER_PHONE_KF_E104", ""),
    "KF-F105": os.environ.get("RIDER_PHONE_KF_F105", ""),
    "KF-G106": os.environ.get("RIDER_PHONE_KF_G106", ""),
    "KF-H107": os.environ.get("RIDER_PHONE_KF_H107", ""),
    "KF-I108": os.environ.get("RIDER_PHONE_KF_I108", ""),
    "KF-J109": os.environ.get("RIDER_PHONE_KF_J109", ""),
    "KF-K110": os.environ.get("RIDER_PHONE_KF_K110", ""),
    "KF-L111": os.environ.get("RIDER_PHONE_KF_L111", ""),
}

MANAGER_PHONE = os.environ.get("MANAGER_PHONE", "")
