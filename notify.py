# notify.py — Twilio SMS alerts for KOFA Battery SOH System

def send_alert(battery_id, soh, soc, faulty=True):
    try:
        from config import TWILIO_SID, TWILIO_TOKEN, TWILIO_NUMBER, RIDER_PHONES
        from twilio.rest import Client

        client = Client(TWILIO_SID, TWILIO_TOKEN)
        phone  = RIDER_PHONES.get(battery_id, "+254797804812")

        if faulty:
            msg = ("KOFA BATTERY ALERT\n"
                   "Battery " + battery_id + " is FAULTY.\n"
                   "SOH: " + str(round(soh,1)) + "% | SOC: " + str(round(soc,1)) + "%\n"
                   "Return to nearest KOFA station immediately.\n"
                   "Do NOT use for your next trip. - KOFA Team")
        else:
            msg = ("KOFA BATTERY CHECK\n"
                   "Battery " + battery_id + " is HEALTHY.\n"
                   "SOH: " + str(round(soh,1)) + "%\n"
                   "Safe to use for your next trip.\n"
                   "Ride safe! - KOFA Team")

        message = client.messages.create(
            body=msg,
            from_=TWILIO_NUMBER,
            to=phone
        )
        return {"status": "sent", "phone": phone, "sid": message.sid}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    print("Testing Twilio SMS ...")
    r = send_alert("KF-B101", soh=67.3, soc=18.2, faulty=True)
    print(r)
