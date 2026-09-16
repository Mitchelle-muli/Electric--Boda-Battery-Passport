# notify.py — Twilio SMS alerts for KOFA Battery SOH System

def send_alert(battery_id, soh, soc, faulty=True):
    try:
        from config import TWILIO_SID, TWILIO_TOKEN, TWILIO_NUMBER, RIDER_PHONES
        from twilio.rest import Client

        client = Client(TWILIO_SID, TWILIO_TOKEN)
        phone = RIDER_PHONES.get(battery_id, "")

        if faulty:
            msg = (
                "KOFA BATTERY ALERT "
                "Battery " + battery_id + " is FAULTY. "
                "SOH: " + str(round(soh, 1)) + "% SOC: " + str(round(soc, 1)) + "% "
                "Return to KOFA station immediately."
            )
        else:
            msg = (
                "KOFA BATTERY CHECK "
                "Battery " + battery_id + " is HEALTHY. "
                "SOH: " + str(round(soh, 1)) + "% "
                "Safe for your next trip. Ride safe! - KOFA Team"
            )

        message = client.messages.create(
            body=msg,
            from_=TWILIO_NUMBER,
            to=phone,
        )
        return {"status": "sent", "phone": phone, "sid": message.sid}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    print("Testing Twilio SMS ...")
    r = send_alert("KF-B101", soh=67.3, soc=18.2, faulty=True)
    print(r)
