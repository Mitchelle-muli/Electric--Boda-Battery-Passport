import sys, os, urllib3, ssl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context
import africastalking
from config import AT_USERNAME, AT_API_KEY, RIDER_PHONES

def init_at():
    africastalking.initialize(AT_USERNAME, AT_API_KEY)
    return africastalking.SMS

def send_faulty_alert(battery_id, drain_rate, soh, soc):
    phone = RIDER_PHONES.get(battery_id)
    if not phone:
        return dict(status="error", message="No phone mapped")
    line1 = "KOFA BATTERY ALERT"
    line2 = "Battery " + battery_id + " is FAULTY."
    line3 = "SOH: " + str(round(soh,1)) + "% | SOC: " + str(round(soc,1)) + "%"
    line4 = "Return to nearest KOFA station immediately."
    msg = line1 + "\n" + line2 + "\n" + line3 + "\n" + line4
    try:
        result = init_at().send(msg, [phone])
        return dict(status="sent", phone=phone, response=result)
    except Exception as e:
        return dict(status="error", message=str(e))

def send_healthy_alert(battery_id, soh):
    phone = RIDER_PHONES.get(battery_id)
    if not phone:
        return dict(status="error", message="No phone mapped")
    line1 = "KOFA BATTERY CHECK"
    line2 = "Battery " + battery_id + " is HEALTHY."
    line3 = "SOH: " + str(round(soh,1)) + "% - Safe to use."
    line4 = "Ride safe! - KOFA Team"
    msg = line1 + "\n" + line2 + "\n" + line3 + "\n" + line4
    try:
        result = init_at().send(msg, [phone])
        return dict(status="sent", phone=phone, response=result)
    except Exception as e:
        return dict(status="error", message=str(e))

if __name__ == "__main__":
    print("Testing SMS ...")
    r = send_faulty_alert("KF-B101", drain_rate=22.5, soh=67.3, soc=18.2)
    print(r)
