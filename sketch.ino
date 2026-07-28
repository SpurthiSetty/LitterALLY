//////////////////DISTANCE ONLY////////////////////////

// #include "Modulino.h"
// #include <Arduino_RouterBridge.h>

// ModulinoDistance distance;

// // Updated for raw millimeter outputs
// const int MIN_DIST_MM = 20;   // 2 cm minimum (ignores optical blindspot/deadzone)
// const int MAX_DIST_MM = 100;  // 10 cm target threshold (100 mm)

// void setup() {
//     Monitor.begin();
//     Modulino.begin();
//     distance.begin();

//     Monitor.println("--- CALIBRATED DISTANCE TEST ---");
// }

// void loop() {
//     if (distance.available()) {
//         int raw_mm = distance.get();
//         int dist_cm = raw_mm / 10; // Convert mm -> cm

//         Monitor.print("Raw: ");
//         Monitor.print(raw_mm);
//         Monitor.print(" mm  |  Calculated: ");
//         Monitor.print(dist_cm);
//         Monitor.println(" cm");

//         // Object within 2cm to 10cm range check
//         if (raw_mm >= MIN_DIST_MM && raw_mm <= MAX_DIST_MM) {
//             Monitor.println(" -> OBJECT IN 10CM TARGET RANGE!");
//         }
//     }

//     delay(100);
// }


///////////////////DISTANCE AND BUZZER////////////////////
// #include "Modulino.h"
// #include <Arduino_RouterBridge.h>

// ModulinoDistance distance;
// ModulinoBuzzer buzzer;

// const int MIN_DEADZONE_MM = 20;            // 2 cm minimum (optical floor)
// const int DIST_THRESHOLD_MM = 100;         // 10 cm target threshold
// const unsigned long HOLD_TIME_MS = 3000;   // 3 seconds required hold
// const unsigned long POLL_INTERVAL_MS = 100; // 10 Hz refresh rate

// unsigned long lastPollTime = 0;
// unsigned long targetStartTime = 0;
// bool trackingActive = false;
// bool buzzerPlayed = false;

// void setup() {
//     Monitor.begin();
//     delay(1000);

//     // Initialize I2C Bus & Modulinos
//     Modulino.begin();
//     distance.begin();
//     buzzer.begin();

//     Monitor.println("====================================");
//     Monitor.println("  SYSTEM READY: INVERTED SENSING    ");
//     Monitor.println("====================================");
// }

// void loop() {
//     unsigned long currentMillis = millis();

//     // 10 Hz Polling Loop
//     if (currentMillis - lastPollTime >= POLL_INTERVAL_MS) {
//         lastPollTime = currentMillis;

//         if (distance.available()) {
//             int raw_mm = distance.get();
//             int dist_cm = raw_mm / 10;

//             // Live print so you see exact values
//             Monitor.print("Live Distance: ");
//             Monitor.print(dist_cm);
//             Monitor.print(" cm (");
//             Monitor.print(raw_mm);
//             Monitor.println(" mm)");

//             // TARGET CONDITION: Distance is WITHIN 10 cm (between 2cm and 10cm)
//             if (raw_mm >= MIN_DEADZONE_MM && raw_mm <= DIST_THRESHOLD_MM) {
//                 if (!trackingActive) {
//                     trackingActive = true;
//                     targetStartTime = currentMillis;
//                     Monitor.println(">>> Object within 10 cm! Starting 3s hold timer... <<<");
//                 } 
//                 else if ((currentMillis - targetStartTime >= HOLD_TIME_MS) && !buzzerPlayed) {
//                     buzzerPlayed = true;
//                     Monitor.println(">>> 3 SECONDS HELD AT < 10CM! BEEP! <<<");
                    
//                     // Sound tone (1000 Hz for 300 ms)
//                     buzzer.tone(1000, 300);
//                 }
//             } else {
//                 // Object pulled away (> 10 cm) -> Reset state instantly
//                 if (trackingActive && !buzzerPlayed) {
//                     Monitor.println("Object removed early. Resetting 3s timer.");
//                 }
//                 trackingActive = false;
//                 buzzerPlayed = false;
//             }
//         }
//     }
// }




////////////////CAMERABUZZDIST/////////////////////
#include "Modulino.h"
#include <Arduino_RouterBridge.h>

ModulinoDistance distance;
ModulinoBuzzer buzzer;

const int MIN_DEADZONE_MM = 20;            // 2 cm optical floor limit
const int DIST_THRESHOLD_MM = 250;         // 25 cm target threshold (UPDATED)
const unsigned long HOLD_TIME_MS = 3000;   // 3 seconds hold requirement
const unsigned long POLL_INTERVAL_MS = 100; // 10 Hz refresh rate

unsigned long lastPollTime = 0;
unsigned long targetStartTime = 0;
bool trackingActive = false;
bool cameraActive = false;

void setup() {
    Monitor.begin();
    delay(1000);

    Modulino.begin();
    distance.begin();
    buzzer.begin();

    Monitor.println("====================================");
    Monitor.println("  SYSTEM READY: CAM + BUZZ + DIST   ");
    Monitor.println("====================================");
}

void loop() {
    unsigned long currentMillis = millis();

    if (currentMillis - lastPollTime >= POLL_INTERVAL_MS) {
        lastPollTime = currentMillis;

        if (distance.available()) {
            int raw_mm = distance.get();

            // Ignore invalid 0 mm optical noise glitch
            if (raw_mm == 0) {
                return;
            }

            bool isWithinRange = (raw_mm >= MIN_DEADZONE_MM && raw_mm <= DIST_THRESHOLD_MM);

            if (isWithinRange) {
                // Hand/object is currently present (< 25 cm)
                if (!trackingActive) {
                    // Start the 3-second timer
                    trackingActive = true;
                    targetStartTime = currentMillis;
                    Monitor.println(">>> Target detected! Starting 3s timer... <<<");
                } 
                else if ((currentMillis - targetStartTime >= HOLD_TIME_MS) && !cameraActive) {
                    // 3 seconds passed -> Turn on camera
                    cameraActive = true;
                    Monitor.println(">>> 3s HOLD COMPLETE! BEEPING & STARTING CAMERA <<<");
                    
                    buzzer.tone(1000, 300); // Audible beep
                    Bridge.notify("camera_cmd", "START");
                }
            } 
            else {
                // Object moved out of range (> 25 cm or removed)
                if (cameraActive) {
                    // Stop condition: Distance > 25 cm while camera was running
                    Monitor.println(">>> OBJECT MOVED OUT OF RANGE (>25cm): STOPPING CAMERA <<<");
                    Bridge.notify("camera_cmd", "STOP");
                    cameraActive = false;
                }
                
                // Reset tracking state completely
                trackingActive = false;
            }
        }
    }
}
