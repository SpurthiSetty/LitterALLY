// Smart Bin - MCU side.
//
// Owns the Qwiic bus and everything the user sees. Detects an item held in front
// of the sensor, tells the MPU once via on_trigger(), then waits for the MPU to
// push a disposal category back via set_feedback(). If Linux never answers, the
// bin still responds - it just shows "unknown".

#include "Modulino.h"
#include <Arduino_RouterBridge.h>

ModulinoDistance distance;
ModulinoBuzzer buzzer;

const int MIN_DEADZONE_MM = 20;               // optical floor of the ToF sensor
const int DIST_THRESHOLD_MM = 250;            // an item this close counts as presented
const unsigned long HOLD_TIME_MS = 3000;      // how long it must be held still
const unsigned long POLL_INTERVAL_MS = 100;   // 10 Hz
const unsigned long FEEDBACK_TIMEOUT_MS = 4000;
const unsigned long RESULT_HOLD_MS = 5000;    // how long the result stays lit

// Categories must match the keys in disposal_rules.yaml. Colour and tone live
// here, not in the rules file, because the MCU decides how things are displayed.
//
// There is no Modulino Pixels on this build, so the colour signal goes to the
// two onboard RGB LEDs instead. That allows only one bit per channel, which is
// still enough for six distinguishable colours - one per category.
struct CategoryStyle {
  const char* name;
  bool r, g, b;
  unsigned int tone;
};

const CategoryStyle STYLES[] = {
  {"recycle",   false, false, true,   880},  // blue
  {"compost",   false, true,  false,  660},  // green
  {"trash",     true,  true,  true,   440},  // white
  {"hazardous", true,  false, false, 1320},  // red
  {"ewaste",    true,  false, true,  1100},  // magenta
  {"unknown",   true,  true,  false,  220},  // yellow
};
const size_t STYLE_COUNT = sizeof(STYLES) / sizeof(STYLES[0]);
const size_t UNKNOWN_STYLE = STYLE_COUNT - 1;

enum State { IDLE, ARMING, WAITING, SHOWING };

State state = IDLE;
unsigned long lastPollTime = 0;
unsigned long holdStartTime = 0;
unsigned long triggerSentTime = 0;
unsigned long resultShownTime = 0;

bool mpuReady = false;
bool feedbackPending = false;
String pendingCategory = "unknown";

// Bridge handlers only record what arrived; loop() drives the hardware so that
// neither the Qwiic bus nor the LEDs are touched from two contexts at once.
void set_feedback(String category) {
  pendingCategory = category;
  feedbackPending = true;
}

void mpu_ready() {
  mpuReady = true;
}

void setLeds(bool r, bool g, bool b) {
  // The onboard RGB LEDs are active low: LOW lights a channel.
  digitalWrite(LED3_R, r ? LOW : HIGH);
  digitalWrite(LED3_G, g ? LOW : HIGH);
  digitalWrite(LED3_B, b ? LOW : HIGH);
  digitalWrite(LED4_R, r ? LOW : HIGH);
  digitalWrite(LED4_G, g ? LOW : HIGH);
  digitalWrite(LED4_B, b ? LOW : HIGH);
}

void clearLeds() {
  setLeds(false, false, false);
}

const CategoryStyle& styleFor(const String& name) {
  for (size_t i = 0; i < STYLE_COUNT; i++) {
    if (name == STYLES[i].name) {
      return STYLES[i];
    }
  }
  return STYLES[UNKNOWN_STYLE];
}

void showCategory(const String& name) {
  const CategoryStyle& style = styleFor(name);
  setLeds(style.r, style.g, style.b);
  buzzer.tone(style.tone, 200);

  Monitor.print(">>> category: ");
  Monitor.println(name);
}

void setup() {
  Bridge.begin();
  Monitor.begin();
  delay(1000);

  pinMode(LED3_R, OUTPUT);
  pinMode(LED3_G, OUTPUT);
  pinMode(LED3_B, OUTPUT);
  pinMode(LED4_R, OUTPUT);
  pinMode(LED4_G, OUTPUT);
  pinMode(LED4_B, OUTPUT);
  clearLeds();

  Modulino.begin();
  distance.begin();
  buzzer.begin();

  Bridge.provide_safe("set_feedback", set_feedback);
  Bridge.provide_safe("mpu_ready", mpu_ready);

  Monitor.println("====================================");
  Monitor.println("  SMART BIN MCU READY               ");
  Monitor.println("====================================");
}

void loop() {
  unsigned long now = millis();

  if (feedbackPending) {
    feedbackPending = false;
    if (state == WAITING) {
      showCategory(pendingCategory);
      resultShownTime = now;
      state = SHOWING;
    }
  }

  if (state == WAITING && now - triggerSentTime >= FEEDBACK_TIMEOUT_MS) {
    Monitor.println(">>> no answer from Linux, showing unknown");
    showCategory("unknown");
    resultShownTime = now;
    state = SHOWING;
  }

  if (state == SHOWING && now - resultShownTime >= RESULT_HOLD_MS) {
    clearLeds();
    state = IDLE;
  }

  if (now - lastPollTime < POLL_INTERVAL_MS) {
    return;
  }
  lastPollTime = now;

  if (!distance.available()) {
    return;
  }

  int raw_mm = distance.get();
  if (raw_mm == 0) {
    return;  // optical noise glitch, not a real reading
  }

  bool present = (raw_mm >= MIN_DEADZONE_MM && raw_mm <= DIST_THRESHOLD_MM);

  switch (state) {
    case IDLE:
      if (present) {
        holdStartTime = now;
        state = ARMING;
        Monitor.println(">>> item detected, holding...");
      }
      break;

    case ARMING:
      if (!present) {
        state = IDLE;
        Monitor.println("item withdrawn, resetting");
      } else if (now - holdStartTime >= HOLD_TIME_MS) {
        buzzer.tone(1000, 150);
        if (mpuReady) {
          Bridge.notify("on_trigger");
          triggerSentTime = now;
          state = WAITING;
          Monitor.println(">>> held 3s, asked Linux to classify");
        } else {
          Monitor.println(">>> held 3s, but Linux is not up yet");
          showCategory("unknown");
          resultShownTime = now;
          state = SHOWING;
        }
      }
      break;

    case WAITING:
    case SHOWING:
      break;
  }
}
